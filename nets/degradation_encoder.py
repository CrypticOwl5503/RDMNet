from torch import nn
from nets.moco import MoCo
import clip
import torch


class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.SiLU(True),
            nn.Conv2d(out_feat, out_feat, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat)
        )

    def forward(self, x):
        return nn.SiLU(True)(self.backbone(x) + self.shortcut(x))


class ResEncoder(nn.Module):
    def __init__(self, in_channel=3, n_feat=32, reduction=8, use_dce=False):
        super(ResEncoder, self).__init__()
        self.use_dce = use_dce
        self.emb_in = nn.Conv2d(in_channel, n_feat, kernel_size=3, padding=1, bias=False)
        self.E1 = ResBlock(in_feat=n_feat, out_feat=n_feat, stride=2)  # 320x320x32
        self.E2 = ResBlock(in_feat=n_feat, out_feat=n_feat * 2, stride=2)  # 160x160x64
        self.E3 = ResBlock(in_feat=n_feat * 2, out_feat=n_feat * 4, stride=2)  # 80x80
        self.E4 = ResBlock(in_feat=n_feat * 4, out_feat=n_feat * 8, stride=2)  # 40x40
        self.E5 = ResBlock(in_feat=n_feat * 8, out_feat=n_feat * 16, stride=2)  # 20x20

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_feat * 16, n_feat * 8),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feat * 8, 256),
        )
        
        # CLIP setup (only if use_dce is True)
        if self.use_dce:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
            # Freeze CLIP parameters
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # Register hook for intermediate features
            self.target_layer = 'visual.transformer.resblocks.11.ln_2'
            self.intermediate_features = {}
            self._register_clip_hook()

    def _register_clip_hook(self):
        """Register forward hook on CLIP to extract intermediate features"""
        layer = self.clip_model
        for name in self.target_layer.split("."):
            layer = getattr(layer, name)
        
        def hook_fn(module, input, output):
            self.intermediate_features[self.target_layer] = output
        
        layer.register_forward_hook(hook_fn)

    def _extract_clip_features(self, context_images):
        """
        Extract CLIP features from context images.
        
        Args:
            context_images: List of [degrad_context, clean_context] tensors
                           Each tensor: B x 3 x 224 x 224 (preprocessed for CLIP)
        
        Returns:
            features: B x 100 x 768 (concatenated degrad + clean features)
        """
        degrad_context, clean_context = context_images
        B = degrad_context.shape[0]
        
        # Process degrad and clean separately to avoid batch dimension issues
        with torch.no_grad():
            # Process degrad context
            _ = self.clip_model.encode_image(degrad_context)
            degrad_features = self.intermediate_features[self.target_layer].clone()
            
            # Process clean context
            _ = self.clip_model.encode_image(clean_context)
            clean_features = self.intermediate_features[self.target_layer].clone()
        
        # CLIP intermediate features are in [L, B, C] format, transpose to [B, L, C]
        # degrad_features shape: [50, B, 768] -> [B, 50, 768]
        degrad_features = degrad_features.transpose(0, 1)
        clean_features = clean_features.transpose(0, 1)
        
        # Debug: print shapes
        print(f"Debug - degrad_features shape after transpose: {degrad_features.shape}")
        print(f"Debug - clean_features shape after transpose: {clean_features.shape}")
        
        # Ensure both have the same batch size
        assert degrad_features.shape[0] == B, f"Expected batch size {B}, got {degrad_features.shape[0]}"
        assert clean_features.shape[0] == B, f"Expected batch size {B}, got {clean_features.shape[0]}"
        
        # Concatenate along sequence dimension: B x 100 x 768
        # degrad_features: B x 50 x 768, clean_features: B x 50 x 768
        # combined: B x 100 x 768
        combined_features = torch.cat([degrad_features, clean_features], dim=1)
        
        return combined_features

    def forward(self, x, context_images=None):
        """
        Forward pass.
        
        Args:
            x: Input images (B x 3 x H x W)
            context_images: Optional list [degrad_context, clean_context] for DCE
                           Each: B x 3 x 224 x 224 (CLIP-preprocessed)
        
        Returns:
            out: Encoded features (B x 256)
            (l1, l2, l3, l4, l5): Intermediate feature maps
        """
        # Extract CLIP features if DCE is enabled and context is provided
        dce_features = None
        if self.use_dce and context_images is not None:
            dce_features = self._extract_clip_features(context_images)
            # dce_features shape: B x 100 x 768
        
        emb = self.emb_in(x)
        l1 = self.E1(emb)
        l2 = self.E2(l1)
        l3 = self.E3(l2)
        l4 = self.E4(l3)
        l5 = self.E5(l4)
        out = self.mlp(l5)

        return out, (l1, l2, l3, l4, l5)


class UDE(nn.Module):
    def __init__(self, use_dce=False):
        super(UDE, self).__init__()
        self.moco = MoCo(base_encoder=ResEncoder, dim=256, K=12 * 256, use_dce=use_dce)

    def forward(self, x_query, x_key=None, context_images=None):
        if self.training:
            # degradation-aware represenetion learning
            logits, labels, inter = self.moco(x_query, x_key, context_images=context_images)

            return logits, labels, inter
        else:
            # degradation-aware represenetion learning
            inter = self.moco(x_query, context_images=context_images)
            return inter

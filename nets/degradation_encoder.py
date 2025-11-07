from torch import nn
import torch
import torch.nn.functional as F
from nets.moco import MoCo
import clip
from einops import rearrange


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


class ContextAttention(nn.Module):
    """Multi-Head Self-Attention for DCE block"""
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        # x: B x 2L x C
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class DCEBlock(nn.Module):
    """Degradation Context Extraction Block
    Input: CLIP features (B x L x 768) from degrad+clean context pairs
    Output: O_DCE^l ∈ R^{2L × C^l}
    """
    def __init__(self, context_dim=768, output_dim=32, heads=4):
        super(DCEBlock, self).__init__()
        # Proj: context_dim -> output_dim
        self.proj = nn.Linear(context_dim, output_dim)
        # GELU activation
        # LayerNorm
        self.norm = nn.LayerNorm(output_dim)
        # MHSA
        self.mhsa = ContextAttention(dim=output_dim, heads=heads, dim_head=64)

    def forward(self, context_embs):
        # context_embs: B x 2L x 768 (concatenated degrad+clean)
        # Proj
        x = self.proj(context_embs)  # B x 2L x output_dim
        # GELU
        x = F.gelu(x)
        # LayerNorm
        x = self.norm(x)  # B x 2L x output_dim
        # MHSA
        out = self.mhsa(x)  # B x 2L x output_dim
        return out


class DCEModulatedResBlock(nn.Module):
    """ResBlock with DCE-based channel modulation"""
    def __init__(self, in_feat, out_feat, stride=1, dce_output_dim=None):
        super(DCEModulatedResBlock, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dce_output_dim = dce_output_dim if dce_output_dim is not None else out_feat
        
        # Original ResBlock
        self.resblock = ResBlock(in_feat, out_feat, stride)
        
        # FFNN on DCE: 2L × C^l -> C2
        # Assuming 2L=100 (50 degrad + 50 clean from CLIP ViT-B/32)
        self.dce_ffnn = nn.Sequential(
            nn.Linear(100 * self.dce_output_dim, out_feat),  # Flatten 2L dims
            nn.GELU(),
            nn.Linear(out_feat, out_feat)
        )
        
        # C2 separate CNNs: one CNN per channel
        # Each CNN processes n x n x 1 -> scalar
        self.channel_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ) for _ in range(out_feat)
        ])
        
        # FFNN shrink: C2 -> C2/2
        self.shrink_ffnn = nn.Sequential(
            nn.Linear(out_feat, out_feat // 2),
            nn.GELU()
        )
        
        # FFNN expand: C2/2 -> C2
        self.expand_ffnn = nn.Sequential(
            nn.Linear(out_feat // 2, out_feat),
            nn.Sigmoid()  # Use sigmoid for modulation weights
        )

    def forward(self, x, dce_output):
        """
        Args:
            x: n x n x C2 (spatial features)
            dce_output: B x 2L x C^l (DCE output)
        """
        B, C, H, W = x.shape
        
        # 1. FFNN on DCE: 2L × C^l -> C2
        dce_flat = dce_output.reshape(B, -1)  # B x (2L * C^l)
        dce_features = self.dce_ffnn(dce_flat)  # B x C2
        
        # 2. C2 separate CNNs on x: n x n x C2 -> C2
        spatial_features = []
        for i in range(self.out_feat):
            # Extract single channel
            channel_feat = x[:, i:i+1, :, :]  # B x 1 x H x W
            # Process through CNN
            channel_scalar = self.channel_cnns[i](channel_feat)  # B x 1
            spatial_features.append(channel_scalar)
        spatial_features = torch.cat(spatial_features, dim=1)  # B x C2
        
        # 3. Multiply: C2 * C2 -> C2
        multiplied = dce_features * spatial_features  # B x C2
        
        # 4. FFNN shrink: C2 -> C2/2
        shrunk = self.shrink_ffnn(multiplied)  # B x C2/2
        
        # 5. FFNN expand: C2/2 -> C2
        modulation_weights = self.expand_ffnn(shrunk)  # B x C2
        
        # 6. Channel-wise multiply: (n x n x C2) ⊙ C2 -> n x n x C2
        modulation_weights = modulation_weights.unsqueeze(-1).unsqueeze(-1)  # B x C2 x 1 x 1
        modulated_x = x * modulation_weights  # B x C x H x W
        
        # 7. Pass to ResBlock
        out = self.resblock(modulated_x)
        return out


class ResEncoder(nn.Module):
    def __init__(self, in_channel=3, n_feat=32, reduction=8, use_dce=False):
        super(ResEncoder, self).__init__()
        self.use_dce = use_dce
        self.n_feat = n_feat
        
        self.emb_in = nn.Conv2d(in_channel, n_feat, kernel_size=3, padding=1, bias=False)
        
        if use_dce:
            # Initialize CLIP model
            self.clip, _ = clip.load("ViT-B/32", device="cpu")
            # Freeze CLIP parameters
            for param in self.clip.parameters():
                param.requires_grad = False
            
            # Register hook for intermediate features
            self.target_layers = ['visual.transformer.resblocks.11.ln_2']
            self.intermediate_features = {}
            
            hooks = []
            for layer_name in self.target_layers:
                layer = self.clip
                for name in layer_name.split("."):
                    layer = getattr(layer, name)
                hook = layer.register_forward_hook(
                    lambda module, input, output, name=layer_name: self.hook_fn(module, input, output, name))
                hooks.append(hook)
            
            # DCE blocks for each ResBlock level
            # Output dimensions: 32, 64, 128, 256, 512
            self.dce1 = DCEBlock(context_dim=768, output_dim=n_feat, heads=4)      # 32
            self.dce2 = DCEBlock(context_dim=768, output_dim=n_feat * 2, heads=4)   # 64
            self.dce3 = DCEBlock(context_dim=768, output_dim=n_feat * 4, heads=4)   # 128
            self.dce4 = DCEBlock(context_dim=768, output_dim=n_feat * 8, heads=4)   # 256
            self.dce5 = DCEBlock(context_dim=768, output_dim=n_feat * 16, heads=4) # 512
            
            # Use DCEModulatedResBlocks
            self.E1 = DCEModulatedResBlock(in_feat=n_feat, out_feat=n_feat, stride=2, dce_output_dim=n_feat)
            self.E2 = DCEModulatedResBlock(in_feat=n_feat, out_feat=n_feat * 2, stride=2, dce_output_dim=n_feat * 2)
            self.E3 = DCEModulatedResBlock(in_feat=n_feat * 2, out_feat=n_feat * 4, stride=2, dce_output_dim=n_feat * 4)
            self.E4 = DCEModulatedResBlock(in_feat=n_feat * 4, out_feat=n_feat * 8, stride=2, dce_output_dim=n_feat * 8)
            self.E5 = DCEModulatedResBlock(in_feat=n_feat * 8, out_feat=n_feat * 16, stride=2, dce_output_dim=n_feat * 16)
        else:
            # Original ResBlocks (backward compatibility)
            self.E1 = ResBlock(in_feat=n_feat, out_feat=n_feat, stride=2)
            self.E2 = ResBlock(in_feat=n_feat, out_feat=n_feat * 2, stride=2)
            self.E3 = ResBlock(in_feat=n_feat * 2, out_feat=n_feat * 4, stride=2)
            self.E4 = ResBlock(in_feat=n_feat * 4, out_feat=n_feat * 8, stride=2)
            self.E5 = ResBlock(in_feat=n_feat * 8, out_feat=n_feat * 16, stride=2)

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_feat * 16, n_feat * 8),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feat * 8, 256),
        )

    def hook_fn(self, module, input, output, layer_name):
        """Hook function to capture CLIP intermediate features"""
        self.intermediate_features[layer_name] = output

    def forward(self, x, context_pairs=None):
        """
        Args:
            x: Input image tensor
            context_pairs: Tuple of (degrad_context, clean_context) tensors, each B x 3 x 224 x 224
        """
        emb = self.emb_in(x)
        
        if self.use_dce and context_pairs is not None:
            degrad_context, clean_context = context_pairs
            
            # Process context pairs through CLIP
            # Concatenate degrad and clean contexts
            context_combined = torch.cat([degrad_context, clean_context], dim=0)  # 2B x 3 x 224 x 224
            
            # Encode through CLIP
            with torch.no_grad():
                _ = self.clip.encode_image(context_combined)
            
            # Get intermediate features
            clip_features = self.intermediate_features[self.target_layers[0]]  # 2B x 50 x 768
            
            # Split back to degrad and clean
            B = degrad_context.shape[0]
            degrad_features = clip_features[:B]  # B x 50 x 768
            clean_features = clip_features[B:]    # B x 50 x 768
            
            # Concatenate along sequence dimension: B x 100 x 768
            context_embs = torch.cat([degrad_features, clean_features], dim=1)
            
            # Process through DCE blocks
            dce1_out = self.dce1(context_embs)  # B x 100 x 32
            dce2_out = self.dce2(context_embs)  # B x 100 x 64
            dce3_out = self.dce3(context_embs)  # B x 100 x 128
            dce4_out = self.dce4(context_embs)  # B x 100 x 256
            dce5_out = self.dce5(context_embs)  # B x 100 x 512
            
            # Forward through modulated ResBlocks
            l1 = self.E1(emb, dce1_out)
            l2 = self.E2(l1, dce2_out)
            l3 = self.E3(l2, dce3_out)
            l4 = self.E4(l3, dce4_out)
            l5 = self.E5(l4, dce5_out)
        else:
            # Original forward pass
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
        self.use_dce = use_dce
        self.moco = MoCo(base_encoder=lambda **kwargs: ResEncoder(**kwargs, use_dce=use_dce), dim=256, K=12 * 256)

    def forward(self, x_query, x_key=None, context_pairs=None):
        if self.training:
            # degradation-aware representation learning
            logits, labels, inter = self.moco(x_query, x_key, context_pairs)
            return logits, labels, inter
        else:
            # degradation-aware representation learning
            inter = self.moco(x_query, context_pairs=context_pairs)
            return inter
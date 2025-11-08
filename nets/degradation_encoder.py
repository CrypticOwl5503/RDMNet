from torch import nn
from nets.moco import MoCo
import clip
import torch
import torch.nn.functional as F


class DCEBlock(nn.Module):
    """
    Degradation Context Extraction Block.
    Implements: Proj -> GELU -> LayerNorm -> Multi-Head Self-Attention
    
    Input: B x 2L x context_dim (CLIP features: 768)
    Output: B x 2L x output_dim (C^l for each ResBlock level)
    """
    def __init__(self, context_dim=768, output_dim=32, num_heads=8):
        super(DCEBlock, self).__init__()
        self.context_dim = context_dim
        self.output_dim = output_dim
        
        # Projection: context_dim -> output_dim
        self.proj = nn.Linear(context_dim, output_dim, bias=False)
        
        # Multi-Head Self-Attention
        # Using PyTorch's built-in MultiheadAttention
        # Note: PyTorch expects (seq_len, batch, embed_dim) format
        self.mhsa = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=False  # We'll handle transpose manually
        )
        
        # Layer Normalization (applied before MHSA)
        self.ln = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (B x 2L x context_dim)
        
        Returns:
            out: DCE output (B x 2L x output_dim)
        """
        # Proj: B x 2L x context_dim -> B x 2L x output_dim
        x = self.proj(x)
        
        # GELU activation
        x = F.gelu(x)
        
        # Layer Normalization
        x = self.ln(x)
        
        # Multi-Head Self-Attention
        # PyTorch MultiheadAttention expects (seq_len, batch, embed_dim)
        # So we transpose: B x 2L x output_dim -> 2L x B x output_dim
        x_transposed = x.transpose(0, 1)  # (2L, B, output_dim)
        
        # Self-attention: query, key, value are all the same
        attn_out, _ = self.mhsa(x_transposed, x_transposed, x_transposed)
        
        # Transpose back: 2L x B x output_dim -> B x 2L x output_dim
        out = attn_out.transpose(0, 1)
        
        return out


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
            
            # DCE blocks for each ResBlock level
            # Each DCE block processes CLIP features (B x 100 x 768) -> (B x 100 x C^l)
            self.dce1 = DCEBlock(context_dim=768, output_dim=n_feat, num_heads=8)        # 32
            self.dce2 = DCEBlock(context_dim=768, output_dim=n_feat * 2, num_heads=8)    # 64
            self.dce3 = DCEBlock(context_dim=768, output_dim=n_feat * 4, num_heads=8)    # 128
            self.dce4 = DCEBlock(context_dim=768, output_dim=n_feat * 8, num_heads=8)    # 256
            self.dce5 = DCEBlock(context_dim=768, output_dim=n_feat * 16, num_heads=8)  # 512
            
            # Replace ResBlocks with DCEModulatedResBlocks
            self.E1 = DCEModulatedResBlock(in_feat=n_feat, out_feat=n_feat, stride=2, 
                                          dce_dim=n_feat, dce_seq_len=100)  # 320x320x32
            self.E2 = DCEModulatedResBlock(in_feat=n_feat, out_feat=n_feat * 2, stride=2,
                                          dce_dim=n_feat * 2, dce_seq_len=100)  # 160x160x64
            self.E3 = DCEModulatedResBlock(in_feat=n_feat * 2, out_feat=n_feat * 4, stride=2,
                                          dce_dim=n_feat * 4, dce_seq_len=100)  # 80x80x128
            self.E4 = DCEModulatedResBlock(in_feat=n_feat * 4, out_feat=n_feat * 8, stride=2,
                                          dce_dim=n_feat * 8, dce_seq_len=100)  # 40x40x256
            self.E5 = DCEModulatedResBlock(in_feat=n_feat * 8, out_feat=n_feat * 16, stride=2,
                                          dce_dim=n_feat * 16, dce_seq_len=100)  # 20x20x512
        else:
            # Original ResBlocks (backward compatibility)
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
        degrad_features = degrad_features.transpose(0, 1)
        clean_features = clean_features.transpose(0, 1)
        
        # Ensure both have the same batch size
        assert degrad_features.shape[0] == B, f"Expected batch size {B}, got {degrad_features.shape[0]}"
        assert clean_features.shape[0] == B, f"Expected batch size {B}, got {clean_features.shape[0]}"
        
        # Concatenate along sequence dimension: B x 100 x 768
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
        # Extract CLIP features and process through DCE blocks if DCE is enabled
        dce_outputs = None
        if self.use_dce and context_images is not None:
            # Extract CLIP features: B x 100 x 768
            clip_features = self._extract_clip_features(context_images)
            
            # Process through DCE blocks to get DCE outputs for each level
            dce_outputs = {
                'dce1': self.dce1(clip_features),  # B x 100 x 32
                'dce2': self.dce2(clip_features),   # B x 100 x 64
                'dce3': self.dce3(clip_features),   # B x 100 x 128
                'dce4': self.dce4(clip_features),   # B x 100 x 256
                'dce5': self.dce5(clip_features),    # B x 100 x 512
            }
        
        emb = self.emb_in(x)
        
        # Forward through ResBlocks with DCE modulation
        if self.use_dce and dce_outputs is not None:
            l1 = self.E1(emb, dce_output=dce_outputs['dce1'])
            l2 = self.E2(l1, dce_output=dce_outputs['dce2'])
            l3 = self.E3(l2, dce_output=dce_outputs['dce3'])
            l4 = self.E4(l3, dce_output=dce_outputs['dce4'])
            l5 = self.E5(l4, dce_output=dce_outputs['dce5'])
        else:
            # No DCE, use original behavior
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


class DCEModulatedResBlock(nn.Module):
    """
    ResBlock with DCE-based channel-wise modulation.
    
    Architecture:
    1. FFNN on DCE: 2L × C1 -> C2
    2. C2 CNNs on x: n × n × C2 -> C2 (one CNN per channel)
    3. Multiply: C2 × C2 -> C2
    4. FFNN shrink: C2 -> C2/2
    5. FFNN expand: C2/2 -> C2
    6. Channel-wise multiply: (n × n × C2) ⊙ C2 -> n × n × C2
    7. Pass to ResBlock
    """
    def __init__(self, in_feat, out_feat, stride=1, dce_dim=None, dce_seq_len=100):
        """
        Args:
            in_feat: Input channels
            out_feat: Output channels (C2)
            stride: Stride for ResBlock
            dce_dim: DCE output dimension (C1), if None, modulation is disabled
            dce_seq_len: Sequence length of DCE output (2L, default 100)
        """
        super(DCEModulatedResBlock, self).__init__()
        self.out_feat = out_feat
        self.use_dce = (dce_dim is not None)
        
        # Original ResBlock
        self.resblock = ResBlock(in_feat, out_feat, stride)
        
        if self.use_dce:
            # 1. FFNN on DCE: reduce sequence dimension and project to C2
            # Input: B × 2L × C1, we want to get C2 scalars
            # Strategy: mean pool over sequence, then project
            self.dce_pool = nn.AdaptiveAvgPool1d(1)  # Reduces 2L dimension
            self.dce_proj = nn.Linear(dce_dim, out_feat)  # C1 -> C2
            
            # 2. C2 separate CNNs: one per channel
            # Each CNN takes one channel and outputs a scalar
            self.channel_cnns = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
                    nn.AdaptiveAvgPool2d(1),  # n × n -> 1 × 1
                    nn.Flatten()  # -> scalar
                ) for _ in range(out_feat)
            ])
            
            # 3. Multiply (element-wise, handled in forward)
            
            # 4. FFNN shrink: C2 -> C2/2
            self.shrink = nn.Sequential(
                nn.Linear(out_feat, out_feat // 2),
                nn.ReLU(inplace=True)
            )
            
            # 5. FFNN expand: C2/2 -> C2
            self.expand = nn.Sequential(
                nn.Linear(out_feat // 2, out_feat),
                nn.Sigmoid()  # Use sigmoid to get modulation factors in [0, 1]
            )
    
    def forward(self, x, dce_output=None):
        """
        Forward pass.
        
        Args:
            x: Input features (B × C_in × n × n)
            dce_output: Optional DCE output (B × 2L × C1)
        
        Returns:
            Output of ResBlock (B × C2 × n × n)
        """
        if not self.use_dce or dce_output is None:
            # No DCE modulation, use original ResBlock
            return self.resblock(x)
        
        B, C_in, H, W = x.shape
        C2 = self.out_feat
        
        # 1. Get C2 inputs from DCE: B × 2L × C1 -> B × C2
        # Transpose for pooling: B × 2L × C1 -> B × C1 × 2L
        dce_transposed = dce_output.transpose(1, 2)  # B × C1 × 2L
        # Pool over sequence: B × C1 × 2L -> B × C1 × 1
        dce_pooled = self.dce_pool(dce_transposed).squeeze(-1)  # B × C1
        # Project to C2: B × C1 -> B × C2
        dce_proj = self.dce_proj(dce_pooled)  # B × C2
        
        # 2. Get C2 inputs from spatial features: B × C2 × n × n -> B × C2
        # Note: x might have C_in channels, but we need C2 channels
        # We'll process the input channels, or if C_in == C2, use directly
        if C_in == C2:
            x_for_cnn = x
        else:
            # If channels don't match, we need to handle this
            # For now, assume we process what we have
            x_for_cnn = x
        
        # Apply C2 separate CNNs (one per output channel)
        cnn_outputs = []
        for i in range(C2):
            if i < C_in:
                # Extract channel i: B × 1 × n × n
                channel_i = x_for_cnn[:, i:i+1, :, :]
                # Apply CNN: B × 1 × n × n -> B × 1
                cnn_out = self.channel_cnns[i](channel_i)
                cnn_outputs.append(cnn_out)
            else:
                # If C2 > C_in, pad with zeros or use last channel
                channel_i = x_for_cnn[:, -1:, :, :]
                cnn_out = self.channel_cnns[i](channel_i)
                cnn_outputs.append(cnn_out)
        
        # Stack: B × C2
        spatial_proj = torch.cat(cnn_outputs, dim=1)  # B × C2
        
        # 3. Multiply: B × C2
        multiplied = dce_proj * spatial_proj  # B × C2
        
        # 4. FFNN shrink: B × C2 -> B × C2/2
        shrunk = self.shrink(multiplied)  # B × C2/2
        
        # 5. FFNN expand: B × C2/2 -> B × C2
        modulation_factors = self.expand(shrunk)  # B × C2
        
        # 6. Channel-wise multiply: (B × C_in × n × n) ⊙ (B × C2 × 1 × 1)
        # Reshape modulation_factors: B × C2 -> B × C2 × 1 × 1
        modulation_factors = modulation_factors.view(B, C2, 1, 1)
        
        # If C_in != C2, we need to project x first
        if C_in != C2:
            # Use a simple 1x1 conv to match channels
            if not hasattr(self, 'channel_proj'):
                self.channel_proj = nn.Conv2d(C_in, C2, kernel_size=1, bias=False).to(x.device)
            x_proj = self.channel_proj(x)
        else:
            x_proj = x
        
        # Channel-wise multiplication
        modulated_x = x_proj * modulation_factors  # B × C2 × n × n
        
        # 7. Pass to ResBlock
        # Note: ResBlock expects C_in input, but we have C2
        # We need to handle this - either modify ResBlock input or use a projection
        # For now, let's assume we pass modulated_x which has C2 channels
        # But ResBlock expects in_feat channels...
        # Actually, we should modulate the input to ResBlock, not replace it
        # Let me reconsider: we modulate x, then pass to ResBlock
        
        # Actually, looking at the requirement again:
        # "multiply with C2 channels of n x n of the original degradation encoder RB input"
        # So we modulate the input, then pass to ResBlock
        # But ResBlock needs in_feat channels, so we need to ensure compatibility
        
        # For now, let's create a wrapper that handles channel mismatch
        if C_in == C2:
            # Direct modulation
            output = self.resblock(modulated_x)
        else:
            # Need to handle channel mismatch
            # Option 1: Project modulated_x back to C_in
            # Option 2: Create a ResBlock that accepts C2
            # For simplicity, let's project back
            if not hasattr(self, 'channel_proj_back'):
                self.channel_proj_back = nn.Conv2d(C2, C_in, kernel_size=1, bias=False).to(x.device)
            modulated_x_back = self.channel_proj_back(modulated_x)
            output = self.resblock(modulated_x_back)
        
        return output

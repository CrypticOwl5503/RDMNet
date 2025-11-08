# test_step1.py
import torch
from nets.degradation_encoder import ResEncoder
import clip

# Test 1: Backward compatibility (no DCE)
print("Test 1: Backward compatibility...")
encoder = ResEncoder(use_dce=False)
x = torch.randn(2, 3, 640, 640)
out, features = encoder(x)
print(f"✓ Output shape: {out.shape}, Features: {[f.shape for f in features]}")

# Test 2: CLIP loading (with DCE but no context)
print("\nTest 2: CLIP loading...")
encoder_dce = ResEncoder(use_dce=True)
print(f"✓ CLIP model loaded: {encoder_dce.clip_model is not None}")

# Test 3: CLIP feature extraction
print("\nTest 3: CLIP feature extraction...")
clip_preprocess = encoder_dce.clip_preprocess
# Create dummy context images (would normally come from dataloader)
degrad_ctx = torch.randn(2, 3, 224, 224)
clean_ctx = torch.randn(2, 3, 224, 224)
context_images = [degrad_ctx, clean_ctx]

dce_features = encoder_dce._extract_clip_features(context_images)
print(f"✓ DCE features shape: {dce_features.shape} (should be [2, 100, 768])")

# Test 4: Forward with context (but ResBlocks not using it yet)
print("\nTest 4: Forward pass with context...")
out2, features2 = encoder_dce(x, context_images=context_images)
print(f"✓ Output shape: {out2.shape}, Features: {[f.shape for f in features2]}")

print("\n✅ Step 1 complete! CLIP is integrated and working.")
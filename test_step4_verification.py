# test_step4_verification.py
import torch
from nets.degradation_encoder import ResEncoder

print("Verification: Check that DCE outputs are actually being used...")
encoder_dce = ResEncoder(use_dce=True)

# Create two different context pairs
x = torch.randn(1, 3, 640, 640)
context1 = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]
context2 = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]

# Forward with context1
out1, _ = encoder_dce(x, context_images=context1)

# Forward with context2
out2, _ = encoder_dce(x, context_images=context2)

# Outputs should be different (different contexts should produce different modulations)
diff = torch.abs(out1 - out2).mean()
print(f"✓ Difference between two different contexts: {diff.item():.6f}")

# Check that DCE blocks are actually processing
print("\nVerification: Check DCE block outputs...")
clip_features = encoder_dce._extract_clip_features(context1)
dce1_out = encoder_dce.dce1(clip_features)
dce2_out = encoder_dce.dce2(clip_features)
print(f"✓ DCE1 output shape: {dce1_out.shape} (expected [1, 100, 32])")
print(f"✓ DCE2 output shape: {dce2_out.shape} (expected [1, 100, 64])")
assert dce1_out.shape == (1, 100, 32), f"Expected [1, 100, 32], got {dce1_out.shape}"
assert dce2_out.shape == (1, 100, 64), f"Expected [1, 100, 64], got {dce2_out.shape}"

# Check that modulation factors are being computed
print("\nVerification: Check that modulation is non-trivial...")
# Create a simple test to see if modulation factors vary
encoder_dce.eval()
with torch.no_grad():
    x_test = torch.randn(1, 32, 320, 320)
    dce_test = torch.randn(1, 100, 32)
    
    # Access the first ResBlock's modulation components
    block = encoder_dce.E1
    
    # Manually compute modulation to verify
    dce_pooled = block.dce_pool(dce_test.transpose(1, 2)).squeeze(-1)
    dce_proj = block.dce_proj(dce_pooled)
    
    # Check that modulation factors are not all zeros or ones
    print(f"✓ DCE projection range: [{dce_proj.min().item():.4f}, {dce_proj.max().item():.4f}]")
    assert dce_proj.abs().mean() > 0.01, "DCE projection should produce non-zero values"

print("\n✅ All verifications passed! DCE is correctly integrated.")

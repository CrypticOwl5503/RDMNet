# test_step4.py
import torch
from nets.degradation_encoder import ResEncoder

print("Test 1: Backward compatibility (no DCE)...")
encoder = ResEncoder(use_dce=False)
x = torch.randn(2, 3, 640, 640)
out, features = encoder(x)
print(f"✓ Output shape: {out.shape}")
print(f"✓ Features shapes: {[f.shape for f in features]}")
assert out.shape == (2, 256), f"Expected [2, 256], got {out.shape}"

print("\nTest 2: ResEncoder with DCE (no context - should work like original)...")
encoder_dce = ResEncoder(use_dce=True)
x = torch.randn(2, 3, 640, 640)
out, features = encoder_dce(x, context_images=None)
print(f"✓ Output shape: {out.shape}")
print(f"✓ Features shapes: {[f.shape for f in features]}")
assert out.shape == (2, 256), f"Expected [2, 256], got {out.shape}"

print("\nTest 3: ResEncoder with DCE and context...")
# Create dummy context images (CLIP-preprocessed: 224x224)
degrad_ctx = torch.randn(2, 3, 224, 224)
clean_ctx = torch.randn(2, 3, 224, 224)
context_images = [degrad_ctx, clean_ctx]

x = torch.randn(2, 3, 640, 640)
out, features = encoder_dce(x, context_images=context_images)
print(f"✓ Output shape: {out.shape}")
print(f"✓ Features shapes: {[f.shape for f in features]}")
assert out.shape == (2, 256), f"Expected [2, 256], got {out.shape}"
assert features[0].shape == (2, 32, 320, 320), f"Expected [2, 32, 320, 320], got {features[0].shape}"
assert features[1].shape == (2, 64, 160, 160), f"Expected [2, 64, 160, 160], got {features[1].shape}"
assert features[2].shape == (2, 128, 80, 80), f"Expected [2, 128, 80, 80], got {features[2].shape}"
assert features[3].shape == (2, 256, 40, 40), f"Expected [2, 256, 40, 40], got {features[3].shape}"
assert features[4].shape == (2, 512, 20, 20), f"Expected [2, 512, 20, 20], got {features[4].shape}"

print("\nTest 4: Gradient flow test...")
x = torch.randn(2, 3, 640, 640, requires_grad=True)
degrad_ctx = torch.randn(2, 3, 224, 224)
clean_ctx = torch.randn(2, 3, 224, 224)
context_images = [degrad_ctx, clean_ctx]

out, features = encoder_dce(x, context_images=context_images)
loss = out.mean()
loss.backward()
print(f"✓ Gradients computed successfully")
print(f"✓ Input gradient shape: {x.grad.shape}")

print("\nTest 5: Compare outputs with and without DCE...")
# Same input
x = torch.randn(2, 3, 640, 640)
degrad_ctx = torch.randn(2, 3, 224, 224)
clean_ctx = torch.randn(2, 3, 224, 224)
context_images = [degrad_ctx, clean_ctx]

# Without DCE
out_no_dce, _ = encoder(x)

# With DCE but no context
out_dce_no_ctx, _ = encoder_dce(x, context_images=None)

# With DCE and context
out_dce_ctx, _ = encoder_dce(x, context_images=context_images)

print(f"✓ Output without DCE: {out_no_dce.shape}")
print(f"✓ Output with DCE (no context): {out_dce_no_ctx.shape}")
print(f"✓ Output with DCE (with context): {out_dce_ctx.shape}")

# Outputs should be different when context is provided
diff = torch.abs(out_dce_ctx - out_dce_no_ctx).mean()
print(f"✓ Mean absolute difference (with vs without context): {diff.item():.6f}")
assert diff.item() > 1e-6, "DCE should produce different outputs when context is provided"

print("\n✅ Step 4 complete! DCE is integrated into ResEncoder.")

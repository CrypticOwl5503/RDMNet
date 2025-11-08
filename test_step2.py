# test_step2.py
import torch
from nets.degradation_encoder import DCEBlock

print("Test 1: DCEBlock initialization...")
# Test with different output dimensions (matching ResBlock channels)
dce1 = DCEBlock(context_dim=768, output_dim=32, num_heads=8)   # For E1
dce2 = DCEBlock(context_dim=768, output_dim=64, num_heads=8)   # For E2
dce3 = DCEBlock(context_dim=768, output_dim=128, num_heads=8)  # For E3
dce4 = DCEBlock(context_dim=768, output_dim=256, num_heads=8)  # For E4
dce5 = DCEBlock(context_dim=768, output_dim=512, num_heads=8)  # For E5
print("✓ All DCEBlocks initialized successfully")

print("\nTest 2: DCEBlock forward pass...")
# Simulate CLIP features: B=2, 2L=100, context_dim=768
clip_features = torch.randn(2, 100, 768)

# Test each DCE block
output1 = dce1(clip_features)
print(f"✓ DCE1 output shape: {output1.shape} (expected [2, 100, 32])")
assert output1.shape == (2, 100, 32), f"Expected [2, 100, 32], got {output1.shape}"

output2 = dce2(clip_features)
print(f"✓ DCE2 output shape: {output2.shape} (expected [2, 100, 64])")
assert output2.shape == (2, 100, 64), f"Expected [2, 100, 64], got {output2.shape}"

output3 = dce3(clip_features)
print(f"✓ DCE3 output shape: {output3.shape} (expected [2, 100, 128])")
assert output3.shape == (2, 100, 128), f"Expected [2, 100, 128], got {output3.shape}"

output4 = dce4(clip_features)
print(f"✓ DCE4 output shape: {output4.shape} (expected [2, 100, 256])")
assert output4.shape == (2, 100, 256), f"Expected [2, 100, 256], got {output4.shape}"

output5 = dce5(clip_features)
print(f"✓ DCE5 output shape: {output5.shape} (expected [2, 100, 512])")
assert output5.shape == (2, 100, 512), f"Expected [2, 100, 512], got {output5.shape}"

print("\nTest 3: Gradient flow test...")
# Test that gradients can flow through
clip_features.requires_grad = True
output = dce1(clip_features)
loss = output.mean()
loss.backward()
print(f"✓ Gradients computed successfully")
print(f"✓ Input gradient shape: {clip_features.grad.shape}")

print("\n✅ Step 2 complete! DCEBlock is working correctly.")

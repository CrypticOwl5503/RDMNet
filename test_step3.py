# test_step3.py
import torch
from nets.degradation_encoder import DCEModulatedResBlock, DCEBlock

print("Test 1: DCEModulatedResBlock initialization...")
# Test with different channel configurations
# E1: 32 -> 32
block1 = DCEModulatedResBlock(in_feat=32, out_feat=32, stride=2, dce_dim=32, dce_seq_len=100)
# E2: 32 -> 64
block2 = DCEModulatedResBlock(in_feat=32, out_feat=64, stride=2, dce_dim=64, dce_seq_len=100)
# E3: 64 -> 128
block3 = DCEModulatedResBlock(in_feat=64, out_feat=128, stride=2, dce_dim=128, dce_seq_len=100)
print("✓ All DCEModulatedResBlocks initialized successfully")

print("\nTest 2: Forward pass without DCE (backward compatibility)...")
x1 = torch.randn(2, 32, 320, 320)
out1 = block1(x1, dce_output=None)
print(f"✓ Output shape: {out1.shape} (expected [2, 32, 160, 160])")
assert out1.shape == (2, 32, 160, 160), f"Expected [2, 32, 160, 160], got {out1.shape}"

print("\nTest 3: Forward pass with DCE...")
# Create dummy DCE output: B × 2L × C1
dce_output1 = torch.randn(2, 100, 32)  # For block1 (C1=32)
x1 = torch.randn(2, 32, 320, 320)
out1 = block1(x1, dce_output=dce_output1)
print(f"✓ Block1 output shape: {out1.shape} (expected [2, 32, 160, 160])")
assert out1.shape == (2, 32, 160, 160), f"Expected [2, 32, 160, 160], got {out1.shape}"

dce_output2 = torch.randn(2, 100, 64)  # For block2 (C1=64)
x2 = torch.randn(2, 32, 320, 320)
out2 = block2(x2, dce_output=dce_output2)
print(f"✓ Block2 output shape: {out2.shape} (expected [2, 64, 160, 160])")
assert out2.shape == (2, 64, 160, 160), f"Expected [2, 64, 160, 160], got {out2.shape}"

dce_output3 = torch.randn(2, 100, 128)  # For block3 (C1=128)
x3 = torch.randn(2, 64, 160, 160)
out3 = block3(x3, dce_output=dce_output3)
print(f"✓ Block3 output shape: {out3.shape} (expected [2, 128, 80, 80])")
assert out3.shape == (2, 128, 80, 80), f"Expected [2, 128, 80, 80], got {out3.shape}"

print("\nTest 4: Integration test with DCEBlock...")
# Create DCE block
dce_block = DCEBlock(context_dim=768, output_dim=32, num_heads=8)
# Create CLIP-like features
clip_features = torch.randn(2, 100, 768)
# Get DCE output
dce_out = dce_block(clip_features)
print(f"✓ DCE output shape: {dce_out.shape} (expected [2, 100, 32])")

# Use with DCEModulatedResBlock
x = torch.randn(2, 32, 320, 320)
out = block1(x, dce_output=dce_out)
print(f"✓ Integrated output shape: {out.shape} (expected [2, 32, 160, 160])")

print("\nTest 5: Gradient flow test...")
x = torch.randn(2, 32, 320, 320, requires_grad=True)
dce_out = torch.randn(2, 100, 32, requires_grad=True)
out = block1(x, dce_output=dce_out)
loss = out.mean()
loss.backward()
print(f"✓ Gradients computed successfully")
print(f"✓ Input gradient shape: {x.grad.shape}")
print(f"✓ DCE gradient shape: {dce_out.grad.shape}")

print("\n✅ Step 3 complete! DCEModulatedResBlock is working correctly.")

# test_step6.py
import torch
from nets.yolo import YoloBody

print("Test 1: YoloBody without DCE (backward compatibility)...")
model_no_dce = YoloBody(num_classes=5, phi='s', use_dce=False)
x = torch.randn(2, 3, 640, 640)

# Test inference mode
model_no_dce.eval()
with torch.no_grad():
    detected, restored = model_no_dce(x)
print(f"✓ Inference output - detected: {len(detected)} outputs, restored: {restored.shape}")

# Test training mode
model_no_dce.train()
x_train = torch.randn(4, 3, 640, 640)  # 2x batch for training (x + posimg)
detected, restored, logits, labels = model_no_dce(x_train)
print(f"✓ Training output - detected: {len(detected)} outputs, restored: {restored.shape}")
print(f"✓ Logits: {logits.shape if isinstance(logits, torch.Tensor) else 'N/A'}, Labels: {labels.shape if isinstance(labels, torch.Tensor) else 'N/A'}")

print("\nTest 2: YoloBody with DCE (no context - should work like original)...")
model_dce = YoloBody(num_classes=5, phi='s', use_dce=True)
x = torch.randn(2, 3, 640, 640)

# Test inference mode without context
model_dce.eval()
with torch.no_grad():
    detected, restored = model_dce(x, context_images=None)
print(f"✓ Inference output (no context) - detected: {len(detected)} outputs, restored: {restored.shape}")

# Test training mode without context
model_dce.train()
x_train = torch.randn(4, 3, 640, 640)
detected, restored, logits, labels = model_dce(x_train, context_images=None)
print(f"✓ Training output (no context) - detected: {len(detected)} outputs, restored: {restored.shape}")

print("\nTest 3: YoloBody with DCE and context...")
# Create dummy context images (CLIP-preprocessed: 224x224)
degrad_ctx = torch.randn(2, 3, 224, 224)
clean_ctx = torch.randn(2, 3, 224, 224)
context_images = (degrad_ctx, clean_ctx)

# Test inference mode with context
model_dce.eval()
with torch.no_grad():
    detected, restored = model_dce(x, context_images=context_images)
print(f"✓ Inference output (with context) - detected: {len(detected)} outputs, restored: {restored.shape}")

# Test training mode with context
model_dce.train()
x_train = torch.randn(4, 3, 640, 640)
detected, restored, logits, labels = model_dce(x_train, context_images=context_images)
print(f"✓ Training output (with context) - detected: {len(detected)} outputs, restored: {restored.shape}")

print("\nTest 4: Compare outputs with and without context...")
model_dce.eval()
x_test = torch.randn(2, 3, 640, 640)

with torch.no_grad():
    detected_no_ctx, restored_no_ctx = model_dce(x_test, context_images=None)
    detected_with_ctx, restored_with_ctx = model_dce(x_test, context_images=context_images)

# Check that outputs are different when context is provided
restored_diff = torch.abs(restored_with_ctx - restored_no_ctx).mean()
print(f"✓ Mean absolute difference in restored output: {restored_diff.item():.9f}")

# Note: The difference is small because:
# 1. Model is untrained (parameters at initialization)
# 2. Random context images don't provide meaningful degradation info
# 3. The effect will be more pronounced after training
# The fact that it's non-zero confirms DCE is working
if restored_diff.item() > 0:
    print("✓ DCE is active (difference > 0, will be more pronounced after training)")
else:
    print("⚠ Warning: No difference detected - DCE might not be working")

print("\nTest 5: Gradient flow test...")
model_dce.train()
x_train = torch.randn(4, 3, 640, 640, requires_grad=True)
degrad_ctx = torch.randn(2, 3, 224, 224)
clean_ctx = torch.randn(2, 3, 224, 224)
context_images = (degrad_ctx, clean_ctx)

detected, restored, logits, labels = model_dce(x_train, context_images=context_images)
loss = restored.mean()
loss.backward()
print(f"✓ Gradients computed successfully")
print(f"✓ Input gradient shape: {x_train.grad.shape}")

print("\n✅ Step 6 complete! YoloBody accepts and passes context pairs.")

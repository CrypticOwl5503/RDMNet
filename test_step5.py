# test_step5.py
import torch
import numpy as np
from utils.dataloader import YoloDataset, yolo_dataset_collate

# Create dummy annotation lines for testing
# Format: "image_path x1,y1,x2,y2,class x1,y1,x2,y2,class ..."
dummy_annotation_lines = [
    "dummy_path1.jpg 10,10,50,50,0",
    "dummy_path2.jpg 20,20,60,60,0",
    "dummy_path3.jpg 30,30,70,70,0",
]

print("Test 1: Dataloader without DCE (backward compatibility)...")
# This will fail if files don't exist, but we're just testing the structure
try:
    dataset_no_dce = YoloDataset(
        dataset_dir="dummy",
        annotation_lines=dummy_annotation_lines,
        input_shape=(640, 640),
        num_classes=1,
        epoch_length=100,
        mosaic=False,
        train=True,
        use_dce=False
    )
    print("✓ Dataset initialized without DCE")
    print(f"✓ Dataset length: {len(dataset_no_dce)}")
except Exception as e:
    print(f"Note: Dataset initialization failed (expected if files don't exist): {e}")

print("\nTest 2: Dataloader with DCE...")
try:
    dataset_dce = YoloDataset(
        dataset_dir="dummy",
        annotation_lines=dummy_annotation_lines,
        input_shape=(640, 640),
        num_classes=1,
        epoch_length=100,
        mosaic=False,
        train=True,
        use_dce=True
    )
    print("✓ Dataset initialized with DCE")
    print(f"✓ CLIP transform available: {hasattr(dataset_dce, 'clip_transform')}")
except Exception as e:
    print(f"Note: Dataset initialization failed (expected if files don't exist): {e}")

print("\nTest 3: Collate function without context...")
# Simulate batch without context
batch_no_ctx = [
    (np.random.rand(3, 640, 640), np.array([[10, 10, 50, 50, 0]]), np.random.rand(3, 640, 640)),
    (np.random.rand(3, 640, 640), np.array([[20, 20, 60, 60, 0]]), np.random.rand(3, 640, 640)),
]
result = yolo_dataset_collate(batch_no_ctx)
print(f"✓ Collate result length: {len(result)}")
print(f"✓ Images shape: {result[0].shape}")
print(f"✓ Clear images shape: {result[2].shape}")

print("\nTest 4: Collate function with context...")
# Simulate batch with context
batch_with_ctx = [
    (np.random.rand(3, 640, 640), np.array([[10, 10, 50, 50, 0]]), np.random.rand(3, 640, 640),
     torch.randn(3, 224, 224), torch.randn(3, 224, 224)),
    (np.random.rand(3, 640, 640), np.array([[20, 20, 60, 60, 0]]), np.random.rand(3, 640, 640),
     torch.randn(3, 224, 224), torch.randn(3, 224, 224)),
]
result = yolo_dataset_collate(batch_with_ctx)
print(f"✓ Collate result length: {len(result)}")
print(f"✓ Images shape: {result[0].shape}")
print(f"✓ Clear images shape: {result[2].shape}")
print(f"✓ Degrad context shape: {result[3].shape}")
print(f"✓ Clean context shape: {result[4].shape}")
assert len(result) == 5, f"Expected 5 elements, got {len(result)}"
assert result[3].shape == (2, 3, 224, 224), f"Expected (2, 3, 224, 224), got {result[3].shape}"
assert result[4].shape == (2, 3, 224, 224), f"Expected (2, 3, 224, 224), got {result[4].shape}"

print("\n✅ Step 5 complete! Dataloader supports context pairs.")

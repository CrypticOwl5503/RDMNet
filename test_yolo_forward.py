import torch
import sys

try:
    from nets.yolo import YoloBody
    model = YoloBody(5, 's')
    x = torch.randn(2, 3, 640, 640)
    print("Testing forward pass...")
    
    # Needs a 2-chunk (input, posimg) in train mode, so we feed 4 batch dimension just in case, but let's set train=False
    model.eval()
    detected, restored = model(x)
    print("Forward pass successful!")
    print(f"Detected length: {len(detected)}")
    print(f"Restored shape: {restored.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

from ultralytics.nn.modules.mobilenetv4 import *
import torch

model = MobileNetV4("MobileNetV4ConvSmall")
x = torch.randn(1, 3, 640, 640)
features = model(x)
for i, f in enumerate(features):
    print(f"Feature {i}: shape = {f.shape}")

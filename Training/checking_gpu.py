import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Capability:", torch.cuda.get_device_capability(0))
else:
    print("No GPU available.")

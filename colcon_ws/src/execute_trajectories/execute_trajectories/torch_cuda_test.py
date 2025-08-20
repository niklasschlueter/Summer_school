#!/usr/bin/env python3
import torch
import subprocess

print("=== NVIDIA / CUDA / PyTorch Check ===")
# 2. Check CUDA availability in PyTorch
print("\n[PyTorch CUDA Check]")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Capability: {torch.cuda.get_device_capability(i)}")
else:
    print("No GPU detected by PyTorch")
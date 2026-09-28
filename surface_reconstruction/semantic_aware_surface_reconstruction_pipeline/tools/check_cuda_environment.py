#!/usr/bin/env python3
import sys
try:
    import torch
except Exception as exc:
    print("FAIL: PyTorch import failed:", exc)
    raise SystemExit(2)

print("Python        :", sys.version.split()[0])
print("PyTorch       :", torch.__version__)
print("torch CUDA    :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count     :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i:<2}        : {p.name} | {p.total_memory/1024**3:.1f} GiB | CC {p.major}.{p.minor}")
if not torch.cuda.is_available():
    print("\nThis environment does not have a CUDA-enabled PyTorch runtime.")
    raise SystemExit(2)

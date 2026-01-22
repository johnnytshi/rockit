#!/usr/bin/env python3
"""
Trace PyTorch matrix multiplication to see actual kernel calls
"""

import torch
import subprocess

device = torch.device('cuda')

print("=" * 70)
print("PyTorch Kernel Trace")
print("=" * 70)

m, n, k = 4096, 4096, 4096
A = torch.randn(m, k, device=device, dtype=torch.float16)
B = torch.randn(k, n, device=device, dtype=torch.float16)

print(f"\nRunning matrix multiplication: ({m}x{k}) @ ({k}x{n})...")

# Warmup
for _ in range(3):
    _ = torch.mm(A, B)
torch.cuda.synchronize()

# Use ROCTRACER to see actual kernel calls
print("\nAttempting to trace HIP kernels...")
print("This will show if PyTorch is using hipBLAS, hipBLASLt, or something else\n")

# Simple trace
import os
os.environ['HIP_TRACE_API'] = '1'
os.environ['AMD_LOG_LEVEL'] = '4'

# Do a single matmul
result = torch.mm(A, B)
torch.cuda.synchronize()

print("\n" + "=" * 70)
print("Note: Check if 'hipblasLt' or 'hipblas' appears in any output above")
print("=" * 70)

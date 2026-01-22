#!/usr/bin/env python3
"""
Check what PyTorch is actually using for matrix multiplication
"""

import torch
import os

# Enable logging
os.environ['PYTORCH_DEBUG'] = '1'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'

device = torch.device('cuda')

print("=" * 70)
print("PyTorch Backend Investigation")
print("=" * 70)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA/HIP available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Check if hipBLASLt is available
print("\nChecking for hipBLASLt...")
try:
    # This is a hint that PyTorch might be using hipBLASLt
    print(f"torch.cuda.get_arch_list(): {torch.cuda.get_arch_list()}")
except:
    pass

# Check compilation flags
print("\nPyTorch build info:")
print(f"CUDA compiled version: {torch.version.cuda}")
if hasattr(torch.version, 'hip'):
    print(f"HIP compiled version: {torch.version.hip}")

# Try to see which BLAS backend is being used
print("\nBLAS backend info:")
try:
    # Create small matrices and check kernel names
    import torch.profiler

    m, n, k = 4096, 4096, 4096
    A = torch.randn(m, k, device=device, dtype=torch.float16)
    B = torch.randn(k, n, device=device, dtype=torch.float16)

    print("\nProfiling matrix multiplication...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=False
    ) as prof:
        C = torch.mm(A, B)
        torch.cuda.synchronize()

    # Print kernel names
    print("\nKernel names used:")
    for evt in prof.key_averages():
        if 'gemm' in evt.key.lower() or 'mm' in evt.key.lower() or 'blas' in evt.key.lower():
            print(f"  {evt.key}")
            print(f"    CUDA time: {evt.cuda_time_total/1000:.2f} us")

    # Full trace
    print("\nTop 10 operations:")
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="cuda_time_total", row_limit=10))

except Exception as e:
    print(f"Profiling failed: {e}")

print("\n" + "=" * 70)

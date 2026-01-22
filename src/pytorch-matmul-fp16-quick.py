#!/usr/bin/env python3
"""
Quick PyTorch Matrix Multiplication Benchmark (FP16)
Testing a few representative shapes
"""

import torch
import time

print("=" * 70)
print("PyTorch Matrix Multiplication Benchmark (FP16)")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("❌ CUDA/ROCm not available")
    exit(1)

print(f"\nDevice: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch version: {torch.__version__}")

# Test a few representative shapes
test_cases = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (2048, 4096, 2048),
]

print("\n" + "=" * 70)
print("Starting benchmark...")
print("=" * 70)

results = []

for i, (m, n, k) in enumerate(test_cases, 1):
    print(f"\n[{i}/{len(test_cases)}] Testing (m={m}, n={n}, k={k})...")

    # Create random matrices in FP16: C(m×n) = A(m×k) × B(k×n)
    A = torch.randn(m, k, device=device, dtype=torch.float16)
    B = torch.randn(k, n, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(3):
        _ = torch.mm(A, B)

    torch.cuda.synchronize()

    # Benchmark
    iterations = 10
    start = time.time()
    for _ in range(iterations):
        C = torch.mm(A, B)

    torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = elapsed / iterations

    # Calculate TOPS (Tera Operations Per Second)
    # Matrix multiplication: 2*m*n*k operations
    tops = (2 * m * n * k) / (avg_time * 1e12)

    print(f"  Time: {avg_time*1000:.2f} ms, Performance: {tops:.2f} TOPS")

    results.append({
        'm': m,
        'n': n,
        'k': k,
        'time_ms': avg_time * 1000,
        'tops': tops
    })

print("\n" + "=" * 70)
print("Results Summary:")
print("=" * 70)
for r in results:
    print(f"  ({r['m']}, {r['n']}, {r['k']}): {r['time_ms']:.2f} ms ({r['tops']:.2f} TOPS)")
print("=" * 70)

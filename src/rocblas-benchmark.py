#!/usr/bin/env python3
"""
rocBLAS Matrix Multiplication Benchmark (BF16)
Direct rocBLAS calls to measure raw compute performance
"""

import ctypes
import time
import sys
import numpy as np

print("=" * 70)
print("rocBLAS Matrix Multiplication Benchmark (BF16)")
print("=" * 70)

# Load ROCm libraries
try:
    hip = ctypes.CDLL("libamdhip64.so")
    rocblas = ctypes.CDLL("librocblas.so")
    print("✅ rocBLAS and HIP libraries loaded successfully")
except OSError as e:
    print(f"❌ Failed to load ROCm libraries: {e}")
    print("Make sure ROCm is installed and LD_LIBRARY_PATH includes /opt/rocm/lib")
    sys.exit(1)

# HIP error codes
HIP_SUCCESS = 0

# rocBLAS types and constants
ROCBLAS_STATUS_SUCCESS = 0
ROCBLAS_OPERATION_NONE = 111  # 'n' for no transpose
ROCBLAS_OPERATION_TRANSPOSE = 112  # 't' for transpose

# rocBLAS data types
ROCBLAS_DATATYPE_F16_R = 150  # 16-bit float (half precision)
ROCBLAS_DATATYPE_F32_R = 151  # 32-bit float
ROCBLAS_DATATYPE_BF16_R = 168  # 16-bit bfloat16

# rocBLAS GEMM algorithm
ROCBLAS_GEMM_DEFAULT = 0

# Helper functions
def hip_check(status, msg="HIP operation failed"):
    if status != HIP_SUCCESS:
        raise RuntimeError(f"{msg}: error code {status}")

def rocblas_check(status, msg="rocBLAS operation failed"):
    if status != ROCBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"{msg}: error code {status}")

# Get device info
device_count = ctypes.c_int()
hip.hipGetDeviceCount(ctypes.byref(device_count))
print(f"\nDevice count: {device_count.value}")

if device_count.value == 0:
    print("❌ No HIP devices found")
    sys.exit(1)

# Get device properties
class hipDeviceProp_t(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        # ... there are more fields but we only need these
    ]

device_prop = hipDeviceProp_t()
hip.hipGetDeviceProperties(ctypes.byref(device_prop), 0)
print(f"Device: {device_prop.name.decode('utf-8')}")
print(f"Total memory: {device_prop.totalGlobalMem / (1024**3):.2f} GB")
print(f"Compute capability: {device_prop.major}.{device_prop.minor}")

# Initialize rocBLAS
handle = ctypes.c_void_p()
rocblas_check(rocblas.rocblas_create_handle(ctypes.byref(handle)), "Failed to create rocBLAS handle")

print("\n" + "=" * 70)
print("Starting benchmark...")
print("=" * 70)

# Test dimensions
dims = [1024, 2048, 4096, 8192]
results = []

# Matrix multiplication: C(m×n) = A(m×k) × B(k×n)
# rocBLAS uses column-major order, so we need to be careful with dimensions
# For row-major C = A × B, we compute: C^T = B^T × A^T in column-major

total_tests = len(dims) ** 3
current_test = 0

for m in dims:
    for n in dims:
        for k in dims:
            current_test += 1
            print(f"\n[{current_test}/{total_tests}] Testing (m={m}, n={n}, k={k})...")

            # Allocate host memory (using float32 for initialization, will convert to bf16)
            # Note: NumPy doesn't natively support bfloat16, so we'll use float32 and let rocBLAS handle conversion
            # In practice, rocBLAS will work with bf16 data directly

            # For bf16, each element is 2 bytes
            element_size = 2  # bytes for bf16

            A_size = m * k * element_size
            B_size = k * n * element_size
            C_size = m * n * element_size

            # Allocate device memory
            d_A = ctypes.c_void_p()
            d_B = ctypes.c_void_p()
            d_C = ctypes.c_void_p()

            hip_check(hip.hipMalloc(ctypes.byref(d_A), A_size), "Failed to allocate d_A")
            hip_check(hip.hipMalloc(ctypes.byref(d_B), B_size), "Failed to allocate d_B")
            hip_check(hip.hipMalloc(ctypes.byref(d_C), C_size), "Failed to allocate d_C")

            # Initialize with random data (use float32 host arrays, copy as bytes)
            # In a real scenario, you'd convert to bf16 format properly
            h_A = np.random.randn(m, k).astype(np.float16)  # Use float16 as proxy for bf16
            h_B = np.random.randn(k, n).astype(np.float16)

            # Copy to device
            hip_check(hip.hipMemcpy(d_A, h_A.ctypes.data, A_size, 1), "Failed to copy A to device")  # 1 = HostToDevice
            hip_check(hip.hipMemcpy(d_B, h_B.ctypes.data, B_size, 1), "Failed to copy B to device")

            # rocBLAS GEMM parameters
            # C = alpha * A * B + beta * C
            alpha = ctypes.c_float(1.0)
            beta = ctypes.c_float(0.0)

            # Leading dimensions (column-major)
            lda = m
            ldb = k
            ldc = m

            # Warmup iterations
            warmup_iters = 3
            for _ in range(warmup_iters):
                status = rocblas.rocblas_gemm_ex(
                    handle,
                    ROCBLAS_OPERATION_NONE,  # transA
                    ROCBLAS_OPERATION_NONE,  # transB
                    m, n, k,
                    ctypes.byref(alpha),
                    d_A, ROCBLAS_DATATYPE_BF16_R, lda,
                    d_B, ROCBLAS_DATATYPE_BF16_R, ldb,
                    ctypes.byref(beta),
                    d_C, ROCBLAS_DATATYPE_BF16_R, ldc,
                    d_C, ROCBLAS_DATATYPE_BF16_R, ldc,
                    ROCBLAS_DATATYPE_F32_R,  # compute type
                    ROCBLAS_GEMM_DEFAULT,
                    0, 0
                )
                rocblas_check(status, "Warmup GEMM failed")

            # Synchronize
            hip_check(hip.hipDeviceSynchronize(), "Failed to synchronize")

            # Benchmark iterations
            bench_iters = 10
            start = time.time()

            for _ in range(bench_iters):
                status = rocblas.rocblas_gemm_ex(
                    handle,
                    ROCBLAS_OPERATION_NONE,  # transA
                    ROCBLAS_OPERATION_NONE,  # transB
                    m, n, k,
                    ctypes.byref(alpha),
                    d_A, ROCBLAS_DATATYPE_BF16_R, lda,
                    d_B, ROCBLAS_DATATYPE_BF16_R, ldb,
                    ctypes.byref(beta),
                    d_C, ROCBLAS_DATATYPE_BF16_R, ldc,
                    d_C, ROCBLAS_DATATYPE_BF16_R, ldc,
                    ROCBLAS_DATATYPE_F32_R,  # compute type
                    ROCBLAS_GEMM_DEFAULT,
                    0, 0
                )
                rocblas_check(status, "Benchmark GEMM failed")

            # Synchronize
            hip_check(hip.hipDeviceSynchronize(), "Failed to synchronize")

            elapsed = time.time() - start
            avg_time = elapsed / bench_iters

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

            # Free device memory
            hip.hipFree(d_A)
            hip.hipFree(d_B)
            hip.hipFree(d_C)

# Cleanup
rocblas.rocblas_destroy_handle(handle)

print("\n" + "=" * 70)
print(f"Summary: {len(results)} combinations tested")
print("=" * 70)

# Sort by TOPS descending and show top 10
results_sorted = sorted(results, key=lambda x: x['tops'], reverse=True)
print("\nTop 10 performers:")
for i, r in enumerate(results_sorted[:10], 1):
    print(f"  {i}. ({r['m']}, {r['n']}, {r['k']}): {r['time_ms']:.2f} ms ({r['tops']:.2f} TOPS)")

print("\nAll results:")
for r in results:
    print(f"  ({r['m']}, {r['n']}, {r['k']}): {r['time_ms']:.2f} ms ({r['tops']:.2f} TOPS)")

print("=" * 70)
print("✅ rocBLAS benchmark complete!")
print("=" * 70)

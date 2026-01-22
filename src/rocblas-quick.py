#!/usr/bin/env python3
"""
Quick rocBLAS Matrix Multiplication Benchmark (BF16)
Testing a few representative shapes
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
ROCBLAS_OPERATION_NONE = 111
ROCBLAS_DATATYPE_BF16_R = 168
ROCBLAS_DATATYPE_F32_R = 151
ROCBLAS_GEMM_DEFAULT = 0

# Define proper function signatures
hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
hip.hipGetDeviceCount.restype = ctypes.c_int

hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
hip.hipMalloc.restype = ctypes.c_int

hip.hipFree.argtypes = [ctypes.c_void_p]
hip.hipFree.restype = ctypes.c_int

hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
hip.hipMemcpy.restype = ctypes.c_int

hip.hipDeviceSynchronize.argtypes = []
hip.hipDeviceSynchronize.restype = ctypes.c_int

rocblas.rocblas_create_handle.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
rocblas.rocblas_create_handle.restype = ctypes.c_int

rocblas.rocblas_destroy_handle.argtypes = [ctypes.c_void_p]
rocblas.rocblas_destroy_handle.restype = ctypes.c_int

# rocblas_gemm_ex signature
rocblas.rocblas_gemm_ex.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # transA
    ctypes.c_int,     # transB
    ctypes.c_int,     # m
    ctypes.c_int,     # n
    ctypes.c_int,     # k
    ctypes.c_void_p,  # alpha
    ctypes.c_void_p,  # A
    ctypes.c_int,     # a_type
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # B
    ctypes.c_int,     # b_type
    ctypes.c_int,     # ldb
    ctypes.c_void_p,  # beta
    ctypes.c_void_p,  # C
    ctypes.c_int,     # c_type
    ctypes.c_int,     # ldc
    ctypes.c_void_p,  # D
    ctypes.c_int,     # d_type
    ctypes.c_int,     # ldd
    ctypes.c_int,     # compute_type
    ctypes.c_int,     # algo
    ctypes.c_int32,   # solution_index
    ctypes.c_uint32,  # flags
]
rocblas.rocblas_gemm_ex.restype = ctypes.c_int

def hip_check(status, msg="HIP operation failed"):
    if status != HIP_SUCCESS:
        raise RuntimeError(f"{msg}: error code {status}")

def rocblas_check(status, msg="rocBLAS operation failed"):
    if status != ROCBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"{msg}: error code {status}")

# Get device info
device_count = ctypes.c_int()
hip_check(hip.hipGetDeviceCount(ctypes.byref(device_count)))
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
    ]

hip.hipGetDeviceProperties.argtypes = [ctypes.POINTER(hipDeviceProp_t), ctypes.c_int]
hip.hipGetDeviceProperties.restype = ctypes.c_int

device_prop = hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(ctypes.byref(device_prop), 0))
print(f"Device: {device_prop.name.decode('utf-8')}")
print(f"Total memory: {device_prop.totalGlobalMem / (1024**3):.2f} GB")

# Initialize rocBLAS
handle = ctypes.c_void_p()
rocblas_check(rocblas.rocblas_create_handle(ctypes.byref(handle)), "Failed to create rocBLAS handle")

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

    # For bf16, each element is 2 bytes
    element_size = 2

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

    # Initialize with random data
    h_A = np.random.randn(m, k).astype(np.float16)
    h_B = np.random.randn(k, n).astype(np.float16)

    # Copy to device (1 = HostToDevice)
    hip_check(hip.hipMemcpy(d_A, h_A.ctypes.data, A_size, 1), "Failed to copy A to device")
    hip_check(hip.hipMemcpy(d_B, h_B.ctypes.data, B_size, 1), "Failed to copy B to device")

    # rocBLAS GEMM parameters
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    # Leading dimensions (column-major)
    lda = m
    ldb = k
    ldc = m

    # Warmup iterations
    for _ in range(3):
        status = rocblas.rocblas_gemm_ex(
            handle,
            ROCBLAS_OPERATION_NONE,
            ROCBLAS_OPERATION_NONE,
            m, n, k,
            ctypes.byref(alpha),
            d_A, ROCBLAS_DATATYPE_BF16_R, lda,
            d_B, ROCBLAS_DATATYPE_BF16_R, ldb,
            ctypes.byref(beta),
            d_C, ROCBLAS_DATATYPE_BF16_R, ldc,
            d_C, ROCBLAS_DATATYPE_BF16_R, ldc,
            ROCBLAS_DATATYPE_F32_R,
            ROCBLAS_GEMM_DEFAULT,
            ctypes.c_int32(0),
            ctypes.c_uint32(0)
        )
        rocblas_check(status, "Warmup GEMM failed")

    hip_check(hip.hipDeviceSynchronize(), "Failed to synchronize")

    # Benchmark iterations
    bench_iters = 10
    start = time.time()

    for _ in range(bench_iters):
        status = rocblas.rocblas_gemm_ex(
            handle,
            ROCBLAS_OPERATION_NONE,
            ROCBLAS_OPERATION_NONE,
            m, n, k,
            ctypes.byref(alpha),
            d_A, ROCBLAS_DATATYPE_BF16_R, lda,
            d_B, ROCBLAS_DATATYPE_BF16_R, ldb,
            ctypes.byref(beta),
            d_C, ROCBLAS_DATATYPE_BF16_R, ldc,
            d_C, ROCBLAS_DATATYPE_BF16_R, ldc,
            ROCBLAS_DATATYPE_F32_R,
            ROCBLAS_GEMM_DEFAULT,
            ctypes.c_int32(0),
            ctypes.c_uint32(0)
        )
        rocblas_check(status, "Benchmark GEMM failed")

    hip_check(hip.hipDeviceSynchronize(), "Failed to synchronize")

    elapsed = time.time() - start
    avg_time = elapsed / bench_iters

    # Calculate TOPS
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
print("Results Summary:")
print("=" * 70)
for r in results:
    print(f"  ({r['m']}, {r['n']}, {r['k']}): {r['time_ms']:.2f} ms ({r['tops']:.2f} TOPS)")
print("=" * 70)

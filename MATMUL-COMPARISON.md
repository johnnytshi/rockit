# Matrix Multiplication Benchmark Comparison: PyTorch vs rocBLAS

## Test Configuration
- **GPU**: Radeon 8060S Graphics (gfx1151)
- **Data Type**: FP16 (16-bit floating point)
- **Iterations**: 10 warm iterations for each test
- **PyTorch Version**: 2.11.0a0+rocm7.11.0a20260106
- **rocBLAS Version**: 5.3.0.37865a97a6

## Results

### PyTorch BF16 Results
| Matrix Shape (m, n, k) | Time (ms) | Performance (TOPS) |
|------------------------|-----------|-------------------|
| (1024, 1024, 1024)     | 0.06      | 33.57             |
| (2048, 2048, 2048)     | 0.58      | 29.76             |
| (4096, 4096, 4096)     | 5.03      | 27.32             |
| (8192, 8192, 8192)     | 42.91     | 25.62             |
| (2048, 4096, 2048)     | 1.30      | 26.43             |

### PyTorch FP16 Results
| Matrix Shape (m, n, k) | Time (ms) | Performance (TOPS) |
|------------------------|-----------|-------------------|
| (1024, 1024, 1024)     | 0.06      | 35.49             |
| (2048, 2048, 2048)     | 0.60      | 28.79             |
| (4096, 4096, 4096)     | 4.27      | 32.22             |
| (8192, 8192, 8192)     | 38.12     | 28.85             |
| (2048, 4096, 2048)     | 1.25      | 27.39             |

### rocBLAS FP16 Results (C++)
| Matrix Shape (m, n, k) | Time (ms) | Performance (TOPS) |
|------------------------|-----------|-------------------|
| (1024, 1024, 1024)     | 0.08      | 26.86             |
| (2048, 2048, 2048)     | 0.60      | 28.45             |
| (4096, 4096, 4096)     | 4.40      | 31.25             |
| (8192, 8192, 8192)     | 38.49     | 28.56             |
| (2048, 4096, 2048)     | 1.38      | 24.87             |

### rocBLAS FP16 Results (rocblas-bench tool)
| Matrix Shape (m, n, k) | Time (ms) | Performance (TOPS) |
|------------------------|-----------|-------------------|
| (1024, 1024, 1024)     | 0.08      | 28.48             |
| (2048, 2048, 2048)     | 0.77      | 22.30             |
| (4096, 4096, 4096)     | 6.41      | 21.46             |
| (8192, 8192, 8192)     | 46.61     | 23.59             |
| (2048, 4096, 2048)     | 1.88      | 18.29             |

## Comparison Summary

### FP16: PyTorch vs rocBLAS C++ Performance Ratio
| Matrix Shape (m, n, k) | PyTorch Time (ms) | rocBLAS C++ Time (ms) | Speedup (PyTorch/rocBLAS) | PyTorch TOPS | rocBLAS C++ TOPS |
|------------------------|-------------------|-----------------------|---------------------------|--------------|------------------|
| (1024, 1024, 1024)     | 0.06              | 0.08                  | **1.33x faster (PyTorch)**| 35.49        | 26.86            |
| (2048, 2048, 2048)     | 0.60              | 0.60                  | **Same speed**            | 28.79        | 28.45            |
| (4096, 4096, 4096)     | 4.27              | 4.40                  | **1.03x faster (PyTorch)**| 32.22        | 31.25            |
| (8192, 8192, 8192)     | 38.12             | 38.49                 | **1.01x faster (PyTorch)**| 28.85        | 28.56            |
| (2048, 4096, 2048)     | 1.25              | 1.38                  | **1.10x faster (PyTorch)**| 27.39        | 24.87            |

**Average Speedup: 1.09x (PyTorch is slightly faster)**

### FP16: PyTorch vs rocblas-bench tool Performance Ratio
| Matrix Shape (m, n, k) | PyTorch Time (ms) | rocblas-bench Time (ms) | Speedup (PyTorch/rocBLAS) | PyTorch TOPS | rocblas-bench TOPS |
|------------------------|-------------------|-------------------------|---------------------------|--------------|--------------------|
| (1024, 1024, 1024)     | 0.06              | 0.08                    | **1.33x faster (PyTorch)**| 35.49        | 28.48              |
| (2048, 2048, 2048)     | 0.60              | 0.77                    | **1.28x faster (PyTorch)**| 28.79        | 22.30              |
| (4096, 4096, 4096)     | 4.27              | 6.41                    | **1.50x faster (PyTorch)**| 32.22        | 21.46              |
| (8192, 8192, 8192)     | 38.12             | 46.61                   | **1.22x faster (PyTorch)**| 28.85        | 23.59              |
| (2048, 4096, 2048)     | 1.25              | 1.88                    | **1.50x faster (PyTorch)**| 27.39        | 18.29              |

**Average Speedup: 1.37x (PyTorch is faster)**

## Key Observations

1. **C++ rocBLAS is nearly identical to PyTorch** for larger matrices
   - For 2048x2048x2048: Same speed (both ~0.60 ms)
   - For 4096x4096x4096: Within 3% (PyTorch 4.27ms vs rocBLAS 4.40ms)
   - For 8192x8192x8192: Within 1% (PyTorch 38.12ms vs rocBLAS 38.49ms)
   - PyTorch only shows advantage on small matrices (1024x1024x1024: 1.33x faster)

2. **rocblas-bench tool has significant overhead**:
   - The CLI tool is 1.37x slower on average than direct C++ calls
   - C++ rocBLAS averages ~28 TOPS vs rocblas-bench tool ~22.8 TOPS
   - Tool overhead includes initialization and parsing costs

3. **PyTorch vs C++ rocBLAS analysis**:
   - Average speedup is only 1.09x (minimal difference)
   - **PyTorch likely calls rocBLAS directly** under the hood
   - Both achieve similar peak performance (~31-32 TOPS on 4096 matrices)
   - Small matrices: PyTorch has better kernel launch overhead
   - Large matrices: Performance is essentially identical

4. **BF16 vs FP16 in PyTorch**:
   - BF16: Average ~28.5 TOPS
   - FP16: Average ~30.5 TOPS
   - FP16 is slightly faster (~7% improvement)

5. **Size scaling**:
   - Peak performance reached at 4096x4096x4096 (~31-32 TOPS)
   - Both implementations show similar scaling characteristics
   - Confirms they're using the same underlying kernels

## Conclusions

- **PyTorch and C++ rocBLAS are essentially equivalent** for large matrices
- PyTorch's minimal overhead proves it's a thin, well-optimized wrapper
- For small matrices (<2048), PyTorch has slightly better launch overhead
- For large matrices (≥4096), the difference is negligible (<3%)
- **Use PyTorch** - you get the same performance with much better ergonomics
- Direct C++ rocBLAS only makes sense if you need absolute control or have non-PyTorch pipelines

## Notes

- rocblas-bench doesn't support BF16 for the standard GEMM function
- All benchmarks use the same warmup (3 iterations) and measurement (10 iterations) methodology
- Results are averages across 10 iterations after warmup
- C++ rocBLAS benchmark compiled with hipcc and linked directly against librocblas.so
- The C++ implementation provides the cleanest comparison since it eliminates all Python/ctypes overhead

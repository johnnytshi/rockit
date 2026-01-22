// rocBLAS Matrix Multiplication Benchmark (FP16) - Detailed Analysis
// Tests different algorithms and configurations

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define ROCBLAS_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            std::cerr << "rocBLAS Error: " << rocblas_status_to_string(status) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

struct BenchmarkResult {
    std::string name;
    double time_ms;
    double tops;
};

// Pre-allocate memory to avoid allocation overhead
struct PreallocatedBuffers {
    _Float16 *d_A, *d_B, *d_C;
    size_t size_A, size_B, size_C;
};

PreallocatedBuffers allocate_buffers(int m, int n, int k) {
    PreallocatedBuffers buffers;

    buffers.size_A = m * k * sizeof(_Float16);
    buffers.size_B = k * n * sizeof(_Float16);
    buffers.size_C = m * n * sizeof(_Float16);

    HIP_CHECK(hipMalloc(&buffers.d_A, buffers.size_A));
    HIP_CHECK(hipMalloc(&buffers.d_B, buffers.size_B));
    HIP_CHECK(hipMalloc(&buffers.d_C, buffers.size_C));

    // Initialize with random data
    std::vector<_Float16> h_A(m * k);
    std::vector<_Float16> h_B(k * n);

    for (size_t i = 0; i < h_A.size(); i++) {
        h_A[i] = static_cast<_Float16>(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    for (size_t i = 0; i < h_B.size(); i++) {
        h_B[i] = static_cast<_Float16>(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    }

    HIP_CHECK(hipMemcpy(buffers.d_A, h_A.data(), buffers.size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buffers.d_B, h_B.data(), buffers.size_B, hipMemcpyHostToDevice));

    return buffers;
}

void free_buffers(PreallocatedBuffers& buffers) {
    HIP_CHECK(hipFree(buffers.d_A));
    HIP_CHECK(hipFree(buffers.d_B));
    HIP_CHECK(hipFree(buffers.d_C));
}

BenchmarkResult benchmark_gemm_ex_algo(
    rocblas_handle handle,
    PreallocatedBuffers& buffers,
    int m, int n, int k,
    rocblas_gemm_algo algo,
    const char* algo_name,
    int32_t solution_index = 0,
    uint32_t flags = 0,
    int warmup_iters = 5,
    int bench_iters = 20
) {
    float alpha = 1.0f;
    float beta = 0.0f;

    rocblas_operation trans_A = rocblas_operation_none;
    rocblas_operation trans_B = rocblas_operation_none;

    int lda = m;
    int ldb = k;
    int ldc = m;

    // Extended warmup
    for (int i = 0; i < warmup_iters; i++) {
        ROCBLAS_CHECK(rocblas_gemm_ex(
            handle,
            trans_A, trans_B,
            m, n, k,
            &alpha,
            buffers.d_A, rocblas_datatype_f16_r, lda,
            buffers.d_B, rocblas_datatype_f16_r, ldb,
            &beta,
            buffers.d_C, rocblas_datatype_f16_r, ldc,
            buffers.d_C, rocblas_datatype_f16_r, ldc,
            rocblas_datatype_f32_r,
            algo,
            solution_index, flags
        ));
    }

    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark with more iterations
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < bench_iters; i++) {
        ROCBLAS_CHECK(rocblas_gemm_ex(
            handle,
            trans_A, trans_B,
            m, n, k,
            &alpha,
            buffers.d_A, rocblas_datatype_f16_r, lda,
            buffers.d_B, rocblas_datatype_f16_r, ldb,
            &beta,
            buffers.d_C, rocblas_datatype_f16_r, ldc,
            buffers.d_C, rocblas_datatype_f16_r, ldc,
            rocblas_datatype_f32_r,
            algo,
            solution_index, flags
        ));
    }

    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double avg_time_sec = elapsed.count() / bench_iters;
    double avg_time_ms = avg_time_sec * 1000.0;

    double ops = 2.0 * m * n * k;
    double tops = ops / (avg_time_sec * 1e12);

    BenchmarkResult result;
    result.name = algo_name;
    result.time_ms = avg_time_ms;
    result.tops = tops;

    return result;
}

BenchmarkResult benchmark_gemm_simple(
    rocblas_handle handle,
    PreallocatedBuffers& buffers,
    int m, int n, int k,
    int warmup_iters = 5,
    int bench_iters = 20
) {
    _Float16 alpha_val = static_cast<_Float16>(1.0f);
    _Float16 beta_val = static_cast<_Float16>(0.0f);

    // Cast to rocblas_half*
    auto alpha = reinterpret_cast<rocblas_half*>(&alpha_val);
    auto beta = reinterpret_cast<rocblas_half*>(&beta_val);
    auto d_A = reinterpret_cast<rocblas_half*>(buffers.d_A);
    auto d_B = reinterpret_cast<rocblas_half*>(buffers.d_B);
    auto d_C = reinterpret_cast<rocblas_half*>(buffers.d_C);

    rocblas_operation trans_A = rocblas_operation_none;
    rocblas_operation trans_B = rocblas_operation_none;

    int lda = m;
    int ldb = k;
    int ldc = m;

    // Extended warmup
    for (int i = 0; i < warmup_iters; i++) {
        ROCBLAS_CHECK(rocblas_hgemm(
            handle,
            trans_A, trans_B,
            m, n, k,
            alpha,
            d_A, lda,
            d_B, ldb,
            beta,
            d_C, ldc
        ));
    }

    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < bench_iters; i++) {
        ROCBLAS_CHECK(rocblas_hgemm(
            handle,
            trans_A, trans_B,
            m, n, k,
            alpha,
            d_A, lda,
            d_B, ldb,
            beta,
            d_C, ldc
        ));
    }

    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double avg_time_sec = elapsed.count() / bench_iters;
    double avg_time_ms = avg_time_sec * 1000.0;

    double ops = 2.0 * m * n * k;
    double tops = ops / (avg_time_sec * 1e12);

    BenchmarkResult result;
    result.name = "rocblas_hgemm (simple)";
    result.time_ms = avg_time_ms;
    result.tops = tops;

    return result;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "rocBLAS Detailed Performance Analysis (FP16)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    // Initialize rocBLAS
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "\nDevice: " << prop.name << std::endl;

    // Test one size with multiple algorithms
    int m = 4096, n = 4096, k = 4096;

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Testing matrix size: " << m << "x" << n << "x" << k << std::endl;
    std::cout << "Comparing different rocBLAS algorithms and configurations" << std::endl;
    std::cout << "======================================================================" << std::endl;

    // Pre-allocate buffers (excluding allocation from timing)
    std::cout << "\nAllocating and initializing buffers..." << std::endl;
    PreallocatedBuffers buffers = allocate_buffers(m, n, k);

    std::vector<BenchmarkResult> results;

    // Test 1: Simple hgemm
    std::cout << "\n[1/6] Testing rocblas_hgemm (simple API)..." << std::endl;
    auto result1 = benchmark_gemm_simple(handle, buffers, m, n, k);
    std::cout << "  " << result1.name << ": " << std::fixed << std::setprecision(2)
              << result1.time_ms << " ms (" << result1.tops << " TOPS)" << std::endl;
    results.push_back(result1);

    // Test 2: gemm_ex with standard algorithm
    std::cout << "\n[2/6] Testing rocblas_gemm_ex (algo: standard)..." << std::endl;
    auto result2 = benchmark_gemm_ex_algo(handle, buffers, m, n, k,
                                          rocblas_gemm_algo_standard,
                                          "gemm_ex (standard)");
    std::cout << "  " << result2.name << ": " << std::fixed << std::setprecision(2)
              << result2.time_ms << " ms (" << result2.tops << " TOPS)" << std::endl;
    results.push_back(result2);

    // Test 3: Solution index optimization
    std::cout << "\n[3/6] Testing rocblas_gemm_ex (solution_index: 1)..." << std::endl;
    try {
        auto result3 = benchmark_gemm_ex_algo(handle, buffers, m, n, k,
                                              rocblas_gemm_algo_standard,
                                              "gemm_ex (solution_idx 1)", 1);
        std::cout << "  " << result3.name << ": " << std::fixed << std::setprecision(2)
                  << result3.time_ms << " ms (" << result3.tops << " TOPS)" << std::endl;
        results.push_back(result3);
    } catch (...) {
        std::cout << "  Solution index 1 not available" << std::endl;
    }

    // Test 4: With flags
    std::cout << "\n[4/6] Testing rocblas_gemm_ex (flags: optimized)..." << std::endl;
    try {
        auto result4 = benchmark_gemm_ex_algo(handle, buffers, m, n, k,
                                              rocblas_gemm_algo_standard,
                                              "gemm_ex (flags 1)", 0, 1);
        std::cout << "  " << result4.name << ": " << std::fixed << std::setprecision(2)
                  << result4.time_ms << " ms (" << result4.tops << " TOPS)" << std::endl;
        results.push_back(result4);
    } catch (...) {
        std::cout << "  Flags not available" << std::endl;
    }

    // Test 5: More warmup iterations
    std::cout << "\n[5/6] Testing with extended warmup (10 iters)..." << std::endl;
    auto result5 = benchmark_gemm_simple(handle, buffers, m, n, k, 10, 20);
    result5.name = "hgemm (warmup=10)";
    std::cout << "  " << result5.name << ": " << std::fixed << std::setprecision(2)
              << result5.time_ms << " ms (" << result5.tops << " TOPS)" << std::endl;
    results.push_back(result5);

    // Test 6: Even more iterations
    std::cout << "\n[6/6] Testing with 50 benchmark iterations..." << std::endl;
    auto result6 = benchmark_gemm_simple(handle, buffers, m, n, k, 10, 50);
    result6.name = "hgemm (iters=50)";
    std::cout << "  " << result6.name << ": " << std::fixed << std::setprecision(2)
              << result6.time_ms << " ms (" << result6.tops << " TOPS)" << std::endl;
    results.push_back(result6);

    // Print summary
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Summary - All Configurations:" << std::endl;
    std::cout << "======================================================================" << std::endl;

    // Sort by performance
    std::sort(results.begin(), results.end(),
              [](const BenchmarkResult& a, const BenchmarkResult& b) {
                  return a.tops > b.tops;
              });

    for (size_t i = 0; i < results.size(); i++) {
        std::string medal = (i == 0) ? "🥇" : (i == 1) ? "🥈" : (i == 2) ? "🥉" : "  ";
        std::cout << medal << " " << std::setw(30) << std::left << results[i].name << ": "
                  << std::fixed << std::setprecision(2) << std::setw(7) << std::right
                  << results[i].time_ms << " ms (" << results[i].tops << " TOPS)" << std::endl;
    }

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Best rocBLAS configuration: " << results[0].name << std::endl;
    std::cout << "Performance: " << results[0].tops << " TOPS" << std::endl;
    std::cout << "======================================================================" << std::endl;

    free_buffers(buffers);
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return 0;
}

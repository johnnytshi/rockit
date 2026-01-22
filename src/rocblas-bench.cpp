// rocBLAS Matrix Multiplication Benchmark (FP16/BF16)
// Direct C++ implementation for maximum performance

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cmath>

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
    int m, n, k;
    double time_ms;
    double tops;
};

void print_device_info() {
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    std::cout << "\nDevice count: " << device_count << std::endl;

    if (device_count == 0) {
        std::cerr << "No HIP devices found!" << std::endl;
        exit(1);
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Total memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0))
              << " GB" << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
}

template<typename T>
BenchmarkResult benchmark_gemm(
    rocblas_handle handle,
    int m, int n, int k,
    rocblas_datatype data_type,
    const char* type_name,
    int warmup_iters = 3,
    int bench_iters = 10
) {
    // Allocate device memory
    size_t size_A = m * k * sizeof(T);
    size_t size_B = k * n * sizeof(T);
    size_t size_C = m * n * sizeof(T);

    T *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));

    // Initialize with random data (using host memory)
    std::vector<T> h_A(m * k);
    std::vector<T> h_B(k * n);

    // Fill with random values (convert from float)
    for (size_t i = 0; i < h_A.size(); i++) {
        float val = static_cast<float>(rand()) / RAND_MAX;
        h_A[i] = static_cast<T>(val);
    }
    for (size_t i = 0; i < h_B.size(); i++) {
        float val = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<T>(val);
    }

    // Copy to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));

    // GEMM parameters
    float alpha = 1.0f;
    float beta = 0.0f;

    rocblas_operation trans_A = rocblas_operation_none;
    rocblas_operation trans_B = rocblas_operation_none;

    int lda = m;
    int ldb = k;
    int ldc = m;

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        ROCBLAS_CHECK(rocblas_gemm_ex(
            handle,
            trans_A, trans_B,
            m, n, k,
            &alpha,
            d_A, data_type, lda,
            d_B, data_type, ldb,
            &beta,
            d_C, data_type, ldc,
            d_C, data_type, ldc,
            rocblas_datatype_f32_r,  // compute type
            rocblas_gemm_algo_standard,
            0, 0
        ));
    }

    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < bench_iters; i++) {
        ROCBLAS_CHECK(rocblas_gemm_ex(
            handle,
            trans_A, trans_B,
            m, n, k,
            &alpha,
            d_A, data_type, lda,
            d_B, data_type, ldb,
            &beta,
            d_C, data_type, ldc,
            d_C, data_type, ldc,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard,
            0, 0
        ));
    }

    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double avg_time_sec = elapsed.count() / bench_iters;
    double avg_time_ms = avg_time_sec * 1000.0;

    // Calculate TOPS
    double ops = 2.0 * m * n * k;
    double tops = ops / (avg_time_sec * 1e12);

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    BenchmarkResult result;
    result.m = m;
    result.n = n;
    result.k = k;
    result.time_ms = avg_time_ms;
    result.tops = tops;

    return result;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "rocBLAS Matrix Multiplication Benchmark (C++)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    print_device_info();

    // Initialize rocBLAS
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // Test cases
    std::vector<std::tuple<int, int, int>> test_cases = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
        {2048, 4096, 2048}
    };

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Starting FP16 benchmark..." << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::vector<BenchmarkResult> results_fp16;

    int count = 0;
    for (const auto& test : test_cases) {
        count++;
        int m = std::get<0>(test);
        int n = std::get<1>(test);
        int k = std::get<2>(test);

        std::cout << "\n[" << count << "/" << test_cases.size()
                  << "] Testing (m=" << m << ", n=" << n << ", k=" << k << ")..." << std::endl;

        auto result = benchmark_gemm<_Float16>(
            handle, m, n, k,
            rocblas_datatype_f16_r,
            "FP16"
        );

        std::cout << "  Time: " << std::fixed << std::setprecision(2)
                  << result.time_ms << " ms, Performance: "
                  << result.tops << " TOPS" << std::endl;

        results_fp16.push_back(result);
    }

    // Skip BF16 benchmark - not supported in rocBLAS GEMM
    std::vector<BenchmarkResult> results_bf16;

    // Print summary
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "FP16 Results Summary:" << std::endl;
    std::cout << "======================================================================" << std::endl;
    for (const auto& r : results_fp16) {
        std::cout << "  (" << r.m << ", " << r.n << ", " << r.k << "): "
                  << std::fixed << std::setprecision(2)
                  << r.time_ms << " ms (" << r.tops << " TOPS)" << std::endl;
    }

    // BF16 results skipped

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "✅ rocBLAS C++ benchmark complete!" << std::endl;
    std::cout << "======================================================================" << std::endl;

    // Cleanup
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return 0;
}

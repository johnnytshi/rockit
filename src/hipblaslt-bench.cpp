// hipBLASLt Matrix Multiplication Benchmark (FP16)
// Direct hipBLASLt implementation to match PyTorch's backend

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
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

#define HIPBLASLT_CHECK(call) \
    do { \
        hipblasStatus_t status = call; \
        if (status != HIPBLAS_STATUS_SUCCESS) { \
            std::cerr << "hipBLASLt Error: " << status \
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

BenchmarkResult benchmark_hipblaslt(
    int m, int n, int k,
    int warmup_iters = 5,
    int bench_iters = 20
) {
    // Allocate device memory
    size_t size_A = m * k * sizeof(_Float16);
    size_t size_B = k * n * sizeof(_Float16);
    size_t size_C = m * n * sizeof(_Float16);

    _Float16 *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));

    // Initialize with random data
    std::vector<_Float16> h_A(m * k);
    std::vector<_Float16> h_B(k * n);

    for (size_t i = 0; i < h_A.size(); i++) {
        h_A[i] = static_cast<_Float16>(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    for (size_t i = 0; i < h_B.size(); i++) {
        h_B[i] = static_cast<_Float16>(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    }

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));

    // Initialize hipBLASLt
    hipblasLtHandle_t handle;
    HIPBLASLT_CHECK(hipblasLtCreate(&handle));

    // Create matrix descriptors
    hipblasLtMatrixLayout_t matA, matB, matC;

    // Matrix A: m x k
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m));

    // Matrix B: k x n
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k));

    // Matrix C: m x n
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));

    // Create operation descriptor
    hipblasLtMatmulDesc_t matmulDesc;
    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    // Set operation (no transpose)
    hipblasOperation_t opA = HIPBLAS_OP_N;
    hipblasOperation_t opB = HIPBLAS_OP_N;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // Scalars
    float alpha = 1.0f;
    float beta = 0.0f;

    // Get the best heuristic
    hipblasLtMatmulPreference_t pref;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));

    // Set workspace size (optional, can help performance)
    size_t workspace_size = 32 * 1024 * 1024; // 32 MB
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    // Allocate workspace
    void* workspace = nullptr;
    HIP_CHECK(hipMalloc(&workspace, workspace_size));

    // Query for the best algorithm
    hipblasLtMatmulHeuristicResult_t heuristicResult[4];
    int returnedAlgoCount;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc,
        matA, matB, matC, matC,
        pref, 4, heuristicResult, &returnedAlgoCount));

    if (returnedAlgoCount == 0) {
        std::cerr << "No suitable algorithm found!" << std::endl;
        exit(1);
    }

    std::cout << "Found " << returnedAlgoCount << " algorithm(s)" << std::endl;

    // Use the best algorithm
    hipblasLtMatmulAlgo_t algo = heuristicResult[0].algo;

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        HIPBLASLT_CHECK(hipblasLtMatmul(
            handle,
            matmulDesc,
            &alpha,
            d_A, matA,
            d_B, matB,
            &beta,
            d_C, matC,
            d_C, matC,
            &algo,
            workspace,
            workspace_size,
            nullptr));
    }

    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < bench_iters; i++) {
        HIPBLASLT_CHECK(hipblasLtMatmul(
            handle,
            matmulDesc,
            &alpha,
            d_A, matA,
            d_B, matB,
            &beta,
            d_C, matC,
            d_C, matC,
            &algo,
            workspace,
            workspace_size,
            nullptr));
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
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtMatrixLayoutDestroy(matA);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtDestroy(handle);

    HIP_CHECK(hipFree(workspace));
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
    std::cout << "hipBLASLt Matrix Multiplication Benchmark (FP16)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    print_device_info();

    // Test cases
    std::vector<std::tuple<int, int, int>> test_cases = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
        {2048, 4096, 2048}
    };

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Starting benchmark..." << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::vector<BenchmarkResult> results;

    int count = 0;
    for (const auto& test : test_cases) {
        count++;
        int m = std::get<0>(test);
        int n = std::get<1>(test);
        int k = std::get<2>(test);

        std::cout << "\n[" << count << "/" << test_cases.size()
                  << "] Testing (m=" << m << ", n=" << n << ", k=" << k << ")..." << std::endl;

        auto result = benchmark_hipblaslt(m, n, k);

        std::cout << "  Time: " << std::fixed << std::setprecision(2)
                  << result.time_ms << " ms, Performance: "
                  << result.tops << " TOPS" << std::endl;

        results.push_back(result);
    }

    // Print summary
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Results Summary:" << std::endl;
    std::cout << "======================================================================" << std::endl;
    for (const auto& r : results) {
        std::cout << "  (" << r.m << ", " << r.n << ", " << r.k << "): "
                  << std::fixed << std::setprecision(2)
                  << r.time_ms << " ms (" << r.tops << " TOPS)" << std::endl;
    }

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "✅ hipBLASLt benchmark complete!" << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}

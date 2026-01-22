// hipBLASLt Benchmark - Test ALL algorithms to find the fastest
// PyTorch might be choosing a different algorithm than the default heuristic

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

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

struct AlgoResult {
    int algo_index;
    double time_ms;
    double tops;
};

void test_all_algorithms(int m, int n, int k) {
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Testing all algorithms for " << m << "x" << n << "x" << k << std::endl;
    std::cout << "======================================================================" << std::endl;

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
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));

    // Create operation descriptor
    hipblasLtMatmulDesc_t matmulDesc;
    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t opA = HIPBLAS_OP_N;
    hipblasOperation_t opB = HIPBLAS_OP_N;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Create preference
    hipblasLtMatmulPreference_t pref;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));

    size_t workspace_size = 32 * 1024 * 1024; // 32 MB
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    void* workspace = nullptr;
    HIP_CHECK(hipMalloc(&workspace, workspace_size));

    // Get ALL algorithms
    hipblasLtMatmulHeuristicResult_t heuristicResult[10];
    int returnedAlgoCount;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc,
        matA, matB, matC, matC,
        pref, 10, heuristicResult, &returnedAlgoCount));

    std::cout << "Found " << returnedAlgoCount << " algorithm(s)" << std::endl;
    std::cout << "\nTesting each algorithm..." << std::endl;

    std::vector<AlgoResult> results;

    // Test each algorithm
    for (int algo_idx = 0; algo_idx < returnedAlgoCount; algo_idx++) {
        hipblasLtMatmulAlgo_t algo = heuristicResult[algo_idx].algo;

        std::cout << "\n  Algorithm " << algo_idx << ":" << std::endl;

        // Warmup
        for (int i = 0; i < 5; i++) {
            HIPBLASLT_CHECK(hipblasLtMatmul(
                handle, matmulDesc,
                &alpha, d_A, matA, d_B, matB,
                &beta, d_C, matC, d_C, matC,
                &algo, workspace, workspace_size, nullptr));
        }
        HIP_CHECK(hipDeviceSynchronize());

        // Benchmark
        int bench_iters = 20;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < bench_iters; i++) {
            HIPBLASLT_CHECK(hipblasLtMatmul(
                handle, matmulDesc,
                &alpha, d_A, matA, d_B, matB,
                &beta, d_C, matC, d_C, matC,
                &algo, workspace, workspace_size, nullptr));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        double avg_time_sec = elapsed.count() / bench_iters;
        double avg_time_ms = avg_time_sec * 1000.0;
        double ops = 2.0 * m * n * k;
        double tops = ops / (avg_time_sec * 1e12);

        std::cout << "    Time: " << std::fixed << std::setprecision(2)
                  << avg_time_ms << " ms, Performance: " << tops << " TOPS" << std::endl;

        AlgoResult result;
        result.algo_index = algo_idx;
        result.time_ms = avg_time_ms;
        result.tops = tops;
        results.push_back(result);
    }

    // Sort by performance
    std::sort(results.begin(), results.end(),
              [](const AlgoResult& a, const AlgoResult& b) {
                  return a.tops > b.tops;
              });

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Algorithm Performance Ranking:" << std::endl;
    std::cout << "======================================================================" << std::endl;

    for (size_t i = 0; i < results.size(); i++) {
        std::string medal = (i == 0) ? "🥇" : (i == 1) ? "🥈" : (i == 2) ? "🥉" : "  ";
        std::cout << medal << " Algorithm " << results[i].algo_index << ": "
                  << std::fixed << std::setprecision(2)
                  << results[i].time_ms << " ms (" << results[i].tops << " TOPS)" << std::endl;
    }

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Best Algorithm: " << results[0].algo_index << std::endl;
    std::cout << "Best Performance: " << results[0].tops << " TOPS" << std::endl;
    std::cout << "PyTorch achieves: ~32.22 TOPS (for 4096x4096x4096)" << std::endl;
    if (m == 4096 && n == 4096 && k == 4096) {
        double speedup = 32.22 / results[0].tops;
        std::cout << "PyTorch is " << std::fixed << std::setprecision(2)
                  << speedup << "x faster" << std::endl;
    }
    std::cout << "======================================================================" << std::endl;

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
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "hipBLASLt Algorithm Analysis" << std::endl;
    std::cout << "Testing ALL algorithms to find the fastest" << std::endl;
    std::cout << "======================================================================" << std::endl;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "\nDevice: " << prop.name << std::endl;

    // Test the key size that PyTorch excels at
    test_all_algorithms(4096, 4096, 4096);

    // Also test smaller sizes
    test_all_algorithms(2048, 2048, 2048);

    return 0;
}

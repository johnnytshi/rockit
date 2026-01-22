// Direct comparison: hipBLAS vs hipBLASLt
// PyTorch links to both - let's see which is faster!

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
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
            std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define HIPBLAS_CHECK(call) \
    do { \
        hipblasStatus_t status = call; \
        if (status != HIPBLAS_STATUS_SUCCESS) { \
            std::cerr << "hipBLAS Error: " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

double benchmark_hipblas(int m, int n, int k, void* d_A, void* d_B, void* d_C) {
    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));

    _Float16 alpha_val = static_cast<_Float16>(1.0f);
    _Float16 beta_val = static_cast<_Float16>(0.0f);

    auto alpha_ptr = reinterpret_cast<hipblasHalf*>(&alpha_val);
    auto beta_ptr = reinterpret_cast<hipblasHalf*>(&beta_val);

    // Warmup
    for (int i = 0; i < 5; i++) {
        HIPBLAS_CHECK(hipblasHgemm(
            handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            m, n, k,
            alpha_ptr,
            (hipblasHalf*)d_A, m,
            (hipblasHalf*)d_B, k,
            beta_ptr,
            (hipblasHalf*)d_C, m
        ));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark
    int iters = 20;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; i++) {
        HIPBLAS_CHECK(hipblasHgemm(
            handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            m, n, k,
            alpha_ptr,
            (hipblasHalf*)d_A, m,
            (hipblasHalf*)d_B, k,
            beta_ptr,
            (hipblasHalf*)d_C, m
        ));
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double avg_time_ms = (elapsed.count() / iters) * 1000.0;
    double ops = 2.0 * m * n * k;
    double tops = ops / ((elapsed.count() / iters) * 1e12);

    HIPBLAS_CHECK(hipblasDestroy(handle));

    return tops;
}

double benchmark_hipblaslt(int m, int n, int k, void* d_A, void* d_B, void* d_C) {
    hipblasLtHandle_t handle;
    HIPBLAS_CHECK(hipblasLtCreate(&handle));

    // Matrix layouts
    hipblasLtMatrixLayout_t matA, matB, matC;
    HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m));
    HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k));
    HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));

    // Operation descriptor
    hipblasLtMatmulDesc_t matmulDesc;
    HIPBLAS_CHECK(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t opA = HIPBLAS_OP_N;
    hipblasOperation_t opB = HIPBLAS_OP_N;
    HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Get best algorithm
    hipblasLtMatmulPreference_t pref;
    HIPBLAS_CHECK(hipblasLtMatmulPreferenceCreate(&pref));

    size_t workspace_size = 32 * 1024 * 1024;
    HIPBLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    void* workspace;
    HIP_CHECK(hipMalloc(&workspace, workspace_size));

    hipblasLtMatmulHeuristicResult_t heuristicResult[1];
    int returnedAlgoCount;
    HIPBLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc,
        matA, matB, matC, matC,
        pref, 1, heuristicResult, &returnedAlgoCount));

    hipblasLtMatmulAlgo_t algo = heuristicResult[0].algo;

    // Warmup
    for (int i = 0; i < 5; i++) {
        HIPBLAS_CHECK(hipblasLtMatmul(
            handle, matmulDesc,
            &alpha, d_A, matA, d_B, matB,
            &beta, d_C, matC, d_C, matC,
            &algo, workspace, workspace_size, nullptr));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark
    int iters = 20;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; i++) {
        HIPBLAS_CHECK(hipblasLtMatmul(
            handle, matmulDesc,
            &alpha, d_A, matA, d_B, matB,
            &beta, d_C, matC, d_C, matC,
            &algo, workspace, workspace_size, nullptr));
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double ops = 2.0 * m * n * k;
    double tops = ops / ((elapsed.count() / iters) * 1e12);

    // Cleanup
    HIP_CHECK(hipFree(workspace));
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtMatrixLayoutDestroy(matA);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtDestroy(handle);

    return tops;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "hipBLAS vs hipBLASLt Direct Comparison" << std::endl;
    std::cout << "======================================================================" << std::endl;

    int m = 4096, n = 4096, k = 4096;

    // Allocate memory once
    size_t size_A = m * k * sizeof(_Float16);
    size_t size_B = k * n * sizeof(_Float16);
    size_t size_C = m * n * sizeof(_Float16);

    void *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));

    std::vector<_Float16> h_A(m * k);
    std::vector<_Float16> h_B(k * n);
    for (size_t i = 0; i < h_A.size(); i++) {
        h_A[i] = static_cast<_Float16>(static_cast<float>(rand()) / RAND_MAX);
    }
    for (size_t i = 0; i < h_B.size(); i++) {
        h_B[i] = static_cast<_Float16>(static_cast<float>(rand()) / RAND_MAX);
    }

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));

    std::cout << "\nMatrix size: " << m << "x" << n << "x" << k << std::endl;
    std::cout << "\n======================================================================" << std::endl;

    std::cout << "\nTesting hipBLAS (Hgemm)..." << std::endl;
    double hipblas_tops = benchmark_hipblas(m, n, k, d_A, d_B, d_C);
    std::cout << "hipBLAS Performance: " << std::fixed << std::setprecision(2)
              << hipblas_tops << " TOPS" << std::endl;

    std::cout << "\nTesting hipBLASLt..." << std::endl;
    double hipblaslt_tops = benchmark_hipblaslt(m, n, k, d_A, d_B, d_C);
    std::cout << "hipBLASLt Performance: " << std::fixed << std::setprecision(2)
              << hipblaslt_tops << " TOPS" << std::endl;

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "COMPARISON" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << "hipBLAS:    " << std::fixed << std::setprecision(2) << hipblas_tops << " TOPS" << std::endl;
    std::cout << "hipBLASLt:  " << std::fixed << std::setprecision(2) << hipblaslt_tops << " TOPS" << std::endl;
    std::cout << "PyTorch:    32.22 TOPS (measured)" << std::endl;

    if (hipblas_tops > hipblaslt_tops) {
        std::cout << "\n🏆 hipBLAS is " << (hipblas_tops / hipblaslt_tops) << "x faster!" << std::endl;
        std::cout << "PyTorch might be using hipBLAS for this operation!" << std::endl;
    } else {
        std::cout << "\n🏆 hipBLASLt is " << (hipblaslt_tops / hipblas_tops) << "x faster!" << std::endl;
    }

    std::cout << "======================================================================" << std::endl;

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return 0;
}

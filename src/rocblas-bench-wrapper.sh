#!/bin/bash
# rocBLAS Benchmark Wrapper for FP16 GEMM (BF16 not supported in rocblas-bench gemm)
# Uses the official rocblas-bench tool

echo "======================================================================"
echo "rocBLAS Matrix Multiplication Benchmark (FP16)"
echo "Note: Using FP16 instead of BF16 (BF16 not supported in rocblas-bench)"
echo "======================================================================"

# Test cases: m, n, k
test_cases=(
    "1024 1024 1024"
    "2048 2048 2048"
    "4096 4096 4096"
    "8192 8192 8192"
    "2048 4096 2048"
)

echo ""
echo "======================================================================"
echo "Starting benchmark..."
echo "======================================================================"

count=0
total=${#test_cases[@]}

for test in "${test_cases[@]}"; do
    count=$((count + 1))
    read -r m n k <<< "$test"

    echo ""
    echo "[$count/$total] Testing (m=$m, n=$n, k=$k)..."

    # Run rocblas-bench for GEMM with FP16
    output=$(rocblas-bench -f gemm -r f16_r -m $m -n $n -k $k -i 10 --transposeA N --transposeB N 2>&1)

    # Extract timing info from the last line of output (CSV format)
    # Format: transA,transB,M,N,K,alpha,lda,beta,ldb,ldc,cold_iters,hot_iters,rocblas-Gflops,us
    last_line=$(echo "$output" | tail -1)

    if echo "$last_line" | grep -q "^N,N,"; then
        # Parse the CSV columns: column 13 is Gflops, column 14 is microseconds
        gflops=$(echo "$last_line" | cut -d',' -f13 | tr -d ' ')
        time_us=$(echo "$last_line" | cut -d',' -f14 | tr -d ' ')

        # Convert to ms and TOPS using awk
        time_ms=$(awk "BEGIN {printf \"%.2f\", $time_us / 1000}")
        tops=$(awk "BEGIN {printf \"%.2f\", $gflops / 1000}")

        echo "  Time: ${time_ms} ms, Performance: ${tops} TOPS"
    else
        echo "  Failed to parse output"
    fi
done

echo ""
echo "======================================================================"
echo "✅ rocBLAS benchmark complete!"
echo "======================================================================"

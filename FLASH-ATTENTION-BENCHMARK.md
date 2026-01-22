# Flash Attention Comprehensive Benchmark

This benchmark compares different attention implementations for prefill (long context) and token generation (short context) scenarios.

## What it tests

### Implementations
1. **Flash Attention 2** - Memory-efficient exact attention (if installed)
2. **Triton Flash Attention** - Triton-jit compiled Flash Attention kernel (if Triton + Flash Attention installed)
3. **xFormers** - Memory-efficient attention library (if installed)
4. **PyTorch SDPA** - Built-in scaled_dot_product_attention (may use Flash Attention backend)
5. **PyTorch SDPA (compiled)** - torch.compile version of SDPA (PyTorch 2.0+)
6. **Manual (Naive)** - Standard QK^T attention for baseline comparison
7. **Manual (compiled)** - torch.compile version of manual attention (PyTorch 2.0+)

### Scenarios

**Prefill (Long Context)**
- Processing initial prompts with long sequences
- Tests: 2048, 4096, 8192 token contexts
- Batch sizes: 1, 4

**Token Generation (Incremental)**
- Generating one token at a time with KV cache
- Query length: 1 token
- KV context: 2048, 4096 tokens
- Batch sizes: 1, 4, 8

## Configuration

- **Attention heads**: 32
- **Head dimension**: 128
- **Warmup iterations**: 3
- **Benchmark iterations**: 10

## How to run

```bash
# Via rockit CLI (recommended)
rockit bench
# Then select: "Flash Attention comprehensive (all implementations)"

# Or directly
cd <your-pytorch-project>
uv run python /path/to/rockit/src/flash-attention-benchmark.py
```

## Output

The benchmark produces:
1. **Per-implementation results** for each configuration (time, throughput, memory)
2. **Best times per configuration** with speedup comparisons
3. **Overall winners** for prefill and generation scenarios

Example output:
```
Configuration: Prefill - Small Batch
Batch size: 1
Query seq length: 2048
Key/Value seq length: 2048

Benchmarking Flash Attention 2...
  ✅ 12.34 ms | 165825 tokens/sec | 0.45 GB

Benchmarking PyTorch SDPA...
  ✅ 13.21 ms | 154912 tokens/sec | 0.47 GB

Benchmarking Manual (Naive)...
  ✅ 45.67 ms | 44831 tokens/sec | 1.23 GB

WINNERS BY SCENARIO
🏆 Best for Prefill: Flash Attention 2 (avg 15.42 ms)
🏆 Best for Generation: Flash Attention 2 (avg 0.87 ms)
```

## Results storage

Results are saved to: `~/.config/rockit/benchmark-results/flash_comprehensive_<timestamp>.json`

The JSON includes:
- Full console output
- Parsed structured data (winners, times, throughput)
- System metadata (GPU, ROCm version, PyTorch version, installed packages)

## Requirements

- PyTorch with CUDA/ROCm support
- (Optional) Flash Attention: `pip install flash-attn --no-build-isolation`
- (Optional) xFormers: `pip install xformers`

Without Flash Attention or xFormers, the benchmark will still compare PyTorch SDPA vs Manual implementations.

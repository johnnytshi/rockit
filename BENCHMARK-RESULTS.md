# Benchmark Results Storage

## Overview
Rockit automatically stores all benchmark results with comprehensive metadata in `~/.config/rockit/benchmark-results/`.

## What's Stored

Each benchmark run saves a JSON file containing:

### 1. System Metadata
- **Operating System**: OS type, platform, release, architecture
- **Hardware**: CPU model, total memory, hostname
- **GPU**: Architecture (e.g., gfx1151), device ID
- **ROCm**: Full version with build tag (e.g., 7.10.0a20251029)
- **Python**: Version used
- **Packages**: Installed versions of torch, torchvision, torchaudio, flash-attn

### 2. Benchmark Results
- **Raw Output**: Complete console output from the benchmark
- **Parsed Results**: Structured data extracted from output
  - Matrix multiplication: Size, time (ms), performance (GFLOPS)
  - Flash Attention: Time (ms), throughput (tokens/sec)

### 3. Timestamp
- ISO 8601 format timestamp for when the benchmark was run

## File Naming
Results are saved with the format:
```
{benchmarkType}_{timestamp}.json
```

Examples:
- `full_2025-10-30T12-34-56-789Z.json`
- `matrix_2025-10-30T13-45-12-345Z.json`
- `flash_2025-10-30T14-56-23-456Z.json`

## Viewing Results

### Via CLI
```bash
rockit bench
# Select "View past results" from the menu
```

This provides a formatted view with:
- System information
- GPU and ROCm details
- Package versions
- Parsed benchmark metrics
- Path to full JSON file

### Programmatically
Results are stored as JSON files that can be parsed by any tool:

```bash
# List all results
ls ~/.config/rockit/benchmark-results/

# View a specific result
cat ~/.config/rockit/benchmark-results/full_2025-10-30T12-34-56-789Z.json | jq

# Compare multiple results
jq '.parsed.matrixMultiplication' ~/.config/rockit/benchmark-results/*.json
```

## Example Result Structure

See `example-benchmark-result.json` for a complete example of the JSON structure.

## Use Cases

1. **Performance Tracking**: Compare results across different ROCm/PyTorch versions
2. **System Validation**: Verify installation success and performance
3. **Debugging**: Failed benchmarks are also saved with error information
4. **Reporting**: Generate performance reports from historical data
5. **Optimization**: Track improvements after system/configuration changes

## Storage Location
```
~/.config/rockit/
├── config.json              # Current configuration
└── benchmark-results/       # All benchmark results
    ├── basic_*.json
    ├── matrix_*.json
    ├── flash_*.json
    └── full_*.json
```

## Tips

- Results include both successful and failed benchmarks for debugging
- Use `jq` to extract specific metrics: `jq '.parsed.flashAttention.tokensPerSec' result.json`
- Archive old results periodically to save space
- Compare results across different configurations to optimize performance

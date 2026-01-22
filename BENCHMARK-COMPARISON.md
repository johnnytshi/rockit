# Benchmark Comparison Examples

Now that your results are properly parsed with structured JSON data, here are useful ways to compare them:

## Quick Comparisons

### 1. Compare peak performance across all runs
```bash
jq -r '["Timestamp", "Max GFLOPS", "Flash Tokens/s"], ([.metadata.timestamp[0:19], (.parsed.matrixMultiplication | max_by(.gflops) | .gflops | tostring), (.parsed.flashAttention.tokensPerSec | tostring)]) | @tsv' ~/.config/rockit/benchmark-results/full_*.json | column -t -s (printf '\t')
```

### 2. Compare matrix performance by size
```bash
# Get 8192x8192x8192 performance from all runs
jq '.parsed.matrixMultiplication[] | select(.m == 8192) | {timestamp: .metadata.timestamp, m, n, k, gflops}' ~/.config/rockit/benchmark-results/*.json

# Compare specific matrix size across all runs
jq -r '["Timestamp", "m", "n", "k", "GFLOPS"], ([.metadata.timestamp[0:19], (.parsed.matrixMultiplication[] | select(.m == 4096) | .m, .n, .k, .gflops)]) | @tsv' ~/.config/rockit/benchmark-results/full_*.json | column -t -s (printf '\t')
```

### 3. Compare Flash Attention throughput
```bash
jq '{timestamp: .metadata.timestamp[0:19], tokensPerSec: .parsed.flashAttention.tokensPerSec}' ~/.config/rockit/benchmark-results/full_*.json
```

### 4. Full comparison table
```bash
jq -r '["Time", "ROCm", "PyTorch", "1024 GFLOPS", "2048 GFLOPS", "4096 GFLOPS", "8192 GFLOPS", "Flash tok/s"], ([.metadata.timestamp[11:19], .metadata.rocm.fullVersion, .metadata.packages.torch[0:15], (.parsed.matrixMultiplication[0].gflops | tostring), (.parsed.matrixMultiplication[1].gflops | tostring), (.parsed.matrixMultiplication[2].gflops | tostring), (.parsed.matrixMultiplication[3].gflops | tostring), (.parsed.flashAttention.tokensPerSec | tostring)]) | @tsv' ~/.config/rockit/benchmark-results/full_*.json | column -t -s (printf '\t')
```

### 5. Find best performing run
```bash
# Best matrix performance
jq -s 'max_by(.parsed.matrixMultiplication | max_by(.gflops) | .gflops) | {timestamp: .metadata.timestamp, maxGflops: (.parsed.matrixMultiplication | max_by(.gflops) | .gflops)}' ~/.config/rockit/benchmark-results/full_*.json

# Best Flash Attention
jq -s 'max_by(.parsed.flashAttention.tokensPerSec) | {timestamp: .metadata.timestamp, tokensPerSec: .parsed.flashAttention.tokensPerSec}' ~/.config/rockit/benchmark-results/full_*.json
```

### 6. Export for analysis
```bash
# Export to CSV
echo "timestamp,m,n,k,timeMs,gflops" > matrix_results.csv
jq -r '.parsed.matrixMultiplication[] | [.metadata.timestamp, .m, .n, .k, .timeMs, .gflops] | @csv' ~/.config/rockit/benchmark-results/full_*.json >> matrix_results.csv

# Export Flash Attention results
echo "timestamp,timeMs,tokensPerSec" > flash_results.csv
jq -r '[.metadata.timestamp, .parsed.flashAttention.timeMs, .parsed.flashAttention.tokensPerSec] | @csv' ~/.config/rockit/benchmark-results/full_*.json >> flash_results.csv
```

## Python Analysis Example

```python
import json
import glob
from pathlib import Path

# Load all results
results_dir = Path.home() / '.config' / 'rockit' / 'benchmark-results'
results = []
for file in results_dir.glob('full_*.json'):
    with open(file) as f:
        results.append(json.load(f))

# Compare performance
for r in results:
    timestamp = r['metadata']['timestamp']
    max_gflops = max(m['gflops'] for m in r['parsed']['matrixMultiplication'])
    tokens_per_sec = r['parsed']['flashAttention']['tokensPerSec']
    print(f"{timestamp}: {max_gflops:.2f} GFLOPS, {tokens_per_sec:,.0f} tokens/s")

# Plot performance over time (requires matplotlib)
import matplotlib.pyplot as plt
timestamps = [r['metadata']['timestamp'] for r in results]
gflops = [max(m['gflops'] for m in r['parsed']['matrixMultiplication']) for r in results]
plt.plot(timestamps, gflops)
plt.xlabel('Timestamp')
plt.ylabel('Max GFLOPS')
plt.title('Performance Over Time')
plt.show()
```

## Current Results

Run this to see your current comparison:
```bash
jq -r '["Timestamp", "ROCm Version", "Max GFLOPS", "Flash Tokens/s"], ["--------", "-------", "----------", "-------------"], ([.metadata.timestamp[0:19], .metadata.rocm.fullVersion, (.parsed.matrixMultiplication | max_by(.gflops) | .gflops | tostring), (.parsed.flashAttention.tokensPerSec | tostring)]) | @tsv' ~/.config/rockit/benchmark-results/full_*.json | column -t -s (printf '\t')
```

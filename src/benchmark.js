#!/usr/bin/env node

const inquirer = require('inquirer');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { execSync } = require('child_process');
const { loadConfig } = require('./config');
const { detectSystem } = require('./system-detect');
const { createPackageManager } = require('./package-managers/factory');

const RESULTS_DIR = path.join(os.homedir(), '.config', 'rockit', 'benchmark-results');

/**
 * Run benchmark in the project environment
 * @param {string} projectPath - Path to project directory
 * @param {string} benchmarkScript - Python code to run
 * @param {string} packageManager - Package manager to use ('uv' or 'pip')
 * @returns {Promise<Object>} Benchmark results
 */
async function runBenchmark(projectPath, benchmarkScript, packageManager = 'uv') {
  const scriptPath = path.join(projectPath, '.benchmark_temp.py');

  try {
    // Write benchmark script to temp file
    fs.writeFileSync(scriptPath, benchmarkScript);

    // Get the appropriate Python command
    const pkgManager = createPackageManager(packageManager);
    const pythonCmd = pkgManager.runPythonCommand(projectPath, scriptPath);

    const output = execSync(pythonCmd, {
      cwd: projectPath,
      encoding: 'utf8',
      stdio: 'pipe'
    });

    return { success: true, output };
  } catch (error) {
    return { success: false, error: error.message, output: error.stdout || '' };
  } finally {
    // Clean up temp file
    if (fs.existsSync(scriptPath)) {
      fs.unlinkSync(scriptPath);
    }
  }
}

/**
 * Basic PyTorch availability check
 */
const BASIC_CHECK = `
import sys
import torch

print("=" * 60)
print("PyTorch Environment Check")
print("=" * 60)
print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  CUDA/ROCm not available!")
    sys.exit(1)

print("=" * 60)
`;

/**
 * Matrix multiplication benchmark
 */
const MATRIX_BENCHMARK = `
import torch
import time

print("\\n" + "=" * 60)
print("Matrix Multiplication Benchmark (BF16)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Data type: bfloat16")

# Test all combinations of matrix dimensions
dims = [1024, 2048, 4096, 8192]
results = []

total_tests = len(dims) ** 3
current_test = 0

for m in dims:
    for n in dims:
        for k in dims:
            current_test += 1
            print(f"\\n[{current_test}/{total_tests}] Testing (m={m}, n={n}, k={k})...")
            
            # Create random matrices in BF16: C(m×n) = A(m×k) × B(k×n)
            A = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            B = torch.randn(k, n, device=device, dtype=torch.bfloat16)
            
            # Warmup
            for _ in range(3):
                _ = torch.mm(A, B)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            iterations = 5
            start = time.time()
            for _ in range(iterations):
                C = torch.mm(A, B)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            avg_time = elapsed / iterations
            
            # Calculate TOPS (Tera Operations Per Second)
            # Matrix multiplication: 2*m*n*k operations
            tops = (2 * m * n * k) / (avg_time * 1e12)
            
            print(f"  Time: {avg_time*1000:.2f} ms, Performance: {tops:.2f} TOPS")
            
            results.append({
                'm': m,
                'n': n,
                'k': k,
                'time_ms': avg_time * 1000,
                'tops': tops
            })

print("\\n" + "=" * 60)
print(f"Summary: {len(results)} combinations tested")
print("=" * 60)
# Sort by TOPS descending and show top 10
results_sorted = sorted(results, key=lambda x: x['tops'], reverse=True)
print("\\nTop 10 performers:")
for i, r in enumerate(results_sorted[:10], 1):
    print(f"  {i}. ({r['m']}, {r['n']}, {r['k']}): {r['time_ms']:.2f} ms ({r['tops']:.2f} TOPS)")

print("\\nAll results:")
for r in results:
    print(f"  ({r['m']}, {r['n']}, {r['k']}): {r['time_ms']:.2f} ms ({r['tops']:.2f} TOPS)")
print("=" * 60)
`;

/**
 * Flash Attention benchmark
 */
const FLASH_ATTENTION_BENCHMARK = `
import torch
import time

print("\\n" + "=" * 60)
print("Flash Attention Benchmark")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

try:
    from flash_attn import flash_attn_func
    print("✅ Flash Attention available")
except ImportError:
    print("❌ Flash Attention not installed")
    print("Install with: uv pip install flash-attn --no-build-isolation")
    import sys
    sys.exit(1)

# Test parameters
batch_size = 4
seq_len = 2048
num_heads = 32
head_dim = 128

print(f"\\nTest configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len}")
print(f"  Number of heads: {num_heads}")
print(f"  Head dimension: {head_dim}")

# Create random inputs
q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)

# Warmup
print("\\nWarming up...")
for _ in range(5):
    _ = flash_attn_func(q, k, v)

if torch.cuda.is_available():
    torch.cuda.synchronize()

# Benchmark
print("Running benchmark...")
iterations = 20
start = time.time()

for _ in range(iterations):
    output = flash_attn_func(q, k, v)

if torch.cuda.is_available():
    torch.cuda.synchronize()

elapsed = time.time() - start
avg_time = elapsed / iterations

print(f"\\nResults:")
print(f"  Average time: {avg_time*1000:.2f} ms")
print(f"  Throughput: {batch_size * seq_len / avg_time:.0f} tokens/sec")
print(f"  Memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("=" * 60)
`;

/**
 * Combined benchmark script
 */
const FULL_BENCHMARK = `
${BASIC_CHECK}
${MATRIX_BENCHMARK}

print("\\n" + "=" * 60)
print("✅ Basic benchmarks completed successfully!")
print("=" * 60)
`;

/**
 * Prompt and run benchmarks
 */
async function listResults() {
  console.log('\n📊 AIBenchy - Benchmark Results\n');
  
  if (!fs.existsSync(RESULTS_DIR)) {
    console.log('No benchmark results found yet.');
    console.log('Run "rockit bench" to create your first benchmark.\n');
    return;
  }
  
  const files = fs.readdirSync(RESULTS_DIR)
    .filter(f => f.endsWith('.json'))
    .sort()
    .reverse(); // Most recent first
  
  if (files.length === 0) {
    console.log('No benchmark results found yet.');
    console.log('Run "rockit bench" to create your first benchmark.\n');
    return;
  }
  
  console.log(`Found ${files.length} result(s) in ${RESULTS_DIR}\n`);
  
  const { selectedFile } = await inquirer.prompt([
    {
      type: 'list',
      name: 'selectedFile',
      message: 'Select result to view:',
      choices: [
        ...files.map(f => ({ name: f, value: f })),
        { name: 'Cancel', value: null }
      ],
      loop: false
    }
  ]);
  
  if (!selectedFile) {
    console.log('\nCancelled.');
    return;
  }
  
  const filepath = path.join(RESULTS_DIR, selectedFile);
  const result = JSON.parse(fs.readFileSync(filepath, 'utf8'));
  
  console.log('\n' + '='.repeat(60));
  console.log(`Benchmark: ${result.benchmarkType}`);
  console.log(`Timestamp: ${result.metadata.timestamp}`);
  console.log('='.repeat(60));
  
  // System info
  console.log('\n📱 System:');
  console.log(`  OS: ${result.metadata.system.os} ${result.metadata.system.release}`);
  console.log(`  CPU: ${result.metadata.system.cpus}`);
  console.log(`  Memory: ${result.metadata.system.totalMemory}`);
  
  // GPU info
  if (result.metadata.gpu) {
    console.log('\n🎮 GPU:');
    console.log(`  Architecture: ${result.metadata.gpu.architecture || 'N/A'}`);
    if (result.metadata.gpu.deviceId) {
      console.log(`  Device ID: ${result.metadata.gpu.deviceId}`);
    }
  }
  
  // ROCm info
  if (result.metadata.rocm && result.metadata.rocm.fullVersion) {
    console.log('\n🔧 ROCm:');
    console.log(`  Version: ${result.metadata.rocm.fullVersion}`);
  }
  
  // Python packages
  if (result.metadata.packages) {
    console.log('\n📦 Packages:');
    for (const [pkg, version] of Object.entries(result.metadata.packages)) {
      console.log(`  ${pkg}: ${version}`);
    }
  }
  
  // Parsed results
  if (result.parsed) {
    if (result.parsed.matrixMultiplication && result.parsed.matrixMultiplication.length > 0) {
      console.log('\n📊 Matrix Multiplication Results:');
      for (const test of result.parsed.matrixMultiplication) {
        const perf = test.tops ? `${test.tops.toFixed(2)} TOPS` : `${test.gflops.toFixed(2)} GFLOPS`;
        console.log(`  (${test.m}, ${test.n}, ${test.k}): ${test.timeMs.toFixed(2)} ms (${perf})`);
      }
    }
    
    if (result.parsed.flashAttention && result.parsed.flashAttention.timeMs) {
      console.log('\n⚡ Flash Attention Results:');
      console.log(`  Time: ${result.parsed.flashAttention.timeMs.toFixed(2)} ms`);
      console.log(`  Throughput: ${result.parsed.flashAttention.tokensPerSec.toFixed(2)} tokens/sec`);
    }
  }
  
  console.log('\n' + '='.repeat(60));
  console.log(`Full results: ${filepath}`);
  console.log('='.repeat(60) + '\n');
}

async function promptBenchmark() {
  console.log('\n🔥 AIBenchy - Performance Benchmark\n');
  
  // Load config to get project path
  const config = loadConfig();
  
  if (!config.projectPath) {
    console.error('❌ No project configured. Run "rockit python" first to set up a project.');
    process.exit(1);
  }
  
  if (!fs.existsSync(config.projectPath)) {
    console.error(`❌ Project not found at: ${config.projectPath}`);
    console.error('Run "rockit python" to set up the project.');
    process.exit(1);
  }
  
  console.log(`Project: ${config.projectPath}\n`);
  
  // Check if PyTorch is installed
  const pyprojectPath = path.join(config.projectPath, 'pyproject.toml');
  if (!fs.existsSync(pyprojectPath)) {
    console.error('❌ Project not initialized. Run "rockit python" first.');
    process.exit(1);
  }
  
  const { benchmarkType } = await inquirer.prompt([
    {
      type: 'list',
      name: 'benchmarkType',
      message: 'Select benchmark to run:',
      choices: [
        { name: 'Basic PyTorch check (quick)', value: 'basic' },
        { name: 'Matrix multiplication benchmark', value: 'matrix' },
        { name: 'Flash Attention benchmark (simple)', value: 'flash' },
        { name: 'Flash Attention comprehensive (all implementations)', value: 'flash_comprehensive' },
        { name: 'Full benchmark suite (all tests)', value: 'full' },
        { name: 'View past results', value: 'results' },
        { name: 'Cancel', value: 'cancel' }
      ],
      loop: false
    }
  ]);
  
  if (benchmarkType === 'cancel') {
    console.log('\nCancelled.');
    process.exit(0);
  }
  
  if (benchmarkType === 'results') {
    await listResults();
    return;
  }
  
  let script;
  let useExternalScript = false;
  let runComprehensiveAfter = false;
  
  switch (benchmarkType) {
    case 'basic':
      script = BASIC_CHECK;
      break;
    case 'matrix':
      script = BASIC_CHECK + MATRIX_BENCHMARK;
      break;
    case 'flash':
      script = BASIC_CHECK + FLASH_ATTENTION_BENCHMARK;
      break;
    case 'flash_comprehensive':
      // Use external comprehensive benchmark script
      useExternalScript = true;
      break;
    case 'full':
      script = FULL_BENCHMARK;
      runComprehensiveAfter = true;
      break;
  }
  
  console.log('\n🚀 Running benchmark...\n');
  console.log('─'.repeat(60));
  
  // Collect system metadata before running benchmark
  console.log('\n📊 Collecting system metadata...');
  const metadata = collectSystemMetadata();
  
  // Get package manager from config (default to uv for backward compatibility)
  const packageManager = config.packageManager || 'uv';
  const pkgManager = createPackageManager(packageManager);

  let result;

  if (useExternalScript) {
    // Run the comprehensive benchmark script directly
    const comprehensiveScriptPath = path.join(__dirname, 'flash-attention-benchmark.py');
    if (!fs.existsSync(comprehensiveScriptPath)) {
      console.error('❌ Comprehensive benchmark script not found:', comprehensiveScriptPath);
      process.exit(1);
    }

    try {
      const pythonCmd = pkgManager.runPythonCommand(config.projectPath, comprehensiveScriptPath);
      const output = execSync(pythonCmd, {
        cwd: config.projectPath,
        encoding: 'utf8',
        stdio: 'pipe'
      });
      result = { success: true, output };
    } catch (error) {
      result = { success: false, error: error.message, output: error.stdout || '' };
    }
  } else {
    result = await runBenchmark(config.projectPath, script, packageManager);
  }
  
  if (result.success) {
    console.log(result.output);
    console.log('─'.repeat(60));
    console.log('\n✅ Benchmark completed successfully!\n');
    
    // Save results with metadata
    saveResults(benchmarkType, result.output, metadata);
    
    // If this is the full suite, now run the comprehensive flash attention benchmark
    if (runComprehensiveAfter) {
      console.log('\n🚀 Running comprehensive Flash Attention benchmark...\n');
      console.log('─'.repeat(60));

      const comprehensiveScriptPath = path.join(__dirname, 'flash-attention-benchmark.py');
      if (!fs.existsSync(comprehensiveScriptPath)) {
        console.error('❌ Comprehensive benchmark script not found:', comprehensiveScriptPath);
      } else {
        try {
          const pythonCmd = pkgManager.runPythonCommand(config.projectPath, comprehensiveScriptPath);
          const comprehensiveOutput = execSync(pythonCmd, {
            cwd: config.projectPath,
            encoding: 'utf8',
            stdio: 'pipe'
          });
          console.log(comprehensiveOutput);
          console.log('─'.repeat(60));
          console.log('\n✅ Comprehensive benchmark completed successfully!\n');
          saveResults('flash_comprehensive', comprehensiveOutput, metadata);
        } catch (error) {
          console.error('\n❌ Comprehensive benchmark failed:', error.message);
          if (error.stdout) {
            console.log(error.stdout);
          }
        }
      }
    }
  } else {
    console.log(result.output);
    console.error('\n❌ Benchmark failed:', result.error);
    console.error('\nTroubleshooting:');
    console.error('  1. Make sure PyTorch is installed: rockit python');
    console.error('  2. Check that ROCm is properly configured');
    console.error('  3. Source environment variables: source .env\n');
    
    // Save failed results too for debugging
    saveResults(benchmarkType + '_failed', result.output + '\n\nError: ' + result.error, metadata);
    process.exit(1);
  }
}

function collectSystemMetadata() {
  const metadata = {
    timestamp: new Date().toISOString(),
    system: {
      os: os.type(),
      platform: os.platform(),
      release: os.release(),
      arch: os.arch(),
      hostname: os.hostname(),
      cpus: os.cpus()[0].model,
      totalMemory: `${(os.totalmem() / (1024 ** 3)).toFixed(2)} GB`,
    },
  };

  // Get GPU info
  try {
    const systemInfo = detectSystem();
    metadata.gpu = {
      architecture: systemInfo.gpuArch,
      deviceId: systemInfo.deviceId || 'unknown',
    };
  } catch (error) {
    metadata.gpu = { error: error.message };
  }

  // Get ROCm version
  try {
    const buildInfoPath = '/opt/rocm/.info/build-info.json';
    if (fs.existsSync(buildInfoPath)) {
      const buildInfo = JSON.parse(fs.readFileSync(buildInfoPath, 'utf8'));
      metadata.rocm = {
        version: buildInfo.rocmVersion,
        buildTag: buildInfo.buildTag,
        fullVersion: `${buildInfo.rocmVersion}${buildInfo.buildTag}`,
      };
    } else {
      metadata.rocm = { installed: false };
    }
  } catch (error) {
    metadata.rocm = { error: error.message };
  }

  // Get Python/PyTorch/Flash Attention versions from config
  try {
    const config = loadConfig();
    if (config.pythonVersion) {
      metadata.python = { version: config.pythonVersion };
    }
    if (config.installedPackages) {
      metadata.packages = config.installedPackages;
    }
  } catch (error) {
    metadata.config = { error: error.message };
  }

  return metadata;
}

function saveResults(benchmarkType, output, metadata) {
  try {
    // Ensure results directory exists
    if (!fs.existsSync(RESULTS_DIR)) {
      fs.mkdirSync(RESULTS_DIR, { recursive: true });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `${benchmarkType}_${timestamp}.json`;
    const filepath = path.join(RESULTS_DIR, filename);

    const result = {
      benchmarkType,
      metadata,
      output,
      rawOutput: output,
    };

    // Parse output for structured data
    if (benchmarkType === 'matrix' || benchmarkType === 'full') {
      result.parsed = parseMatrixResults(output);
    }
    if (benchmarkType === 'flash' || benchmarkType === 'full') {
      result.parsed = { ...result.parsed, ...parseFlashResults(output) };
    }
    if (benchmarkType === 'flash_comprehensive') {
      result.parsed = parseComprehensiveFlashResults(output);
    }

    fs.writeFileSync(filepath, JSON.stringify(result, null, 2));
    console.log(`\n💾 Results saved to: ${filepath}`);

    return filepath;
  } catch (error) {
    console.error('⚠️  Failed to save results:', error.message);
    return null;
  }
}

function parseMatrixResults(output) {
  const results = { matrixMultiplication: [] };
  const lines = output.split('\n');
  
  for (const line of lines) {
    // Match TOPS format: "  (1024, 2048, 4096): 12.34 ms (1.23 TOPS)"
    let match = line.match(/\((\d+),\s*(\d+),\s*(\d+)\):\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+TOPS\)/);
    if (match) {
      results.matrixMultiplication.push({
        m: parseInt(match[1]),
        n: parseInt(match[2]),
        k: parseInt(match[3]),
        timeMs: parseFloat(match[4]),
        tops: parseFloat(match[5]),
      });
      continue;
    }
    
    // Match GFLOPS format: "  (1024, 2048, 4096): 12.34 ms (1234.56 GFLOPS)"
    match = line.match(/\((\d+),\s*(\d+),\s*(\d+)\):\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+GFLOPS\)/);
    if (match) {
      results.matrixMultiplication.push({
        m: parseInt(match[1]),
        n: parseInt(match[2]),
        k: parseInt(match[3]),
        timeMs: parseFloat(match[4]),
        gflops: parseFloat(match[5]),
      });
      continue;
    }
    
    // Match old format: "  1024x1024: 1.07 ms (2010.63 GFLOPS)" or "Size 1024x1024: ..."
    match = line.match(/(?:Size\s+)?(\d+)x(\d+):\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+GFLOPS\)/);
    if (match) {
      const m = parseInt(match[1]);
      const n = parseInt(match[2]);
      const k = m; // For square matrices, k = m = n
      results.matrixMultiplication.push({
        m: m,
        n: n,
        k: k,
        timeMs: parseFloat(match[3]),
        gflops: parseFloat(match[4]),
      });
    }
  }
  
  return results;
}

function parseFlashResults(output) {
  const results = { flashAttention: {} };
  const lines = output.split('\n');
  
  for (const line of lines) {
    // Match "Average time: 25.43 ms" or "Time: 25.43 ms"
    const timeMatch = line.match(/(?:Average )?[Tt]ime:\s+([\d.]+)\s+ms/);
    // Match "Throughput: 322180 tokens/sec"
    const throughputMatch = line.match(/Throughput:\s+([\d.]+)\s+tokens\/sec/);
    
    if (timeMatch) {
      results.flashAttention.timeMs = parseFloat(timeMatch[1]);
    }
    if (throughputMatch) {
      results.flashAttention.tokensPerSec = parseFloat(throughputMatch[1]);
    }
  }
  
  return results;
}

function parseComprehensiveFlashResults(output) {
  const results = { comprehensiveFlashAttention: [] };
  const lines = output.split('\n');
  
  let currentConfig = 'Unknown';
  let currentScenario = 'Unknown';
  let currentImpl = 'Unknown';

  for (const line of lines) {
    // Capture the current configuration
    let configMatch = line.match(/^Configuration: (.*)/);
    if (configMatch) {
      currentConfig = configMatch[1].trim();
      currentScenario = currentConfig.toLowerCase().includes('prefill') ? 'prefill' : 'generation';
      continue;
    }

    // Capture the implementation being benchmarked
    let implMatch = line.match(/^Benchmarking (.*)\.\.\./);
    if (implMatch) {
      currentImpl = implMatch[1].trim();
      continue;
    }

    // Capture the successful result for the current implementation
    const successMatch = line.match(/✅\s+([\d.]+)\s+ms\s+\|\s+([\d.]+)\s+tokens\/sec\s+\|\s+([\d.]+)\s+GB/);
    if (successMatch) {
      results.comprehensiveFlashAttention.push({
        implementation: currentImpl,
        config: currentConfig,
        scenario: currentScenario,
        timeMs: parseFloat(successMatch[1]),
        tokensPerSec: parseFloat(successMatch[2]),
        memoryGb: parseFloat(successMatch[3]),
        success: true,
      });
      continue;
    }

    // Capture the failed result for the current implementation
    const failMatch = line.match(/⚠️\s+(.*) failed: (.*)/);
    if (failMatch && failMatch[1].trim() === currentImpl) {
      results.comprehensiveFlashAttention.push({
        implementation: currentImpl,
        config: currentConfig,
        scenario: currentScenario,
        success: false,
        error: failMatch[2].trim(),
      });
    }
  }
  
  // Parse winner summary
  // Format: "🏆 Best for Prefill: Flash Attention 2 (avg 12.34 ms)"
  const prefillWinner = output.match(/🏆 Best for Prefill:\s+([^(]+)\s+\(avg\s+([\d.]+)\s+ms\)/);
  if (prefillWinner) {
    results.prefillWinner = {
      implementation: prefillWinner[1].trim(),
      avgTimeMs: parseFloat(prefillWinner[2])
    };
  }
  
  const generationWinner = output.match(/🏆 Best for Generation:\s+([^(]+)\s+\(avg\s+([\d.]+)\s+ms\)/);
  if (generationWinner) {
    results.generationWinner = {
      implementation: generationWinner[1].trim(),
      avgTimeMs: parseFloat(generationWinner[2])
    };
  }
  
  return results;
}

module.exports = { promptBenchmark, listResults };

// Run if called directly
if (require.main === module) {
  promptBenchmark().catch(error => {
    console.error('\n❌ Error:', error.message);
    process.exit(1);
  });
}

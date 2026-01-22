# Rockit – Project Summary and Function Reference

This document summarizes what’s implemented in the repo, and lists the key functions, what they do, and where any outputs are written.

## Overview

- ROCm installer and artifact discovery (interactive CLI)
- PyTorch installer for AMD ROCm nightlies (interactive CLI)
- System/GPU detection and compatibility filtering
- PyPI helper for Flash Attention versions
- Benchmark runner and results parser/saver
- Utilities to list/nightly builds and reprocess old benchmark results
- Web-based benchmark result viewer.

Top-level CLI: `bin/rockit.js`
- Commands: `rockit rocm`, `rockit python`, `rockit detect`, `rockit config`, `rockit bench`, `rockit view`

## Data and output locations

- ROCm install path: `/opt/rocm` (writes `.info/build-info.json` on install)
- Downloads cache: `~/.cache/rockit/`
- Config file: `~/.config/rockit/config.json`
- Benchmark results: `~/.config/rockit/benchmark-results/*.json`
- Project env file (PyTorch): `<your project>/.env`

---

## CLI entry point

### `bin/rockit.js`
- Wires subcommands to their respective modules.
- Exposes: `rocm`, `python`, `detect`, `config`, `bench`.
- Output: console output per command; most side effects are delegated to modules below.

---

## ROCm installation flow

### `src/cli.js`
- `checkCurrentRocm()`
   - Detects existing ROCm under `/opt/rocm` and reads version/build metadata.
   - Output: returns info object; prints details if invoked via `promptInstallation()`.
- `promptInstallation()`
   - Full interactive ROCm install workflow: detect system → fetch artifacts → select version/variant/build → download → install → setup → verify.
   - Output: console logs; installs to `/opt/rocm`; may back up existing install to `/opt/rocm.backup.<timestamp>`.
- `getFileSize(url)`
   - Estimates the tarball size via HTTP HEAD.
   - Output: returns MB estimate (string).

### `src/installer.js`
- `downloadFile(url, destPath)`
   - Streams a file to disk with progress.
   - Output: writes file to `destPath`.
- `extractTarGz(tarPath, destDir)`
   - Extracts tar.gz into a directory.
   - Output: files in `destDir`.
- `requiresSudo(dir)`
   - Checks if write access requires sudo.
   - Output: boolean.
- `installRocm(tarPath, installDir = '/opt/rocm', buildInfo)`
   - Extracts, backs up existing `/opt/rocm`, installs new files, sets permissions, and writes `.info/build-info.json`.
   - Output: files under `installDir`; metadata in `installDir/.info/build-info.json`.
- `setupEnvironment(installDir)`
   - Prints environment export lines for bash/fish.
   - Output: console only (user applies).
- `verifyInstallation(installDir)`
   - Runs `rocminfo` to validate.
   - Output: console logs; return boolean.

### `src/rocm-parser.js`
- `parseRocmArtifacts()`
   - Fetches ROCm nightly index and parses available tarballs.
   - Output: returns array of artifact objects.
- `parseArtifactInfo(url, filename)`
   - Parses filename into fields: platform/gpu/variant/rocmVersion/buildTag/buildDate.
   - Output: returns normalized artifact object.
- `filterArtifacts(artifacts, filters)`
   - Filters by platform/gpu/variant/rocmVersion and can pick latest per group.
   - Output: filtered array.
- `getUniqueValues(artifacts, field)`
   - Distinct values for a field.
   - Output: string[]

### `src/system-detect.js`
- `detectPlatform()` → `'linux' | 'windows'`.
- `detectGpuArch()` → detects AMD GPU arch via `rocminfo`/`lspci`/name heuristics.
- `mapDeviceIdToArch(deviceId)` → maps PCI IDs to gfx arch.
- `mapArchToRocmGpu(arch)` → maps `gfxXXXX` to ROCm GPU families.
- `detectSystem()` → aggregates platform, arch, ROCm families, OS info.
- `findCompatibleArtifacts(artifacts, systemInfo, options)` → filters artifacts for the machine.
   - Output: return values only; console warnings on missing GPU.

---

## PyTorch installation flow

### `src/pytorch-cli.js`
- `checkCurrentRocm()`
   - Reads `/opt/rocm/.info/build-info.json` or `.info/version` for display.
   - Output: version string or null.
- `promptPyTorchInstallation()`
   - Full interactive flow to select Python version and PyTorch nightly wheels for the detected GPU; initializes a `uv` project if needed; installs torch/vision/audio and optional Flash Attention; writes `.env`.
   - Output: packages installed into the uv environment at your chosen project path; writes `<project>/.env` and updates `~/.config/rockit/config.json`.

### `src/pytorch-installer.js`
- `isUvInstalled()` → checks for `uv`.
- `getPythonVersion()` → system Python major.minor.
- `initializeUvProject(projectPath, pythonVersion)` → runs `uv init`.
- `installPyTorchPackages(projectPath, gpuArch, packages, options)` → installs packages from AMD ROCm index; optional Flash Attention.
   - Output: packages in uv environment under `projectPath`.
- `updatePyTorchPackages(projectPath, gpuArch, packages)` → updates specific packages to chosen versions.
- `isProjectInitialized(projectPath)` → `pyproject.toml` exists.
- `getInstalledPackages(projectPath)` → returns `{ name: version }` map via `uv pip list`.
- `setupEnvironmentVariables(projectPath, options)` → writes `<projectPath>/.env` with ROCm and optional Flash Attention flags.

### `src/pytorch-parser.js`
- `parsePackageVersions(gpuArch, packageName)` → lists wheels for a package.
- `parsePyTorchPackages(gpuArch)` → aggregates torch/vision/audio versions for arch.
- `parsePackageName(filename)` → parses wheel filename metadata.
- `compareVersions(a, b)` → version comparator.
- `groupPackagesByName(packages)` → map of name → sorted versions.
- `getLatestVersions(packages)` → latest per package.
- `filterByPythonVersion(packages, py)`; `filterByPlatform(packages, platform)`.
- `getAvailablePythonVersions(packages)` → distinct supported Python versions.
   - Output: return values only.

### `src/pypi-parser.js`
- `fetchPyPiVersions(packageName)` → versions from PyPI (e.g., `flash-attn`).
- `compareVersions(a, b)` → version comparator.
- `fetchPackageInfo(packageName)` → PyPI package metadata.
   - Output: return values only.

### `src/list-pytorch.js`
- `listPyTorchVersions()`
   - Prints available PyTorch nightly versions for the detected GPU and platform, grouped by Python versions, with install guidance.
   - Output: console only.

### `src/config.js`
- `getDefaultConfig()` → default config structure.
- `loadConfig()` / `saveConfig(config)` / `updateConfig(updates)` / `displayConfig(config)` / `resetConfig()`.
   - Output: `save/update/reset` write `~/.config/rockit/config.json`; `display` prints.

---

## Benchmarking

### `src/benchmark.js`
- `runBenchmark(projectPath, benchmarkScript)`
   - Writes a temp Python file, runs it via `uv run python`, captures output.
   - Output: returns `{ success, output }`; temp file is deleted.
- `listResults()`
   - Interactive selector to view previously saved results.
   - Output: console display; reads from `~/.config/rockit/benchmark-results/`.
- `promptBenchmark()`
   - Interactive runner for: basic torch env check, matrix-mult benchmark (BF16), Flash Attention benchmark (simple), or the comprehensive Flash Attention benchmark. Saves structured results.
   - Output: writes JSON result files to `~/.config/rockit/benchmark-results/`.
- `collectSystemMetadata()`
   - Captures OS, CPU, GPU, ROCm build info and installed package versions from config.
   - Output: metadata object embedded in result JSON.
- `saveResults(benchmarkType, output, metadata)`
   - Serializes benchmark output with parsed sections to a timestamped JSON file.
   - Output: `~/.config/rockit/benchmark-results/<type>_<timestamp>.json`.
- `parseMatrixResults(output)`
   - Parses bf16 GEMM logs into rows with m/n/k, timeMs, TOPS or GFLOPS.
   - Output: `{ matrixMultiplication: [...] }`.
- `parseFlashResults(output)`
   - Parses Flash Attention time and tokens/sec.
   - Output: `{ flashAttention: { timeMs, tokensPerSec } }`.
- `parseComprehensiveFlashResults(output)`
  - Parses the detailed output of the comprehensive Flash Attention benchmark, associating results with specific implementations and configurations.
  - Output: `{ comprehensiveFlashAttention: [...] }`.

### `src/flash-attention-benchmark.py`
- A standalone Python script for running a comprehensive benchmark across multiple attention implementations.
- Implementations tested: Flash Attention 2, Triton, xFormers, PyTorch SDPA, and `torch.compile` versions.
- Scenarios: "Prefill" (long context) and "Token Generation" (incremental with KV cache).
- Output: Detailed console logs with performance metrics (ms, tokens/sec, memory) and a summary of the best-performing implementation for each scenario. This output is captured and parsed by `benchmark.js`.

### `scripts/serve-benchmark-viewer.js`
- Serves a static web page to visualize benchmark results.
- Provides an API endpoint `/api/results` to fetch all benchmark data.

### Viewer assets
- The benchmark viewer is located in the `frontend/` directory.
- It expects benchmark JSON files shaped like those written by `saveResults`.

---

## Utilities and helpers

### `src/detect-and-list.js`
- `main()` (top-level) – combines detection and listing of compatible ROCm builds with recommendations.
- Output: console only.

### `src/debug-html.js`
- `debug()` – fetches the ROCm nightly index HTML and prints debug info for parsing.
- Output: console only.

---

## Try it

1) Install and link the CLI
```bash
cd /home/johnny/playground/rockit
npm install
npm link
```

2) Commands
```bash
rockit detect      # show platform/GPU and compatible ROCm families
rockit rocm        # guided ROCm installer
rockit python      # guided PyTorch + Flash-Attn installer
rockit bench       # run benchmarks and save results
rockit config      # print current config
rockit view        # view benchmark results
```

3) Where outputs go
- `/opt/rocm` – installed ROCm files, with `.info/build-info.json`
- `~/.cache/rockit/` – downloads, temp extracts
- `~/.config/rockit/config.json` – persisted CLI configuration
- `~/.config/rockit/benchmark-results/*.json` – benchmark result snapshots

---

If you want this summary to include additional internal helpers or tests, say the word and I’ll add them.

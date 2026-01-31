# Rockit 🚀 - ROCm Toolkit for Linux

One CLI to set up AMD ROCm + PyTorch nightlies and run quick benchmarks. Fun, fast, and focused on AMD GPUs.

Important: AMD only (for now)
- You need an AMD GPU. (No NVIDIA support, I don't think I can distrubute CUDA or cuDNN since its close sourced)
- Linux is the target platform right now.

## TL;DR

```bash
# 1) Install and link the CLI
npm install
npm link

# 2) Detect your system
rockit detect

# 3) Install ROCm (interactive, uses sudo for /opt/rocm)
rockit rocm

# 4) Install PyTorch for AMD ROCm (interactive)
rockit python

# 5) Run benchmarks and save results
rockit bench

# 6) View benchmark results
rockit view
```

That’s it. The tool guides you with friendly prompts and sensible defaults.

## What you get

- Auto-detect AMD GPU + compatible ROCm families
- Pick from ROCm nightly builds and install to `/opt/rocm`
- Set up a Python project with AMD ROCm nightly wheels (torch/vision/audio)
- **Choose your package manager**: `uv` (fast, recommended) or `pip` (standard)
- Optional Flash Attention install
- Simple performance benchmarks that save JSON snapshots
- A web-based viewer to visualize and compare benchmark results.

## Where things go

- ROCm install: `/opt/rocm` (writes `.info/build-info.json`)
- Download cache: `~/.cache/rockit/`
- CLI config: `~/.config/rockit/config.json`
- Benchmark results: `~/.config/rockit/benchmark-results/*.json`
- Project env file: `<your-project>/.env`

## Requirements

- AMD GPU with ROCm support (RDNA/CDNA, e.g., gfx10xx/gfx11xx/gfx9x)
- Linux, with sudo access for installing to `/opt/rocm`
- Python package manager: `uv` (recommended) or `pip`
  - Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - pip: Usually comes with Python

Tip: If you use fish shell, the tool prints fish-friendly env exports as well.

## Package Manager Support

Rockit supports both `uv` and `pip` for Python dependency management:

- **uv** (recommended): Faster, more reliable, handles Python versions automatically
- **pip**: Standard Python package manager, works with existing venv setups

The tool will prompt you to select your preferred package manager on first use. For existing projects, it automatically detects which package manager was used. See [PACKAGE_MANAGER_SUPPORT.md](PACKAGE_MANAGER_SUPPORT.md) for details.

## Commands cheat sheet

- `rockit detect` — Show platform, GPU arch, and ROCm compatibility
- `rockit rocm` — Guided ROCm install (download → backup → install → verify)
- `rockit python [path]` — Guided PyTorch/Flash-Attn install for AMD ROCm nightlies
  - Prompts for package manager selection (uv or pip)
  - Auto-detects existing project setup
- `rockit bench` — Run basic GPU checks and small benchmarks; saves JSON
- `rockit config` — Print current config (includes selected package manager)
- `rockit view` — Starts a web server to view benchmark results.

## Friendly notes

- The installer backs up any existing `/opt/rocm` to a timestamped `.backup` folder before installing.
- Benchmarks write results you can diff over time and post-process however you like.
- NVIDIA and CPU-only support are on the wishlist. If you need them, open an issue—PRs welcome!

## License

MIT

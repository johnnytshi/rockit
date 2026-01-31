# Package Manager Support: pip and uv

## Overview

Rockit now supports both `pip` and `uv` as package managers for Python dependency management. Users can choose their preferred package manager during installation, and the tool will automatically detect which package manager is used in existing projects.

## Implementation Summary

### Package Manager Abstraction

A new abstraction layer was created using the Strategy pattern:

```
src/package-managers/
├── base.js           # Abstract base class defining the interface
├── factory.js        # Factory for creating and detecting package managers
├── uv-manager.js     # uv implementation
└── pip-manager.js    # pip implementation
```

### Key Features

1. **Automatic Detection**: When working with existing projects, the tool detects which package manager was used to initialize the project
2. **User Selection**: For new projects, users can choose between available package managers
3. **Conflict Resolution**: If a user's configured preference differs from an existing project's package manager, the tool prompts for a choice
4. **Backward Compatibility**: Existing configurations default to `uv` to maintain compatibility

### Modified Files

1. **src/config.js**: Added `packageManager` field to configuration
2. **src/pytorch-cli.js**: Added Step 1 for package manager selection, updated all package operations to use abstraction
3. **src/benchmark.js**: Updated to use package manager abstraction for running Python scripts
4. **src/pytorch-installer.js**: Kept for backward compatibility (getPythonVersion, setupEnvironmentVariables)

## Usage

### New Installation

When installing PyTorch in a new project:

```bash
rockit python /path/to/project
```

**Step 1** will prompt you to select a package manager:
- `uv` (recommended - faster and more reliable)
- `pip` (standard Python package manager)

The tool will only show package managers that are installed on your system.

### Existing Projects

When working with an existing project, rockit automatically detects which package manager was used:

```bash
rockit python /path/to/existing-project
```

If your configured preference differs from the project's package manager, you'll be prompted:
- Continue with the existing package manager (recommended)
- Reinitialize with your preferred package manager
- Cancel

### Configuration

Your package manager preference is saved in `~/.config/rockit/config.json`:

```json
{
  "packageManager": "uv",
  "projectPath": "/home/user/pytorch-projects",
  ...
}
```

View your configuration:
```bash
rockit config
```

## Package Manager Comparison

| Feature | uv | pip |
|---------|----|----|
| Speed | Very fast | Standard |
| Project files | pyproject.toml | Just .venv/ |
| Initialization | `uv init --python X` | `pythonX -m venv .venv` |
| Installation | `uv pip install` | `.venv/bin/pip install` |
| Run Python | `uv run python` | `.venv/bin/python` |
| Prerelease flag | `--prerelease allow` | `--pre` |

## Technical Details

### Package Manager Interface

All package managers implement these methods:

- `isInstalled()`: Check if package manager is available
- `initializeProject(path, version)`: Initialize a new project
- `installPackages(path, arch, packages, options)`: Install PyTorch packages
- `updatePackages(path, arch, packages)`: Update packages
- `listInstalledPackages(path)`: List installed packages
- `runPythonCommand(path, script)`: Get command to run Python
- `getVenvPath(path)`: Get virtual environment path
- `isProjectInitialized(path)`: Check if project is initialized
- `getName()`: Get package manager name

### Project Detection

Projects are detected by checking for specific files/directories:
- **uv projects**: Look for `pyproject.toml`
- **pip projects**: Look for `.venv/bin/python`

### Benchmarks

The benchmark command automatically uses the correct package manager for the project:

```bash
rockit benchmark full /path/to/project
```

The package manager is read from the config (or defaults to `uv` for backward compatibility).

## Migration Guide

### From uv to pip

If you want to switch an existing uv project to pip:

1. Run `rockit python /path/to/project`
2. Select pip when prompted about the mismatch
3. Choose "Reinitialize with pip"
4. The project will be reinitialized with pip

### From pip to uv

Same process as above, but select uv when prompted.

**Note**: Switching package managers will reinitialize the virtual environment, so you'll need to reinstall packages.

## Troubleshooting

### Neither package manager found

If you see an error about no package managers being found:

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or ensure pip is available
python -m pip --version
```

### Python version not found (pip only)

pip requires the specific Python version to be installed on your system:

```bash
# Error: Python 3.11 is not installed
# Solution: Install Python 3.11 or choose a different version
```

uv can automatically download and manage Python versions, so this is less of an issue with uv.

## Testing

The implementation includes comprehensive testing:

1. **Syntax validation**: All files pass Node.js syntax checks
2. **Package manager detection**: Both uv and pip are correctly detected when installed
3. **Project initialization**: Both package managers can initialize projects
4. **Project detection**: Existing projects are correctly identified by package manager
5. **Configuration integration**: Package manager preference is saved and loaded correctly

## Future Enhancements

Possible future improvements:
- Support for additional package managers (poetry, conda, etc.)
- Automatic migration between package managers
- Package manager-specific optimizations
- Better handling of Python version mismatches

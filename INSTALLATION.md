## Installation Demo

Here's what you'll see when you run `rockit install`:

### 1. System Detection
```
🚀 Rockit - ROCm Installation Tool

=== Step 1: Detecting Your System ===

Platform: linux
Architecture: x64
GPU: gfx1151 ✅
Compatible with: gfx1151
```

### 2. Available Builds
```
=== Step 2: Fetching Available ROCm Builds ===

Fetching ROCm artifacts from: https://therock-nightly-tarball.s3.amazonaws.com/index.html
Parsed 1240 file entries from the index
Found 1240 ROCm artifacts

Found 164 compatible builds
```

### 3. Select Version
```
=== Step 3: Select ROCm Version ===

? Which ROCm version would you like to install? (Use arrow keys)
  ROCm 7.10.0
❯ ROCm 7.9.0
  ROCm 7.0.0
  ROCm 6.5.0
  ROCm 6.4.0
```

### 4. Select Build Date
```
? Select build date: (Use arrow keys)
❯ 2025-10-08 (rc20251008) - 10/8/2025
  2025-10-07 (rc20251007) - 10/7/2025
  2025-10-06 (rc20251006) - 10/6/2025
  2025-10-05 (rc20251005) - 10/5/2025
```

### 5. Confirmation
```
=== Installation Summary ===

ROCm Version: 7.9.0
GPU: gfx1151
Variant: default
Build Date: 2025-10-08
File: therock-dist-linux-gfx1151-7.9.0rc20251008.tar.gz
Size: ~2500 MB (estimated)
Install Location: /opt/rocm

? Proceed with installation? (Y/n)
```

### 6. Download
```
=== Step 4: Downloading ROCm ===

Downloading: therock-dist-linux-gfx1151-7.9.0rc20251008.tar.gz
From: https://therock-nightly-tarball.s3.amazonaws.com/...
Progress: 45% (1125MB / 2500MB)
```

### 7. Installation
```
=== Step 5: Installing ROCm ===

⚠️  Installation to /opt/rocm requires sudo privileges
You may be prompted for your password

Step 1: Extracting archive...
Extracting to: /opt/rocm
✅ Extraction complete

Step 2: Backing up existing installation to /opt/rocm.backup.1730318400

Step 3: Installing to /opt/rocm
[sudo] password for johnny: 

Step 4: Setting permissions...

✅ ROCm installation complete!
```

### 8. Environment Setup
```
=== Environment Setup ===

Add these lines to your shell configuration (~/.bashrc or ~/.config/fish/config.fish):

  export ROCM_PATH="/opt/rocm"
  export PATH="/opt/rocm/bin:$PATH"
  export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

For Fish shell users:

  set -gx ROCM_PATH "/opt/rocm"
  set -gx PATH "/opt/rocm/bin" $PATH
  set -gx LD_LIBRARY_PATH "/opt/rocm/lib" $LD_LIBRARY_PATH
```

### 9. Verification (Optional)
```
? Run verification test? (Y/n) Yes

=== Verifying Installation ===

Running rocminfo...

ROCk module is loaded
=====================    
HSA System Attributes    
=====================    
Runtime Version:         1.1
System Timestamp Freq.:  1000.000000MHz
Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
...

✅ ROCm is working correctly!
```

### 10. Complete!
```
✨ Installation complete! ✨

Next steps:
  1. Restart your shell or source your shell config
  2. Run: rocminfo
  3. Install PyTorch with ROCm support
```

## Commands

```bash
# Run the installer
rockit install

# Check your system
rockit detect

# Get help
rockit --help
```

## Notes

- The installer requires **sudo** access to install to `/opt/rocm`
- Downloads are cached in `~/.cache/rockit` for reuse
- Existing ROCm installations are backed up before installing
- File sizes vary from 1GB to 5GB depending on the build

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const PackageManager = require('./base');

/**
 * Package manager implementation for pip
 */
class PipManager extends PackageManager {
  getName() {
    return 'pip';
  }

  isInstalled() {
    try {
      execSync('pip --version', { stdio: 'pipe' });
      return true;
    } catch (error) {
      return false;
    }
  }

  async initializeProject(projectPath, pythonVersion) {
    console.log(`\nInitializing Python virtual environment at: ${projectPath}`);

    if (!fs.existsSync(projectPath)) {
      fs.mkdirSync(projectPath, { recursive: true });
    }

    const cwd = process.cwd();
    try {
      process.chdir(projectPath);

      // Create virtual environment using the specified Python version
      const pythonCmd = `python${pythonVersion}`;

      // Check if the specific Python version is available
      try {
        execSync(`${pythonCmd} --version`, { stdio: 'pipe' });
      } catch (error) {
        throw new Error(`Python ${pythonVersion} is not installed. Please install it first.`);
      }

      execSync(`${pythonCmd} -m venv .venv`, { stdio: 'inherit' });
      console.log('✅ Virtual environment created');
    } catch (error) {
      console.error('Failed to initialize virtual environment:', error.message);
      throw error;
    } finally {
      process.chdir(cwd);
    }
  }

  async installPackages(projectPath, gpuArch, packages, options = {}) {
    console.log('\n=== Installing PyTorch Packages ===\n');

    const cwd = process.cwd();
    try {
      process.chdir(projectPath);

      const indexUrl = `https://rocm.nightlies.amd.com/v2/${gpuArch}/`;
      const pipPath = path.join(projectPath, '.venv', 'bin', 'pip');

      // Verify venv exists
      if (!fs.existsSync(pipPath)) {
        throw new Error('Virtual environment not found. Please initialize the project first.');
      }

      // Build package specifications
      const packageSpecs = packages.map(pkg => {
        if (pkg.version) {
          return `${pkg.name}==${pkg.version}`;
        }
        return pkg.name;
      });

      console.log(`Installing packages: ${packageSpecs.join(', ')}`);
      console.log(`Using index: ${indexUrl}\n`);

      const cmd = `${pipPath} install --index-url ${indexUrl} --pre --upgrade ${packageSpecs.join(' ')}`;
      console.log(`Running: ${cmd}\n`);
      execSync(cmd, { stdio: 'inherit' });
      console.log('\n✅ PyTorch packages installed successfully');

      // Install flash-attn if requested
      if (options.installFlashAttn) {
        console.log('\n=== Installing Flash Attention ===\n');
        try {
          const version = options.flashAttnVersion && options.flashAttnVersion !== 'latest'
            ? `==${options.flashAttnVersion}`
            : '';
          const flashAttnCmd = `${pipPath} install flash-attn${version} --no-build-isolation`;
          console.log(`Running: ${flashAttnCmd}\n`);
          execSync(flashAttnCmd, { stdio: 'inherit' });
          console.log('\n✅ Flash Attention installed successfully');
        } catch (error) {
          console.error('\n⚠️  Flash Attention installation failed:', error.message);
          console.error('You can try installing it manually later with:');
          const version = options.flashAttnVersion && options.flashAttnVersion !== 'latest'
            ? `==${options.flashAttnVersion}`
            : '';
          console.error(`  ${pipPath} install flash-attn${version} --no-build-isolation`);
        }
      }

      return true;
    } catch (error) {
      console.error('\n❌ Installation failed:', error.message);
      console.error('\nTip: Make sure your virtual environment is properly initialized');
      return false;
    } finally {
      process.chdir(cwd);
    }
  }

  async updatePackages(projectPath, gpuArch, packages) {
    console.log('\n=== Updating PyTorch Packages ===\n');

    const cwd = process.cwd();
    try {
      process.chdir(projectPath);

      const indexUrl = `https://rocm.nightlies.amd.com/v2/${gpuArch}/`;
      const pipPath = path.join(projectPath, '.venv', 'bin', 'pip');

      for (const pkg of packages) {
        try {
          console.log(`Updating ${pkg.name} to ${pkg.version}...`);
          const cmd = `${pipPath} install --index-url ${indexUrl} --pre --upgrade ${pkg.name}==${pkg.version}`;
          execSync(cmd, { stdio: 'inherit' });
          console.log(`✅ ${pkg.name} updated\n`);
        } catch (error) {
          console.error(`❌ Failed to update ${pkg.name}:`, error.message);
        }
      }

      return true;
    } finally {
      process.chdir(cwd);
    }
  }

  async listInstalledPackages(projectPath) {
    const cwd = process.cwd();
    try {
      process.chdir(projectPath);
      const pipPath = path.join(projectPath, '.venv', 'bin', 'pip');

      const output = execSync(`${pipPath} list --format json`, { encoding: 'utf8' });
      const packages = JSON.parse(output);

      return packages.map(pkg => ({
        name: pkg.name,
        version: pkg.version
      }));
    } catch (error) {
      return [];
    } finally {
      process.chdir(cwd);
    }
  }

  runPythonCommand(projectPath, scriptPath) {
    const pythonPath = path.join(projectPath, '.venv', 'bin', 'python');
    return `${pythonPath} ${scriptPath}`;
  }

  getVenvPath(projectPath) {
    return path.join(projectPath, '.venv');
  }

  isProjectInitialized(projectPath) {
    const venvPath = path.join(projectPath, '.venv');
    const pythonPath = path.join(venvPath, 'bin', 'python');
    return fs.existsSync(pythonPath);
  }
}

module.exports = PipManager;

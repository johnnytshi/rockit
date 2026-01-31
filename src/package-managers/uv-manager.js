const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const PackageManager = require('./base');

/**
 * Package manager implementation for uv
 */
class UvManager extends PackageManager {
  getName() {
    return 'uv';
  }

  isInstalled() {
    try {
      execSync('uv --version', { stdio: 'pipe' });
      return true;
    } catch (error) {
      return false;
    }
  }

  async initializeProject(projectPath, pythonVersion) {
    console.log(`\nInitializing uv project at: ${projectPath}`);

    if (!fs.existsSync(projectPath)) {
      fs.mkdirSync(projectPath, { recursive: true });
    }

    const cwd = process.cwd();
    try {
      process.chdir(projectPath);

      execSync(`uv init --python ${pythonVersion}`, { stdio: 'inherit' });
      console.log('✅ Project initialized');
    } catch (error) {
      console.error('Failed to initialize project:', error.message);
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

      // Build package specifications
      const packageSpecs = packages.map(pkg => {
        if (pkg.version) {
          return `${pkg.name}==${pkg.version}`;
        }
        return pkg.name;
      });

      console.log(`Installing packages: ${packageSpecs.join(', ')}`);
      console.log(`Using index: ${indexUrl}\n`);

      const cmd = `uv pip install --index-url ${indexUrl} --prerelease allow --upgrade ${packageSpecs.join(' ')}`;
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
          const flashAttnCmd = `uv pip install flash-attn${version} --no-build-isolation`;
          console.log(`Running: ${flashAttnCmd}\n`);
          execSync(flashAttnCmd, { stdio: 'inherit' });
          console.log('\n✅ Flash Attention installed successfully');
        } catch (error) {
          console.error('\n⚠️  Flash Attention installation failed:', error.message);
          console.error('You can try installing it manually later with:');
          const version = options.flashAttnVersion && options.flashAttnVersion !== 'latest'
            ? `==${options.flashAttnVersion}`
            : '';
          console.error(`  uv pip install flash-attn${version} --no-build-isolation`);
        }
      }

      return true;
    } catch (error) {
      console.error('\n❌ Installation failed:', error.message);
      console.error('\nTip: Make sure your project has a valid pyproject.toml');
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

      for (const pkg of packages) {
        try {
          console.log(`Updating ${pkg.name} to ${pkg.version}...`);
          const cmd = `uv pip install --index-url ${indexUrl} --prerelease allow --upgrade ${pkg.name}==${pkg.version}`;
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
      const output = execSync('uv pip list --format json', { encoding: 'utf8' });
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
    return `uv run python ${scriptPath}`;
  }

  getVenvPath(projectPath) {
    return path.join(projectPath, '.venv');
  }

  isProjectInitialized(projectPath) {
    const pyprojectPath = path.join(projectPath, 'pyproject.toml');
    return fs.existsSync(pyprojectPath);
  }
}

module.exports = UvManager;

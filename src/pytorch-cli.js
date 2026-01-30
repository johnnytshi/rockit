#!/usr/bin/env node

const inquirer = require('inquirer');
const path = require('path');
const fs = require('fs');
const { detectSystem } = require('./system-detect');
const { 
  parsePyTorchPackages, 
  filterByPythonVersion,
  filterByPlatform,
  getAvailablePythonVersions,
  groupPackagesByName
} = require('./pytorch-parser');
const {
  fetchPyPiVersions
} = require('./pypi-parser');
const {
  isUvInstalled,
  getPythonVersion,
  initializeUvProject,
  installPyTorchPackages,
  updatePyTorchPackages,
  isProjectInitialized,
  getInstalledPackages,
  setupEnvironmentVariables
} = require('./pytorch-installer');
const {
  loadConfig,
  saveConfig,
  displayConfig,
  updateConfig
} = require('./config');

/**
 * Check current ROCm installation
 * @returns {Object|null}
 */
function checkCurrentRocm() {
  const versionFile = '/opt/rocm/.info/version';
  const buildInfoFile = '/opt/rocm/.info/build-info.json';
  
  let version = null;
  
  if (fs.existsSync(buildInfoFile)) {
    try {
      const buildInfo = JSON.parse(fs.readFileSync(buildInfoFile, 'utf8'));
      if (buildInfo.rocmVersion && buildInfo.buildTag) {
        version = `${buildInfo.rocmVersion}${buildInfo.buildTag}`;
      } else {
        version = buildInfo.rocmVersion;
      }
    } catch (error) {
      // Ignore
    }
  }
  
  if (!version && fs.existsSync(versionFile)) {
    try {
      version = fs.readFileSync(versionFile, 'utf8').trim();
    } catch (error) {
      // Ignore
    }
  }
  
  return version;
}

async function promptPyTorchInstallation(projectPath) {
  console.log('\n🔥 AIBenchy - PyTorch Installation Tool\n');
  
  // Step 1: Check prerequisites
  console.log('=== Step 1: Checking Prerequisites ===\n');
  
  // Check uv
  if (!isUvInstalled()) {
    console.error('❌ uv is not installed.');
    console.log('\nInstall uv with:');
    console.log('  curl -LsSf https://astral.sh/uv/install.sh | sh');
    console.log('  or: pip install uv\n');
    process.exit(1);
  }
  console.log('✅ uv is installed');
  
  // Detect system
  const systemInfo = detectSystem();
  console.log(`✅ Platform: ${systemInfo.platform}`);
  
  if (!systemInfo.detected) {
    console.error('❌ No AMD GPU detected');
    process.exit(1);
  }
  console.log(`✅ GPU: ${systemInfo.gpuArch}`);
  
  // Check ROCm
  const rocmVersion = checkCurrentRocm();
  if (rocmVersion) {
    console.log(`✅ ROCm: ${rocmVersion}`);
  } else {
    console.log('⚠️  ROCm not detected (optional)');
  }
  
  // Get Python version
  const systemPythonVersion = getPythonVersion();
  if (systemPythonVersion) {
    console.log(`✅ Python: ${systemPythonVersion}`);
  }
  
  // Step 2: Load or create config
  console.log('\n=== Step 2: Configuration ===\n');
  
  let config = loadConfig();
  
  // Update auto-detected values
  config.gpuArch = systemInfo.gpuArch;
  config.rocmVersion = rocmVersion;

  // Use provided project path (already resolved to absolute in CLI)
  config.projectPath = projectPath;
  console.log(`Using project path: ${config.projectPath}`);
  
  // Check if project exists
  const projectExists = isProjectInitialized(config.projectPath);
  
  if (projectExists) {
    console.log(`\n✅ Project found at: ${config.projectPath}`);
    
    // Get installed packages
    try {
      const installed = getInstalledPackages(config.projectPath);
      if (installed.torch) {
        console.log(`   torch: ${installed.torch}`);
        config.installedPackages.torch = installed.torch;
      }
      if (installed.torchvision) {
        console.log(`   torchvision: ${installed.torchvision}`);
        config.installedPackages.torchvision = installed.torchvision;
      }
      if (installed.torchaudio) {
        console.log(`   torchaudio: ${installed.torchaudio}`);
        config.installedPackages.torchaudio = installed.torchaudio;
      }
    } catch (error) {
      // Ignore
    }
    
    const { action } = await inquirer.prompt([
      {
        type: 'list',
        name: 'action',
        message: 'What would you like to do?',
        choices: [
          { name: 'Update packages to new versions', value: 'update' },
          { name: 'Reinstall packages', value: 'reinstall' },
          { name: 'Cancel', value: 'cancel' }
        ],
        loop: false
      }
    ]);
    
    if (action === 'cancel') {
      console.log('\nCancelled.');
      process.exit(0);
    }
  } else {
    console.log(`\n📁 New project will be created at: ${config.projectPath}`);
  }
  
  // Step 3: Select Python version
  console.log('\n=== Step 3: Fetching Available Packages ===\n');
  
  const allPackages = await parsePyTorchPackages(config.gpuArch);
  
  // Filter by platform
  const platformPackages = filterByPlatform(allPackages, systemInfo.platform);
  console.log(`Found ${allPackages.length} packages (${platformPackages.length} for ${systemInfo.platform})`);
  
  const availablePythonVersions = getAvailablePythonVersions(platformPackages);
  console.log(`Supported Python versions: ${availablePythonVersions.join(', ')}\n`);
  
  // Sort Python versions in descending order (newest first)
  const sortedPythonVersions = availablePythonVersions.sort((a, b) => {
    const partsA = a.split('.').map(x => parseInt(x) || 0);
    const partsB = b.split('.').map(x => parseInt(x) || 0);
    
    for (let i = 0; i < Math.max(partsA.length, partsB.length); i++) {
      const pa = partsA[i] || 0;
      const pb = partsB[i] || 0;
      if (pb > pa) return 1;
      if (pb < pa) return -1;
    }
    return 0;
  });
  
  // Filter to only show Python versions that make sense
  let pythonChoices = sortedPythonVersions.map(v => ({
    name: `Python ${v}${systemPythonVersion === v ? ' (system version)' : ''}`,
    value: v
  }));
  
  const { pythonVersion } = await inquirer.prompt([
    {
      type: 'list',
      name: 'pythonVersion',
      message: 'Select Python version:',
      choices: pythonChoices,
      default: systemPythonVersion || availablePythonVersions[availablePythonVersions.length - 1],
      loop: false
    }
  ]);
  
  config.pythonVersion = pythonVersion;
  
  // Filter packages by Python version and platform
  let compatiblePackages = filterByPythonVersion(platformPackages, pythonVersion);
  
  // Filter by ROCm version if installed (before grouping to ensure proper sorting)
  if (rocmVersion) {
    const baseRocmVersion = rocmVersion.match(/^[0-9]+\.[0-9]+\.[0-9]+/)?.[0];
    const rocmFiltered = compatiblePackages.filter(pkg => 
      pkg.rocmVersion && pkg.rocmVersion.startsWith(baseRocmVersion)
    );
    
    if (rocmFiltered.length > 0) {
      compatiblePackages = rocmFiltered;
      console.log(`\nFiltering packages to ROCm ${baseRocmVersion} compatible versions\n`);
    }
  }
  
  // Group and sort packages
  const grouped = groupPackagesByName(compatiblePackages);
  
  // Step 4: Select versions for each package
  console.log('=== Step 4: Select Package Versions ===\n');
  
  const packagesToInstall = [];
  const packageNames = ['torch', 'torchvision', 'torchaudio'];
  
  for (const pkgName of packageNames) {
    if (!grouped[pkgName] || grouped[pkgName].length === 0) {
      console.log(`⚠️  ${pkgName} not available for Python ${pythonVersion}`);
      continue;
    }
    
    // Get filtered and sorted versions from grouped data
    let filteredVersions = grouped[pkgName];
    
    // Show only latest 10 versions
    const versionChoices = filteredVersions.slice(0, 10).map(pkg => {
      const dateStr = pkg.devDate 
        ? `${pkg.devDate.slice(0, 4)}-${pkg.devDate.slice(4, 6)}-${pkg.devDate.slice(6, 8)}`
        : '';
      return {
        name: `${pkg.version} (ROCm ${pkg.rocmVersion}, ${dateStr})`,
        value: pkg.version
      };
    });
    
    if (versionChoices.length > 10) {
      versionChoices.push({
        name: `... and ${filteredVersions.length - 10} more versions`,
        value: null
      });
    }
    
    const { version } = await inquirer.prompt([
      {
        type: 'list',
        name: 'version',
        message: `Select ${pkgName} version:`,
        choices: versionChoices,
        default: versionChoices[0].value,
        loop: false
      }
    ]);
    
    if (version) {
      packagesToInstall.push({
        name: pkgName,
        version: version
      });
    }
  }
  
  // Step 5: Flash Attention option
  const { installFlashAttn } = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'installFlashAttn',
      message: 'Install Flash Attention? (Recommended for faster attention operations)',
      default: true
    }
  ]);
  
  let flashAttnVersion = null;
  if (installFlashAttn) {
    console.log('\nFetching flash-attn versions...');
    const flashAttnVersions = await fetchPyPiVersions('flash-attn');
    
    if (flashAttnVersions.length > 0) {
      const versionChoices = flashAttnVersions.slice(0, 10).map(v => ({
        name: v,
        value: v
      }));
      
      if (flashAttnVersions.length > 10) {
        versionChoices.push({
          name: `... and ${flashAttnVersions.length - 10} more versions`,
          value: null
        });
      }
      
      const { selectedFlashAttnVersion } = await inquirer.prompt([
        {
          type: 'list',
          name: 'selectedFlashAttnVersion',
          message: 'Select flash-attn version:',
          choices: versionChoices,
          default: versionChoices[0].value,
          loop: false
        }
      ]);
      
      flashAttnVersion = selectedFlashAttnVersion;
    } else {
      console.log('⚠️  Could not fetch flash-attn versions');
      const { continueWithoutVersion } = await inquirer.prompt([
        {
          type: 'confirm',
          name: 'continueWithoutVersion',
          message: 'Install latest available version anyway?',
          default: false
        }
      ]);
      
      if (continueWithoutVersion) {
        flashAttnVersion = 'latest';
      } else {
        flashAttnVersion = null;
      }
    }
  }
  
  // Step 6: Confirm installation
  console.log('\n=== Installation Summary ===\n');
  console.log(`Project Path: ${config.projectPath}`);
  console.log(`Python Version: ${pythonVersion}`);
  console.log(`GPU: ${config.gpuArch}`);
  if (rocmVersion) {
    console.log(`ROCm Version: ${rocmVersion}`);
  }
  console.log('\nPackages to install:');
  packagesToInstall.forEach(pkg => {
    console.log(`  • ${pkg.name} ${pkg.version}`);
  });
  if (installFlashAttn && flashAttnVersion) {
    console.log(`  • flash-attn ${flashAttnVersion === 'latest' ? '(latest)' : flashAttnVersion}`);
  }
  
  const { confirmInstall } = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'confirmInstall',
      message: 'Proceed with installation?',
      default: true
    }
  ]);
  
  if (!confirmInstall) {
    console.log('\nCancelled.');
    process.exit(0);
  }
  
  // Step 7: Initialize project if needed
  if (!projectExists) {
    console.log('\n=== Step 5: Initializing Project ===');
    initializeUvProject(config.projectPath, pythonVersion);
  }
  
  // Step 8: Install packages
  const success = await installPyTorchPackages(
    config.projectPath,
    config.gpuArch,
    packagesToInstall,
    { installFlashAttn, flashAttnVersion }
  );
  
  if (!success) {
    console.error('\n❌ Installation failed');
    process.exit(1);
  }
  
  // Step 9: Setup environment variables
  setupEnvironmentVariables(config.projectPath, { hasFlashAttn: installFlashAttn });
  
  // Step 10: Update config
  packagesToInstall.forEach(pkg => {
    config.installedPackages[pkg.name] = pkg.version;
  });
  
  if (installFlashAttn) {
    config.installedPackages['flash-attn'] = flashAttnVersion;
  }
  
  saveConfig(config);
  
  console.log('\n✨ Installation complete! ✨\n');
  console.log('Next steps:');
  console.log(`  1. cd ${config.projectPath}`);
  console.log(`  2. Source environment variables: source .env`);
  console.log(`  3. uv run python`);
  console.log(`  4. import torch; print(torch.cuda.is_available())\n`);
}

module.exports = { promptPyTorchInstallation };

// Run if called directly
if (require.main === module) {
  const projectPath = process.argv[2] ? path.resolve(process.argv[2]) : process.cwd();
  promptPyTorchInstallation(projectPath).catch(error => {
    console.error('\n❌ Error:', error.message);
    process.exit(1);
  });
}

#!/usr/bin/env node

const inquirer = require('inquirer');
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');
const { detectSystem, findCompatibleArtifacts } = require('./system-detect');
const { parseRocmArtifacts, getUniqueValues } = require('./rocm-parser');
const { downloadFile, installRocm, setupEnvironment, verifyInstallation } = require('./installer');

/**
 * Check if ROCm is already installed and get version
 * @returns {Object|null} ROCm installation info or null
 */
function checkCurrentRocm() {
  const possiblePaths = [
    '/opt/rocm',
    '/opt/rocm-*',
  ];
  
  let rocmPath = null;
  let version = null;
  let buildInfo = null;
  
  // Check if /opt/rocm exists and has .info/version file
  if (fs.existsSync('/opt/rocm')) {
    rocmPath = '/opt/rocm';
    
    // Try to read build metadata first (most detailed)
    const buildInfoFile = path.join(rocmPath, '.info', 'build-info.json');
    if (fs.existsSync(buildInfoFile)) {
      try {
        buildInfo = JSON.parse(fs.readFileSync(buildInfoFile, 'utf8'));
        // Construct full version from build info
        if (buildInfo.rocmVersion && buildInfo.buildTag) {
          version = `${buildInfo.rocmVersion}${buildInfo.buildTag}`;
        } else {
          version = buildInfo.rocmVersion || null;
        }
      } catch (error) {
        // Ignore parse errors
      }
    }
    
    // Fallback: Try to read version from .info/version file
    if (!version) {
      const versionFile = path.join(rocmPath, '.info', 'version');
      if (fs.existsSync(versionFile)) {
        try {
          version = fs.readFileSync(versionFile, 'utf8').trim();
        } catch (error) {
          // Ignore read errors
        }
      }
    }
    
    // Fallback: try to get version from rocminfo
    if (!version) {
      try {
        const output = execSync('rocminfo --version 2>/dev/null', { 
          encoding: 'utf8',
          shell: '/bin/bash'
        });
        const match = output.match(/ROCm version:\s*([0-9.]+)/i) || output.match(/([0-9]+\.[0-9]+\.[0-9]+)/);
        if (match) {
          version = match[1];
        }
      } catch (error) {
        // rocminfo not available or failed
      }
    }
    
    // Check if ROCm directory has actual content
    try {
      const contents = fs.readdirSync(rocmPath);
      if (contents.length > 1 || (contents.length === 1 && contents[0] !== '.info')) {
        return {
          installed: true,
          path: rocmPath,
          version: version || 'unknown',
          buildInfo: buildInfo,
          hasContent: true
        };
      }
    } catch (error) {
      // Ignore
    }
  }
  
  return {
    installed: false,
    path: null,
    version: null,
    buildInfo: null,
    hasContent: false
  };
}

async function promptInstallation() {
  console.log('\n🚀 Rockit - ROCm Installation Tool\n');
  
  // Check current ROCm installation
  const currentRocm = checkCurrentRocm();
  if (currentRocm.installed && currentRocm.hasContent) {
    console.log('📦 Current ROCm Installation:');
    console.log(`   Path: ${currentRocm.path}`);
    console.log(`   Version: ${currentRocm.version}`);
    
    // Show additional build info if available
    if (currentRocm.buildInfo) {
      if (currentRocm.buildInfo.gpu) {
        console.log(`   GPU: ${currentRocm.buildInfo.gpu}`);
      }
      if (currentRocm.buildInfo.buildDate) {
        console.log(`   Build Date: ${currentRocm.buildInfo.buildDate}`);
      }
      if (currentRocm.buildInfo.installedAt) {
        const installedDate = new Date(currentRocm.buildInfo.installedAt).toLocaleString();
        console.log(`   Installed: ${installedDate}`);
      }
    }
    console.log('');
    
    const { continueInstall } = await inquirer.prompt([
      {
        type: 'confirm',
        name: 'continueInstall',
        message: 'ROCm is already installed. Do you want to reinstall/upgrade?',
        default: false
      }
    ]);
    
    if (!continueInstall) {
      console.log('\nInstallation cancelled.');
      process.exit(0);
    }
    console.log('');
  } else if (fs.existsSync('/opt/rocm')) {
    console.log('⚠️  /opt/rocm exists but appears incomplete or empty\n');
  }
  
  // Step 1: Detect system
  console.log('=== Step 1: Detecting Your System ===\n');
  
  // Step 1: Detect system
  console.log('=== Step 1: Detecting Your System ===\n');
  const systemInfo = detectSystem();
  
  console.log(`Platform: ${systemInfo.platform}`);
  console.log(`Architecture: ${systemInfo.osInfo.arch}`);
  
  if (systemInfo.detected) {
    console.log(`GPU: ${systemInfo.gpuArch} ✅`);
    console.log(`Compatible with: ${systemInfo.rocmGpuFamilies.join(', ')}`);
  } else {
    console.log('GPU: Not detected ⚠️');
    const { continueWithoutGpu } = await inquirer.prompt([
      {
        type: 'confirm',
        name: 'continueWithoutGpu',
        message: 'AMD GPU not detected. Continue anyway?',
        default: false
      }
    ]);
    
    if (!continueWithoutGpu) {
      console.log('\nInstallation cancelled.');
      process.exit(0);
    }
  }
  
  // Step 2: Fetch artifacts
  console.log('\n=== Step 2: Fetching Available ROCm Builds ===\n');
  const allArtifacts = await parseRocmArtifacts();
  
  let compatible = systemInfo.detected 
    ? findCompatibleArtifacts(allArtifacts, systemInfo)
    : allArtifacts.filter(a => a.platform === systemInfo.platform);
  
  if (compatible.length === 0) {
    console.error('❌ No compatible ROCm builds found.');
    process.exit(1);
  }
  
  console.log(`Found ${compatible.length} compatible builds\n`);
  
  // Step 3: Select ROCm version
  console.log('=== Step 3: Select ROCm Version ===\n');
  
  const versions = getUniqueValues(compatible, 'rocmVersion');
  const { selectedVersion } = await inquirer.prompt([
    {
      type: 'list',
      name: 'selectedVersion',
      message: 'Which ROCm version would you like to install?',
      choices: versions.map(v => ({
        name: `ROCm ${v}`,
        value: v
      })),
      default: versions[versions.length - 1], // Latest version
      loop: false
    }
  ]);
  
  // Filter by selected version
  compatible = compatible.filter(a => a.rocmVersion === selectedVersion);
  
  // Step 4: Select variant (if multiple available)
  const variants = getUniqueValues(compatible, 'variant');
  let selectedVariant = variants[0];
  
  if (variants.length > 1) {
    const variantChoices = variants
      .filter(v => v !== 'ADHOCBUILD') // Exclude adhoc builds by default
      .map(v => ({
        name: v === 'default' ? 'default (standard build)' : v,
        value: v
      }));
    
    if (variantChoices.length > 1) {
      const result = await inquirer.prompt([
        {
          type: 'list',
          name: 'selectedVariant',
          message: 'Select build variant:',
          choices: variantChoices,
          default: 'default',
          loop: false
        }
      ]);
      selectedVariant = result.selectedVariant;
    }
  }
  
  // Step 5: Select specific build
  const builds = compatible
    .filter(a => a.variant === selectedVariant)
    .sort((a, b) => (b.buildDate || '').localeCompare(a.buildDate || ''));
  
  if (builds.length === 0) {
    console.error('❌ No builds found for selected options.');
    process.exit(1);
  }
  
  const { selectedBuild } = await inquirer.prompt([
    {
      type: 'list',
      name: 'selectedBuild',
      message: 'Select build date:',
      choices: builds.slice(0, 10).map(b => ({
        name: `${b.buildDate} (${b.buildTag}) - ${(b.mtime ? new Date(b.mtime * 1000).toLocaleDateString() : 'N/A')}`,
        value: b
      })),
      default: builds[0],
      loop: false
    }
  ]);
  
  // Step 6: Confirm installation
  console.log('\n=== Installation Summary ===\n');
  console.log(`ROCm Version: ${selectedBuild.rocmVersion}`);
  console.log(`GPU: ${selectedBuild.gpu}`);
  console.log(`Variant: ${selectedBuild.variant}`);
  console.log(`Build Date: ${selectedBuild.buildDate}`);
  console.log(`File: ${selectedBuild.filename}`);
  console.log(`Size: ~${(await getFileSize(selectedBuild.url))} MB (estimated)`);
  console.log(`Install Location: /opt/rocm`);
  
  const { confirmInstall } = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'confirmInstall',
      message: 'Proceed with installation?',
      default: true
    }
  ]);
  
  if (!confirmInstall) {
    console.log('\nInstallation cancelled.');
    process.exit(0);
  }
  
  // Step 7: Download
  console.log('\n=== Step 4: Downloading ROCm ===');
  
  const downloadDir = path.join(process.env.HOME, '.cache', 'rockit');
  if (!fs.existsSync(downloadDir)) {
    fs.mkdirSync(downloadDir, { recursive: true });
  }
  
  const downloadPath = path.join(downloadDir, selectedBuild.filename);
  
  // Check if already downloaded
  if (fs.existsSync(downloadPath)) {
    const { useExisting } = await inquirer.prompt([
      {
        type: 'confirm',
        name: 'useExisting',
        message: `File already exists. Use existing download?`,
        default: true
      }
    ]);
    
    if (!useExisting) {
      fs.unlinkSync(downloadPath);
    }
  }
  
  if (!fs.existsSync(downloadPath)) {
    await downloadFile(selectedBuild.url, downloadPath);
  } else {
    console.log('\n✅ Using existing download');
  }
  
  // Step 8: Install
  console.log('\n=== Step 5: Installing ROCm ===');
  
  const installSuccess = await installRocm(downloadPath, '/opt/rocm', selectedBuild);
  
  if (!installSuccess) {
    console.error('\n❌ Installation failed. Please check the error messages above.');
    process.exit(1);
  }
  
  // Step 9: Setup environment
  setupEnvironment('/opt/rocm');
  
  // Step 10: Verify (optional)
  const { runVerify } = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'runVerify',
      message: 'Run verification test?',
      default: true
    }
  ]);
  
  if (runVerify) {
    verifyInstallation('/opt/rocm');
  }
  
  console.log('\n✨ Installation complete! ✨\n');
  console.log('Next steps:');
  console.log('  1. Restart your shell or source your shell config');
  console.log('  2. Run: rocminfo');
  console.log('  3. Install PyTorch with ROCm support\n');
}

async function getFileSize(url) {
  try {
    const response = await require('axios').head(url);
    const bytes = parseInt(response.headers['content-length'], 10);
    return (bytes / 1024 / 1024).toFixed(0);
  } catch {
    return '1000-5000';
  }
}

module.exports = { promptInstallation };

// Run if called directly
if (require.main === module) {
  promptInstallation().catch(error => {
    console.error('\n❌ Error:', error.message);
    process.exit(1);
  });
}

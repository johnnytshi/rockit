const fs = require('fs');
const path = require('path');
const os = require('os');

const CONFIG_DIR = path.join(os.homedir(), '.config', 'rockit');
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');

/**
 * Get default config
 * @returns {Object}
 */
function getDefaultConfig() {
  return {
    packageManager: null, // 'uv' or 'pip'
    projectPath: path.join(os.homedir(), 'pytorch-projects'),
    pythonVersion: null, // Will be auto-detected
    gpuArch: null, // Will be auto-detected
    rocmVersion: null, // Will be read from installation
    lastUpdated: null,
    installedPackages: {
      torch: null,
      torchvision: null,
      torchaudio: null,
      'flash-attn': null
    }
  };
}

/**
 * Load config from file
 * @returns {Object} Configuration object
 */
function loadConfig() {
  try {
    if (fs.existsSync(CONFIG_FILE)) {
      const data = fs.readFileSync(CONFIG_FILE, 'utf8');
      return { ...getDefaultConfig(), ...JSON.parse(data) };
    }
  } catch (error) {
    console.warn('Warning: Could not load config file:', error.message);
  }
  
  return getDefaultConfig();
}

/**
 * Save config to file
 * @param {Object} config - Configuration object
 */
function saveConfig(config) {
  try {
    // Ensure config directory exists
    if (!fs.existsSync(CONFIG_DIR)) {
      fs.mkdirSync(CONFIG_DIR, { recursive: true });
    }
    
    // Update lastUpdated timestamp
    config.lastUpdated = new Date().toISOString();
    
    // Write config file
    fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2), 'utf8');
    console.log(`\n✅ Configuration saved to: ${CONFIG_FILE}`);
  } catch (error) {
    console.error('Error saving config:', error.message);
    throw error;
  }
}

/**
 * Update specific config values
 * @param {Object} updates - Object with config keys to update
 * @returns {Object} Updated config
 */
function updateConfig(updates) {
  const config = loadConfig();
  Object.assign(config, updates);
  saveConfig(config);
  return config;
}

/**
 * Display current configuration
 * @param {Object} config - Configuration object
 */
function displayConfig(config) {
  console.log('\n=== Current Configuration ===\n');
  console.log(`Package Manager: ${config.packageManager || 'Not set'}`);
  console.log(`Project Path: ${config.projectPath}`);
  console.log(`Python Version: ${config.pythonVersion || 'Not set'}`);
  console.log(`GPU Architecture: ${config.gpuArch || 'Not detected'}`);
  console.log(`ROCm Version: ${config.rocmVersion || 'Not installed'}`);

  console.log('\nInstalled Packages:');
  const pkgs = config.installedPackages;
  console.log(`  torch: ${pkgs.torch || 'Not installed'}`);
  console.log(`  torchvision: ${pkgs.torchvision || 'Not installed'}`);
  console.log(`  torchaudio: ${pkgs.torchaudio || 'Not installed'}`);
  console.log(`  flash-attn: ${pkgs['flash-attn'] || 'Not installed'}`);

  if (config.lastUpdated) {
    console.log(`\nLast Updated: ${new Date(config.lastUpdated).toLocaleString()}`);
  }
  console.log('');
}

/**
 * Reset config to defaults
 */
function resetConfig() {
  const config = getDefaultConfig();
  saveConfig(config);
  console.log('✅ Configuration reset to defaults');
  return config;
}

module.exports = {
  CONFIG_FILE,
  loadConfig,
  saveConfig,
  updateConfig,
  displayConfig,
  resetConfig,
  getDefaultConfig
};

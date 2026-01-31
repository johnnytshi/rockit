const UvManager = require('./uv-manager');
const PipManager = require('./pip-manager');

/**
 * Create a package manager instance by type
 * @param {string} type - Package manager type ('uv' or 'pip')
 * @returns {PackageManager} Package manager instance
 */
function createPackageManager(type) {
  switch (type) {
    case 'uv':
      return new UvManager();
    case 'pip':
      return new PipManager();
    default:
      throw new Error(`Unknown package manager: ${type}`);
  }
}

/**
 * Get list of available package managers installed on the system
 * @returns {Array<string>} Array of available package manager names
 */
function getAvailableManagers() {
  const managers = [];

  const uv = new UvManager();
  if (uv.isInstalled()) {
    managers.push('uv');
  }

  const pip = new PipManager();
  if (pip.isInstalled()) {
    managers.push('pip');
  }

  return managers;
}

/**
 * Detect which package manager a project is using
 * @param {string} projectPath - Path to project directory
 * @returns {string|null} Package manager name or null if not detected
 */
function detectProjectPackageManager(projectPath) {
  const uv = new UvManager();
  if (uv.isProjectInitialized(projectPath)) {
    return 'uv';
  }

  const pip = new PipManager();
  if (pip.isProjectInitialized(projectPath)) {
    return 'pip';
  }

  return null;
}

module.exports = {
  createPackageManager,
  getAvailableManagers,
  detectProjectPackageManager
};

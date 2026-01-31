/**
 * Abstract base class for package managers
 * Defines the interface that all package managers must implement
 */
class PackageManager {
  /**
   * Check if the package manager is installed and available
   * @returns {boolean} true if installed, false otherwise
   */
  isInstalled() {
    throw new Error('isInstalled() must be implemented');
  }

  /**
   * Initialize a new project with the package manager
   * @param {string} projectPath - Path to the project directory
   * @param {string} pythonVersion - Python version to use (e.g., '3.11')
   * @returns {Promise<void>}
   */
  async initializeProject(projectPath, pythonVersion) {
    throw new Error('initializeProject() must be implemented');
  }

  /**
   * Install PyTorch packages
   * @param {string} projectPath - Path to the project directory
   * @param {string} gpuArch - GPU architecture (e.g., 'gfx1100')
   * @param {Array<string>} packages - List of packages to install
   * @param {Object} options - Installation options (e.g., { rocmVersion: '6.0' })
   * @returns {Promise<void>}
   */
  async installPackages(projectPath, gpuArch, packages, options = {}) {
    throw new Error('installPackages() must be implemented');
  }

  /**
   * Update installed packages
   * @param {string} projectPath - Path to the project directory
   * @param {string} gpuArch - GPU architecture
   * @param {Array<string>} packages - List of packages to update
   * @returns {Promise<void>}
   */
  async updatePackages(projectPath, gpuArch, packages) {
    throw new Error('updatePackages() must be implemented');
  }

  /**
   * List installed packages
   * @param {string} projectPath - Path to the project directory
   * @returns {Promise<Array<{name: string, version: string}>>}
   */
  async listInstalledPackages(projectPath) {
    throw new Error('listInstalledPackages() must be implemented');
  }

  /**
   * Run a Python script using the package manager's environment
   * @param {string} projectPath - Path to the project directory
   * @param {string} scriptPath - Path to the Python script
   * @returns {string} Command to execute
   */
  runPythonCommand(projectPath, scriptPath) {
    throw new Error('runPythonCommand() must be implemented');
  }

  /**
   * Get the path to the virtual environment
   * @param {string} projectPath - Path to the project directory
   * @returns {string} Path to venv
   */
  getVenvPath(projectPath) {
    throw new Error('getVenvPath() must be implemented');
  }

  /**
   * Check if project is initialized with this package manager
   * @param {string} projectPath - Path to the project directory
   * @returns {boolean} true if initialized, false otherwise
   */
  isProjectInitialized(projectPath) {
    throw new Error('isProjectInitialized() must be implemented');
  }

  /**
   * Get the name of the package manager
   * @returns {string} Package manager name
   */
  getName() {
    throw new Error('getName() must be implemented');
  }
}

module.exports = PackageManager;

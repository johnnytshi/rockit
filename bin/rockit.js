#!/usr/bin/env node

const path = require('path');
const { Command } = require('commander');
const { spawn } = require('child_process');
const { promptInstallation } = require(path.join(__dirname, '..', 'src', 'cli'));
const { promptPyTorchInstallation } = require(path.join(__dirname, '..', 'src', 'pytorch-cli'));
const { detectSystem } = require(path.join(__dirname, '..', 'src', 'system-detect'));
const { displayConfig, loadConfig } = require(path.join(__dirname, '..', 'src', 'config'));
const { promptBenchmark } = require(path.join(__dirname, '..', 'src', 'benchmark'));

const program = new Command();

program
  .name('rockit')
  .description('CLI tool to automate installing ROCm and updating PyTorch')
  .version('1.0.0');

program
  .command('rocm')
  .description('Install ROCm with interactive prompts')
  .action(async () => {
    try {
      await promptInstallation();
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  });

program
  .command('python')
  .description('Install or update PyTorch with ROCm support')
  .argument('[path]', 'Project directory path (defaults to current directory)')
  .action(async (pathArg) => {
    try {
      // Determine project path: use CLI arg if provided, else current directory
      const projectPath = pathArg ? path.resolve(pathArg) : process.cwd();
      await promptPyTorchInstallation(projectPath);
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  });

program
  .command('detect')
  .description('Detect system and show compatible ROCm versions')
  .action(async () => {
    try {
      const systemInfo = detectSystem();
      console.log('\n=== System Information ===\n');
      console.log('Platform:', systemInfo.platform);
      console.log('OS:', `${systemInfo.osInfo.type} ${systemInfo.osInfo.release}`);
      console.log('Architecture:', systemInfo.osInfo.arch);
      
      if (systemInfo.detected) {
        console.log('\n✅ AMD GPU Detected!');
        console.log('GPU Architecture:', systemInfo.gpuArch);
        console.log('Compatible ROCm GPU Families:', systemInfo.rocmGpuFamilies.join(', '));
      } else {
        console.log('\n⚠️  AMD GPU not detected');
      }
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  });

program
  .command('config')
  .description('Show current configuration')
  .action(() => {
    try {
      const config = loadConfig();
      displayConfig(config);
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  });

program
  .command('bench')
  .description('Run performance benchmarks')
  .action(async () => {
    try {
      await promptBenchmark();
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  });

program
  .command('view')
  .description('View benchmark results in a web interface')
  .action(() => {
    serveBenchmarkViewer();
  });

program.parse();

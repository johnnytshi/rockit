#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const os = require('os');

const RESULTS_DIR = path.join(os.homedir(), '.config', 'rockit', 'benchmark-results');

function parseMatrixResults(output) {
  const results = { matrixMultiplication: [] };
  const lines = output.split('\n');
  
  for (const line of lines) {
    // Match TOPS format: "  (1024, 2048, 4096): 12.34 ms (1.23 TOPS)"
    let match = line.match(/\((\d+),\s*(\d+),\s*(\d+)\):\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+TOPS\)/);
    if (match) {
      results.matrixMultiplication.push({
        m: parseInt(match[1]),
        n: parseInt(match[2]),
        k: parseInt(match[3]),
        timeMs: parseFloat(match[4]),
        tops: parseFloat(match[5]),
      });
      continue;
    }
    
    // Match GFLOPS format: "  (1024, 2048, 4096): 12.34 ms (1234.56 GFLOPS)"
    match = line.match(/\((\d+),\s*(\d+),\s*(\d+)\):\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+GFLOPS\)/);
    if (match) {
      results.matrixMultiplication.push({
        m: parseInt(match[1]),
        n: parseInt(match[2]),
        k: parseInt(match[3]),
        timeMs: parseFloat(match[4]),
        gflops: parseFloat(match[5]),
      });
      continue;
    }
    
    // Match old format: "  1024x1024: 1.07 ms (2010.63 GFLOPS)" or "Size 1024x1024: ..."
    match = line.match(/(?:Size\s+)?(\d+)x(\d+):\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+GFLOPS\)/);
    if (match) {
      const m = parseInt(match[1]);
      const n = parseInt(match[2]);
      const k = m; // For square matrices, k = m = n
      results.matrixMultiplication.push({
        m: m,
        n: n,
        k: k,
        timeMs: parseFloat(match[3]),
        gflops: parseFloat(match[4]),
      });
    }
  }
  return results;
}

function parseFlashResults(output) {
  const results = { flashAttention: {} };
  const lines = output.split('\n');
  
  for (const line of lines) {
    const timeMatch = line.match(/(?:Average )?[Tt]ime:\s+([\d.]+)\s+ms/);
    const throughputMatch = line.match(/Throughput:\s+([\d.]+)\s+tokens\/sec/);
    
    if (timeMatch) {
      results.flashAttention.timeMs = parseFloat(timeMatch[1]);
    }
    if (throughputMatch) {
      results.flashAttention.tokensPerSec = parseFloat(throughputMatch[1]);
    }
  }
  return results;
}

// Reprocess all existing results
if (!fs.existsSync(RESULTS_DIR)) {
  console.log('No results directory found.');
  process.exit(0);
}

const files = fs.readdirSync(RESULTS_DIR).filter(f => f.endsWith('.json'));
console.log(`Found ${files.length} result file(s) to reprocess...\n`);

let updated = 0;
for (const file of files) {
  const filepath = path.join(RESULTS_DIR, file);
  const result = JSON.parse(fs.readFileSync(filepath, 'utf8'));
  
  // Check if parsing needs update (check for old 'size' field or missing m/n/k)
  const needsUpdate = 
    (result.benchmarkType.includes('matrix') || result.benchmarkType === 'full') &&
    result.output &&
    (!result.parsed?.matrixMultiplication?.[0]?.m || result.parsed?.matrixMultiplication?.[0]?.size);
  
  if (needsUpdate) {
    console.log(`Reprocessing: ${file}`);
    
    // Reparse the output
    if (result.benchmarkType === 'matrix' || result.benchmarkType === 'full') {
      result.parsed = parseMatrixResults(result.output);
    }
    if (result.benchmarkType === 'flash' || result.benchmarkType === 'full') {
      result.parsed = { ...result.parsed, ...parseFlashResults(result.output) };
    }
    
    // Save updated result
    fs.writeFileSync(filepath, JSON.stringify(result, null, 2));
    console.log(`  ✅ Updated: ${result.parsed.matrixMultiplication?.length || 0} matrix results with m, n, k format\n`);
    updated++;
  }
}

console.log(`\n✅ Reprocessed ${updated} file(s)`);

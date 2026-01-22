const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

const PORT = 8080;
const FRONTEND_DIR = path.join(__dirname, '..', 'frontend');
const RESULTS_DIR = path.join(os.homedir(), '.config', 'rockit', 'benchmark-results');

const server = http.createServer((req, res) => {
    if (req.url === '/api/results') {
        fs.readdir(RESULTS_DIR, (err, files) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Failed to read results directory' }));
                return;
            }

            const jsonFiles = files.filter(file => path.extname(file) === '.json');
            const results = [];
            let filesRead = 0;

            if (jsonFiles.length === 0) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify([]));
                return;
            }

            jsonFiles.forEach(file => {
                fs.readFile(path.join(RESULTS_DIR, file), 'utf8', (err, content) => {
                    if (err) {
                        console.error(`Error reading file ${file}:`, err);
                    } else {
                        try {
                            results.push(JSON.parse(content));
                        } catch (e) {
                            console.error(`Error parsing JSON from file ${file}:`, e);
                        }
                    }
                    filesRead++;
                    if (filesRead === jsonFiles.length) {
                        res.writeHead(200, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify(results));
                    }
                });
            });
        });
    } else {
        let filePath = path.join(FRONTEND_DIR, req.url === '/' ? 'benchmark-viewer.html' : req.url);
        const extname = String(path.extname(filePath)).toLowerCase();
        const mimeTypes = {
            '.html': 'text/html',
            '.js': 'text/javascript',
            '.css': 'text/css',
        };

        const contentType = mimeTypes[extname] || 'application/octet-stream';

        fs.readFile(filePath, (err, content) => {
            if (err) {
                if (err.code == 'ENOENT') {
                    res.writeHead(404, { 'Content-Type': 'text/html' });
                    res.end('404: File Not Found');
                } else {
                    res.writeHead(500);
                    res.end('Sorry, check with the site admin for error: ' + err.code + ' ..\n');
                }
            } else {
                res.writeHead(200, { 'Content-Type': contentType });
                res.end(content, 'utf-8');
            }
        });
    }
});

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
    console.log('Open this URL in your browser to view the benchmark results.');
});

const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    mainWindow.loadFile('frontend/index.html');
}

function startPythonBackend() {
    pythonProcess = spawn('python', ['backend/api.py']);
    
    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python: ${data}`);
    });
}

app.whenReady().then(() => {
    createWindow();
    startPythonBackend();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
    if (pythonProcess) {
        pythonProcess.kill();
    }
}); 
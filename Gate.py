# Gate.py
# Credits: Dr_Animalis

import argparse
import json
import sys
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
import threading
import queue
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_JSON = "registered_codes.json"

# Global state for the display
scan_history = queue.Queue(maxsize=100)
current_scan = None
scan_lock = threading.Lock()

app = Flask(__name__)


def _load_codes(path: Path) -> set[str]:
    """Load registered codes from JSON file."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if isinstance(data, dict):
        data = data.get("codes", [])
    if not isinstance(data, list):
        return set()
    return {str(item).strip() for item in data if str(item).strip()}


def _save_codes(path: Path, codes: set[str]) -> None:
    """Save registered codes to JSON file."""
    payload = {"codes": sorted(codes)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def register_mode(path: Path) -> int:
    """CLI mode for registering codes."""
    codes = _load_codes(path)
    while True:
        raw = input("Enter new code (blank to finish): ").strip()
        if not raw:
            break
        codes.add(raw)
        print(f"Registered: {raw}")
    _save_codes(path, codes)
    print(f"Saved {len(codes)} codes to {path}")
    return 0


class BarcodeScanner:
    """Handles barcode detection from camera frames."""
    
    def __init__(self, codes_path: Path, debounce_seconds: float = 2.0):
        self.codes_path = codes_path
        self.debounce_seconds = debounce_seconds
        self.last_code = None
        self.last_time = 0.0
        self.allowed_symbols = [
            ZBarSymbol.CODE128,
            ZBarSymbol.CODE39,
            ZBarSymbol.EAN13,
            ZBarSymbol.EAN8,
            ZBarSymbol.UPCA,
            ZBarSymbol.UPCE,
        ]
        self._reload_codes()
    
    def _reload_codes(self):
        """Reload registered codes from file."""
        self.registered_codes = _load_codes(self.codes_path)
    
    def process_frame(self, frame_data: bytes) -> dict:
        """
        Process a camera frame and detect barcodes.
        Returns a dict with scan results.
        """
        # Reload codes in case they've been updated
        self._reload_codes()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "error": "Invalid image data"}
        
        # Preprocess for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Try multiple decode passes
        barcodes = []
        candidates = [gray, cv2.bitwise_not(gray)]
        upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        candidates.append(upscaled)
        candidates.append(cv2.bitwise_not(upscaled))
        
        for candidate in candidates:
            barcodes = pyzbar.decode(candidate, symbols=self.allowed_symbols)
            if barcodes:
                break
        
        if not barcodes:
            return {"success": True, "scanned": False}
        
        # Process the first barcode found
        barcode = barcodes[0]
        code = barcode.data.decode("utf-8", errors="ignore").strip()
        
        if not code:
            return {"success": True, "scanned": False}
        
        # Check debounce
        now = time.time()
        if code == self.last_code and (now - self.last_time) < self.debounce_seconds:
            return {"success": True, "scanned": False, "debounced": True}
        
        self.last_code = code
        self.last_time = now
        
        # Determine if code is authorized
        is_authorized = code in self.registered_codes
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Update global scan state
        scan_result = {
            "code": code,
            "authorized": is_authorized,
            "timestamp": timestamp,
            "message": "GATE OPEN" if is_authorized else "GATE CLOSED"
        }
        
        with scan_lock:
            global current_scan
            current_scan = scan_result
            # Add to history
            try:
                scan_history.put_nowait(scan_result.copy())
            except queue.Full:
                # Remove oldest and add new
                try:
                    scan_history.get_nowait()
                    scan_history.put_nowait(scan_result.copy())
                except queue.Empty:
                    pass
        
        logger.info(f"Scanned: {code} - {'Authorized' if is_authorized else 'Unauthorized'}")
        
        return {
            "success": True,
            "scanned": True,
            "code": code,
            "authorized": is_authorized,
            "message": scan_result["message"]
        }


# Initialize scanner (will be configured in main)
scanner = None


# Flask Routes

@app.route('/')
def index():
    """Mobile scanner page."""
    return render_template('scanner.html')


@app.route('/display')
def display():
    """Admin display page for showing scanned items."""
    return render_template('display.html')


@app.route('/api/scan', methods=['POST'])
def api_scan():
    """API endpoint for submitting scanned frames."""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty file"}), 400
    
    frame_data = file.read()
    result = scanner.process_frame(frame_data)
    
    return jsonify(result)


@app.route('/api/current-scan')
def api_current_scan():
    """Get the most recent scan result."""
    with scan_lock:
        if current_scan:
            return jsonify(current_scan)
        return jsonify({"code": None})


@app.route('/api/scan-history')
def api_scan_history():
    """Get scan history."""
    history = []
    # Get all items from queue without removing them
    with scan_lock:
        # Create a copy of the queue items
        temp_queue = queue.Queue()
        while not scan_history.empty():
            try:
                item = scan_history.get_nowait()
                history.append(item)
                temp_queue.put(item)
            except queue.Empty:
                break
        
        # Restore items
        while not temp_queue.empty():
            try:
                scan_history.put(temp_queue.get_nowait())
            except queue.Full:
                pass
    
    return jsonify(history[::-1])  # Return in reverse order (newest first)


@app.route('/api/registered-codes')
def api_registered_codes():
    """Get list of registered codes."""
    codes = list(scanner.registered_codes)
    return jsonify({"count": len(codes), "codes": codes})


@app.route('/api/register-code', methods=['POST'])
def api_register_code():
    """Register a new code."""
    data = request.get_json()
    if not data or 'code' not in data:
        return jsonify({"error": "No code provided"}), 400
    
    code = data['code'].strip()
    if not code:
        return jsonify({"error": "Empty code"}), 400
    
    codes = _load_codes(scanner.codes_path)
    codes.add(code)
    _save_codes(scanner.codes_path, codes)
    
    # Reload codes in scanner
    scanner._reload_codes()
    
    return jsonify({"success": True, "code": code})


def create_html_templates():
    """Create HTML templates if they don't exist."""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Scanner template
    scanner_template = """<!DOCTYPE html>
<html>
<head>
    <title>Barcode Scanner</title>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #000;
            color: #fff;
            height: 100vh;
            overflow: hidden;
            position: relative;
        }
        
        #container {
            position: relative;
            width: 100%;
            height: 100%;
        }
        
        #video-container {
            position: relative;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        
        #video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        #overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 20px 20px calc(20px + env(safe-area-inset-bottom));
            z-index: 10;
        }
        
        .scan-area {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .scan-frame {
            width: 80%;
            max-width: 300px;
            height: 200px;
            border: 3px solid rgba(255, 255, 255, 0.5);
            border-radius: 20px;
            box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5);
            position: relative;
        }
        
        .scan-line {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: #00ff00;
            animation: scan 2s linear infinite;
            box-shadow: 0 0 10px #00ff00;
        }
        
        @keyframes scan {
            0% { top: 0; }
            50% { top: 100%; }
            100% { top: 0; }
        }
        
        #result {
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            transition: all 0.3s;
            backdrop-filter: blur(5px);
        }
        
        .authorized {
            color: #00ff00;
            border-left: 4px solid #00ff00;
        }
        
        .unauthorized {
            color: #ff0000;
            border-left: 4px solid #ff0000;
        }
        
        #controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            padding: 15px;
            margin-bottom: 24px;
            pointer-events: auto;
        }
        
        button {
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 30px;
            font-size: 16px;
            backdrop-filter: blur(5px);
            cursor: pointer;
            transition: background 0.3s;
            pointer-events: auto;
        }
        
        button:active {
            background: rgba(255, 255, 255, 0.4);
        }
        
        #server-status {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            z-index: 20;
        }
        
        #fps-counter {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.7);
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            z-index: 20;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="video-container">
            <video id="video" playsinline autoplay></video>
            <canvas id="canvas" style="display: none;"></canvas>
        </div>
        
        <div id="overlay">
            <div id="server-status"> Connecting...</div>
            <div id="fps-counter">0 fps</div>
            
            <div class="scan-area">
                <div class="scan-frame">
                    <div class="scan-line"></div>
                </div>
            </div>
            
            <div id="result">Ready to scan</div>
            
            <div id="controls">
                <button id="toggleCamera">Switch Camera</button>
            </div>
        </div>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const resultDiv = document.getElementById('result');
        const serverStatus = document.getElementById('server-status');
        const fpsCounter = document.getElementById('fps-counter');
        
        let currentStream = null;
        let scanning = true;
        let lastScanTime = 0;
        const scanCooldown = 2000; // 2 seconds cooldown between scans
        let frameCount = 0;
        let lastFpsUpdate = performance.now();
        let useBackCamera = true;
        
        // Check server connection
        async function checkServer() {
            try {
                const response = await fetch('/api/current-scan');
                if (response.ok) {
                    serverStatus.innerHTML = 'Connected';
                    serverStatus.style.color = '#00ff00';
                    return true;
                }
            } catch (e) {
                serverStatus.innerHTML = 'Disconnected';
                serverStatus.style.color = '#ff0000';
            }
            return false;
        }
        
        setInterval(checkServer, 3000);
        
        // Start camera
        async function startCamera(useBack = true) {
            if (currentStream) {
                currentStream.getTracks().forEach(track => track.stop());
            }
            
            const constraints = {
                video: {
                    facingMode: useBack ? 'environment' : 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };
            
            try {
                currentStream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = currentStream;
                await video.play();
                scanFrame();
            } catch (err) {
                resultDiv.innerHTML = 'Camera error: ' + err.message;
            }
        }
        
        // Toggle camera
        document.getElementById('toggleCamera').addEventListener('click', () => {
            useBackCamera = !useBackCamera;
            startCamera(useBackCamera);
        });
        
        // Scan frame and send to server
        async function scanFrame() {
            if (!scanning) return;
            
            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('image', blob, 'frame.jpg');
                    
                    try {
                        const response = await fetch('/api/scan', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (result.success && result.scanned) {
                            const now = Date.now();
                            if (now - lastScanTime > scanCooldown) {
                                lastScanTime = now;
                                
                                resultDiv.innerHTML = result.message + '<br>' + result.code;
                                resultDiv.className = result.authorized ? 'authorized' : 'unauthorized';
                            }
                        }
                    } catch (e) {
                        // Silently fail - server might be busy
                    }
                    
                    // Update FPS counter
                    frameCount++;
                    const now = performance.now();
                    if (now - lastFpsUpdate >= 1000) {
                        const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
                        fpsCounter.innerHTML = `${fps} fps`;
                        frameCount = 0;
                        lastFpsUpdate = now;
                    }
                    
                    // Continue scanning
                    requestAnimationFrame(scanFrame);
                }, 'image/jpeg', 0.8);
            } else {
                requestAnimationFrame(scanFrame);
            }
        }
        
        // Start with back camera
        startCamera(true);
    </script>
</body>
</html>"""
    
    # Display template
    display_template = """<!DOCTYPE html>
<html>
<head>
    <title>Gate Display - Scanned Items</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #00ff00;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .current-scan {
            background: #2a2a2a;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            border-left: 5px solid #00ff00;
            transition: all 0.3s;
        }
        
        .current-scan.unauthorized {
            border-left-color: #ff0000;
        }
        
        .timestamp {
            color: #888;
            font-size: 14px;
            margin-bottom: 10px;
        }
        
        .code {
            font-size: 48px;
            font-weight: bold;
            margin: 20px 0;
            word-break: break-all;
            font-family: monospace;
        }
        
        .message {
            font-size: 24px;
            font-weight: bold;
        }
        
        .message.authorized {
            color: #00ff00;
        }
        
        .message.unauthorized {
            color: #ff0000;
        }
        
        .history-section {
            background: #2a2a2a;
            border-radius: 15px;
            padding: 20px;
        }
        
        h2 {
            margin-bottom: 20px;
            color: #00ff00;
        }
        
        .history-list {
            list-style: none;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .history-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            border-bottom: 1px solid #3a3a3a;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .history-item.authorized {
            border-left: 3px solid #00ff00;
        }
        
        .history-item.unauthorized {
            border-left: 3px solid #ff0000;
        }
        
        .history-code {
            font-family: monospace;
            font-size: 16px;
            font-weight: bold;
        }
        
        .history-time {
            color: #888;
            font-size: 12px;
        }
        
        .history-status {
            font-size: 12px;
            padding: 3px 8px;
            border-radius: 12px;
        }
        
        .history-status.authorized {
            background: rgba(0, 255, 0, 0.2);
            color: #00ff00;
        }
        
        .history-status.unauthorized {
            background: rgba(255, 0, 0, 0.2);
            color: #ff0000;
        }
        
        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .stat-box {
            background: #333;
            border-radius: 10px;
            padding: 15px;
            flex: 1;
            min-width: 120px;
            text-align: center;
        }
        
        .stat-label {
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
        }
        
        #registered-codes {
            background: #333;
            border-radius: 8px;
            padding: 10px;
            margin-top: 20px;
        }
        
        .code-badge {
            display: inline-block;
            background: #444;
            padding: 5px 10px;
            border-radius: 15px;
            margin: 3px;
            font-family: monospace;
            font-size: 12px;
        }
        
        #refresh-btn {
            background: #00ff00;
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            margin-left: 10px;
        }
        
        #refresh-btn:hover {
            background: #00cc00;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gate Scanner Display</h1>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Total Scans Today</div>
                <div class="stat-value" id="totalScans">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Authorized</div>
                <div class="stat-value" id="authorizedCount">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Unauthorized</div>
                <div class="stat-value" id="unauthorizedCount">0</div>
            </div>
        </div>
        
        <div id="current-scan" class="current-scan">
            <div class="timestamp">Waiting for scan...</div>
            <div class="code">-</div>
            <div class="message">Point your phone camera at a barcode</div>
        </div>
        
        <div class="history-section">
            <h2>Recent Scans</h2>
            <ul id="history-list" class="history-list">
                <li style="text-align: center; color: #888; padding: 20px;">No scans yet</li>
            </ul>
        </div>
        
        <div style="margin-top: 20px; text-align: center;">
            <small style="color: #666;">Scan with your phone at <span id="server-url"></span></small>
        </div>
    </div>
    
    <script>
        const currentScanDiv = document.getElementById('current-scan');
        const historyList = document.getElementById('history-list');
        const totalScansSpan = document.getElementById('totalScans');
        const authorizedCountSpan = document.getElementById('authorizedCount');
        const unauthorizedCountSpan = document.getElementById('unauthorizedCount');
        const serverUrlSpan = document.getElementById('server-url');
        
        // Display server URL
        serverUrlSpan.textContent = window.location.hostname + ':' + window.location.port;
        
        let lastScanKey = null;
        
        // Update current scan and history
        async function updateDisplay() {
            try {
                // Get current scan
                const currentResponse = await fetch('/api/current-scan');
                const currentData = await currentResponse.json();
                
                if (currentData.code) {
                    const isAuthorized = currentData.authorized;
                    const className = isAuthorized ? 'authorized' : 'unauthorized';
                    
                    currentScanDiv.className = 'current-scan ' + className;
                    currentScanDiv.innerHTML = `
                        <div class="timestamp">${currentData.timestamp || new Date().toLocaleTimeString()}</div>
                        <div class="code">${currentData.code}</div>
                        <div class="message ${className}">${currentData.message}</div>
                    `;
                    
                    // Treat each scan event as unique using code + timestamp.
                    // This allows repeated scans of the same barcode to be counted.
                    const scanKey = `${currentData.code}|${currentData.timestamp || ''}`;
                    if (scanKey !== lastScanKey) {
                        lastScanKey = scanKey;
                    }
                }
                
                // Get history
                const historyResponse = await fetch('/api/scan-history');
                const historyData = await historyResponse.json();
                
                if (historyData.length > 0) {
                    let html = '';
                    historyData.forEach(item => {
                        const className = item.authorized ? 'authorized' : 'unauthorized';
                        html += `
                            <li class="history-item ${className}">
                                <div>
                                    <div class="history-code">${item.code}</div>
                                    <div class="history-time">${item.timestamp}</div>
                                </div>
                                <div class="history-status ${className}">${item.message}</div>
                            </li>
                        `;
                    });
                    historyList.innerHTML = html;
                }
                
                // Update stats from history so counts stay accurate.
                const scanCount = historyData.length;
                const authorizedCount = historyData.filter(item => item.authorized).length;
                const unauthorizedCount = scanCount - authorizedCount;

                totalScansSpan.textContent = scanCount;
                authorizedCountSpan.textContent = authorizedCount;
                unauthorizedCountSpan.textContent = unauthorizedCount;
                
            } catch (e) {
                console.error('Error updating display:', e);
            }
        }
        
        // Update every second
        setInterval(updateDisplay, 1000);
        updateDisplay();
    </script>
</body>
</html>"""
    
    # Write templates
    (templates_dir / "scanner.html").write_text(scanner_template)
    (templates_dir / "display.html").write_text(display_template)
    logger.info("HTML templates created in ./templates directory")


def run_flask(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Run the Flask application."""
    create_html_templates()
    logger.info(f"Starting Flask server on http://{host}:{port}")
    logger.info(f"Access the scanner on your phone at http://YOUR_IP:{port}/")
    logger.info(f"View the display at http://localhost:{port}/display")
    app.run(host=host, port=port, debug=debug, threaded=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Barcode gate system with Flask")
    parser.add_argument(
        "--mode",
        choices=["gate", "register", "server"],
        default="server",
        help="Run the gate scanner, register codes, or start the Flask server",
    )
    parser.add_argument(
        "--json",
        default=DEFAULT_JSON,
        help="Path to registered codes json file",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for gate mode (default 0)",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Seconds to ignore repeated scans of the same code",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind Flask server to (default 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for Flask server (default 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug mode",
    )
    
    args = parser.parse_args(argv)
    path = Path(args.json)
    
    if args.mode == "register":
        return register_mode(path)
    elif args.mode == "gate":
        return gate_mode(path, args.camera, args.debounce)
    else:  # server mode
        global scanner
        scanner = BarcodeScanner(path, args.debounce)
        run_flask(host=args.host, port=args.port, debug=args.debug)
        return 0


if __name__ == "__main__":
    # Keep the original gate_mode function for backward compatibility
    def gate_mode(path: Path, camera_index: int, debounce_seconds: float) -> int:
        """Original gate mode using local camera."""
        try:
            import cv2
            from pyzbar import pyzbar
            from pyzbar.pyzbar import ZBarSymbol
        except Exception as exc:
            print("Missing dependencies. Install with: pip install opencv-python pyzbar")
            print(f"Error: {exc}")
            return 1

        codes = _load_codes(path)
        if not codes:
            print("Warning: no registered codes yet. Use --mode register to add codes.")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Could not open camera index {camera_index}")
            return 1

        last_code = None
        last_time = 0.0

        allowed_symbols = [
            ZBarSymbol.CODE128,
            ZBarSymbol.CODE39,
            ZBarSymbol.EAN13,
            ZBarSymbol.EAN8,
            ZBarSymbol.UPCA,
            ZBarSymbol.UPCE,
        ]

        preview_enabled = True
        print("Starting gate scanner. Press Ctrl+C or press 'q' in the preview window to stop.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed.")
                    time.sleep(0.1)
                    continue

                # Preprocess to help 1D barcode detection without over-thresholding.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

                if preview_enabled:
                    try:
                        # Sharpness meter: variance of Laplacian.
                        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                        cv2.putText(
                            frame,
                            f"Sharpness: {sharpness:.0f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                        cv2.imshow("Gate Camera Preview", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        preview_enabled = False
                        print(
                            "Preview disabled: OpenCV GUI functions not available. "
                            "Install opencv-python (not headless) to enable preview."
                        )

                # Try multiple decode passes to improve detection.
                barcodes = []
                candidates = [gray, cv2.bitwise_not(gray)]
                upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                candidates.append(upscaled)
                candidates.append(cv2.bitwise_not(upscaled))

                for candidate in candidates:
                    barcodes = pyzbar.decode(candidate, symbols=allowed_symbols)
                    if barcodes:
                        break

                # Draw barcode bounding boxes when preview is on.
                if preview_enabled and barcodes:
                    for barcode in barcodes:
                        x, y, w, h = barcode.rect
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if not barcodes:
                    continue

                for barcode in barcodes:
                    code = barcode.data.decode("utf-8", errors="ignore").strip()
                    if not code:
                        continue

                    now = time.time()
                    if code == last_code and (now - last_time) < debounce_seconds:
                        continue

                    last_code = code
                    last_time = now

                    if code in codes:
                        print(f"GATE OPEN - {code}")
                    else:
                        print(f"GATE CLOSED - {code}")
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            cap.release()
            if preview_enabled:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

        return 0
    
    raise SystemExit(main(sys.argv[1:]))

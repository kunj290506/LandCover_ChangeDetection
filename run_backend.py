#!/usr/bin/env python3
"""
Simple Backend Server - Land Cover Change Detection
Runs inference server on localhost with basic HTTP interface
"""

import os
import sys
import json
import socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse as urlparse
import cgi
from pathlib import Path
from datetime import datetime
import base64
import cv2
import numpy as np
import torch

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from models.snunet import SNUNet

class ChangeDetectionServer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        print(f"[INFO] Server initialized with device: {self.device}")
        
    def load_model(self):
        """Load trained model"""
        model_path = project_root / "checkpoints" / "best_model.pth"
        model = SNUNet(3, 1).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def predict_from_files(self, image1_path, image2_path):
        """Run inference on image files"""
        try:
            # Load and preprocess images
            img1 = cv2.imread(str(image1_path))
            img2 = cv2.imread(str(image2_path))
            
            if img1 is None or img2 is None:
                return {"error": "Could not load images"}
            
            # Preprocess
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img1, (256, 256))
            img2 = cv2.resize(img2, (256, 256))
            
            # Normalize and convert to tensor
            img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float() / 255.0
            img2 = torch.from_numpy(img2.transpose(2, 0, 1)).float() / 255.0
            img1 = img1.unsqueeze(0).to(self.device)
            img2 = img2.unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(img1, img2)
                probs = torch.sigmoid(logits)
                mask = (probs > 0.5).float()
                
            # Convert to numpy
            mask_np = mask.squeeze().cpu().numpy()
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            
            # Calculate metrics
            change_pixels = int(np.sum(mask_np > 0.5))
            total_pixels = int(mask_np.size)
            change_percentage = (change_pixels / total_pixels) * 100
            
            return {
                "status": "success",
                "change_detected": bool(change_pixels > 0),
                "change_pixels": change_pixels,
                "total_pixels": total_pixels,
                "change_percentage": round(change_percentage, 2),
                "mask_shape": list(mask_np.shape),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Initialize server
cd_server = ChangeDetectionServer()

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/":
            self.send_web_interface()
        elif self.path == "/health":
            self.send_health_check()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == "/detect":
            self.handle_detect_request()
        else:
            self.send_error(404, "Not Found")
    
    def do_OPTIONS(self):
        """Handle preflight requests for CORS"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def handle_detect_request(self):
        """Handle change detection request"""
        try:
            # Parse multipart form data
            content_type = self.headers.get('Content-Type', '')
            if not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Expected multipart/form-data")
                return
            
            # Get form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            # Get uploaded files
            image_before = form['image_before']
            image_after = form['image_after']
            
            if not image_before.file or not image_after.file:
                self.send_error(400, "Missing image files")
                return
            
            # Read image data
            before_data = image_before.file.read()
            after_data = image_after.file.read()
            
            # Run inference
            result = self.predict_from_bytes(before_data, after_data)
            
            # Send response
            self.send_response(200)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            error_response = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self.send_response(500)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def predict_from_bytes(self, image1_bytes, image2_bytes):
        """Run inference on image bytes"""
        try:
            # Decode images from bytes
            nparr1 = np.frombuffer(image1_bytes, np.uint8)
            nparr2 = np.frombuffer(image2_bytes, np.uint8)
            
            img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
            
            if img1 is None or img2 is None:
                return {"status": "error", "error": "Could not decode images"}
            
            # Preprocess
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img1, (256, 256))
            img2 = cv2.resize(img2, (256, 256))
            
            # Normalize (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(cd_server.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(cd_server.device)
            
            img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float() / 255.0
            img2 = torch.from_numpy(img2.transpose(2, 0, 1)).float() / 255.0
            
            img1 = img1.unsqueeze(0).to(cd_server.device)
            img2 = img2.unsqueeze(0).to(cd_server.device)
            
            # Apply Normalization
            img1 = (img1 - mean) / std
            img2 = (img2 - mean) / std
            
            # Inference
            with torch.no_grad():
                logits = cd_server.model(img1, img2)
                probs = torch.sigmoid(logits)
                
                # Debug logging
                print(f"[DEBUG] Probs: Min={probs.min().item():.4f}, Max={probs.max().item():.4f}, Mean={probs.mean().item():.4f}")
                
                mask = (probs > 0.5).float()
                
            # Convert to numpy and create visualization
            mask_np = mask.squeeze().cpu().numpy()
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            
            # Create colored mask for visualization
            mask_colored = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_HOT)
            
            # Encode mask as base64
            _, buffer = cv2.imencode('.png', mask_colored)
            mask_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Calculate metrics
            change_pixels = int(np.sum(mask_np > 0.5))
            total_pixels = int(mask_np.size)
            change_percentage = (change_pixels / total_pixels) * 100
            
            return {
                "status": "success",
                "change_detected": bool(change_pixels > 0),
                "change_pixels": change_pixels,
                "total_pixels": total_pixels,
                "change_percentage": round(change_percentage, 2),
                "mask_base64": mask_b64,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def send_web_interface(self):
        """Send simple web interface"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Land Cover Change Detection</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f0f8ff; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #2e7d32; text-align: center; }}
                .status {{ text-align: center; background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .example {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .code {{ background: #263238; color: #ffffff; padding: 10px; border-radius: 5px; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌍 Land Cover Change Detection API</h1>
                
                <div class="status">
                    <h3>✅ API Server Running</h3>
                    <p>Model: SNUNet-CBAM</p>
                    <p>Device: {cd_server.device}</p>
                    <p>Status: Ready for inference</p>
                </div>
                
                <h3>🎨 Frontend Interface</h3>
                <div class="example">
                    <p>Access the web interface at:</p>
                    <div class="code">
                        <a href="http://localhost:3000" style="color: #4caf50;">http://localhost:3000</a>
                    </div>
                </div>
                
                <h3>🚀 Quick Test</h3>
                <div class="example">
                    <p>Test the API with sample images from your dataset:</p>
                    <div class="code">
                        python test_api.py
                    </div>
                </div>
                
                <h3>📡 API Endpoints</h3>
                <div class="example">
                    <p><strong>Health Check:</strong> <a href="/health">/health</a></p>
                    <p><strong>Change Detection:</strong> POST /detect</p>
                </div>
                
                <h3>💻 Usage Example</h3>
                <div class="example">
                    <p>Use Python requests or curl to send images:</p>
                    <div class="code">
import requests<br>
# Load your before/after images<br>
response = requests.post('http://localhost:8080/detect', <br>
&nbsp;&nbsp;files={{"image_before": open("before.jpg", "rb"), "image_after": open("after.jpg", "rb")}})
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_cors_headers()
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def send_health_check(self):
        """Send health check response"""
        health_data = {
            "status": "healthy",
            "model_loaded": True,
            "device": str(cd_server.device),
            "timestamp": datetime.now().isoformat()
        }
        
        self.send_response(200)
        self.send_cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(health_data).encode())

def run_server(port=8080):
    """Run the HTTP server"""
    try:
        server = HTTPServer(('localhost', port), RequestHandler)
        print("=" * 60)
        print("LAND COVER CHANGE DETECTION - BACKEND RUNNING")
        print("=" * 60)
        print(f"Server URL: http://localhost:{port}")
        print(f"Health Check: http://localhost:{port}/health")
        print(f"Device: {cd_server.device}")
        print(f"Model: Loaded and ready")
        print()
        print("Open http://localhost:8080 in your browser")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n\nServer stopped")
        server.shutdown()
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    run_server()
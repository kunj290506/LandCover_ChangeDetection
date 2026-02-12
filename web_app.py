#!/usr/bin/env python3
"""
Land Cover Change Detection Web API Server
FastAPI server with web interface for change detection
"""

import os
import sys
import io
import base64
import torch
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Request
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from models.snunet import SNUNet

class ChangeDetectionAPI:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
    def load_model(self):
        """Load the trained SNUNet model"""
        model_path = project_root / "checkpoints" / "best_model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = SNUNet(3, 1).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def preprocess_image(self, image_bytes, target_size=(256, 256)):
        """Preprocess uploaded image"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = image.unsqueeze(0).to(self.device)
        
        return image
    
    def predict(self, image_before_bytes, image_after_bytes, threshold=0.5):
        """Predict change detection"""
        try:
            # Preprocess
            img_before = self.preprocess_image(image_before_bytes)
            img_after = self.preprocess_image(image_after_bytes)
            
            # Inference
            with torch.no_grad():
                logits = self.model(img_before, img_after)
            
            # Post-process
            probs = torch.sigmoid(logits)
            mask = (probs > threshold).float()
            mask_np = mask.squeeze().cpu().numpy()
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            
            # Calculate metrics
            change_pixels = int(np.sum(mask_np > 0.5))
            total_pixels = int(mask_np.size)
            change_percentage = float((change_pixels / total_pixels) * 100)
            
            # Convert mask to base64 image
            mask_bgr = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
            _, buffer = cv2.imencode('.png', mask_bgr)
            mask_b64 = base64.b64encode(buffer).decode('utf-8')
            
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
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Initialize API
cd_api = ChangeDetectionAPI()

if FASTAPI_AVAILABLE:
    # FastAPI Application
    app = FastAPI(title="Land Cover Change Detection API", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def home():
        """Serve web interface"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Land Cover Change Detection</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; margin-bottom: 30px; }
                .upload-section { display: flex; gap: 20px; margin-bottom: 20px; }
                .upload-box { flex: 1; border: 2px dashed #ddd; padding: 20px; text-align: center; border-radius: 5px; }
                .upload-box.dragover { border-color: #4CAF50; background: #f9f9f9; }
                .btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                .btn:hover { background: #45a049; }
                .results { margin-top: 20px; }
                .result-images { display: flex; gap: 20px; margin-top: 20px; }
                .result-images img { max-width: 200px; border-radius: 5px; }
                .metrics { background: #f0f0f0; padding: 15px; border-radius: 5px; margin-top: 10px; }
                .error { color: red; background: #ffebee; padding: 10px; border-radius: 5px; }
                .success { color: green; background: #e8f5e8; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌍 Land Cover Change Detection</h1>
                    <p>Upload before and after satellite images to detect changes</p>
                </div>
                
                <form id="uploadForm">
                    <div class="upload-section">
                        <div class="upload-box">
                            <h3>📷 Before Image</h3>
                            <input type="file" id="beforeImage" accept="image/*" style="display: none;">
                            <div id="beforePreview">Click to select image</div>
                            <button type="button" onclick="document.getElementById('beforeImage').click()">Choose File</button>
                        </div>
                        
                        <div class="upload-box">
                            <h3>📷 After Image</h3>
                            <input type="file" id="afterImage" accept="image/*" style="display: none;">
                            <div id="afterPreview">Click to select image</div>
                            <button type="button" onclick="document.getElementById('afterImage').click()">Choose File</button>
                        </div>
                    </div>
                    
                    <div style="text-align: center;">
                        <button type="button" class="btn" onclick="detectChanges()">🔍 Detect Changes</button>
                    </div>
                </form>
                
                <div id="results" class="results" style="display: none;">
                    <h3>📊 Results</h3>
                    <div id="resultContent"></div>
                </div>
            </div>

            <script>
                // File preview functionality
                function setupFilePreview(inputId, previewId) {
                    const input = document.getElementById(inputId);
                    const preview = document.getElementById(previewId);
                    
                    input.addEventListener('change', function(e) {
                        const file = e.target.files[0];
                        if (file) {
                            const reader = new FileReader();
                            reader.onload = function(e) {
                                preview.innerHTML = '<img src="' + e.target.result + '" style="max-width: 150px; border-radius: 5px;">';
                            };
                            reader.readAsDataURL(file);
                        }
                    });
                }
                
                setupFilePreview('beforeImage', 'beforePreview');
                setupFilePreview('afterImage', 'afterPreview');
                
                async function detectChanges() {
                    const beforeFile = document.getElementById('beforeImage').files[0];
                    const afterFile = document.getElementById('afterImage').files[0];
                    
                    if (!beforeFile || !afterFile) {
                        alert('Please select both before and after images');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('image_before', beforeFile);
                    formData.append('image_after', afterFile);
                    
                    // Show loading
                    const results = document.getElementById('results');
                    const resultContent = document.getElementById('resultContent');
                    results.style.display = 'block';
                    resultContent.innerHTML = '<p>🔄 Processing images...</p>';
                    
                    try {
                        const response = await fetch('/detect', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (data.status === 'success') {
                            resultContent.innerHTML = `
                                <div class="success">
                                    <h4>✅ Analysis Complete</h4>
                                    <div class="metrics">
                                        <p><strong>Change Detected:</strong> ${data.change_detected ? 'Yes' : 'No'}</p>
                                        <p><strong>Change Percentage:</strong> ${data.change_percentage}%</p>
                                        <p><strong>Changed Pixels:</strong> ${data.change_pixels.toLocaleString()} / ${data.total_pixels.toLocaleString()}</p>
                                        <p><strong>Analysis Time:</strong> ${new Date(data.timestamp).toLocaleString()}</p>
                                    </div>
                                    <h4>📊 Change Detection Mask</h4>
                                    <img src="data:image/png;base64,${data.mask_base64}" style="max-width: 400px; border-radius: 5px;">
                                    <p><em>White areas indicate detected changes</em></p>
                                </div>
                            `;
                        } else {
                            resultContent.innerHTML = `<div class="error"><h4>❌ Error</h4><p>${data.error}</p></div>`;
                        }
                        
                    } catch (error) {
                        resultContent.innerHTML = `<div class="error"><h4>❌ Network Error</h4><p>${error.message}</p></div>`;
                    }
                }
            </script>
        </body>
        </html>
        """
        return html_content

    @app.post("/detect")
    async def detect_changes(
        image_before: UploadFile = File(...),
        image_after: UploadFile = File(...)
    ):
        """API endpoint for change detection"""
        try:
            # Read uploaded files
            before_bytes = await image_before.read()
            after_bytes = await image_after.read()
            
            # Run inference
            result = cd_api.predict(before_bytes, after_bytes)
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": str(cd_api.device),
            "timestamp": datetime.now().isoformat()
        }

def start_web_server():
    """Start the web server"""
    print("🌟 STARTING WEB APPLICATION...")
    print(f"🖥️  Device: {cd_api.device}")
    print(f"✅ Model loaded successfully")
    print()
    print("🌐 Starting web server...")
    print("📱 Open your browser to: http://localhost:8080")
    print("📚 API documentation: http://localhost:8080/docs")
    print("💻 Health check: http://localhost:8080/health")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")

def main():
    """Main function"""
    print("=" * 60)
    print("🚀 LAND COVER CHANGE DETECTION - WEB APPLICATION")
    print("=" * 60)
    
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Running basic test...")
        # Run basic test
        return
    
    try:
        start_web_server()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
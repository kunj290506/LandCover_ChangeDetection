"""
Land Cover Change Detection - Flask Application
Minimal, production-ready web server with model inference
"""

import os
import uuid
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
import torch
from PIL import Image

from inference import ChangeDetector

# Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Ensure upload folder exists
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Initialize model (lazy loading)
detector = None


def get_detector():
    """Lazy load the change detector model."""
    global detector
    if detector is None:
        model_path = Path(__file__).parent.parent / 'checkpoints_production' / 'best_model.pth'
        if not model_path.exists():
            # Fallback paths
            model_path = Path(__file__).parent.parent / 'best_model.pth'
            if not model_path.exists():
                 model_path = Path(__file__).parent.parent / 'checkpoints_optimized' / 'best_model.pth'
        detector = ChangeDetector(str(model_path) if model_path.exists() else None)
    return detector


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Landing page route."""
    return render_template('index.html')


@app.route('/app')
def application():
    """Main application page route."""
    return render_template('app.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for change detection inference.
    Expects: image1, image2 as multipart/form-data
    Returns: JSON with results and image URLs
    """
    start_time = time.time()
    
    # Validate request
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Both images are required'}), 400
    
    file1 = request.files['image1']
    file2 = request.files['image2']
    
    if not file1.filename or not file2.filename:
        return jsonify({'error': 'No selected files'}), 400
    
    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Generate unique ID for this request
        request_id = str(uuid.uuid4())[:8]
        
        # Save uploaded files
        filename1 = f"{request_id}_t1_{secure_filename(file1.filename)}"
        filename2 = f"{request_id}_t2_{secure_filename(file2.filename)}"
        
        path1 = app.config['UPLOAD_FOLDER'] / filename1
        path2 = app.config['UPLOAD_FOLDER'] / filename2
        
        file1.save(str(path1))
        file2.save(str(path2))
        
        # Run inference
        detector = get_detector()
        result = detector.predict(str(path1), str(path2))
        
        # Save result images
        mask_filename = f"{request_id}_mask.png"
        overlay_filename = f"{request_id}_overlay.png"
        
        mask_path = app.config['UPLOAD_FOLDER'] / mask_filename
        overlay_path = app.config['UPLOAD_FOLDER'] / overlay_filename
        
        result['mask'].save(str(mask_path))
        result['overlay'].save(str(overlay_path))
        
        # Calculate metrics
        total_pixels = result['mask'].size[0] * result['mask'].size[1]
        changed_pixels = result['changed_pixels']
        change_percentage = (changed_pixels / total_pixels) * 100
        
        process_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'image1_url': url_for('uploaded_file', filename=filename1),
            'image2_url': url_for('uploaded_file', filename=filename2),
            'mask_url': url_for('uploaded_file', filename=mask_filename),
            'overlay_url': url_for('uploaded_file', filename=overlay_filename),
            'changed_pixels': changed_pixels,
            'total_pixels': total_pixels,
            'change_percentage': change_percentage,
            'confidence': result.get('confidence', 0.95),
            'process_time': round(process_time, 2)
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'cuda_available': torch.cuda.is_available()
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Land Cover Change Detection Server")
    print("="*50)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("Starting server at http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

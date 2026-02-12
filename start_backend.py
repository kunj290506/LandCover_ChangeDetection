#!/usr/bin/env python3
"""
Backend Application Launcher
Starts the Land Cover Change Detection services locally
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "services" / "common"))
sys.path.insert(0, str(project_root / "services" / "gateway" / "app"))
sys.path.insert(0, str(project_root / "services" / "inference" / "app"))
sys.path.insert(0, str(project_root / "services" / "training" / "app"))

def run_service(name, module, port, cwd=None):
    """Run a FastAPI service using uvicorn"""
    try:
        cmd = [
            sys.executable, "-m", "uvicorn", 
            f"{module}:app", 
            "--host", "0.0.0.0", 
            "--port", str(port),
            "--reload"
        ]
        
        print(f"🚀 Starting {name} on port {port}...")
        
        if cwd:
            process = subprocess.Popen(cmd, cwd=cwd)
        else:
            process = subprocess.Popen(cmd)
            
        return process
        
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None

def check_model_exists():
    """Check if trained model exists"""
    model_path = project_root / "checkpoints" / "best_model.pth"
    return model_path.exists()

def setup_environment():
    """Setup environment variables for local development"""
    os.environ.setdefault("POSTGRES_URL", "sqlite:///landcover_local.db")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("MINIO_ENDPOINT", "http://localhost:9000")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("DEBUG", "true")
    os.environ.setdefault("ENVIRONMENT", "development")

def start_minimal_stack():
    """Start minimal services for local development"""
    print("=" * 60)
    print("🌟 LAND COVER CHANGE DETECTION - BACKEND STARTUP")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Check if model exists
    if check_model_exists():
        print("✅ Trained model found: checkpoints/best_model.pth")
    else:
        print("⚠️  No trained model found - inference may not work")
    
    # Start services
    services = []
    
    try:
        # Start Inference Service (core functionality)
        inference_cwd = str(project_root / "services" / "inference")
        inference_proc = run_service(
            "Inference Service", 
            "app.main", 
            8001, 
            cwd=inference_cwd
        )
        if inference_proc:
            services.append(("Inference", inference_proc, 8001))
            time.sleep(3)  # Let it start
        
        # Start Gateway Service (API entry point)  
        gateway_cwd = str(project_root / "services" / "gateway")
        gateway_proc = run_service(
            "Gateway Service", 
            "app.main", 
            8000,
            cwd=gateway_cwd
        )
        if gateway_proc:
            services.append(("Gateway", gateway_proc, 8000))
            time.sleep(3)  # Let it start
        
        # Start Training Service
        training_cwd = str(project_root / "services" / "training")
        training_proc = run_service(
            "Training Service", 
            "app.main", 
            8002,
            cwd=training_cwd
        )
        if training_proc:
            services.append(("Training", training_proc, 8002))
        
        if services:
            print("\n🎉 BACKEND SERVICES STARTED SUCCESSFULLY!")
            print("=" * 60)
            print("📡 API ENDPOINTS:")
            for name, proc, port in services:
                print(f"   {name}: http://localhost:{port}")
                
            print("\n📚 API DOCUMENTATION:")
            print("   Gateway API: http://localhost:8000/docs")
            print("   Inference API: http://localhost:8001/docs")
            print("   Training API: http://localhost:8002/docs")
            
            print("\n🔥 READY FOR REQUESTS!")
            print("=" * 60)
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n⏹️  Stopping services...")
                for name, proc, port in services:
                    proc.terminate()
                print("✅ Backend stopped")
                
        else:
            print("❌ No services started successfully")
            
    except Exception as e:
        print(f"❌ Error starting services: {e}")

def main():
    """Main launcher function"""
    try:
        start_minimal_stack()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
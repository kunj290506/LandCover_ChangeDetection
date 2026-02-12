#!/usr/bin/env python3
"""
Memory-Optimized Training Launcher
Run this script to start training immediately
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "services" / "training" / "app"))
sys.path.insert(0, str(project_root / "services" / "common"))

# Import training components
from trainer import run_training
from landcover_common.settings import Settings


def setup_logging():
    """Setup simple console logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_memory():
    """Check available memory and GPU"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("WARNING: No GPU detected - using CPU (will be slower)")
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")


def check_dataset():
    """Check if dataset exists"""
    dataset_path = project_root / "data" / "LEVIR-CD-patches"
    train_list = project_root / "train_list.txt"
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please ensure LEVIR-CD dataset is available")
        return False
    
    if not train_list.exists():
        print(f"ERROR: Training list not found at {train_list}")
        return False
    
    print(f"SUCCESS: Dataset found at {dataset_path}")
    return True


def start_training():
    """Start optimized training"""
    print("Starting Memory-Optimized Training...")
    
    # Set memory efficient settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Use local paths for training
    dataset_uri = str(project_root / "data" / "LEVIR-CD-patches")
    config_uri = str(project_root / "config" / "train_config.yaml")
    
    try:
        # Start training
        run_id = run_training(
            dataset_uri=dataset_uri,
            config_uri=config_uri,
            register_model=True
        )
        
        print(f"TRAINING COMPLETED! MLflow Run ID: {run_id}")
        print(f"View results at: http://localhost:5000")
        
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("LAND COVER CHANGE DETECTION - TRAINING READY")
    print("=" * 60)
    
    setup_logging()
    check_memory()
    
    if check_dataset():
        print("\nAll systems ready for training!")
        print("Type 'go' to start training or Ctrl+C to exit")
        
        # Wait for user input
        try:
            user_input = input("\nCommand: ").strip().lower()
            if user_input == "go":
                start_training()
            else:
                print("Training cancelled.")
        except KeyboardInterrupt:
            print("\nTraining cancelled.")
    else:
        print("\nERROR: Please fix dataset issues before training.")
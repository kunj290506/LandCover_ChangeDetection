#!/usr/bin/env python3
"""
Training Monitor - Check Progress
"""
import os
import glob
from pathlib import Path

def check_training_progress():
    print("=" * 50)
    print("TRAINING PROGRESS MONITOR")
    print("=" * 50)
    
    # Check if training is running
    checkpoints_dir = Path("./checkpoints")
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*.pth"))
        print(f"✓ Checkpoints found: {len(checkpoints)}")
        
        for ckpt in sorted(checkpoints):
            size_mb = ckpt.stat().st_size / (1024*1024)
            print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
    else:
        print("✗ No checkpoints directory found")
    
    # Check logs
    log_files = glob.glob("*.log")
    if log_files:
        print(f"✓ Log files found: {len(log_files)}")
        for log in log_files:
            print(f"  - {log}")
    
    print("\n" + "=" * 50)
    print("Run this script anytime to check training progress!")
    print("=" * 50)

if __name__ == "__main__":
    check_training_progress()
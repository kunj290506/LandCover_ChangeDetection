#!/usr/bin/env python3
"""
Training Results Analyzer
Extract final training results from saved checkpoint
"""

import torch
import os
from datetime import datetime

def analyze_checkpoint(checkpoint_path):
    """Analyze saved checkpoint for training results"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        # Load checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        
        # Check if it's a full checkpoint or just state dict
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            print(f"📊 Available data: {keys}")
            
            # Extract training metrics
            if 'epoch' in checkpoint:
                print(f"🎯 Training completed at Epoch: {checkpoint['epoch']}")
            
            if 'best_f1' in checkpoint:
                print(f"🏆 Best F1 Score: {checkpoint['best_f1']:.4f}")
            
            if 'val_metrics' in checkpoint:
                metrics = checkpoint['val_metrics']
                print("📈 Validation Metrics:")
                for key, value in metrics.items():
                    print(f"   {key.upper()}: {value:.4f}")
            
            if 'train_loss' in checkpoint:
                print(f"📉 Final Training Loss: {checkpoint['train_loss']:.4f}")
            
            if 'val_loss' in checkpoint:
                print(f"📉 Final Validation Loss: {checkpoint['val_loss']:.4f}")
                
            return checkpoint
        else:
            print("ℹ️  State dict only - no training metrics saved")
            return None
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def get_file_info(filepath):
    """Get file size and modification time"""
    if os.path.exists(filepath):
        stat = os.stat(filepath)
        size_mb = stat.st_size / (1024*1024)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        return size_mb, mod_time
    return None, None

def main():
    print("=" * 60)
    print("🎯 LAND COVER CHANGE DETECTION - TRAINING RESULTS")
    print("=" * 60)
    
    # Check best model
    best_model_path = './checkpoints/best_model.pth'
    size, mod_time = get_file_info(best_model_path)
    
    if size:
        print(f"📁 Best Model: {best_model_path}")
        print(f"📏 Size: {size:.1f} MB")
        print(f"⏰ Last Updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Analyze checkpoint
        checkpoint = analyze_checkpoint(best_model_path)
        
    else:
        print("❌ No best model found")
    
    # Check for other checkpoints
    checkpoint_dir = './checkpoints'
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') or f.endswith('.pt')]
        print(f"\n📂 Total Checkpoints Found: {len(checkpoints)}")
        
        for ckpt in sorted(checkpoints):
            ckpt_path = os.path.join(checkpoint_dir, ckpt)
            size, mod_time = get_file_info(ckpt_path)
            print(f"   {ckpt} ({size:.1f} MB) - {mod_time.strftime('%H:%M:%S')}")
    
    print("\n" + "=" * 60)
    print("🚀 TRAINING ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
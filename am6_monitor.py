#!/usr/bin/env python3
"""
6 AM Training Stopper and Model Saver
Run this script to monitor training and stop at 6 AM
"""

import os
import time
import psutil
import subprocess
from datetime import datetime, time as dt_time
import shutil

def get_current_time_info():
    """Get current time and time until 6 AM"""
    now = datetime.now()
    current_time = now.time()
    
    # Calculate time until 6 AM
    if current_time < dt_time(6, 0):
        # 6 AM is today
        target = datetime.combine(now.date(), dt_time(6, 0))
    else:
        # 6 AM is tomorrow
        from datetime import timedelta
        target = datetime.combine(now.date() + timedelta(days=1), dt_time(6, 0))
    
    time_remaining = target - now
    return now, target, time_remaining

def check_training_process():
    """Check if Python training process is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'train.py' in cmdline:
                    return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def stop_training_process(pid):
    """Safely stop the training process"""
    try:
        proc = psutil.Process(pid)
        proc.terminate()  # Send SIGTERM
        time.sleep(5)  # Wait for graceful shutdown
        
        if proc.is_running():
            proc.kill()  # Force kill if still running
        
        print(f"✅ Training process {pid} stopped successfully")
        return True
    except Exception as e:
        print(f"❌ Error stopping process {pid}: {e}")
        return False

def find_best_checkpoint():
    """Find the best checkpoint file based on validation metrics"""
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        return None
    
    best_file = None
    best_f1 = 0.0
    
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth') and 'best' in filename:
            filepath = os.path.join(checkpoint_dir, filename)
            best_file = filepath
            break
    
    # If no 'best' file, look for latest checkpoint
    if not best_file:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
            best_file = os.path.join(checkpoint_dir, latest)
    
    return best_file

def save_final_model():
    """Save the best model as final model for deployment"""
    best_checkpoint = find_best_checkpoint()
    
    if best_checkpoint:
        final_model_path = './checkpoints/final_best_model_6am.pth'
        deployment_path = './best_model_ready.pth'
        
        # Copy to final location
        shutil.copy2(best_checkpoint, final_model_path)
        shutil.copy2(best_checkpoint, deployment_path)
        
        print(f"💾 Best model saved:")
        print(f"   Source: {best_checkpoint}")
        print(f"   Final: {final_model_path}")
        print(f"   Ready: {deployment_path}")
        return final_model_path
    else:
        print("❌ No checkpoint file found!")
        return None

def main():
    print("=" * 80)
    print("🌙 6 AM TRAINING MONITOR & STOPPER")
    print("=" * 80)
    
    while True:
        now, target, remaining = get_current_time_info()
        
        print(f"\n⏰ Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target Stop Time: {target.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏳ Time Remaining: {str(remaining).split('.')[0]}")
        
        # Check if it's 6 AM or later
        if remaining.total_seconds() <= 0:
            print("\n🌅 6 AM REACHED! Stopping training...")
            
            # Find and stop training process
            pid = check_training_process()
            if pid:
                print(f"🔍 Found training process: PID {pid}")
                stop_training_process(pid)
            else:
                print("ℹ️  No active training process found")
            
            # Save best model
            print("\n💾 Saving best model...")
            final_model = save_final_model()
            
            if final_model:
                print("\n🎉 TRAINING COMPLETE!")
                print("✅ Best model saved and ready for deployment")
            
            break
        
        # Check if training is still running
        pid = check_training_process()
        if pid:
            print(f"✅ Training active (PID: {pid}")
            
            # Show checkpoint status
            checkpoint_dir = './checkpoints'
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                print(f"📊 Checkpoints saved: {len(checkpoints)}")
        else:
            print("⚠️  No training process detected")
        
        # Wait 5 minutes before next check
        print("💤 Waiting 5 minutes...")
        time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitor stopped by user")
        print("🔍 Checking for best model...")
        save_final_model()
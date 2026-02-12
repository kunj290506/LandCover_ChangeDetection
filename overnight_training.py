#!/usr/bin/env python3
"""
Overnight Training Script - Run Until 6 AM
Automatically stops at 6 AM and saves best model
"""

import os
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from datetime import datetime, time as dt_time

# Add paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "src"))

from dataset import ChangeDetectionDataset
from models.snunet import SNUNet
from utils.losses import HybridLoss
from utils.metrics import get_metrics, AverageMeter
from utils.general import set_seed, EarlyStopping

def should_stop_training():
    """Check if current time is 6 AM or later"""
    current_time = datetime.now().time()
    stop_time = dt_time(6, 0)  # 6:00 AM
    
    # If it's past 6 AM, stop training
    if current_time >= stop_time:
        return True
    
    return False

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()
    
    print(f"\n🚀 Starting Epoch {epoch+1} at {datetime.now().strftime('%H:%M:%S')}")
    loop = tqdm(loader, desc=f"Epoch {epoch+1} Training")
    
    for i, batch in enumerate(loop):
        # Check time before each batch
        if should_stop_training():
            print(f"\n⏰ 6 AM reached! Stopping training at batch {i}")
            return losses.avg, True  # Return True to indicate early stop
        
        img1 = batch['image1'].to(device)
        img2 = batch['image2'].to(device)
        label = batch['label'].to(device)
        
        optimizer.zero_grad()
        output = model(img1, img2)
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), img1.size(0))
        loop.set_postfix(loss=losses.avg, time=datetime.now().strftime('%H:%M:%S'))
        
    return losses.avg, False

def validate(model, loader, criterion, device, epoch):
    model.eval()
    losses = AverageMeter()
    metrics = {
        'f1': AverageMeter(),
        'iou': AverageMeter(),
        'kappa': AverageMeter(),
    }
    
    with torch.no_grad():
        loop = tqdm(loader, desc=f"Epoch {epoch+1} Validation")
        for batch in loop:
            # Quick time check during validation too
            if should_stop_training():
                print(f"\n⏰ 6 AM reached during validation! Stopping...")
                break
                
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            label = batch['label'].to(device)
            
            output = model(img1, img2)
            loss = criterion(output, label)
            
            losses.update(loss.item(), img1.size(0))
            
            # Calculate metrics
            batch_metrics = get_metrics(output, label)
            for k in metrics:
                metrics[k].update(batch_metrics[k], img1.size(0))
            
            loop.set_postfix(loss=losses.avg, f1=metrics['f1'].avg)
            
    return losses.avg, {k: v.avg for k, v in metrics.items()}

def main():
    print("=" * 80)
    print("🌙 OVERNIGHT TRAINING - RUN UNTIL 6 AM")
    print("=" * 80)
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Stop Time: 6:00 AM")
    print("=" * 80)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Data paths
    data_root = './data/LEVIR-CD-patches'
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Dataset
    train_dataset = ChangeDetectionDataset(data_root, 'train_list.txt', mode='train', patch_size=256)
    val_dataset = ChangeDetectionDataset(data_root, 'val_list.txt', mode='val', patch_size=256)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)
    
    print(f"📊 Train Dataset: {len(train_dataset)} samples")
    print(f"📊 Train Batches: {len(train_loader)}")
    
    # Model
    model = SNUNet(3, 1).to(device)
    
    # Loss & Optimizer
    criterion = HybridLoss(weight_bce=0.7, weight_dice=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # Best model tracking
    best_f1 = 0.0
    best_model_path = os.path.join(checkpoint_dir, 'best_model_6am.pth')
    
    # Training loop with time check
    epoch = 0
    while True:
        # Check if we should stop before starting new epoch
        if should_stop_training():
            print(f"\n⏰ 6 AM reached! Stopping before epoch {epoch+1}")
            break
        
        print(f"\n{'='*60}")
        print(f"🌙 Overnight Training - Epoch {epoch+1}")
        print(f"⏰ Current Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        # Train epoch
        train_loss, early_stop = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        if early_stop:  # Stop due to time
            break
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val F1: {val_metrics['f1']:.4f}")
        print(f"   Val IoU: {val_metrics['iou']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            
            print(f"💾 New best model saved! F1: {best_f1:.4f}")
        
        # Regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
        }, checkpoint_path)
        
        epoch += 1
        
        # Final time check
        if should_stop_training():
            print(f"\n⏰ 6 AM reached after epoch {epoch}! Stopping...")
            break
    
    # Final summary
    print("\n" + "=" * 80)
    print("🌅 TRAINING COMPLETED AT 6 AM!")
    print("=" * 80)
    print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏆 Best F1 Score: {best_f1:.4f}")
    print(f"💾 Best Model Saved: {best_model_path}")
    print(f"📁 Total Epochs Completed: {epoch}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
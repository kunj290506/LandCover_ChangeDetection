"""
Optimized Training Script - Fixed Configuration
Based on proven settings from RESULTS.md and BENCHMARK_REPORT.md
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

sys.path.append('src')
from dataset import ChangeDetectionDataset
from models.snunet import SNUNet
from utils.losses import HybridLoss
from utils.metrics import get_metrics, AverageMeter
from utils.general import set_seed, EarlyStopping


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    
    loop = tqdm(loader, desc="Training")
    for batch in loop:
        img1 = batch['image1'].to(device)
        img2 = batch['image2'].to(device)
        label = batch['label'].to(device)
        
        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output, label)
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), img1.size(0))
        loop.set_postfix(loss=losses.avg)
        
    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    metrics = {
        'f1': AverageMeter(),
        'iou': AverageMeter(),
        'kappa': AverageMeter(),
    }
    
    with torch.no_grad():
        loop = tqdm(loader, desc="Validation")
        for batch in loop:
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            label = batch['label'].to(device)
            
            output = model(img1, img2)
            loss = criterion(output, label)
            
            losses.update(loss.item(), img1.size(0))
            
            batch_metrics = get_metrics(output, label)
            for k in metrics:
                metrics[k].update(batch_metrics[k], img1.size(0))
            
            loop.set_postfix(loss=losses.avg, f1=metrics['f1'].avg)
            
    return losses.avg, {k: v.avg for k, v in metrics.items()}


def main(args):
    print("="*70)
    print(" LAND COVER CHANGE DETECTION - OPTIMIZED TRAINING")
    print("="*70)
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ChangeDetectionDataset(
        args.data_root, 
        args.train_list, 
        mode='train', 
        patch_size=args.patch_size
    )
    val_dataset = ChangeDetectionDataset(
        args.data_root, 
        args.val_list, 
        mode='val', 
        patch_size=args.patch_size
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    print(f"\nInitializing SNUNet (base_channel={args.base_channel})...")
    model = SNUNet(
        in_channels=3, 
        num_classes=1, 
        base_channel=args.base_channel,
        use_attention=args.use_attention
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss & Optimizer
    criterion = HybridLoss(weight_bce=0.7, weight_dice=0.3)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience, 
        verbose=True, 
        path=os.path.join(args.checkpoint_dir, 'best_model.pth')
    )
    
    # Training loop
    print("\n" + "="*70)
    print(" TRAINING START")
    print("="*70)
    
    best_f1 = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val F1:     {val_metrics['f1']:.4f}")
        print(f"  Val IoU:    {val_metrics['iou']:.4f}")
        print(f"  Val Kappa:  {val_metrics['kappa']:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }, os.path.join(args.checkpoint_dir, f'best_model_f1_{best_f1:.4f}.pth'))
            print(f"  >> Best model saved (F1={best_f1:.4f})")
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered")
            break
    
    print("\n" + "="*70)
    print(f" TRAINING COMPLETE - Best F1: {best_f1:.4f}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--train_list', type=str, default='train_list.txt')
    parser.add_argument('--val_list', type=str, default='val_list.txt')
    parser.add_argument('--patch_size', type=int, default=256)
    
    # Model
    parser.add_argument('--base_channel', type=int, default=32, help='Base channel (16/32/64)')
    parser.add_argument('--use_attention', action='store_true', default=True)
    parser.add_argument('--no_attention', dest='use_attention', action='store_false')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # System
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_optimized')
    parser.add_argument('--patience', type=int, default=15)
    
    args = parser.parse_args()
    main(args)

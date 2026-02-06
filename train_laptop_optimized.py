"""
LAPTOP-OPTIMIZED TRAINING SCRIPT FOR LAND COVER CHANGE DETECTION
=================================================================
Optimized for: GTX 1650 (4GB VRAM) + Intel i5-10300H
Training Time: ~5-6 hours
Goal: Best F1 score without overheating

Key optimizations:
1. Small batch size (4) with gradient accumulation (effective batch = 16)
2. Mixed precision training (FP16) - faster + less memory
3. Periodic cooling pauses to prevent overheating
4. OneCycle LR scheduler for faster convergence
5. Enhanced augmentations for better generalization
6. Label smoothing for better calibration
7. Test-time augmentation (TTA) for validation
8. Cosine annealing with warm restarts
"""

import os
import sys
import time
import csv
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
try:
    from torch.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
from tqdm import tqdm
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from dataset import ChangeDetectionDataset
from models.snunet import SNUNet
from utils.losses import HybridLoss
from utils.metrics import get_metrics, AverageMeter
from utils.general import set_seed


class EnhancedDataset(ChangeDetectionDataset):
    """Enhanced dataset with stronger augmentations for better results"""
    
    def __init__(self, root_dir, list_path, mode='train', patch_size=256, use_advanced_aug=True):
        super().__init__(root_dir, list_path, mode, patch_size)
        self.use_advanced_aug = use_advanced_aug
        
    def _transform(self, img1, img2, label):
        """Enhanced augmentation pipeline"""
        # Always resize to patch_size first
        img1 = TF.resize(img1, [self.patch_size, self.patch_size])
        img2 = TF.resize(img2, [self.patch_size, self.patch_size])
        label = TF.resize(label, [self.patch_size, self.patch_size], interpolation=Image.NEAREST)
        
        # Random Horizontal Flip
        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            label = TF.hflip(label)

        # Random Vertical Flip
        if random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            label = TF.vflip(label)
            
        # Random Rotation (90 degree increments)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            label = TF.rotate(label, angle)
        
        if self.use_advanced_aug:
            # Color jitter (only on images, not label) - helps with different lighting conditions
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                saturation = random.uniform(0.8, 1.2)
                img1 = TF.adjust_brightness(img1, brightness)
                img1 = TF.adjust_contrast(img1, contrast)
                img1 = TF.adjust_saturation(img1, saturation)
                img2 = TF.adjust_brightness(img2, brightness)
                img2 = TF.adjust_contrast(img2, contrast)
                img2 = TF.adjust_saturation(img2, saturation)
            
            # Gaussian blur (simulates different sensor qualities)
            if random.random() > 0.7:
                kernel_size = random.choice([3, 5])
                img1 = TF.gaussian_blur(img1, kernel_size)
                img2 = TF.gaussian_blur(img2, kernel_size)

        return img1, img2, label
    
    def __getitem__(self, idx):
        img1_path = os.path.join(self.root_dir, self.files[idx][0])
        img2_path = os.path.join(self.root_dir, self.files[idx][1])
        label_path = os.path.join(self.root_dir, self.files[idx][2])

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img1, img2, label = self._transform(img1, img2, label)
        else:
            # Resize for validation
            img1 = TF.resize(img1, [self.patch_size, self.patch_size])
            img2 = TF.resize(img2, [self.patch_size, self.patch_size])
            label = TF.resize(label, [self.patch_size, self.patch_size], interpolation=Image.NEAREST)

        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        label = TF.to_tensor(label)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img1 = TF.normalize(img1, mean, std)
        img2 = TF.normalize(img2, mean, std)

        return {'image1': img1, 'image2': img2, 'label': label, 'name': self.files[idx][0]}


class LabelSmoothingBCE(nn.Module):
    """BCE with label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(inputs, targets)


class ImprovedHybridLoss(nn.Module):
    """Improved loss with focal component for hard examples"""
    def __init__(self, weight_bce=0.5, weight_dice=0.3, weight_focal=0.2, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        # Label smoothing
        smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # BCE Loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, smooth_targets)
        
        # Dice Loss
        probs = torch.sigmoid(inputs)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + 1) / (probs_flat.sum() + targets_flat.sum() + 1)
        dice_loss = 1 - dice
        
        # Focal Loss (focuses on hard examples)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = (focal_weight * nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')).mean()
        
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss + self.weight_focal * focal_loss


def cooling_pause(duration=30):
    """Pause training to let laptop cool down"""
    print(f"\n🌡️  Cooling pause ({duration}s) - preventing overheating...")
    for i in range(duration, 0, -10):
        print(f"   Resuming in {i}s...", end='\r')
        time.sleep(10)
    print("   Cooling complete! Resuming training...     ")


def train_epoch(model, loader, criterion, optimizer, scaler, device, accumulation_steps=4, use_amp=True):
    """Training with gradient accumulation and optional mixed precision"""
    model.train()
    losses = AverageMeter()
    
    optimizer.zero_grad()
    loop = tqdm(loader, desc="Training", leave=False)
    
    for i, batch in enumerate(loop):
        img1 = batch['image1'].to(device)
        img2 = batch['image2'].to(device)
        label = batch['label'].to(device)
        
        if use_amp and device.type == 'cuda':
            # Mixed precision forward pass (GPU only)
            with autocast(device_type='cuda'):
                output = model(img1, img2)
                loss = criterion(output, label) / accumulation_steps
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard forward pass (CPU)
            output = model(img1, img2)
            loss = criterion(output, label) / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        losses.update(loss.item() * accumulation_steps, img1.size(0))
        loop.set_postfix(loss=f"{losses.avg:.4f}")
    
    return losses.avg


def validate_with_tta(model, loader, criterion, device, use_tta=True, use_amp=True):
    """Validation with optional Test-Time Augmentation"""
    model.eval()
    losses = AverageMeter()
    metrics = {
        'f1': AverageMeter(),
        'iou': AverageMeter(),
        'kappa': AverageMeter(),
        'precision': AverageMeter(),
        'recall': AverageMeter(),
    }
    
    with torch.no_grad():
        loop = tqdm(loader, desc="Validation", leave=False)
        for batch in loop:
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            label = batch['label'].to(device)
            
            if use_tta:
                # Test-Time Augmentation: average predictions from multiple augmented versions
                outputs = []
                
                # Original
                outputs.append(torch.sigmoid(model(img1, img2)))
                
                # Horizontal flip
                outputs.append(torch.sigmoid(model(
                    torch.flip(img1, dims=[3]), 
                    torch.flip(img2, dims=[3])
                )).flip(dims=[3]))
                
                # Vertical flip
                outputs.append(torch.sigmoid(model(
                    torch.flip(img1, dims=[2]), 
                    torch.flip(img2, dims=[2])
                )).flip(dims=[2]))
                
                # Average all augmented outputs
                output_avg = torch.stack(outputs).mean(dim=0)
                # Convert back to logits for loss calculation
                output = torch.log(output_avg / (1 - output_avg + 1e-8))
            else:
                output = model(img1, img2)
            
            loss = criterion(output, label)
            
            losses.update(loss.item(), img1.size(0))
            
            batch_metrics = get_metrics(output, label)
            for k in metrics:
                if k in batch_metrics:
                    metrics[k].update(batch_metrics[k], img1.size(0))
            
            loop.set_postfix(loss=f"{losses.avg:.4f}", f1=f"{metrics['f1'].avg:.4f}")
    
    return losses.avg, {k: v.avg for k, v in metrics.items()}


def save_training_log(log_path, epoch, train_loss, val_loss, metrics, lr):
    """Save training metrics to CSV"""
    file_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'f1', 'iou', 'kappa', 'precision', 'recall', 'lr'])
        writer.writerow([
            epoch, 
            f"{train_loss:.6f}", 
            f"{val_loss:.6f}",
            f"{metrics.get('f1', 0):.6f}",
            f"{metrics.get('iou', 0):.6f}",
            f"{metrics.get('kappa', 0):.6f}",
            f"{metrics.get('precision', 0):.6f}",
            f"{metrics.get('recall', 0):.6f}",
            f"{lr:.8f}"
        ])


def main(args):
    print("=" * 70)
    print("  🛰️  LAND COVER CHANGE DETECTION - LAPTOP OPTIMIZED TRAINING")
    print("=" * 70)
    print(f"  GPU: GTX 1650 (4GB) | CPU: Intel i5-10300H")
    print(f"  Target: Best F1 in ~{args.target_hours} hours without overheating")
    print("=" * 70)
    
    # Set seeds for reproducibility
    set_seed(args.seed)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Faster training
        torch.backends.cuda.matmul.allow_tf32 = True  # Tensor cores if available
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️  Running on CPU - training will be slower")
        print("   TIP: Make sure NVIDIA drivers are installed and try running")
        print("   this script from a regular command prompt (not VS Code terminal)")
        print("   You can also try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Calculate training schedule
    print(f"\n📊 Dataset Configuration:")
    
    # Load datasets
    train_dataset = EnhancedDataset(
        args.data_root,
        args.train_list,
        mode='train',
        patch_size=args.patch_size,
        use_advanced_aug=True
    )
    
    val_dataset = EnhancedDataset(
        args.data_root,
        args.val_list,
        mode='val',
        patch_size=args.patch_size,
        use_advanced_aug=False
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Optimized data loaders for laptop (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # Small batch for low VRAM
        shuffle=True,
        num_workers=0,  # Use 0 workers on Windows for stability
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    effective_batch = args.batch_size * args.accumulation_steps
    print(f"   Batch size: {args.batch_size} (effective: {effective_batch} with accumulation)")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Estimate training time
    est_epoch_time = len(train_loader) * 0.5 + len(val_loader) * 0.3  # seconds
    total_est_time = (est_epoch_time * args.epochs) / 3600
    print(f"   Estimated total time: ~{total_est_time:.1f} hours")
    
    # Initialize model
    print(f"\n🏗️  Model Configuration:")
    model = SNUNet(
        in_channels=3,
        num_classes=1,
        base_channel=args.base_channel,
        use_attention=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Architecture: SNUNet with CBAM attention")
    print(f"   Base channels: {args.base_channel}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Load pretrained if specified
    if args.resume:
        print(f"\n📂 Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   Loaded model from epoch {checkpoint.get('epoch', '?')}")
            print(f"   Previous best F1: {checkpoint.get('best_f1', '?')}")
        else:
            model.load_state_dict(checkpoint)
    
    # Loss function with improvements
    print(f"\n📉 Training Configuration:")
    criterion = ImprovedHybridLoss(
        weight_bce=0.5,
        weight_dice=0.3,
        weight_focal=0.2,
        gamma=2.0,
        smoothing=0.05
    )
    print(f"   Loss: Hybrid (BCE:0.5 + Dice:0.3 + Focal:0.2) with label smoothing")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    print(f"   Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    
    # Learning rate scheduler - OneCycleLR for faster convergence
    steps_per_epoch = len(train_loader) // args.accumulation_steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,  # Peak LR
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # Warmup for 10% of training
        anneal_strategy='cos',
        div_factor=10,  # Start LR = max_lr / 10
        final_div_factor=100  # End LR = max_lr / 1000
    )
    print(f"   Scheduler: OneCycleLR (max_lr={args.lr * 10})")
    
    # Mixed precision scaler (only for GPU)
    use_amp = device.type == 'cuda'
    if use_amp:
        scaler = GradScaler('cuda')
        print(f"   Mixed precision: Enabled (FP16)")
    else:
        scaler = None
        print(f"   Mixed precision: Disabled (CPU mode)")
    print(f"   Gradient accumulation: {args.accumulation_steps} steps")
    print(f"   Gradient clipping: max_norm=1.0")
    
    # Training tracking
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    log_path = os.path.join(args.checkpoint_dir, 'training_log.csv')
    
    print("\n" + "=" * 70)
    print("  🚀 TRAINING STARTED")
    print("=" * 70)
    
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            
            print(f"\n{'─' * 70}")
            print(f"  Epoch {epoch}/{args.epochs}")
            print(f"{'─' * 70}")
            
            # Training
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                accumulation_steps=args.accumulation_steps, use_amp=use_amp
            )
            
            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Validation with TTA
            val_loss, val_metrics = validate_with_tta(
                model, val_loader, criterion, device,
                use_tta=args.use_tta, use_amp=use_amp
            )
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Print results
            print(f"\n  📊 Results:")
            print(f"     Train Loss: {train_loss:.4f}")
            print(f"     Val Loss:   {val_loss:.4f}")
            print(f"     F1 Score:   {val_metrics['f1']:.4f}")
            print(f"     IoU:        {val_metrics['iou']:.4f}")
            print(f"     Kappa:      {val_metrics['kappa']:.4f}")
            print(f"     Precision:  {val_metrics.get('precision', 0):.4f}")
            print(f"     Recall:     {val_metrics.get('recall', 0):.4f}")
            print(f"     LR:         {current_lr:.2e}")
            print(f"     Time:       {epoch_time:.0f}s (Total: {total_time/3600:.2f}h)")
            
            # Save training log
            save_training_log(log_path, epoch, train_loss, val_loss, val_metrics, current_lr)
            
            # Check for improvement
            if val_metrics['f1'] > best_f1:
                improvement = val_metrics['f1'] - best_f1
                best_f1 = val_metrics['f1']
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                save_path = os.path.join(args.checkpoint_dir, f'best_model_f1_{best_f1:.4f}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1': best_f1,
                    'val_metrics': val_metrics,
                }, save_path)
                
                # Also save as best_model.pth for easy loading
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
                
                print(f"\n  🏆 NEW BEST! F1: {best_f1:.4f} (+{improvement:.4f})")
                print(f"     Saved to: {save_path}")
            else:
                patience_counter += 1
                print(f"\n  ⏳ No improvement for {patience_counter}/{args.patience} epochs")
                print(f"     Best F1: {best_f1:.4f} (epoch {best_epoch})")
            
            # Cooling pause every N epochs to prevent overheating
            if epoch % args.cooling_interval == 0 and epoch < args.epochs:
                cooling_pause(args.cooling_duration)
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                break
            
            # Time limit check
            if total_time / 3600 > args.target_hours:
                print(f"\n⏰ Time limit reached ({args.target_hours}h)")
                break
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("  🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Total time:    {total_time/3600:.2f} hours")
    print(f"  Best F1:       {best_f1:.4f}")
    print(f"  Best epoch:    {best_epoch}")
    print(f"  Checkpoints:   {args.checkpoint_dir}")
    print(f"  Training log:  {log_path}")
    print("=" * 70)
    
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Laptop-Optimized Change Detection Training')
    
    # Data
    parser.add_argument('--data_root', type=str, default='./data/LEVIR-CD-patches',
                        help='Path to dataset root')
    parser.add_argument('--train_list', type=str, default='train_list.txt',
                        help='Path to training list file')
    parser.add_argument('--val_list', type=str, default='val_list.txt',
                        help='Path to validation list file')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Input patch size')
    
    # Model
    parser.add_argument('--base_channel', type=int, default=32,
                        help='Base channel count for SNUNet')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Training - Optimized for GTX 1650 (4GB) + preventing overheating
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (small for low VRAM)')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * accumulation)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    
    # Early stopping & patience
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Laptop heat management
    parser.add_argument('--cooling_interval', type=int, default=10,
                        help='Pause for cooling every N epochs')
    parser.add_argument('--cooling_duration', type=int, default=30,
                        help='Cooling pause duration in seconds')
    parser.add_argument('--target_hours', type=float, default=5.5,
                        help='Target training time in hours')
    
    # Validation
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Use test-time augmentation for validation')
    
    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_laptop',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    main(args)

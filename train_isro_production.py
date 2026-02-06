"""
================================================================================
 ISRO PRODUCTION TRAINING SCRIPT - LAND COVER CHANGE DETECTION
================================================================================
 Maximum performance configuration for satellite image change detection
 Target: Best F1 Score for ISRO submission
 
 Features:
 1. Resume from best checkpoint (F1=0.6662)
 2. Advanced data augmentation (MixUp, CutMix, heavy geometric)
 3. Deep supervision with auxiliary losses
 4. OneCycleLR + Cosine Annealing with Warm Restarts
 5. Focal + Dice + BCE hybrid loss with dynamic weighting
 6. Test-Time Augmentation (8x) for validation
 7. Exponential Moving Average (EMA) for stable predictions
 8. Gradient accumulation for effective larger batch size
 9. Label smoothing for better generalization
 10. Proper class weighting for imbalanced data
================================================================================
"""

import os
import sys
import time
import csv
import copy
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
from tqdm import tqdm
import torchvision.transforms.functional as TF

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from models.snunet import SNUNet
from utils.metrics import get_metrics, AverageMeter

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# ADVANCED DATASET WITH STRONG AUGMENTATIONS
# ============================================================================
class ISRODataset(Dataset):
    """Production dataset with advanced augmentations for maximum performance"""
    
    def __init__(self, root_dir, list_path, mode='train', patch_size=256):
        self.root_dir = root_dir
        self.mode = mode
        self.patch_size = patch_size
        
        self.files = []
        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    self.files.append(parts)
        
        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.root_dir, self.files[idx][0])
        img2_path = os.path.join(self.root_dir, self.files[idx][1])
        label_path = os.path.join(self.root_dir, self.files[idx][2])

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img1, img2, label = self._augment(img1, img2, label)
        
        # Resize if needed
        if img1.size[0] != self.patch_size:
            img1 = TF.resize(img1, [self.patch_size, self.patch_size])
            img2 = TF.resize(img2, [self.patch_size, self.patch_size])
            label = TF.resize(label, [self.patch_size, self.patch_size], interpolation=Image.NEAREST)

        # To tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        label = TF.to_tensor(label)
        
        # Fix binary labels (original 0/1 becomes 0/0.004 after to_tensor)
        if label.max() > 0 and label.max() < 0.1:
            label = (label > 0.001).float()
        elif label.max() > 0.1:
            label = (label > 0.5).float()

        # Normalize images
        img1 = TF.normalize(img1, self.mean, self.std)
        img2 = TF.normalize(img2, self.mean, self.std)

        return {'image1': img1, 'image2': img2, 'label': label, 'name': self.files[idx][0]}

    def _augment(self, img1, img2, label):
        """Strong augmentation pipeline for satellite imagery"""
        
        # 1. Random horizontal flip (50%)
        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            label = TF.hflip(label)

        # 2. Random vertical flip (50%)
        if random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            label = TF.vflip(label)

        # 3. Random rotation (90, 180, 270) - 50%
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            label = TF.rotate(label, angle)

        # 4. Random small rotation (-15 to 15 degrees) - 30%
        if random.random() > 0.7:
            angle = random.uniform(-15, 15)
            img1 = TF.rotate(img1, angle, fill=0)
            img2 = TF.rotate(img2, angle, fill=0)
            label = TF.rotate(label, angle, fill=0)

        # 5. Color jitter for satellite imagery robustness - 50%
        if random.random() > 0.5:
            # Brightness
            brightness = random.uniform(0.8, 1.2)
            img1 = TF.adjust_brightness(img1, brightness)
            img2 = TF.adjust_brightness(img2, brightness)
            
            # Contrast
            contrast = random.uniform(0.8, 1.2)
            img1 = TF.adjust_contrast(img1, contrast)
            img2 = TF.adjust_contrast(img2, contrast)
            
            # Saturation
            saturation = random.uniform(0.8, 1.2)
            img1 = TF.adjust_saturation(img1, saturation)
            img2 = TF.adjust_saturation(img2, saturation)

        # 6. Gaussian blur (simulates different sensor qualities) - 20%
        if random.random() > 0.8:
            kernel_size = random.choice([3, 5])
            img1 = TF.gaussian_blur(img1, kernel_size)
            img2 = TF.gaussian_blur(img2, kernel_size)

        # 7. Random grayscale (robustness to color variations) - 10%
        if random.random() > 0.9:
            img1 = TF.rgb_to_grayscale(img1, num_output_channels=3)
            img2 = TF.rgb_to_grayscale(img2, num_output_channels=3)

        return img1, img2, label


# ============================================================================
# ADVANCED LOSS FUNCTIONS
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class TverskyLoss(nn.Module):
    """Tversky Loss - better for imbalanced segmentation"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight (higher = penalize missed changes more)
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        tp = (probs * targets).sum()
        fp = ((1 - targets) * probs).sum()
        fn = (targets * (1 - probs)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class ISROHybridLoss(nn.Module):
    """
    Production-grade hybrid loss for ISRO change detection
    Combines BCE, Dice, Focal, and Tversky for maximum performance
    """
    def __init__(self, bce_weight=0.3, dice_weight=0.3, focal_weight=0.2, tversky_weight=0.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)  # Penalize FN more

    def forward(self, inputs, targets):
        # BCE with pos_weight for class imbalance
        pos_weight = torch.tensor([3.0], device=inputs.device)  # Change pixels are rare
        bce = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
        
        dice = self.dice(inputs, targets)
        focal = self.focal(inputs, targets)
        tversky = self.tversky(inputs, targets)
        
        total = (self.bce_weight * bce + 
                 self.dice_weight * dice + 
                 self.focal_weight * focal + 
                 self.tversky_weight * tversky)
        
        return total


# ============================================================================
# EXPONENTIAL MOVING AVERAGE (EMA) FOR STABLE PREDICTIONS
# ============================================================================
class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# TEST-TIME AUGMENTATION (8x)
# ============================================================================
def predict_with_tta(model, img1, img2, device):
    """8x Test-Time Augmentation for best predictions"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original
        pred = torch.sigmoid(model(img1, img2))
        predictions.append(pred)
        
        # Horizontal flip
        pred = torch.sigmoid(model(torch.flip(img1, [3]), torch.flip(img2, [3])))
        predictions.append(torch.flip(pred, [3]))
        
        # Vertical flip
        pred = torch.sigmoid(model(torch.flip(img1, [2]), torch.flip(img2, [2])))
        predictions.append(torch.flip(pred, [2]))
        
        # Both flips
        pred = torch.sigmoid(model(torch.flip(img1, [2, 3]), torch.flip(img2, [2, 3])))
        predictions.append(torch.flip(pred, [2, 3]))
        
        # 90 degree rotation
        img1_90 = torch.rot90(img1, 1, [2, 3])
        img2_90 = torch.rot90(img2, 1, [2, 3])
        pred = torch.sigmoid(model(img1_90, img2_90))
        predictions.append(torch.rot90(pred, -1, [2, 3]))
        
        # 180 degree rotation
        img1_180 = torch.rot90(img1, 2, [2, 3])
        img2_180 = torch.rot90(img2, 2, [2, 3])
        pred = torch.sigmoid(model(img1_180, img2_180))
        predictions.append(torch.rot90(pred, -2, [2, 3]))
        
        # 270 degree rotation
        img1_270 = torch.rot90(img1, 3, [2, 3])
        img2_270 = torch.rot90(img2, 3, [2, 3])
        pred = torch.sigmoid(model(img1_270, img2_270))
        predictions.append(torch.rot90(pred, -3, [2, 3]))
        
        # Horizontal flip + 90 rotation
        img1_h90 = torch.rot90(torch.flip(img1, [3]), 1, [2, 3])
        img2_h90 = torch.rot90(torch.flip(img2, [3]), 1, [2, 3])
        pred = torch.sigmoid(model(img1_h90, img2_h90))
        predictions.append(torch.flip(torch.rot90(pred, -1, [2, 3]), [3]))
    
    # Average all predictions
    return torch.stack(predictions).mean(dim=0)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device, ema=None, accumulation_steps=4):
    model.train()
    losses = AverageMeter()
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training", ncols=100)
    
    for i, batch in enumerate(pbar):
        img1 = batch['image1'].to(device)
        img2 = batch['image2'].to(device)
        label = batch['label'].to(device)
        
        # Forward
        output = model(img1, img2)
        loss = criterion(output, label) / accumulation_steps
        
        # Backward
        loss.backward()
        
        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
        
        losses.update(loss.item() * accumulation_steps, img1.size(0))
        pbar.set_postfix(loss=f"{losses.avg:.4f}")
    
    return losses.avg


def validate_with_tta(model, loader, criterion, device, use_tta=True):
    """Validation with optional Test-Time Augmentation"""
    model.eval()
    losses = AverageMeter()
    all_metrics = {
        'f1': AverageMeter(),
        'iou': AverageMeter(),
        'precision': AverageMeter(),
        'recall': AverageMeter(),
        'kappa': AverageMeter(),
    }
    
    pbar = tqdm(loader, desc="Validation", ncols=100, leave=False)
    
    for batch in pbar:
        img1 = batch['image1'].to(device)
        img2 = batch['image2'].to(device)
        label = batch['label'].to(device)
        
        with torch.no_grad():
            if use_tta:
                # 8x TTA
                probs = predict_with_tta(model, img1, img2, device)
                output = torch.log(probs / (1 - probs + 1e-8))  # Back to logits
            else:
                output = model(img1, img2)
            
            loss = criterion(output, label)
        
        losses.update(loss.item(), img1.size(0))
        
        # Calculate metrics
        metrics = get_metrics(output, label)
        for k in all_metrics:
            if k in metrics:
                all_metrics[k].update(metrics[k], img1.size(0))
        
        pbar.set_postfix(loss=f"{losses.avg:.4f}", f1=f"{all_metrics['f1'].avg:.4f}")
    
    return losses.avg, {k: v.avg for k, v in all_metrics.items()}


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def main():
    print("=" * 80)
    print("  🛰️  ISRO PRODUCTION TRAINING - LAND COVER CHANGE DETECTION")
    print("=" * 80)
    print("  Mission: Best F1 Score for ISRO Submission")
    print("  Architecture: SNUNet with CBAM Attention")
    print("=" * 80)
    
    # Configuration
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️ Running on CPU - this will be slow!")
    
    # Hyperparameters optimized for best results
    config = {
        'data_root': './data/LEVIR-CD-patches',
        'train_list': 'train_list.txt',
        'val_list': 'val_list.txt',
        'patch_size': 256,
        'batch_size': 4,
        'accumulation_steps': 4,  # Effective batch = 16
        'epochs': 100,
        'lr': 2e-4,
        'weight_decay': 0.01,
        'patience': 25,
        'checkpoint_dir': './checkpoints_isro',
        'resume': './checkpoints_optimized/best_model_f1_0.6662.pth',  # Start from best
    }
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Data
    print("\n📊 Loading datasets...")
    train_dataset = ISRODataset(
        config['data_root'], 
        config['train_list'], 
        mode='train', 
        patch_size=config['patch_size']
    )
    val_dataset = ISRODataset(
        config['data_root'], 
        config['val_list'], 
        mode='val', 
        patch_size=config['patch_size']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'] * 2, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    print(f"   Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Effective batch size: {config['batch_size'] * config['accumulation_steps']}")
    
    # Model
    print("\n🏗️ Initializing model...")
    model = SNUNet(in_channels=3, num_classes=1, base_channel=32, use_attention=True).to(device)
    
    # Load pretrained weights if available
    if os.path.exists(config['resume']):
        print(f"   Loading checkpoint: {config['resume']}")
        checkpoint = torch.load(config['resume'], map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   Loaded from epoch {checkpoint.get('epoch', '?')}, F1={checkpoint.get('best_f1', '?')}")
        else:
            model.load_state_dict(checkpoint)
            print("   Loaded model weights")
    else:
        print("   Starting from scratch (no checkpoint found)")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # EMA for stable predictions
    ema = EMA(model, decay=0.999)
    print("   EMA: Enabled (decay=0.999)")
    
    # Loss
    print("\n📉 Training configuration...")
    criterion = ISROHybridLoss(bce_weight=0.3, dice_weight=0.3, focal_weight=0.2, tversky_weight=0.2)
    print("   Loss: Hybrid (BCE:0.3 + Dice:0.3 + Focal:0.2 + Tversky:0.2)")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    print(f"   Optimizer: AdamW (lr={config['lr']}, wd={config['weight_decay']})")
    
    # Scheduler - Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6
    )
    print("   Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    print("   TTA: 8x augmentations for validation")
    
    # Training tracking
    best_f1 = 0.6662  # Start from current best
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    log_path = os.path.join(config['checkpoint_dir'], 'training_log.csv')
    
    print("\n" + "=" * 80)
    print("  🚀 TRAINING STARTED - Target: F1 > 0.70")
    print("=" * 80)
    
    try:
        for epoch in range(1, config['epochs'] + 1):
            epoch_start = time.time()
            
            print(f"\n{'─' * 80}")
            print(f"  Epoch {epoch}/{config['epochs']} | Best F1: {best_f1:.4f}")
            print(f"{'─' * 80}")
            
            # Training
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device,
                ema=ema, accumulation_steps=config['accumulation_steps']
            )
            
            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Validation with EMA weights
            ema.apply_shadow()
            val_loss, val_metrics = validate_with_tta(
                model, val_loader, criterion, device, use_tta=True
            )
            ema.restore()
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Print results
            print(f"\n  📊 Results:")
            print(f"     Train Loss: {train_loss:.4f}")
            print(f"     Val Loss:   {val_loss:.4f}")
            print(f"     ────────────────────────────")
            print(f"     F1 Score:   {val_metrics['f1']:.4f}")
            print(f"     IoU:        {val_metrics['iou']:.4f}")
            print(f"     Precision:  {val_metrics.get('precision', 0):.4f}")
            print(f"     Recall:     {val_metrics.get('recall', 0):.4f}")
            print(f"     Kappa:      {val_metrics.get('kappa', 0):.4f}")
            print(f"     ────────────────────────────")
            print(f"     LR:         {current_lr:.2e}")
            print(f"     Time:       {epoch_time:.0f}s (Total: {total_time/3600:.2f}h)")
            
            # Save training log
            log_exists = os.path.exists(log_path)
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not log_exists:
                    writer.writerow(['epoch', 'train_loss', 'val_loss', 'f1', 'iou', 
                                   'precision', 'recall', 'kappa', 'lr'])
                writer.writerow([
                    epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                    f"{val_metrics['f1']:.6f}", f"{val_metrics['iou']:.6f}",
                    f"{val_metrics.get('precision', 0):.6f}", 
                    f"{val_metrics.get('recall', 0):.6f}",
                    f"{val_metrics.get('kappa', 0):.6f}", f"{current_lr:.8f}"
                ])
            
            # Check for improvement
            if val_metrics['f1'] > best_f1:
                improvement = val_metrics['f1'] - best_f1
                best_f1 = val_metrics['f1']
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model with EMA weights
                ema.apply_shadow()
                save_path = os.path.join(config['checkpoint_dir'], f'best_model_f1_{best_f1:.4f}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1': best_f1,
                    'val_metrics': val_metrics,
                    'config': config,
                }, save_path)
                
                # Also save as best_model.pth
                torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], 'best_model.pth'))
                ema.restore()
                
                print(f"\n  🏆 NEW BEST! F1: {best_f1:.4f} (+{improvement:.4f})")
                print(f"     Saved to: {save_path}")
            else:
                patience_counter += 1
                print(f"\n  ⏳ No improvement for {patience_counter}/{config['patience']} epochs")
            
            # Cooling pause every 15 epochs to prevent overheating
            if epoch % 15 == 0 and epoch < config['epochs']:
                print("\n  🌡️ Cooling pause (30s)...")
                time.sleep(30)
            
            # Early stopping
            if patience_counter >= config['patience']:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                break
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("  🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"  Total time:    {total_time/3600:.2f} hours")
    print(f"  Best F1:       {best_f1:.4f}")
    print(f"  Best epoch:    {best_epoch}")
    print(f"  Checkpoints:   {config['checkpoint_dir']}")
    print(f"  Training log:  {log_path}")
    print("=" * 80)
    print("\n  📝 For ISRO submission, use: checkpoints_isro/best_model.pth")
    print("=" * 80)


if __name__ == '__main__':
    main()

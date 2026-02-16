import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import logging
from datetime import datetime

from dataset import ChangeDetectionDataset
from models.snunet import SNUNet
from utils.losses import HybridLoss
from utils.metrics import get_metrics, AverageMeter
from utils.general import set_seed, EarlyStopping

# Mixup implementation
def mixup_data(x1, x2, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index, :]
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def setup_logging(checkpoint_dir):
    log_file = os.path.join(checkpoint_dir, f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def train_epoch(model, loader, criterion, optimizer, device, epoch, args):
    model.train()
    losses = AverageMeter()
    
    loop = tqdm(loader, desc=f"Training Epoch {epoch+1}")
    for i, batch in enumerate(loop):
        img1 = batch['image1'].to(device)
        img2 = batch['image2'].to(device)
        label = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Apply Mixup
        if args.mixup and random.random() < 0.5:
            img1, img2, label_a, label_b, lam = mixup_data(img1, img2, label, alpha=0.4, use_cuda=True)
            output = model(img1, img2)
            # Loss needs to handle mixup
            # HybridLoss combines Focal and Dice. 
            # We apply mixup to the loss calculation:
            loss = lam * criterion(output, label_a) + (1 - lam) * criterion(output, label_b)
        else:
            output = model(img1, img2)
            loss = criterion(output, label)
        
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
        'precision': AverageMeter(),
        'recall': AverageMeter()
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
            
            # Calculate metrics
            batch_metrics = get_metrics(output, label)
            for k in metrics:
                if k in batch_metrics:
                    metrics[k].update(batch_metrics[k], img1.size(0))
            
            loop.set_postfix(loss=losses.avg, f1=metrics['f1'].avg)
            
    return losses.avg, {k: v.avg for k, v in metrics.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_advanced', help='path to save checkpoints')
    parser.add_argument('--weights', type=str, default=None, help='pretrained weights')
    parser.add_argument('--mixup', action='store_true', help='enable mixup augmentation')
    
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logging(args.checkpoint_dir)
    logger.info(f"Using device: {device}")
    logger.info(f"Arguments: {args}")
    
    # Data
    train_dataset = ChangeDetectionDataset(args.data_root, 'train_list.txt', mode='train', patch_size=256)
    val_dataset = ChangeDetectionDataset(args.data_root, 'val_list.txt', mode='val', patch_size=256)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4 if os.name != 'nt' else 0, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4 if os.name != 'nt' else 0, 
        pin_memory=True
    )
    
    # Model
    model = SNUNet(3, 1, base_channel=32, use_attention=True).to(device)
    
    if args.weights and os.path.exists(args.weights):
        logger.info(f"Loading pretrained weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
    # Loss & Optimizer
    # Using the new HybridLoss (Focal + Dice)
    criterion = HybridLoss(weight_bce=0.5, weight_dice=0.5)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    # Advanced Scheduler: Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=1e-6
    )
    
    # Early Stopping
    early_stopping = EarlyStopping(
        patience=20, 
        verbose=True, 
        path=os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    )
    
    best_f1 = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        logger.info(f"Val Metrics: F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1,
            }, os.path.join(args.checkpoint_dir, 'best_model_advanced.pth'))
            logger.info(f"Saved new best model with F1: {best_f1:.4f}")
            
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

if __name__ == "__main__":
    main()

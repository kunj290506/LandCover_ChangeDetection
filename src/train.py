import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from dataset import ChangeDetectionDataset
from models.snunet import SNUNet
from utils.losses import HybridLoss
from utils.metrics import get_metrics, AverageMeter
from utils.general import set_seed, EarlyStopping

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    
    loop = tqdm(loader, desc="Training")
    loop = tqdm(loader, desc="Training")
    for i, batch in enumerate(loop):
        print(f"Processing Batch {i}/{len(loader)}")
        img1 = batch['image1'].to(device)
        img2 = batch['image2'].to(device)
        label = batch['label'].to(device)
        
        optimizer.zero_grad()
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
            
            # Calculate metrics
            batch_metrics = get_metrics(output, label)
            for k in metrics:
                metrics[k].update(batch_metrics[k], img1.size(0))
            
            loop.set_postfix(loss=losses.avg, f1=metrics['f1'].avg)
            
    return losses.avg, {k: v.avg for k, v in metrics.items()}

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Data
    # Assuming lists of files are generated elsewhere or splitting is done here
    # For now, using placeholders for list paths
    train_dataset = ChangeDetectionDataset(args.data_root, 'train_list.txt', mode='train', patch_size=256)
    val_dataset = ChangeDetectionDataset(args.data_root, 'val_list.txt', mode='val', patch_size=256)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    print(f"DEBUG: Train Dataset Len: {len(train_dataset)}")
    print(f"DEBUG: Train Loader Len: {len(train_loader)}")
    print(f"DEBUG: Batch Size: {args.batch_size}")
    
    # Model
    model = SNUNet(3, 1).to(device) # num_classes=1 for binary (using BCE)

    # Load pretrained weights if specified
    if args.weights:
        if os.path.isfile(args.weights):
            print(f"Loading pretrained weights from {args.weights}")
            checkpoint = torch.load(args.weights, map_location=device)
            # Handle both full checkpoint (with optimizer state) and state_dict only
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Weights loaded successfully.")
        else:
            print(f"Warning: Weights file not found at {args.weights}. Training from scratch.")
    
    # Loss & Optimizer
    criterion = HybridLoss(weight_bce=0.7, weight_dice=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # User specified lr=3e-4
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=15, verbose=True, path=os.path.join(args.checkpoint_dir, 'checkpoint.pt'))
    
    # Loop
    best_f1 = 0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        
        # Save best model based on F1
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            
        # Early Stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--weights', type=str, default=None, help='path to pretrained model weights for fine-tuning')
    
    args = parser.parse_args()
    
    # Check if lists exist, otherwise warn
    if not os.path.exists('train_list.txt'):
        print("Warning: 'train_list.txt' not found. Please generate data lists before running.")
    
    main(args)

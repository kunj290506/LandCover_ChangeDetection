import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from sklearn.model_selection import KFold
import numpy as np

from dataset import ChangeDetectionDataset
from models.snunet import SNUNet
from utils.losses import HybridLoss
from utils.general import set_seed, EarlyStopping
# Import training functions from train.py
from train import train_epoch, validate

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Dataset - Load entire dataset then split
    # For now assuming 'train_list.txt' contains all training pairs
    full_dataset = ChangeDetectionDataset(args.data_root, 'train_list.txt', mode='train', patch_size=256)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f'FOLD {fold+1}')
        print('--------------------------------')
        
        # Subsets
        train_subsampler = Subset(full_dataset, train_ids)
        val_subsampler = Subset(full_dataset, val_ids)
        
        train_loader = DataLoader(train_subsampler, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subsampler, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Init Model per fold
        model = SNUNet(3, 1, use_attention=True).to(device)
        
        criterion = HybridLoss(weight_bce=0.7, weight_dice=0.3)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
        # Early Stopping per fold
        fold_checkpoint_path = os.path.join(args.checkpoint_dir, f'model_fold_{fold+1}.pth')
        early_stopping = EarlyStopping(patience=15, verbose=True, path=fold_checkpoint_path)
        
        best_fold_f1 = 0
        
        for epoch in range(args.epochs):
            print(f'Epoch {epoch+1}/{args.epochs}')
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
            
            if val_metrics['f1'] > best_fold_f1:
                best_fold_f1 = val_metrics['f1']
            
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        results[fold] = best_fold_f1
        print(f'Fold {fold+1} Best F1: {best_fold_f1}')
    
    print('--------------------------------')
    print('K-FOLD CROSS VALIDATION RESULTS')
    print('--------------------------------')
    sum_f1 = 0.0
    for key, value in results.items():
        print(f'Fold {key+1}: {value} F1')
        sum_f1 += value
    print(f'Average F1: {sum_f1/len(results)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs per fold') # Usually less than full training
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/kfold', help='path to save checkpoints')
    
    args = parser.parse_args()
    
    if not os.path.exists('train_list.txt'):
         print("Warning: 'train_list.txt' not found.")
         
    main(args)

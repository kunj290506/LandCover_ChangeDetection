import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from dataset import ChangeDetectionDataset
from models.baselines import FCEF, FCSiamConc_Fixed, FCSiamDiff
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
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), img1.size(0))
        loop.set_postfix(loss=losses.avg)
    return losses.avg

def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    metrics = {'f1': AverageMeter(), 'iou': AverageMeter()}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            label = batch['label'].to(device)
            output = model(img1, img2)
            loss = criterion(output, label)
            losses.update(loss.item(), img1.size(0))
            batch_metrics = get_metrics(output, label)
            for k in metrics: metrics[k].update(batch_metrics[k], img1.size(0))
    return losses.avg, {k: v.avg for k, v in metrics.items()}

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for {args.model}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train_dataset = ChangeDetectionDataset(args.data_root, 'train_list.txt', mode='train', patch_size=256)
    val_dataset = ChangeDetectionDataset(args.data_root, 'val_list.txt', mode='val', patch_size=256)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    if args.model == 'fcef':
        model = FCEF(in_channels=6, num_classes=1).to(device)
    elif args.model == 'fcsiam_conc':
        model = FCSiamConc_Fixed(in_channels=3, num_classes=1).to(device)
    elif args.model == 'fcsiam_diff':
        model = FCSiamDiff(in_channels=3, num_classes=1).to(device)
    else:
        raise ValueError("Invalid model")
    
    criterion = HybridLoss(weight_bce=0.7, weight_dice=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(args.checkpoint_dir, f'{args.model}_best.pth'))
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val F1: {val_metrics['f1']:.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['fcef', 'fcsiam_conc', 'fcsiam_diff'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3) # Baselines often need higher LR or different
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    main(args)

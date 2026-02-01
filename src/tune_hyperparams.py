import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import optuna

from dataset import ChangeDetectionDataset
from models.snunet import SNUNet
from utils.losses import HybridLoss
from train import train_epoch, validate
from utils.general import set_seed

def objective(trial, args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    # Loss weights
    w_bce = trial.suggest_float("w_bce", 0.5, 0.9)
    # w_dice = 1.0 - w_bce # Or independent
    w_dice = trial.suggest_float("w_dice", 0.1, 0.5)
    
    # Model config
    base_channel = trial.suggest_categorical("base_channel", [16, 32])
    
    model = SNUNet(3, 1, base_channel=base_channel, use_attention=True).to(device)
    
    criterion = HybridLoss(weight_bce=w_bce, weight_dice=w_dice)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Data - use a smaller subset or just one split for speed
    full_dataset = ChangeDetectionDataset(args.data_root, 'train_list.txt', mode='train', patch_size=256)
    
    # Use 20% validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Train for limited epochs for tuning
    for epoch in range(10): 
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Pruning
        trial.report(val_metrics['f1'], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_metrics['f1']

def main(args):
    # Check if optuna is installed
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Please run 'pip install optuna'")
        return

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_trials', type=int, default=20)
    args = parser.parse_args()
    
    if not os.path.exists('train_list.txt'):
         print("Warning: train_list.txt missing")
         
    main(args)

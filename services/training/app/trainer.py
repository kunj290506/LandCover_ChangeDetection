import os
import tempfile
import yaml
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score

from landcover_common.settings import Settings
from landcover_common.storage import get_s3_client
from dataset import ChangeDetectionDataset
from models.snunet import SNUNet
from utils.losses import HybridLoss
from utils.metrics import calculate_iou


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _download_from_s3(uri: str, settings: Settings) -> str:
    """Download file from S3 and return local path"""
    if not uri.startswith("s3://"):
        return uri
    
    client = get_s3_client(settings)
    _, path = uri.split("s3://", 1)
    bucket, key = path.split("/", 1)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(key)[1])
    client.download_file(bucket, key, tmp.name)
    return tmp.name


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Calculate comprehensive metrics for change detection"""
    # Convert to binary predictions
    pred_bin = (pred > 0.5).float()
    
    # Flatten for sklearn metrics
    pred_np = pred_bin.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    # Calculate metrics
    precision = precision_score(target_np, pred_np, zero_division=0)
    recall = recall_score(target_np, pred_np, zero_division=0)
    f1 = f1_score(target_np, pred_np, zero_division=0)
    iou = calculate_iou(pred_bin, target)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou.item()
    }


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        img1 = batch['image1'].to(device)
        img2 = batch['image2'].to(device)
        target = batch['label'].to(device)
        
        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), img1.size(0))
        
        # Log batch metrics to MLflow every 100 batches
        if batch_idx % 100 == 0:
            mlflow.log_metric("batch_loss", loss.item(), step=batch_idx)
    
    return losses.avg


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                   device: torch.device) -> tuple[float, Dict[str, float]]:
    """Validate for one epoch"""
    model.eval()
    losses = AverageMeter()
    all_metrics = {'precision': [], 'recall': [], 'f1': [], 'iou': []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            target = batch['label'].to(device)
            
            output = model(img1, img2)
            loss = criterion(output, target)
            
            losses.update(loss.item(), img1.size(0))
            
            # Calculate metrics for this batch
            probs = torch.sigmoid(output)
            batch_metrics = calculate_metrics(probs, target)
            
            for key, value in batch_metrics.items():
                all_metrics[key].append(value)
    
    # Average metrics across all batches
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    return losses.avg, avg_metrics


def run_training(dataset_uri: str, config_uri: str, register_model: bool) -> str:
    """Main training function with MLflow tracking"""
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    # Download config if from S3
    config_path = _download_from_s3(config_uri, settings)
    config = _load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("dataset_uri", dataset_uri)
        mlflow.log_param("config_uri", config_uri)
        mlflow.log_param("device", str(device))
        mlflow.log_params(config)
        
        # Download dataset if from S3
        dataset_path = _download_from_s3(dataset_uri, settings)
        
        # Create datasets
        train_dataset = ChangeDetectionDataset(
            dataset_path, 
            'train_list.txt', 
            mode='train',
            patch_size=config.get('patch_size', 256)
        )
        val_dataset = ChangeDetectionDataset(
            dataset_path, 
            'val_list.txt', 
            mode='val',
            patch_size=config.get('patch_size', 256)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=torch.cuda.is_available()
        )
        
        # Create model
        model = SNUNet(
            in_channels=3,
            num_classes=1,
            base_channel=config.get('base_channel', 32),
            use_attention=config.get('use_attention', True)
        ).to(device)
        
        # Loss function
        criterion = HybridLoss(
            weight_bce=config.get('weight_bce', 0.7),
            weight_dice=config.get('weight_dice', 0.3)
        )
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('epochs', 50),
            eta_min=config.get('eta_min', 1e-6)
        )
        
        # Training loop
        best_f1 = 0.0
        patience_counter = 0
        patience = config.get('patience', 15)
        
        for epoch in range(config.get('epochs', 50)):
            print(f"Epoch {epoch+1}/{config.get('epochs', 50)}")
            
            # Training
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'val_iou': val_metrics['iou'],
                'learning_rate': scheduler.get_last_lr()[0]
            }, step=epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0
                
                # Save model to MLflow
                mlflow.pytorch.log_model(
                    model, 
                    "model",
                    registered_model_name=settings.model_name if register_model else None
                )
                
                # Log best metrics
                mlflow.log_metrics({
                    'best_f1': best_f1,
                    'best_precision': val_metrics['precision'],
                    'best_recall': val_metrics['recall'],
                    'best_iou': val_metrics['iou']
                })
                
                print(f"New best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                mlflow.log_metric('early_stop_epoch', epoch)
                break
        
        # Final model registration with stage transition
        if register_model and best_f1 > 0.5:  # Only register if reasonable performance
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(settings.model_name, stages=["None"])[0]
            client.transition_model_version_stage(
                name=settings.model_name,
                version=model_version.version,
                stage="Staging"
            )
            print(f"Model registered and moved to Staging: {settings.model_name} v{model_version.version}")
        
        return run.info.run_id

import torch
import numpy as np


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Calculate Intersection over Union (IoU)"""
    # Apply threshold to predictions
    pred_binary = (pred > threshold).float()
    
    # Calculate intersection and union
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    # Handle edge case where union is zero
    if union == 0:
        return torch.tensor(1.0 if intersection == 0 else 0.0)
    
    return intersection / union


def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Calculate pixel-wise accuracy"""
    pred_binary = (pred > threshold).float()
    correct = (pred_binary == target).float()
    return correct.mean()


def calculate_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Calculate confusion matrix components"""
    pred_binary = (pred > threshold).float()
    
    tp = ((pred_binary == 1) & (target == 1)).sum().float()
    tn = ((pred_binary == 0) & (target == 0)).sum().float()
    fp = ((pred_binary == 1) & (target == 0)).sum().float()
    fn = ((pred_binary == 0) & (target == 1)).sum().float()
    
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def calculate_kappa(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Calculate Cohen's Kappa coefficient"""
    cm = calculate_confusion_matrix(pred, target, threshold)
    
    tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']
    total = tp + tn + fp + fn
    
    if total == 0:
        return torch.tensor(0.0)
    
    # Observed accuracy
    po = (tp + tn) / total
    
    # Expected accuracy
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (total ** 2)
    
    # Cohen's Kappa
    if pe == 1:
        return torch.tensor(0.0)
    
    kappa = (po - pe) / (1 - pe)
    return kappa
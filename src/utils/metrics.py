import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, cohen_kappa_score, precision_score, recall_score

def get_metrics(inputs, targets, threshold=0.5):
    """
    inputs: logits or probabilities (B, 1, H, W)
    targets: binary labels (B, 1, H, W)
    """
    # Apply sigmoid if logits
    inputs = torch.sigmoid(inputs)
    
    # Binarize
    preds = (inputs > threshold).float()
    
    # Move to CPU numpy
    preds = preds.detach().cpu().numpy().flatten().astype(int)
    targets = targets.detach().cpu().numpy().flatten().astype(int)
    
    f1 = f1_score(targets, preds, zero_division=0)
    iou = jaccard_score(targets, preds, zero_division=0)
    kappa = cohen_kappa_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    
    return {
        'f1': f1,
        'iou': iou,
        'kappa': kappa,
        'precision': precision,
        'recall': recall
    }

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

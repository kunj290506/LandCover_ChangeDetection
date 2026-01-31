import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # Comment out if your model contains a sigmoid or if inputs are not logits
        inputs = torch.sigmoid(inputs)       
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class HybridLoss(nn.Module):
    def __init__(self, weight_bce=0.7, weight_dice=0.3):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_bce/0.3])) # Weighted BCE if needed, or just standard
        # Note: The user requested weighted BCE (change=0.7, no-change=0.3). 
        # Standard BCEWithLogitsLoss uses pos_weight to weigh the positive class (change) more.
        # If change=0.7 and no-change=0.3, ratio is ~2.33. 
        # But here we will mix BCE and Dice.
        self.dice = DiceLoss()
        self.alpha = weight_bce # User said weights: change=0.7, no-change=0.3 for BCE. 
        # Usually implies the weighting *inside* BCE.
        # "Weighted Binary Cross-Entropy (weights: change=0.7, no-change=0.3) + Dice Loss (alpha=0.5)"
        
        # Let's interpret strictly: 
        # Loss = BCE(weighted) + Dice
        
        # For BCE weight:
        # We can implement manual weighted BCE or use pos_weight.
        self.pos_weight = torch.tensor(0.7/0.3) 
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice(inputs, targets)
        return bce + dice


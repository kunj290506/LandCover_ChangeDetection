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
        self.dice = DiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        # Store pos_weight value, will be converted to correct device during forward
        self.pos_weight_value = weight_bce / weight_dice

    def forward(self, inputs, targets):
        # Ensure pos_weight is on the same device as inputs
        pos_weight = torch.tensor([self.pos_weight_value], device=inputs.device)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=pos_weight
        )
        dice = self.dice(inputs, targets)
        return bce_loss + dice

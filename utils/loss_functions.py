import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    
    """
    Implementation of the Focal Loss as a PyTorch module.

    Parameters:
    - alpha (Tensor): Weighting factor for the positive class.
    - gamma (float): Focusing parameter to adjust the rate at which easy examples contribute to the loss.
    
    """
    
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.cuda()
        self.gamma = gamma
        
    def forward(self, inputs, targets):
       
            
        targets_long = targets.clone().long()
        targets_long  = torch.argmax(targets_long , dim=1)
        
        ce_loss = F.cross_entropy(inputs, targets_long, reduction='mean')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets_long] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss

class FocalLossBCE(nn.Module):
    """
    Implementation of the Focal Loss with Binary Cross-Entropy (BCE) as a PyTorch module.

    Parameters:
    - alpha (Tensor): Weighting factor for the positive class.
    - gamma (float): Focusing parameter to adjust the rate at which easy examples contribute to the loss.
    """

    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.cuda() if alpha is not None else None
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid for binary classification
        
        # BCE Loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Compute the focal weight
        pt = torch.exp(-bce_loss)
        focal_weight = ((1 - pt) ** self.gamma) * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        
        loss = (focal_weight * bce_loss).mean()
        
        
        return loss        
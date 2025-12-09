"""
Loss functions for depth estimation
"""

import torch
import torch.nn as nn


class SIlogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss (SIlog)
    Widely used in recent depth estimation papers (e.g., DPT, MiDaS)
    
    SIlog = sqrt(mean((log(pred) - log(gt))^2) - lambda * (mean(log(pred) - log(gt)))^2)
    
    This loss is scale-invariant and handles the log-space differences,
    making it robust to scale ambiguities in depth estimation.
    """
    def __init__(self, lambda_scale=0.5, epsilon=1e-6):
        """
        Args:
            lambda_scale: Weight for the scale term (typically 0.5)
            epsilon: Small value to avoid log(0)
        """
        super(SIlogLoss, self).__init__()
        self.lambda_scale = lambda_scale
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted depth (in meters, denormalized)
            target: Ground truth depth (in meters, denormalized)
        """
        # Clamp to avoid log(0) or log(negative)
        pred = torch.clamp(pred, min=self.epsilon)
        target = torch.clamp(target, min=self.epsilon)
        
        # Compute log differences
        log_diff = torch.log(pred) - torch.log(target)
        
        # Scale-invariant logarithmic loss
        # First term: mean squared log difference
        # Second term: squared mean log difference (scale term)
        # Use max(0, ...) to ensure non-negative value under sqrt (numerical safety)
        variance_term = torch.mean(log_diff ** 2) - self.lambda_scale * (torch.mean(log_diff) ** 2)
        si_log = torch.sqrt(torch.clamp(variance_term, min=0.0))
        
        return si_log

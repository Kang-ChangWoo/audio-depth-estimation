"""Loss functions for depth estimation."""

import torch
import torch.nn as nn


class SIlogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss."""

    def __init__(self, lambda_scale=0.5, epsilon=1e-6):
        super(SIlogLoss, self).__init__()
        self.lambda_scale = lambda_scale
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=self.epsilon)
        target = torch.clamp(target, min=self.epsilon)
        log_diff = torch.log(pred) - torch.log(target)
        variance_term = torch.mean(log_diff ** 2) - self.lambda_scale * (torch.mean(log_diff) ** 2)
        return torch.sqrt(torch.clamp(variance_term, min=0.0))

"""
Loss Functions for Binaural Attention Model

Includes:
    - Depth reconstruction loss
    - Edge-aware loss (preserve boundaries)
    - Smoothness loss (reduce artifacts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinauralAttentionLoss(nn.Module):
    """
    Combined loss for Binaural Attention depth estimation
    
    Components:
        1. Reconstruction Loss (L1): Main depth prediction accuracy
        2. Edge-Aware Loss: Preserve depth discontinuities
        3. Smoothness Loss: Encourage smooth predictions in uniform regions
    
    Args:
        lambda_recon: Weight for reconstruction loss
        lambda_edge: Weight for edge-aware loss
        lambda_smooth: Weight for smoothness loss
    """
    
    def __init__(
        self,
        lambda_recon=1.0,
        lambda_edge=0.2,
        lambda_smooth=0.1
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_edge = lambda_edge
        self.lambda_smooth = lambda_smooth
        
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0))
    
    def forward(self, pred_depth, gt_depth):
        """
        Args:
            pred_depth: [B, 1, H, W] - Predicted depth
            gt_depth: [B, 1, H, W] - Ground truth depth
            
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary of individual losses
        """
        # 1. Reconstruction Loss (L1)
        valid_mask = (gt_depth > 0).float()  # Ignore invalid depth
        
        if valid_mask.sum() > 0:
            loss_recon = F.l1_loss(
                pred_depth * valid_mask,
                gt_depth * valid_mask,
                reduction='sum'
            ) / (valid_mask.sum() + 1e-6)
        else:
            loss_recon = torch.tensor(0.0, device=pred_depth.device)
        
        # 2. Edge-Aware Loss
        loss_edge = self.edge_aware_loss(pred_depth, gt_depth, valid_mask)
        
        # 3. Smoothness Loss
        loss_smooth = self.smoothness_loss(pred_depth, gt_depth, valid_mask)
        
        # Total loss
        total_loss = (
            self.lambda_recon * loss_recon +
            self.lambda_edge * loss_edge +
            self.lambda_smooth * loss_smooth
        )
        
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_recon': loss_recon.item(),
            'loss_edge': loss_edge.item(),
            'loss_smooth': loss_smooth.item()
        }
        
        return total_loss, loss_dict
    
    def edge_aware_loss(self, pred_depth, gt_depth, valid_mask):
        """
        Compute edge-aware loss to preserve depth discontinuities
        
        Args:
            pred_depth: [B, 1, H, W]
            gt_depth: [B, 1, H, W]
            valid_mask: [B, 1, H, W]
        """
        # Compute gradients
        pred_grad_x = F.conv2d(pred_depth, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_depth, self.sobel_y, padding=1)
        
        gt_grad_x = F.conv2d(gt_depth, self.sobel_x, padding=1)
        gt_grad_y = F.conv2d(gt_depth, self.sobel_y, padding=1)
        
        # Edge magnitude
        pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)
        gt_grad = torch.sqrt(gt_grad_x ** 2 + gt_grad_y ** 2 + 1e-6)
        
        # Mask for valid regions
        valid_mask_eroded = F.max_pool2d(valid_mask, kernel_size=3, stride=1, padding=1)
        
        if valid_mask_eroded.sum() > 0:
            loss = F.l1_loss(
                pred_grad * valid_mask_eroded,
                gt_grad * valid_mask_eroded,
                reduction='sum'
            ) / (valid_mask_eroded.sum() + 1e-6)
        else:
            loss = torch.tensor(0.0, device=pred_depth.device)
        
        return loss
    
    def smoothness_loss(self, pred_depth, gt_depth, valid_mask):
        """
        Encourage smoothness in regions where GT is smooth
        
        Args:
            pred_depth: [B, 1, H, W]
            gt_depth: [B, 1, H, W]
            valid_mask: [B, 1, H, W]
        """
        # Compute second derivatives (Laplacian)
        pred_grad_x = F.conv2d(pred_depth, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_depth, self.sobel_y, padding=1)
        
        gt_grad_x = F.conv2d(gt_depth, self.sobel_x, padding=1)
        gt_grad_y = F.conv2d(gt_depth, self.sobel_y, padding=1)
        
        # Edge weight (inverse of GT gradient)
        gt_edge = torch.sqrt(gt_grad_x ** 2 + gt_grad_y ** 2 + 1e-6)
        edge_weight = torch.exp(-gt_edge)  # High weight in smooth regions
        
        # Smoothness penalty
        smoothness = torch.abs(pred_grad_x) + torch.abs(pred_grad_y)
        
        if valid_mask.sum() > 0:
            loss = (smoothness * edge_weight * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        else:
            loss = torch.tensor(0.0, device=pred_depth.device)
        
        return loss


class AdaptiveBinauralAttentionLoss(nn.Module):
    """
    Adaptive loss with curriculum learning
    
    Loss weights change during training:
        - Early: Focus on reconstruction (learn basic structure)
        - Mid: Add edge preservation (refine boundaries)
        - Late: Add smoothness (reduce artifacts)
    """
    
    def __init__(
        self,
        warmup_epochs=20,
        total_epochs=200
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        # Base loss module
        self.base_loss = BinauralAttentionLoss(
            lambda_recon=1.0,
            lambda_edge=0.0,  # Will be adjusted
            lambda_smooth=0.0  # Will be adjusted
        )
        
        # Sobel for edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0))
    
    def forward(self, pred_depth, gt_depth, epoch):
        """
        Args:
            pred_depth: [B, 1, H, W]
            gt_depth: [B, 1, H, W]
            epoch: Current epoch number
        """
        # Adjust weights based on epoch
        progress = min(epoch / self.total_epochs, 1.0)
        
        if epoch < self.warmup_epochs:
            # Early: Focus on reconstruction only
            lambda_recon = 1.0
            lambda_edge = 0.0
            lambda_smooth = 0.0
        elif epoch < self.warmup_epochs * 3:
            # Mid: Add edge preservation
            lambda_recon = 1.0
            lambda_edge = 0.2 * (epoch - self.warmup_epochs) / (self.warmup_epochs * 2)
            lambda_smooth = 0.0
        else:
            # Late: Full loss
            lambda_recon = 1.0
            lambda_edge = 0.2
            lambda_smooth = 0.1 * min((epoch - self.warmup_epochs * 3) / self.warmup_epochs, 1.0)
        
        # Update base loss weights
        self.base_loss.lambda_recon = lambda_recon
        self.base_loss.lambda_edge = lambda_edge
        self.base_loss.lambda_smooth = lambda_smooth
        
        # Compute loss
        total_loss, loss_dict = self.base_loss(pred_depth, gt_depth)
        
        # Add weight info to dict
        loss_dict['lambda_recon'] = lambda_recon
        loss_dict['lambda_edge'] = lambda_edge
        loss_dict['lambda_smooth'] = lambda_smooth
        
        return total_loss, loss_dict


def create_binaural_loss(
    loss_type='standard',
    lambda_recon=1.0,
    lambda_edge=0.2,
    lambda_smooth=0.1,
    warmup_epochs=20,
    total_epochs=200
):
    """
    Factory function to create loss function
    
    Args:
        loss_type: 'standard' or 'adaptive'
        lambda_recon: Weight for reconstruction loss
        lambda_edge: Weight for edge-aware loss
        lambda_smooth: Weight for smoothness loss
        warmup_epochs: Epochs for curriculum warmup (adaptive only)
        total_epochs: Total training epochs (adaptive only)
    """
    if loss_type == 'adaptive':
        return AdaptiveBinauralAttentionLoss(
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs
        )
    else:
        return BinauralAttentionLoss(
            lambda_recon=lambda_recon,
            lambda_edge=lambda_edge,
            lambda_smooth=lambda_smooth
        )


if __name__ == "__main__":
    # Test loss functions
    print("Testing Binaural Attention Loss...")
    
    # Create dummy data
    pred = torch.randn(4, 1, 256, 256)
    gt = torch.randn(4, 1, 256, 256).abs()
    
    # Test standard loss
    criterion = create_binaural_loss(loss_type='standard')
    loss, loss_dict = criterion(pred, gt)
    print(f"\nStandard Loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # Test adaptive loss
    criterion_adaptive = create_binaural_loss(loss_type='adaptive')
    for epoch in [0, 20, 60, 100]:
        loss, loss_dict = criterion_adaptive(pred, gt, epoch)
        print(f"\nAdaptive Loss (Epoch {epoch}): {loss.item():.4f}")
        for k, v in loss_dict.items():
            print(f"  {k}: {v:.4f}")
    
    print("\n✅ Loss test passed!")


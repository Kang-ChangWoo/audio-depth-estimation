"""
Custom Loss Functions for Base + Residual Depth Model

Three-Component Loss:
    1. Reconstruction Loss: Final depth matches GT
    2. Structural Guidance: Base depth captures low-frequency structure
    3. Sparsity Penalty: Residual stays small (Taylor series epsilon)

Mathematical Formulation:
    L_total = λ1 * ||D_gt - (D_base + D_res)||_1
            + λ2 * ||D_base - LowPass(D_gt)||_1
            + λ3 * ||D_res||_1

Design Philosophy:
    - Without explicit layout GT, we use depth map's low-frequency as proxy
    - Base learns room structure (walls, floor, ceiling)
    - Residual learns object details (furniture, small objects)
    - This decomposition makes under-constrained problem tractable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseResidualLoss(nn.Module):
    """
    Custom loss for Base + Residual depth estimation
    
    Args:
        lambda_recon: Weight for reconstruction loss (final depth accuracy)
        lambda_base: Weight for base structural guidance (layout learning)
        lambda_sparse: Weight for residual sparsity (keep residual small)
        lowpass_kernel: Kernel size for low-pass filtering GT (larger = coarser base)
        use_l1: Use L1 loss (True) or L2 loss (False)
    """
    
    def __init__(self, 
                 lambda_recon=1.0,
                 lambda_base=0.8,
                 lambda_sparse=0.2,
                 lowpass_kernel=16,
                 use_l1=True):
        super().__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_base = lambda_base
        self.lambda_sparse = lambda_sparse
        self.lowpass_kernel = lowpass_kernel
        self.use_l1 = use_l1
        
        # Loss function
        if use_l1:
            self.loss_fn = F.l1_loss
        else:
            self.loss_fn = F.mse_loss
    
    def forward(self, base_depth, residual, final_depth, gt_depth, valid_mask=None):
        """
        Compute three-component loss
        
        Args:
            base_depth: [B, 1, H, W] - Predicted base depth
            residual: [B, 1, H, W] - Predicted residual
            final_depth: [B, 1, H, W] - base_depth + residual
            gt_depth: [B, 1, H, W] - Ground truth depth
            valid_mask: [B, 1, H, W] - Mask of valid depth pixels (optional)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # ==========================================
        # 2. Structural Guidance Loss (compute BEFORE masking)
        # ==========================================
        # Base should match low-frequency (coarse structure) of GT
        # This acts as implicit "room layout" supervision
        
        with torch.no_grad():
            # Low-pass filter GT to get structural component
            # Use average pooling to remove high-frequency details
            gt_struct = F.avg_pool2d(
                gt_depth, 
                kernel_size=self.lowpass_kernel,
                stride=1,
                padding=self.lowpass_kernel // 2
            )
            
            # Resize back to original size
            if gt_struct.shape != gt_depth.shape:
                gt_struct = F.interpolate(
                    gt_struct,
                    size=gt_depth.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
        
        # Apply valid mask if provided (after computing structural target)
        if valid_mask is not None:
            final_depth_masked = final_depth[valid_mask]
            base_depth_masked = base_depth[valid_mask]
            residual_masked = residual[valid_mask]
            gt_depth_masked = gt_depth[valid_mask]
            gt_struct_masked = gt_struct[valid_mask]
        else:
            final_depth_masked = final_depth
            base_depth_masked = base_depth
            residual_masked = residual
            gt_depth_masked = gt_depth
            gt_struct_masked = gt_struct
        
        # ==========================================
        # 1. Reconstruction Loss
        # ==========================================
        # Final prediction should match GT exactly
        loss_recon = self.loss_fn(final_depth_masked, gt_depth_masked)
        
        # Structural guidance loss
        loss_base = self.loss_fn(base_depth_masked, gt_struct_masked)
        
        # ==========================================
        # 3. Sparsity Penalty
        # ==========================================
        # Residual should be small (epsilon in Taylor series)
        # This encourages base to do most of the work
        loss_sparse = torch.mean(torch.abs(residual_masked))
        
        # ==========================================
        # Total Loss
        # ==========================================
        total_loss = (
            self.lambda_recon * loss_recon +
            self.lambda_base * loss_base +
            self.lambda_sparse * loss_sparse
        )
        
        # Return loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'recon': loss_recon.item(),
            'base': loss_base.item(),
            'sparse': loss_sparse.item()
        }
        
        return total_loss, loss_dict


class AdaptiveBaseResidualLoss(nn.Module):
    """
    Adaptive version that adjusts weights during training
    
    Strategy:
        - Early training: High lambda_base (learn structure first)
        - Late training: High lambda_recon (refine accuracy)
        - This mimics curriculum learning
    
    Args:
        lambda_recon_init: Initial reconstruction weight
        lambda_base_init: Initial base weight
        lambda_sparse: Sparsity weight (fixed)
        warmup_epochs: Number of epochs to transition weights
        lowpass_kernel: Kernel size for low-pass filtering
    """
    
    def __init__(self,
                 lambda_recon_init=0.5,
                 lambda_base_init=1.5,
                 lambda_sparse=0.3,
                 warmup_epochs=20,
                 lowpass_kernel=16):
        super().__init__()
        
        self.lambda_recon_init = lambda_recon_init
        self.lambda_recon_final = 1.0
        self.lambda_base_init = lambda_base_init
        self.lambda_base_final = 0.3
        self.lambda_sparse = lambda_sparse
        self.warmup_epochs = warmup_epochs
        self.lowpass_kernel = lowpass_kernel
        
        self.current_epoch = 0
        self.base_loss = BaseResidualLoss(
            lambda_recon=lambda_recon_init,
            lambda_base=lambda_base_init,
            lambda_sparse=lambda_sparse,
            lowpass_kernel=lowpass_kernel
        )
    
    def set_epoch(self, epoch):
        """Update loss weights based on current epoch"""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear interpolation
            alpha = epoch / self.warmup_epochs
            self.base_loss.lambda_recon = (
                self.lambda_recon_init + 
                alpha * (self.lambda_recon_final - self.lambda_recon_init)
            )
            self.base_loss.lambda_base = (
                self.lambda_base_init +
                alpha * (self.lambda_base_final - self.lambda_base_init)
            )
        else:
            self.base_loss.lambda_recon = self.lambda_recon_final
            self.base_loss.lambda_base = self.lambda_base_final
    
    def forward(self, base_depth, residual, final_depth, gt_depth, valid_mask=None):
        """Forward pass (same as BaseResidualLoss)"""
        return self.base_loss(base_depth, residual, final_depth, gt_depth, valid_mask)
    
    def get_current_weights(self):
        """Get current loss weights for logging"""
        return {
            'lambda_recon': self.base_loss.lambda_recon,
            'lambda_base': self.base_loss.lambda_base,
            'lambda_sparse': self.base_loss.lambda_sparse
        }


class FrequencyAwareBaseResidualLoss(nn.Module):
    """
    Frequency-domain aware loss (experimental)
    
    Explicitly separates frequency components using FFT
    - Base should match low frequencies
    - Residual should match high frequencies
    
    More sophisticated but computationally expensive
    """
    
    def __init__(self,
                 lambda_recon=1.0,
                 lambda_base_low=0.5,
                 lambda_res_high=0.3,
                 lambda_sparse=0.1,
                 freq_cutoff=0.1):
        super().__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_base_low = lambda_base_low
        self.lambda_res_high = lambda_res_high
        self.lambda_sparse = lambda_sparse
        self.freq_cutoff = freq_cutoff  # Fraction of frequencies to consider "low"
    
    def separate_frequencies(self, depth_map):
        """
        Separate depth map into low and high frequency components using FFT
        
        Args:
            depth_map: [B, 1, H, W]
        
        Returns:
            low_freq: Low frequency component
            high_freq: High frequency component
        """
        B, C, H, W = depth_map.shape
        
        # Apply FFT
        fft = torch.fft.fft2(depth_map)
        fft_shift = torch.fft.fftshift(fft)
        
        # Create low-pass mask
        center_h, center_w = H // 2, W // 2
        cutoff_h = int(H * self.freq_cutoff)
        cutoff_w = int(W * self.freq_cutoff)
        
        mask_low = torch.zeros_like(fft_shift, dtype=torch.bool)
        mask_low[..., 
                 center_h - cutoff_h:center_h + cutoff_h,
                 center_w - cutoff_w:center_w + cutoff_w] = True
        
        # Separate frequencies
        fft_low = fft_shift * mask_low
        fft_high = fft_shift * (~mask_low)
        
        # Inverse FFT
        low_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_low)).real
        high_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_high)).real
        
        return low_freq, high_freq
    
    def forward(self, base_depth, residual, final_depth, gt_depth, valid_mask=None):
        """Compute frequency-aware loss"""
        
        # 1. Reconstruction loss
        if valid_mask is not None:
            loss_recon = F.l1_loss(final_depth[valid_mask], gt_depth[valid_mask])
        else:
            loss_recon = F.l1_loss(final_depth, gt_depth)
        
        # 2. Frequency separation (only if no mask - FFT requires full spatial dims)
        if valid_mask is None:
            gt_low, gt_high = self.separate_frequencies(gt_depth)
            
            # Base should match low frequencies
            loss_base_low = F.l1_loss(base_depth, gt_low)
            
            # Residual should match high frequencies
            loss_res_high = F.l1_loss(residual, gt_high)
        else:
            # Fallback to spatial domain if mask is used
            loss_base_low = F.l1_loss(base_depth[valid_mask], gt_depth[valid_mask])
            loss_res_high = torch.mean(torch.abs(residual[valid_mask]))
        
        # 3. Sparsity penalty
        loss_sparse = torch.mean(torch.abs(residual))
        
        # Total loss
        total_loss = (
            self.lambda_recon * loss_recon +
            self.lambda_base_low * loss_base_low +
            self.lambda_res_high * loss_res_high +
            self.lambda_sparse * loss_sparse
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': loss_recon.item(),
            'base_low': loss_base_low.item(),
            'res_high': loss_res_high.item(),
            'sparse': loss_sparse.item()
        }
        
        return total_loss, loss_dict


# ==========================================
# Test Code
# ==========================================
if __name__ == '__main__':
    print("Testing Base + Residual Loss Functions")
    print("=" * 60)
    
    # Create dummy data
    B, H, W = 4, 256, 256
    base_depth = torch.randn(B, 1, H, W)
    residual = torch.randn(B, 1, H, W) * 0.1  # Smaller residual
    final_depth = base_depth + residual
    gt_depth = torch.randn(B, 1, H, W)
    valid_mask = torch.ones(B, 1, H, W, dtype=torch.bool)
    
    # Test BaseResidualLoss
    print("\n1. Testing BaseResidualLoss...")
    loss_fn = BaseResidualLoss(lambda_recon=1.0, lambda_base=0.5, lambda_sparse=0.1)
    total_loss, loss_dict = loss_fn(base_depth, residual, final_depth, gt_depth)
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Components: {loss_dict}")
    
    # Test AdaptiveBaseResidualLoss
    print("\n2. Testing AdaptiveBaseResidualLoss...")
    adaptive_loss = AdaptiveBaseResidualLoss(warmup_epochs=10)
    
    print("   Epoch 0 (early):")
    adaptive_loss.set_epoch(0)
    total_loss, _ = adaptive_loss(base_depth, residual, final_depth, gt_depth)
    weights = adaptive_loss.get_current_weights()
    print(f"     Weights: {weights}")
    print(f"     Loss: {total_loss.item():.4f}")
    
    print("   Epoch 20 (late):")
    adaptive_loss.set_epoch(20)
    total_loss, _ = adaptive_loss(base_depth, residual, final_depth, gt_depth)
    weights = adaptive_loss.get_current_weights()
    print(f"     Weights: {weights}")
    print(f"     Loss: {total_loss.item():.4f}")
    
    # Test FrequencyAwareBaseResidualLoss
    print("\n3. Testing FrequencyAwareBaseResidualLoss...")
    freq_loss = FrequencyAwareBaseResidualLoss()
    total_loss, loss_dict = freq_loss(base_depth, residual, final_depth, gt_depth)
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Components: {loss_dict}")
    
    print("\n✅ All loss functions tested successfully!")



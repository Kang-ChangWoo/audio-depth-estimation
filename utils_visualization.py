"""
Visualization utilities for saving evaluation results
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def save_depth_comparison(pred_depth, gt_depth, save_path, epoch, idx=0):
    """
    Save depth prediction and ground truth comparison image
    
    Args:
        pred_depth: predicted depth (numpy array or torch tensor, shape: [H, W] or [1, H, W])
        gt_depth: ground truth depth (numpy array or torch tensor, shape: [H, W] or [1, H, W])
        save_path: path to save the image
        epoch: current epoch number
        idx: sample index in batch
    """
    # Convert to numpy if torch tensor
    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.detach().cpu().numpy()
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.detach().cpu().numpy()
    
    # Handle different shapes
    if pred_depth.ndim == 3:
        pred_depth = pred_depth[0]  # Remove channel dimension
    if gt_depth.ndim == 3:
        gt_depth = gt_depth[0]
    
    # Denormalize if needed (assuming depth was normalized to [0, 1])
    # You may need to adjust this based on your normalization
    if pred_depth.max() <= 1.0 and pred_depth.min() >= 0.0:
        # Assume normalized, but we'll keep original values for visualization
        pass
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground truth
    im1 = axes[0].imshow(gt_depth, cmap='jet', vmin=gt_depth.min(), vmax=gt_depth.max())
    axes[0].set_title(f'Ground Truth (Epoch {epoch})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Prediction
    im2 = axes[1].imshow(pred_depth, cmap='jet', vmin=gt_depth.min(), vmax=gt_depth.max())
    axes[1].set_title(f'Prediction (Epoch {epoch})')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Error map
    error_map = np.abs(pred_depth - gt_depth)
    # Mask out zero values in GT
    mask = gt_depth > 0
    error_map[~mask] = 0
    
    im3 = axes[2].imshow(error_map, cmap='hot', vmin=0, vmax=error_map[mask].max() if mask.any() else 1)
    axes[2].set_title(f'Absolute Error (Epoch {epoch})')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_batch_visualization(pred_depths, gt_depths, save_path, epoch, num_samples=4):
    """
    Save batch of depth predictions and ground truths with error maps and histograms
    
    Args:
        pred_depths: predicted depths (torch tensor, shape: [B, 1, H, W] or [B, H, W])
        gt_depths: ground truth depths (torch tensor, shape: [B, 1, H, W] or [B, H, W])
        save_path: path to save the image
        epoch: current epoch number
        num_samples: number of samples to visualize
    
    Visualization rows:
        Row 1: Ground Truth depth maps
        Row 2: Predicted depth maps
        Row 3: Error maps (absolute difference)
        Row 4: GT depth histograms
        Row 5: Pred depth histograms
    """
    # Convert to numpy if torch tensor
    if isinstance(pred_depths, torch.Tensor):
        pred_depths = pred_depths.detach().cpu().numpy()
    if isinstance(gt_depths, torch.Tensor):
        gt_depths = gt_depths.detach().cpu().numpy()
    
    batch_size = min(pred_depths.shape[0], num_samples)
    
    # 5 rows: GT, Pred, Error, GT Histogram, Pred Histogram
    fig, axes = plt.subplots(5, batch_size, figsize=(5*batch_size, 22))
    if batch_size == 1:
        axes = axes.reshape(5, 1)
    
    for i in range(batch_size):
        # Get depth maps
        pred = pred_depths[i]
        gt = gt_depths[i]
        
        # Handle channel dimension
        if pred.ndim == 3:
            pred = pred[0]
        if gt.ndim == 3:
            gt = gt[0]
        
        # Create valid mask (GT > 0)
        valid_mask = gt > 0
        
        # Determine common color range for GT and Pred
        vmin = gt[valid_mask].min() if valid_mask.any() else 0
        vmax = gt[valid_mask].max() if valid_mask.any() else 1
        
        # Row 1: Ground truth
        im1 = axes[0, i].imshow(gt, cmap='jet', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'GT Sample {i+1}')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Row 2: Prediction
        im2 = axes[1, i].imshow(pred, cmap='jet', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Pred Sample {i+1}')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i])
        
        # Row 3: Error map (absolute difference)
        error_map = np.abs(pred - gt)
        error_map[~valid_mask] = 0  # Mask out invalid regions
        
        # Calculate error statistics for valid regions
        if valid_mask.any():
            error_valid = error_map[valid_mask]
            mean_error = error_valid.mean()
            max_error = error_valid.max()
            # Use 95th percentile for better visualization
            error_vmax = np.percentile(error_valid, 95) if len(error_valid) > 0 else 1
        else:
            mean_error = 0
            max_error = 0
            error_vmax = 1
        
        im3 = axes[2, i].imshow(error_map, cmap='hot', vmin=0, vmax=error_vmax)
        axes[2, i].set_title(f'Error (MAE:{mean_error:.2f}, Max:{max_error:.2f})')
        axes[2, i].axis('off')
        plt.colorbar(im3, ax=axes[2, i])
        
        # Row 4: GT depth histogram
        if valid_mask.any():
            gt_valid = gt[valid_mask].flatten()
            # Fine-grained bins for detailed distribution
            n_bins = 100
            bin_range = (vmin, vmax)
            
            axes[3, i].hist(gt_valid, bins=n_bins, range=bin_range, 
                           color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[3, i].set_title(f'GT Dist (μ:{gt_valid.mean():.2f}, σ:{gt_valid.std():.2f})')
            axes[3, i].set_xlabel('Depth (m)')
            axes[3, i].set_ylabel('Frequency')
            axes[3, i].grid(True, alpha=0.3)
        else:
            axes[3, i].text(0.5, 0.5, 'No valid data', ha='center', va='center')
            axes[3, i].set_title('GT Distribution')
        
        # Row 5: Pred depth histogram
        if valid_mask.any():
            pred_valid = pred[valid_mask].flatten()
            
            axes[4, i].hist(pred_valid, bins=n_bins, range=bin_range,
                           color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[4, i].set_title(f'Pred Dist (μ:{pred_valid.mean():.2f}, σ:{pred_valid.std():.2f})')
            axes[4, i].set_xlabel('Depth (m)')
            axes[4, i].set_ylabel('Frequency')
            axes[4, i].grid(True, alpha=0.3)
            
            # Add vertical lines for mean values
            axes[3, i].axvline(gt_valid.mean(), color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {gt_valid.mean():.2f}')
            axes[4, i].axvline(pred_valid.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {pred_valid.mean():.2f}')
            axes[3, i].legend(fontsize=8)
            axes[4, i].legend(fontsize=8)
        else:
            axes[4, i].text(0.5, 0.5, 'No valid data', ha='center', va='center')
            axes[4, i].set_title('Pred Distribution')
    
    plt.suptitle(f'Epoch {epoch} - Depth Predictions with Error Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


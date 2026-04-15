"""Visualization utilities."""

import os
import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_gt_rgb(dataset_dir, scene, sample_idx, depth_type, h=256, w=512):
    """Load ground-truth RGB image for visualization."""
    if depth_type == 'erp':
        rgb_path = os.path.join(dataset_dir, scene, 'erp_rgb', f'erp_{sample_idx}.png')
    else:
        rgb_path = os.path.join(dataset_dir, scene, 'pinhole_rgb', f'pinhole_{sample_idx}.png')
    if os.path.exists(rgb_path):
        img = Image.open(rgb_path).convert('RGB').resize((w, h), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0
    return None


def save_batch_visualization(pred_depths, gt_depths, save_path, epoch, num_samples=4):
    """Save batch of depth predictions with error maps and histograms."""
    if isinstance(pred_depths, torch.Tensor):
        pred_depths = pred_depths.detach().cpu().numpy()
    if isinstance(gt_depths, torch.Tensor):
        gt_depths = gt_depths.detach().cpu().numpy()

    batch_size = min(pred_depths.shape[0], num_samples)
    num_rows = 5

    fig, axes = plt.subplots(num_rows, batch_size, figsize=(5 * batch_size, 4.4 * num_rows))
    if batch_size == 1:
        axes = axes.reshape(num_rows, 1)

    for i in range(batch_size):
        pred = pred_depths[i]
        gt = gt_depths[i]
        if pred.ndim == 3:
            pred = pred[0]
        if gt.ndim == 3:
            gt = gt[0]

        valid_mask = gt > 0
        vmin = gt[valid_mask].min() if valid_mask.any() else 0
        vmax = gt[valid_mask].max() if valid_mask.any() else 1

        im1 = axes[0, i].imshow(gt, cmap='plasma', vmin=0, vmax=vmax)
        axes[0, i].set_title(f'GT Sample {i + 1}'); axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i])

        im2 = axes[1, i].imshow(pred, cmap='plasma', vmin=0, vmax=vmax)
        axes[1, i].set_title(f'Pred Sample {i + 1}'); axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i])

        error_map = np.abs(pred - gt)
        error_map[~valid_mask] = 0
        if valid_mask.any():
            error_valid = error_map[valid_mask]
            mean_error = error_valid.mean()
            max_error = error_valid.max()
            error_vmax = np.percentile(error_valid, 95) if len(error_valid) > 0 else 1
        else:
            mean_error, max_error, error_vmax = 0, 0, 1

        im3 = axes[2, i].imshow(error_map, cmap='hot', vmin=0, vmax=error_vmax)
        axes[2, i].set_title(f'Error (MAE:{mean_error:.2f}, Max:{max_error:.2f})')
        axes[2, i].axis('off')
        plt.colorbar(im3, ax=axes[2, i])

        if valid_mask.any():
            gt_valid = gt[valid_mask].flatten()
            pred_valid = pred[valid_mask].flatten()
            n_bins = 100
            bin_range = (vmin, vmax)

            axes[3, i].hist(gt_valid, bins=n_bins, range=bin_range, color='blue', alpha=0.7,
                            edgecolor='black', linewidth=0.5)
            axes[3, i].set_title(f'GT Dist (u:{gt_valid.mean():.2f}, s:{gt_valid.std():.2f})')
            axes[3, i].set_xlabel('Depth (m)'); axes[3, i].grid(True, alpha=0.3)

            axes[4, i].hist(pred_valid, bins=n_bins, range=bin_range, color='red', alpha=0.7,
                            edgecolor='black', linewidth=0.5)
            axes[4, i].set_title(f'Pred Dist (u:{pred_valid.mean():.2f}, s:{pred_valid.std():.2f})')
            axes[4, i].set_xlabel('Depth (m)'); axes[4, i].grid(True, alpha=0.3)
        else:
            axes[3, i].text(0.5, 0.5, 'No valid data', ha='center', va='center')
            axes[4, i].text(0.5, 0.5, 'No valid data', ha='center', va='center')

    plt.suptitle(f'Epoch {epoch} - Depth Predictions with Error Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_depth_image(depth, save_path, vmin=0, vmax=None, cmap='plasma', title=None):
    """Save a single depth map as a clean image with colorbar."""
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if depth.ndim == 3:
        depth = depth[0]
    if vmax is None:
        valid = depth[depth > 0]
        vmax = valid.max() if len(valid) > 0 else 1.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(depth, cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title, fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_test_visualizations(pred_depth, gt_depth, save_dir, sample_name, epoch):
    """Save merged (5-row) + separate GT/Pred depth images.

    Outputs:
        {save_dir}/merged/{sample_name}.png   -- full 5-row visualization
        {save_dir}/gt_depth/{sample_name}.png  -- GT depth only
        {save_dir}/pred_depth/{sample_name}.png -- predicted depth only
    """
    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.detach().cpu().numpy()
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.detach().cpu().numpy()
    if pred_depth.ndim == 4:
        pred_depth = pred_depth[0]
    if gt_depth.ndim == 4:
        gt_depth = gt_depth[0]
    if pred_depth.ndim == 3:
        pred_depth = pred_depth[0]
    if gt_depth.ndim == 3:
        gt_depth = gt_depth[0]

    valid_mask = gt_depth > 0
    vmax = gt_depth[valid_mask].max() if valid_mask.any() else 1.0

    # ── Merged (5-row) ──
    merged_dir = os.path.join(save_dir, 'merged')
    os.makedirs(merged_dir, exist_ok=True)
    merged_path = os.path.join(merged_dir, f'{sample_name}.png')
    # reuse save_batch_visualization with single sample
    save_batch_visualization(
        pred_depth[np.newaxis, np.newaxis],
        gt_depth[np.newaxis, np.newaxis],
        merged_path, epoch, num_samples=1)

    # ── Separate GT depth ──
    gt_dir = os.path.join(save_dir, 'gt_depth')
    os.makedirs(gt_dir, exist_ok=True)
    save_depth_image(gt_depth, os.path.join(gt_dir, f'{sample_name}.png'),
                     vmax=vmax, title='GT Depth')

    # ── Separate Pred depth ──
    pred_dir = os.path.join(save_dir, 'pred_depth')
    os.makedirs(pred_dir, exist_ok=True)
    save_depth_image(pred_depth, os.path.join(pred_dir, f'{sample_name}.png'),
                     vmax=vmax, title='Pred Depth')

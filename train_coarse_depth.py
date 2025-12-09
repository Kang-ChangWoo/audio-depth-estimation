"""
Training script for Coarse Depth Classification Model.

Trains a model to predict coarse/sparse depth maps using classification.
Each pixel is classified into one of N bins.

Features:
- Supports different sparse depth preprocessing methods (downup, superpixel, etc.)
- Multiple binning strategies (linear, log, SID)
- Soft cross-entropy with label smoothing
- Mixed classification + regression loss

Usage:
    python train_coarse_depth.py --sparse_method downup_015 --n_bins 128

Example sparse_method values:
    - downup_015 (downup with scale=0.15)
    - downup_005 (downup with scale=0.05)
    - superpixel_100 (superpixel with n=100)
    - quantized_32 (quantized with 32 levels)
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader.SparseDepth_Dataset import SparseDepthDataset, BinnedDepthDataset
from models.coarse_depth_model import (
    define_coarse_depth_model,
    CoarseDepthLoss,
    CoarseWithOffsetModel,
    CoarseOffsetLoss,
    DualRegressionModel,
    DualRegressionLoss,
)
from utils_criterion import compute_errors
from utils_visualization import save_batch_visualization
from config_loader import load_config

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import matplotlib.pyplot as plt


def save_hybrid_visualization(gt_batch, coarse_batch, offset_batch, final_batch, 
                               save_path, epoch, num_samples=4):
    """
    Visualization for hybrid/dual_reg model showing:
    - Column 1: Sparse Depth Target (GT)
    - Column 2: Coarse Depth (predicted - before offset)
    - Column 3: Offset (positive=red, negative=blue)
    - Column 4: Final Depth (Coarse + Offset)
    - Column 5: Difference (Final - GT)
    
    Layout: 5 columns x num_samples rows
    """
    num_samples = min(num_samples, gt_batch.shape[0])
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Get vmin/vmax for consistent colorbar (depth values)
    gt_np = gt_batch.cpu().numpy()
    vmin = gt_np[gt_np > 0].min() if (gt_np > 0).any() else 0
    vmax = gt_np.max()
    
    # Offset can be negative, use symmetric colorbar
    offset_np = offset_batch.cpu().numpy()
    offset_max = max(abs(offset_np.min()), abs(offset_np.max()), 0.1)
    
    for i in range(num_samples):
        # Column 1: Sparse Depth Target (GT)
        gt = gt_batch[i, 0].cpu().numpy()
        ax = axes[i, 0]
        im = ax.imshow(gt, cmap='plasma', vmin=vmin, vmax=vmax)
        ax.set_title('Sparse Target (GT)' if i == 0 else '')
        ax.axis('off')
        if i == 0:
            plt.colorbar(im, ax=ax, fraction=0.046, label='m')
        
        # Column 2: Coarse Depth (predicted)
        coarse = coarse_batch[i, 0].cpu().numpy()
        ax = axes[i, 1]
        im = ax.imshow(coarse, cmap='plasma', vmin=vmin, vmax=vmax)
        ax.set_title('Coarse Pred' if i == 0 else '')
        ax.axis('off')
        if i == 0:
            plt.colorbar(im, ax=ax, fraction=0.046, label='m')
        
        # Column 3: Offset (diverging colormap)
        offset = offset_batch[i, 0].cpu().numpy()
        ax = axes[i, 2]
        im = ax.imshow(offset, cmap='RdBu_r', vmin=-offset_max, vmax=offset_max)
        ax.set_title(f'Offset [{offset.min():.2f}, {offset.max():.2f}]' if i == 0 else f'[{offset.min():.2f}, {offset.max():.2f}]')
        ax.axis('off')
        if i == 0:
            plt.colorbar(im, ax=ax, fraction=0.046, label='m')
        
        # Column 4: Final Depth
        final = final_batch[i, 0].cpu().numpy()
        ax = axes[i, 3]
        im = ax.imshow(final, cmap='plasma', vmin=vmin, vmax=vmax)
        ax.set_title('Final (Coarse+Off)' if i == 0 else '')
        ax.axis('off')
        if i == 0:
            plt.colorbar(im, ax=ax, fraction=0.046, label='m')
        
        # Column 5: Difference (Final - GT)
        diff = final - gt
        diff_max = max(abs(diff.min()), abs(diff.max()), 0.1)
        ax = axes[i, 4]
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        ax.set_title(f'Error [{diff.min():.2f}, {diff.max():.2f}]' if i == 0 else f'[{diff.min():.2f}, {diff.max():.2f}]')
        ax.axis('off')
        if i == 0:
            plt.colorbar(im, ax=ax, fraction=0.046, label='m')
    
    plt.suptitle(f'Epoch {epoch} | Sparse Target vs Coarse vs Final', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Coarse Depth Classification Model')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='batvisionv2',
                        choices=['batvisionv1', 'batvisionv2'])
    parser.add_argument('--sparse_method', type=str, default='downup_015',
                        help='Sparse depth method (e.g., downup_015, superpixel_100)')
    
    # Binning
    parser.add_argument('--n_bins', type=int, default=128,
                        help='Number of depth bins for classification')
    parser.add_argument('--bin_mode', type=str, default='linear',
                        choices=['linear', 'log', 'sid'],
                        help='Binning strategy')
    parser.add_argument('--sid_alpha', type=float, default=0.6,
                        help='Alpha for SID binning')
    
    # Model
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'lite', 'hybrid', 'dual_reg'],
                        help='Model architecture (hybrid=CE+offset, dual_reg=regression+offset)')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel count')
    
    # Offset regression (for hybrid/dual_reg model)
    parser.add_argument('--offset_reg_weight', type=float, default=0.01,
                        help='Offset regularization weight (keep offsets small)')
    parser.add_argument('--coarse_weight', type=float, default=1.0,
                        help='Coarse loss weight (for dual_reg)')
    parser.add_argument('--final_weight', type=float, default=1.0,
                        help='Final loss weight (for dual_reg)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['Adam', 'AdamW', 'SGD'])
    
    # Loss weights
    parser.add_argument('--ce_weight', type=float, default=1.0,
                        help='Classification loss weight')
    parser.add_argument('--regression_weight', type=float, default=0.5,
                        help='Depth regression loss weight')
    parser.add_argument('--use_focal', action='store_true',
                        help='Use focal loss instead of CE')
    parser.add_argument('--soft_ce_sigma', type=float, default=2.0,
                        help='Sigma for soft cross-entropy')
    
    # Validation
    parser.add_argument('--validation', type=lambda x: str(x).lower() == 'true',
                        default=True)
    parser.add_argument('--validation_iter', type=int, default=2)
    
    # Experiment
    parser.add_argument('--experiment_name', type=str, default='exp1')
    parser.add_argument('--checkpoints', type=int, default=None,
                        help='Resume from checkpoint epoch')
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='coarse-depth')
    parser.add_argument('--wandb_entity', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(dataset_name=args.dataset, mode='train',
                      experiment_name=args.experiment_name)
    
    # Override config
    cfg.mode.batch_size = args.batch_size
    cfg.mode.learning_rate = args.learning_rate
    cfg.mode.epochs = args.epochs
    cfg.mode.validation = args.validation
    cfg.mode.validation_iter = args.validation_iter
    
    print("=" * 60)
    print("Coarse Depth Classification Training")
    print("=" * 60)
    print(f"Sparse method: {args.sparse_method}")
    print(f"Number of bins: {args.n_bins}")
    print(f"Bin mode: {args.bin_mode}")
    print(f"Model type: {args.model_type}")
    print(f"Base channels: {args.base_channels}")
    
    # GPU setup
    if torch.cuda.is_available():
        n_GPU = torch.cuda.device_count()
        max_gpus = min(n_GPU, 4)
        gpu_ids = list(range(max_gpus))
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"Using {len(gpu_ids)} GPU(s)")
    else:
        gpu_ids = []
        device = torch.device('cpu')
        print("Using CPU")
    
    # Experiment name
    experiment_name = f"coarse_{args.sparse_method}_{args.bin_mode}{args.n_bins}_{args.model_type}_{args.experiment_name}"
    print(f"Experiment: {experiment_name}")
    
    # Dataset
    print("\nLoading dataset...")
    train_set = BinnedDepthDataset(
        cfg=cfg,
        annotation_file=cfg.dataset.annotation_file_train,
        sparse_depth_method=args.sparse_method,
        n_bins=args.n_bins,
        bin_mode=args.bin_mode,
        sid_alpha=args.sid_alpha,
        use_original_depth=False,
    )
    
    if cfg.mode.validation:
        val_set = BinnedDepthDataset(
            cfg=cfg,
            annotation_file=cfg.dataset.annotation_file_val,
            sparse_depth_method=args.sparse_method,
            n_bins=args.n_bins,
            bin_mode=args.bin_mode,
            sid_alpha=args.sid_alpha,
            use_original_depth=False,
        )
    
    print(f"Train: {len(train_set)} samples")
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=cfg.mode.num_threads,
        pin_memory=True,
    )
    
    if cfg.mode.validation:
        print(f"Val: {len(val_set)} samples")
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=cfg.mode.num_threads,
            pin_memory=True,
        )
    
    # Model
    is_hybrid = args.model_type == 'hybrid'
    is_dual_reg = args.model_type == 'dual_reg'
    
    if is_dual_reg:
        # Pure regression model: coarse regression + offset regression
        model = DualRegressionModel(
            input_channels=2,
            base_channels=args.base_channels,
            output_size=cfg.dataset.images_size,
        )
        if len(gpu_ids) > 0:
            model = model.to(device)
            if len(gpu_ids) > 1:
                model = torch.nn.DataParallel(model, gpu_ids)
    elif is_hybrid:
        # Hybrid model: coarse classification + offset regression
        model = CoarseWithOffsetModel(
            input_channels=2,
            n_bins=args.n_bins,
            base_channels=args.base_channels,
            output_size=cfg.dataset.images_size,
        )
        if len(gpu_ids) > 0:
            model = model.to(device)
            if len(gpu_ids) > 1:
                model = torch.nn.DataParallel(model, gpu_ids)
    else:
        model = define_coarse_depth_model(
            model_type=args.model_type,
            input_channels=2,
            n_bins=args.n_bins,
            base_channels=args.base_channels,
            output_size=cfg.dataset.images_size,
            gpu_ids=gpu_ids,
        )
    
    # Set bin centers for depth reconstruction (not needed for dual_reg)
    model_unwrapped = model.module if hasattr(model, 'module') else model
    if not is_dual_reg:
        bin_centers = train_set.bin_centers.to(device)
        if cfg.dataset.depth_norm:
            bin_centers = bin_centers / cfg.dataset.max_depth  # Normalize
        model_unwrapped.set_bin_centers(bin_centers)
    
    print(f"Parameters: {model_unwrapped.get_num_params():,}")
    if is_hybrid:
        print(f"  - Hybrid model (CE + offset)")
    elif is_dual_reg:
        print(f"  - Dual regression model (coarse reg + offset reg)")
    
    # Loss
    if is_dual_reg:
        criterion = DualRegressionLoss(
            coarse_weight=args.coarse_weight,
            final_weight=args.final_weight,
            offset_reg_weight=args.offset_reg_weight,
        )
    elif is_hybrid:
        criterion = CoarseOffsetLoss(
            ce_weight=args.ce_weight,
            regression_weight=args.regression_weight,
            offset_reg_weight=args.offset_reg_weight,
            regression_loss='l1',
            label_smoothing=0.1,
        )
    else:
        criterion = CoarseDepthLoss(
            n_bins=args.n_bins,
            ce_weight=args.ce_weight,
            regression_weight=args.regression_weight,
            use_focal=args.use_focal,
            use_soft_ce=not args.use_focal,
            soft_ce_sigma=args.soft_ce_sigma,
        )
    
    # Optimizer
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Wandb
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment_name,
            config={
                'model_type': args.model_type,
                'sparse_method': args.sparse_method,
                'n_bins': args.n_bins,
                'bin_mode': args.bin_mode,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'ce_weight': args.ce_weight,
                'regression_weight': args.regression_weight,
            },
        )
    
    # Directories
    log_dir = f"./logs/{experiment_name}/"
    results_dir = f"./results/{experiment_name}/"
    checkpoint_dir = f"./checkpoints/{experiment_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Resume
    if args.checkpoints is not None:
        checkpoint_path = f'{checkpoint_dir}/checkpoint_{args.checkpoints}.pth'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {args.checkpoints}")
    else:
        start_epoch = 1
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        
        losses_train = {'total': [], 'coarse': [], 'final': [], 'offset_reg': [], 'ce': [], 'regression': []}
        model.train()
        
        for i, batch_data in enumerate(train_loader):
            # Unpack batch
            audio = batch_data[0].to(device)
            target_bins = batch_data[1].to(device)
            target_depth = batch_data[2].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            if is_dual_reg:
                # Dual regression returns: coarse_depth, offset, final_depth
                coarse_depth, offset, pred_depth = model(audio)
                
                # Loss
                loss, loss_dict = criterion(
                    coarse_depth, offset, pred_depth, target_depth
                )
                losses_train['coarse'].append(loss_dict['coarse'].item())
                losses_train['final'].append(loss_dict['final'].item())
                losses_train['offset_reg'].append(loss_dict['offset_reg'].item())
            elif is_hybrid:
                # Hybrid model returns: logits, coarse_depth, offset, final_depth
                logits, coarse_depth, offset, pred_depth = model(audio)
                
                # Loss
                loss, loss_dict = criterion(
                    logits, coarse_depth, offset, pred_depth,
                    target_depth, target_bins,
                )
                losses_train['ce'].append(loss_dict['ce'].item())
                losses_train['regression'].append(loss_dict['regression'].item())
                losses_train['offset_reg'].append(loss_dict['offset_reg'].item())
            else:
                logits, pred_depth = model(audio)
                
                # Loss
                valid_mask = target_depth > 0
                loss_dict = criterion(
                    logits, pred_depth, 
                    target_bins, target_depth,
                    valid_mask=valid_mask,
                )
                loss = loss_dict['total']
                losses_train['ce'].append(loss_dict['ce'].item())
                losses_train['regression'].append(loss_dict['regression'].item())
            
            losses_train['total'].append(loss_dict['total'].item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Debug first batch
            if epoch == 1 and i == 0:
                print(f"\nDebug:")
                print(f"  Audio: {audio.shape}")
                print(f"  Pred depth: [{pred_depth.min():.4f}, {pred_depth.max():.4f}]")
                print(f"  Target depth: [{target_depth.min():.4f}, {target_depth.max():.4f}]")
                if is_dual_reg or is_hybrid:
                    print(f"  Coarse depth: [{coarse_depth.min():.4f}, {coarse_depth.max():.4f}]")
                    print(f"  Offset: [{offset.min():.4f}, {offset.max():.4f}]")
                if not is_dual_reg:
                    print(f"  Target bins: [{target_bins.min()}, {target_bins.max()}]")
        
        scheduler.step()
        
        epoch_time = time.time() - t0
        
        if is_dual_reg:
            print(f"Epoch {epoch}: total={np.mean(losses_train['total']):.4f}, "
                  f"coarse={np.mean(losses_train['coarse']):.4f}, "
                  f"final={np.mean(losses_train['final']):.4f}, "
                  f"off={np.mean(losses_train['offset_reg']):.4f}, "
                  f"time={epoch_time:.1f}s")
        elif is_hybrid and losses_train['offset_reg']:
            print(f"Epoch {epoch}: total={np.mean(losses_train['total']):.4f}, "
                  f"ce={np.mean(losses_train['ce']):.4f}, "
                  f"reg={np.mean(losses_train['regression']):.4f}, "
                  f"off={np.mean(losses_train['offset_reg']):.4f}, "
                  f"time={epoch_time:.1f}s")
        else:
            print(f"Epoch {epoch}: total={np.mean(losses_train['total']):.4f}, "
                  f"ce={np.mean(losses_train['ce']):.4f}, "
                  f"reg={np.mean(losses_train['regression']):.4f}, "
                  f"time={epoch_time:.1f}s")
        
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss_total': np.mean(losses_train['total']),
                'train/loss_ce': np.mean(losses_train['ce']),
                'train/loss_regression': np.mean(losses_train['regression']),
                'train/lr': scheduler.get_last_lr()[0],
            }, step=epoch)
        
        # Validation
        if cfg.mode.validation and epoch % cfg.mode.validation_iter == 0:
            model.eval()
            losses_val = []
            errors = []
            val_preds = []
            val_gts = []
            val_coarse = []  # For hybrid/dual_reg model
            val_offsets = []  # For hybrid/dual_reg model
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(val_loader):
                    audio_val = batch_data[0].to(device)
                    target_bins_val = batch_data[1].to(device)
                    target_depth_val = batch_data[2].to(device)
                    
                    if is_dual_reg:
                        coarse_depth_val, offset_val, pred_depth_val = model(audio_val)
                        loss_val, loss_dict_val = criterion(
                            coarse_depth_val, offset_val, pred_depth_val, target_depth_val
                        )
                    elif is_hybrid:
                        logits_val, coarse_depth_val, offset_val, pred_depth_val = model(audio_val)
                        loss_val, loss_dict_val = criterion(
                            logits_val, coarse_depth_val, offset_val, pred_depth_val,
                            target_depth_val, target_bins_val,
                        )
                    else:
                        logits_val, pred_depth_val = model(audio_val)
                        valid_mask_val = target_depth_val > 0
                        loss_dict_val = criterion(
                            logits_val, pred_depth_val,
                            target_bins_val, target_depth_val,
                            valid_mask=valid_mask_val,
                        )
                    
                    losses_val.append(loss_dict_val['total'].item())
                    
                    # Save first batch for visualization
                    if batch_idx == 0:
                        if cfg.dataset.depth_norm:
                            val_preds.append(pred_depth_val * cfg.dataset.max_depth)
                            val_gts.append(target_depth_val * cfg.dataset.max_depth)
                            if is_hybrid or is_dual_reg:
                                val_coarse.append(coarse_depth_val * cfg.dataset.max_depth)
                                val_offsets.append(offset_val * cfg.dataset.max_depth)
                        else:
                            val_preds.append(pred_depth_val)
                            val_gts.append(target_depth_val)
                            if is_hybrid or is_dual_reg:
                                val_coarse.append(coarse_depth_val)
                                val_offsets.append(offset_val)
                    
                    # Compute metrics
                    for idx in range(pred_depth_val.shape[0]):
                        gt_map = target_depth_val[idx].cpu().numpy()
                        pred_map = pred_depth_val[idx].cpu().numpy()
                        
                        if gt_map.ndim == 3:
                            gt_map = gt_map[0]
                        if pred_map.ndim == 3:
                            pred_map = pred_map[0]
                        
                        if cfg.dataset.depth_norm:
                            gt_map = gt_map * cfg.dataset.max_depth
                            pred_map = pred_map * cfg.dataset.max_depth
                        
                        pred_map = np.clip(pred_map, 1e-3, cfg.dataset.max_depth)
                        gt_map = np.maximum(gt_map, 0.0)
                        
                        error_metrics = compute_errors(gt_map, pred_map, min_depth_threshold=0.0)
                        errors.append(error_metrics)
            
            mean_errors = np.array(errors).mean(0)
            val_loss = np.mean(losses_val)
            abs_rel, rmse, delta1, delta2, delta3, log10, mae = mean_errors[:7]
            
            print(f'Val - Loss: {val_loss:.4f}, RMSE: {rmse:.3f}, ABS_REL: {abs_rel:.3f}, Delta1: {delta1:.3f}')
            
            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'val/loss': val_loss,
                    'val/abs_rel': abs_rel,
                    'val/rmse': rmse,
                    'val/delta1': delta1,
                    'val/delta2': delta2,
                    'val/delta3': delta3,
                }, step=epoch)
            
            # Visualization
            if len(val_preds) > 0:
                pred_batch = torch.cat(val_preds, dim=0)
                gt_batch = torch.cat(val_gts, dim=0)
                vis_path = os.path.join(results_dir, f'epoch_{epoch:04d}_validation.png')
                
                if (is_hybrid or is_dual_reg) and len(val_coarse) > 0:
                    # Custom visualization for hybrid/dual_reg model: GT, Coarse, Offset, Final
                    coarse_batch = torch.cat(val_coarse, dim=0)
                    offset_batch = torch.cat(val_offsets, dim=0)
                    save_hybrid_visualization(
                        gt_batch, coarse_batch, offset_batch, pred_batch,
                        vis_path, epoch, num_samples=min(4, pred_batch.shape[0])
                    )
                else:
                    save_batch_visualization(pred_batch, gt_batch, vis_path, epoch,
                                            num_samples=min(4, pred_batch.shape[0]))
                print(f'Saved: {vis_path}')
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = f'{checkpoint_dir}/best.pth'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'bin_centers': train_set.bin_centers,
                    'bin_edges': train_set.bin_edges,
                }, best_path)
                print(f'New best! (val_loss={val_loss:.4f})')
        
        # Checkpoint
        if epoch % cfg.mode.saving_checkpoints == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'bin_centers': train_set.bin_centers,
                'bin_edges': train_set.bin_edges,
            }
            torch.save(state, f'{checkpoint_dir}/checkpoint_{epoch}.pth')
    
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()


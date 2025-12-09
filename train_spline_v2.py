"""
Training script for SplineGS-inspired Audio-to-Depth model v2.

Key improvements over v1:
1. Higher rank (16) and control points (16) by default
2. Per-rank audio conditioning to prevent mode collapse
3. Diversity loss to encourage different ranks to learn different patterns
4. Edge-aware loss for better depth discontinuities
5. Reduced smoothness regularization
6. Residual CNN for sample-specific details

Usage:
    python train_spline_v2.py --dataset batvisionv2 --experiment_name spline_v2_exp1
    
    # With custom settings
    python train_spline_v2.py --dataset batvisionv2 --rank 16 --ctrl_x 16 \\
        --diversity_weight 0.01 --edge_weight 0.1 --smooth_weight 0.001
"""

from dataloader.BatvisionV1_Dataset import BatvisionV1Dataset
from dataloader.BatvisionV2_Dataset import BatvisionV2Dataset

from models.spline_depth_model_v2 import (
    define_spline_depth_v2,
    AudioSplineDepthV2,
    DiversityLoss,
    EdgeAwareLoss,
    RankVarianceLoss,
    SplineSmoothLossV2,
    BoundaryLoss,
)

from utils_criterion import compute_errors
from utils_visualization import save_batch_visualization
from utils_loss import SIlogLoss

import time
import os 
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from config_loader import load_config

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description='Train SplineDepth v2 model')
    
    # Dataset and experiment
    parser.add_argument('--dataset', type=str, default='batvisionv2', 
                        choices=['batvisionv1', 'batvisionv2'])
    parser.add_argument('--experiment_name', type=str, default='spline_v2_exp1')
    parser.add_argument('--checkpoints', type=int, default=None,
                        help='Checkpoint epoch to resume from')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (smaller due to larger model)')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate (lower for stability)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    
    # Model architecture - INCREASED defaults
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--rank', type=int, default=16,
                        help='Number of ranks (INCREASED from 8)')
    parser.add_argument('--ctrl_x', type=int, default=16,
                        help='Control points in x (INCREASED from 8)')
    parser.add_argument('--ctrl_y', type=int, default=16,
                        help='Control points in y (INCREASED from 8)')
    parser.add_argument('--use_residual', type=lambda x: str(x).lower() == 'true',
                        default=True, help='Use residual CNN')
    parser.add_argument('--depth_activation', type=str, default='none',
                        choices=['none', 'sigmoid', 'softplus'])
    
    # Loss weights - REBALANCED
    parser.add_argument('--smooth_weight', type=float, default=0.001,
                        help='Smoothness regularization (REDUCED from 0.01)')
    parser.add_argument('--diversity_weight', type=float, default=0.01,
                        help='Diversity loss weight (NEW)')
    parser.add_argument('--edge_weight', type=float, default=0.1,
                        help='Edge-aware loss weight (NEW)')
    parser.add_argument('--rank_var_weight', type=float, default=0.001,
                        help='Rank variance loss weight (NEW)')
    parser.add_argument('--boundary_weight', type=float, default=0.1,
                        help='Boundary smoothness loss weight (FIXES border artifacts)')
    
    # Main loss
    parser.add_argument('--criterion', type=str, default='L1', 
                        choices=['L1', 'SIlog', 'Combined'])
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--silog_lambda', type=float, default=0.5)
    parser.add_argument('--l1_weight', type=float, default=0.5)
    parser.add_argument('--silog_weight', type=float, default=0.5)
    
    # Validation
    parser.add_argument('--validation', type=lambda x: str(x).lower() == 'true', 
                        default=True)
    parser.add_argument('--validation_iter', type=int, default=2)
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='batvision-spline-v2')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'])
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(dataset_name=args.dataset, mode='train', 
                      experiment_name=args.experiment_name)
    
    cfg.mode.batch_size = args.batch_size
    cfg.mode.learning_rate = args.learning_rate
    cfg.mode.epochs = args.epochs
    cfg.mode.criterion = args.criterion
    cfg.mode.optimizer = args.optimizer
    cfg.mode.validation = args.validation
    cfg.mode.validation_iter = args.validation_iter
    
    print(f"Training SplineDepth V2 with rank={args.rank}, ctrl={args.ctrl_x}")
    print(f"Loss weights: smooth={args.smooth_weight}, diversity={args.diversity_weight}, edge={args.edge_weight}")
    
    # ------------ GPU config ------------
    if torch.cuda.is_available():
        n_GPU = torch.cuda.device_count()
        max_gpus = min(n_GPU, 4)
        gpu_ids = list(range(max_gpus))
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"Using {len(gpu_ids)} GPU(s): {gpu_ids}")
    else:
        gpu_ids = []
        device = torch.device('cpu')
        print("WARNING: Using CPU")

    batch_size = cfg.mode.batch_size
    
    # ------------ Experiment name -----------
    experiment_name = f"spline_v2_{cfg.dataset.name}_R{args.rank}_C{args.ctrl_x}_BS{batch_size}_{args.experiment_name}"
    print(f"Experiment: {experiment_name}")
    
    # ------------ Dataset -----------
    if cfg.dataset.name == 'batvisionv1':
        train_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_train)
        if cfg.mode.validation:
            val_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_val)
    else:
        train_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_train) 
        if cfg.mode.validation:
            val_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_val) 

    print(f'Train Dataset: {len(train_set)} samples')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=cfg.mode.shuffle, 
                              num_workers=cfg.mode.num_threads) 

    if cfg.mode.validation:
        print(f'Validation Dataset: {len(val_set)} samples')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                                num_workers=cfg.mode.num_threads)

    # ---------- Model ----------
    model = define_spline_depth_v2(
        cfg=cfg,
        input_nc=2,
        output_h=cfg.dataset.images_size,
        output_w=cfg.dataset.images_size,
        latent_dim=args.latent_dim,
        rank=args.rank,
        ctrl_x=args.ctrl_x,
        ctrl_y=args.ctrl_y,
        use_residual=args.use_residual,
        depth_activation=args.depth_activation,
        init_type='kaiming',
        init_gain=0.02,
        gpu_ids=gpu_ids,
    )
    
    model_unwrapped = model.module if hasattr(model, 'module') else model
    
    print(f'Model: SplineDepth V2')
    print(f'  - Parameters: {model_unwrapped.get_num_params():,}')
    print(f'  - Spline complexity: {model_unwrapped.get_spline_complexity()}')
   
    # ---------- Losses ----------
    max_depth = cfg.dataset.max_depth if cfg.dataset.max_depth else 30.0
    
    # Main depth loss
    if cfg.mode.criterion == 'L1':
        criterion = nn.L1Loss().to(device)
        l1_criterion = None
        silog_criterion = None
    elif cfg.mode.criterion == 'SIlog':
        criterion = SIlogLoss(lambda_scale=args.silog_lambda).to(device)
        l1_criterion = None
        silog_criterion = None
    else:  # Combined
        l1_criterion = nn.L1Loss().to(device)
        silog_criterion = SIlogLoss(lambda_scale=args.silog_lambda).to(device)
        criterion = None
    
    # Spline-specific losses (REBALANCED)
    smooth_loss_fn = SplineSmoothLossV2(weight=args.smooth_weight).to(device)
    diversity_loss_fn = DiversityLoss(weight=args.diversity_weight).to(device)
    edge_loss_fn = EdgeAwareLoss(weight=args.edge_weight).to(device)
    rank_var_loss_fn = RankVarianceLoss(weight=args.rank_var_weight).to(device)
    boundary_loss_fn = BoundaryLoss(weight=args.boundary_weight).to(device)
    
    print(f"Loss functions:")
    print(f"  - Main: {cfg.mode.criterion}")
    print(f"  - Smooth: weight={args.smooth_weight}")
    print(f"  - Diversity: weight={args.diversity_weight}")
    print(f"  - Edge: weight={args.edge_weight}")
    print(f"  - Rank variance: weight={args.rank_var_weight}")
    print(f"  - Boundary: weight={args.boundary_weight}")
    
    # Optimizer
    learning_rate = cfg.mode.learning_rate
    if cfg.mode.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif cfg.mode.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # ---------- Wandb ----------
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment_name,
            mode=args.wandb_mode,
            config={
                'model': 'spline_depth_v2',
                'dataset': cfg.dataset.name,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'optimizer': cfg.mode.optimizer,
                'criterion': cfg.mode.criterion,
                'epochs': cfg.mode.epochs,
                'rank': args.rank,
                'ctrl_x': args.ctrl_x,
                'ctrl_y': args.ctrl_y,
                'use_residual': args.use_residual,
                'smooth_weight': args.smooth_weight,
                'diversity_weight': args.diversity_weight,
                'edge_weight': args.edge_weight,
                'rank_var_weight': args.rank_var_weight,
                'boundary_weight': args.boundary_weight,
                'num_params': model_unwrapped.get_num_params(),
            },
            tags=[cfg.dataset.name, 'spline_v2', f'rank{args.rank}']
        )
    
    # Directories
    log_dir = f"./logs/{experiment_name}/"
    results_dir = f"./results/{experiment_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(f"Model: SplineDepth V2\n")
        f.write(f"Dataset: {cfg.dataset.name}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Rank: {args.rank}\n")
        f.write(f"Control points: {args.ctrl_x} x {args.ctrl_y}\n")
        f.write(f"Use residual: {args.use_residual}\n")
        f.write(f"\nLoss weights:\n")
        f.write(f"  Smooth: {args.smooth_weight}\n")
        f.write(f"  Diversity: {args.diversity_weight}\n")
        f.write(f"  Edge: {args.edge_weight}\n")
        f.write(f"  Rank var: {args.rank_var_weight}\n")
        f.write(f"\nTotal parameters: {model_unwrapped.get_num_params():,}\n")

    # Resume from checkpoint
    if args.checkpoints is not None:
        checkpoint_path = f'./checkpoints/{experiment_name}/checkpoint_{args.checkpoints}.pth'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        checkpoint_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {args.checkpoints}")
    else:
        checkpoint_epoch = 1

    nb_epochs = cfg.mode.epochs
    best_val_loss = float('inf')
    
    # ---------- Training Loop ----------
    for epoch in range(checkpoint_epoch, nb_epochs + 1):
        t0 = time.time()
        
        # Loss accumulators
        losses = {
            'total': [], 'depth': [], 'smooth': [], 
            'diversity': [], 'edge': [], 'rank_var': [], 'boundary': []
        }
        losses_val = []

        model.train()

        for i, (audio, gtdepth) in enumerate(train_loader):
            audio = audio.to(device)
            gtdepth = gtdepth.to(device)

            optimizer.zero_grad()

            # Forward with info
            if hasattr(model, 'module'):
                depth_pred, info = model.module.forward_with_info(audio)
            else:
                depth_pred, info = model.forward_with_info(audio)
            
            Px = info['Px']
            Py = info['Py']
            depth_components = info['depth_components']
            rank_weights = info['rank_weights']
            
            # Valid mask
            valid_mask = gtdepth > 0
            
            # === Main depth loss ===
            if cfg.dataset.depth_norm:
                pred_denorm = depth_pred[valid_mask] * cfg.dataset.max_depth
                gt_denorm = gtdepth[valid_mask] * cfg.dataset.max_depth
                
                if cfg.mode.criterion == 'Combined':
                    depth_loss = args.l1_weight * l1_criterion(pred_denorm, gt_denorm) + \
                                 args.silog_weight * silog_criterion(pred_denorm, gt_denorm)
                else:
                    depth_loss = criterion(pred_denorm, gt_denorm)
            else:
                if cfg.mode.criterion == 'Combined':
                    depth_loss = args.l1_weight * l1_criterion(depth_pred[valid_mask], gtdepth[valid_mask]) + \
                                 args.silog_weight * silog_criterion(depth_pred[valid_mask], gtdepth[valid_mask])
                else:
                    depth_loss = criterion(depth_pred[valid_mask], gtdepth[valid_mask])
            
            # === Regularization losses ===
            smooth_loss = smooth_loss_fn(Px, Py)
            diversity_loss = diversity_loss_fn(depth_components)
            edge_loss = edge_loss_fn(depth_pred, gtdepth)
            rank_var_loss = rank_var_loss_fn(rank_weights)
            boundary_loss = boundary_loss_fn(depth_pred, gtdepth)
            
            # === Total loss ===
            total_loss = depth_loss + smooth_loss + diversity_loss + edge_loss + rank_var_loss + boundary_loss
            
            # Record losses
            losses['total'].append(total_loss.item())
            losses['depth'].append(depth_loss.item())
            losses['smooth'].append(smooth_loss.item())
            losses['diversity'].append(diversity_loss.item())
            losses['edge'].append(edge_loss.item())
            losses['rank_var'].append(rank_var_loss.item())
            losses['boundary'].append(boundary_loss.item())
            
            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Debug first batch
            if epoch == 1 and i == 0:
                print(f"\nDebug - First batch:")
                print(f"  Pred range: [{depth_pred.min():.4f}, {depth_pred.max():.4f}]")
                print(f"  GT range: [{gtdepth.min():.4f}, {gtdepth.max():.4f}]")
                print(f"  Rank weights: {rank_weights[:5].tolist()}")
                print(f"  Losses: depth={depth_loss.item():.4f}, div={diversity_loss.item():.4f}")
                if 'residual' in info:
                    print(f"  Residual range: [{info['residual'].min():.4f}, {info['residual'].max():.4f}]")

        scheduler.step()
        
        epoch_time = time.time() - t0
        
        # Log epoch stats
        log_str = f"Epoch {epoch}: "
        log_str += f"total={np.mean(losses['total']):.4f}, "
        log_str += f"depth={np.mean(losses['depth']):.4f}, "
        log_str += f"smooth={np.mean(losses['smooth']):.6f}, "
        log_str += f"div={np.mean(losses['diversity']):.6f}, "
        log_str += f"edge={np.mean(losses['edge']):.6f}, "
        log_str += f"boundary={np.mean(losses['boundary']):.6f}, "
        log_str += f"time={epoch_time:.1f}s"
        print(log_str)
        
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss_total': np.mean(losses['total']),
                'train/loss_depth': np.mean(losses['depth']),
                'train/loss_smooth': np.mean(losses['smooth']),
                'train/loss_diversity': np.mean(losses['diversity']),
                'train/loss_edge': np.mean(losses['edge']),
                'train/loss_rank_var': np.mean(losses['rank_var']),
                'train/loss_boundary': np.mean(losses['boundary']),
                'train/lr': scheduler.get_last_lr()[0],
            }, step=epoch)

        # ------- Validation ------------
        if cfg.mode.validation and epoch % cfg.mode.validation_iter == 0:
            model.eval()
            errors = []
            val_preds = []
            val_gts = []
            
            with torch.no_grad():
                for batch_idx, (audio_val, gtdepth_val) in enumerate(val_loader):
                    audio_val = audio_val.to(device)
                    gtdepth_val = gtdepth_val.to(device)

                    depth_pred_val = model(audio_val)
                    
                    valid_mask_val = gtdepth_val > 0
                    
                    if cfg.dataset.depth_norm:
                        pred_denorm = depth_pred_val[valid_mask_val] * cfg.dataset.max_depth
                        gt_denorm = gtdepth_val[valid_mask_val] * cfg.dataset.max_depth
                        
                        if cfg.mode.criterion == 'Combined':
                            loss_val = args.l1_weight * l1_criterion(pred_denorm, gt_denorm) + \
                                       args.silog_weight * silog_criterion(pred_denorm, gt_denorm)
                        else:
                            loss_val = criterion(pred_denorm, gt_denorm)
                    else:
                        if cfg.mode.criterion == 'Combined':
                            loss_val = args.l1_weight * l1_criterion(depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val]) + \
                                       args.silog_weight * silog_criterion(depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val])
                        else:
                            loss_val = criterion(depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val])
                    
                    losses_val.append(loss_val.item())

                    # Store for visualization
                    if batch_idx == 0:
                        if cfg.dataset.depth_norm:
                            val_preds.append(depth_pred_val * cfg.dataset.max_depth)
                            val_gts.append(gtdepth_val * cfg.dataset.max_depth)
                        else:
                            val_preds.append(depth_pred_val)
                            val_gts.append(gtdepth_val)

                    # Compute metrics
                    for idx in range(depth_pred_val.shape[0]):
                        gt_map = gtdepth_val[idx].cpu().numpy()
                        pred_map = depth_pred_val[idx].cpu().numpy()
                        
                        if gt_map.ndim == 3:
                            gt_map = gt_map[0]
                        if pred_map.ndim == 3:
                            pred_map = pred_map[0]
                        
                        if cfg.dataset.depth_norm:
                            gt_map = gt_map * cfg.dataset.max_depth
                            pred_map = pred_map * cfg.dataset.max_depth
                        
                        epsilon = 1e-3
                        pred_map = np.clip(pred_map, epsilon, cfg.dataset.max_depth)
                        gt_map = np.maximum(gt_map, 0.0)
                        
                        error_metrics = compute_errors(gt_map, pred_map, min_depth_threshold=0.0)
                        errors.append(error_metrics)

            mean_errors = np.array(errors).mean(0)
            val_loss = np.mean(losses_val)
            abs_rel, rmse, delta1, delta2, delta3, log10, mae = mean_errors[:7]
            
            print(f'Val - Loss: {val_loss:.4f}, RMSE: {rmse:.3f}, ABS_REL: {abs_rel:.3f}, Delta1: {delta1:.3f}')
            
            log_dict = {}
            if WANDB_AVAILABLE and args.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'val/loss': val_loss,
                    'val/abs_rel': abs_rel,
                    'val/rmse': rmse,
                    'val/log10': log10,
                    'val/delta1': delta1,
                    'val/delta2': delta2,
                    'val/delta3': delta3,
                    'val/mae': mae,
                }
            
            # Visualization
            if len(val_preds) > 0:
                pred_batch = torch.cat(val_preds, dim=0)
                gt_batch = torch.cat(val_gts, dim=0)
                vis_path = os.path.join(results_dir, f'epoch_{epoch:04d}_validation.png')
                save_batch_visualization(pred_batch, gt_batch, vis_path, epoch, 
                                         num_samples=min(4, pred_batch.shape[0]))
                print(f'Saved: {vis_path}')
                
                if WANDB_AVAILABLE and args.use_wandb:
                    log_dict['val/visualization'] = wandb.Image(vis_path)
            
            if WANDB_AVAILABLE and args.use_wandb and log_dict:
                wandb.log(log_dict, step=epoch)
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = f'./checkpoints/{experiment_name}/best.pth'
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, best_path)
                print(f'New best model saved (val_loss={val_loss:.4f})')

        # Checkpoint
        if epoch % cfg.mode.saving_checkpoints == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            path_check = f'./checkpoints/{experiment_name}/'
            os.makedirs(path_check, exist_ok=True)
            torch.save(state, f'{path_check}/checkpoint_{epoch}.pth')

    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()


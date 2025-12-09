"""
Training script for SplineGS-inspired Audio-to-Depth model.

This script trains the AudioSplineDepth model which uses low-rank x,y spline
decomposition for depth prediction from binaural audio.

Key differences from baseline UNet training:
1. Uses spline-based depth representation
2. Includes spline-specific regularization (smoothness, sparsity)
3. Supports coarse-to-fine training schedule

Usage:
    python train_spline.py --dataset batvisionv2 --experiment_name spline_exp1
    python train_spline.py --dataset batvisionv2 --rank 8 --ctrl_x 8 --multi_scale
"""

from dataloader.BatvisionV1_Dataset import BatvisionV1Dataset
from dataloader.BatvisionV2_Dataset import BatvisionV2Dataset

from models.spline_depth_model import (
    define_spline_depth,
    AudioSplineDepth,
    SplineSmoothLoss,
    SplineSparsityLoss,
)

from utils_criterion import compute_errors
from utils_visualization import save_batch_visualization
from utils_loss import SIlogLoss

import time
import os 
import numpy as np 
import math
import pickle

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
    parser = argparse.ArgumentParser(description='Train SplineDepth model on Batvision dataset')
    
    # Dataset and experiment
    parser.add_argument('--dataset', type=str, default='batvisionv2', 
                        choices=['batvisionv1', 'batvisionv2'],
                        help='Dataset to use')
    parser.add_argument('--experiment_name', type=str, default='spline_exp1',
                        help='Name of the experiment')
    parser.add_argument('--checkpoints', type=int, default=None,
                        help='Checkpoint epoch to resume from (None to start from scratch)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64, smaller than UNet due to more parameters)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    
    # Spline architecture parameters
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent representation dimension')
    parser.add_argument('--rank', type=int, default=8,
                        help='Number of ranks in low-rank decomposition')
    parser.add_argument('--ctrl_x', type=int, default=8,
                        help='Number of control points in x direction')
    parser.add_argument('--ctrl_y', type=int, default=8,
                        help='Number of control points in y direction')
    parser.add_argument('--encoder_type', type=str, default='standard',
                        choices=['standard', 'unet'],
                        help='Audio encoder type')
    parser.add_argument('--multi_scale', action='store_true', default=False,
                        help='Use multi-scale spline (coarse + fine)')
    parser.add_argument('--depth_activation', type=str, default='none',
                        choices=['none', 'sigmoid', 'softplus'],
                        help='Depth output activation')
    
    # Regularization
    parser.add_argument('--smooth_weight', type=float, default=0.01,
                        help='Spline smoothness regularization weight')
    parser.add_argument('--sparsity_weight', type=float, default=0.001,
                        help='Control point sparsity regularization weight')
    
    # Loss function
    parser.add_argument('--criterion', type=str, default='L1', 
                        choices=['L1', 'SIlog', 'Combined'],
                        help='Loss function')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['Adam', 'AdamW', 'SGD'],
                        help='Optimizer')
    parser.add_argument('--silog_lambda', type=float, default=0.5,
                        help='SIlog lambda parameter')
    parser.add_argument('--l1_weight', type=float, default=0.5,
                        help='L1 weight for combined loss')
    parser.add_argument('--silog_weight', type=float, default=0.5,
                        help='SIlog weight for combined loss')
    
    # Validation
    parser.add_argument('--validation', type=lambda x: (str(x).lower() == 'true'), 
                        default=True, help='Enable validation')
    parser.add_argument('--validation_iter', type=int, default=2,
                        help='Validation frequency: evaluate every N epochs')
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='batvision-spline-depth',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity/team name')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='W&B logging mode')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(dataset_name=args.dataset, mode='train', experiment_name=args.experiment_name)
    
    # Override config with args
    cfg.mode.batch_size = args.batch_size
    cfg.mode.learning_rate = args.learning_rate
    cfg.mode.epochs = args.epochs
    cfg.mode.criterion = args.criterion
    cfg.mode.optimizer = args.optimizer
    cfg.mode.validation = args.validation
    cfg.mode.validation_iter = args.validation_iter
    
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")
    print(f"Training SplineDepth model with rank={args.rank}, ctrl_x={args.ctrl_x}, ctrl_y={args.ctrl_y}")
    
    # ------------ GPU config ------------
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible}")
    
    if torch.cuda.is_available():
        n_GPU = torch.cuda.device_count()
        max_gpus = min(n_GPU, 4)
        gpu_ids = list(range(max_gpus))
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"{n_GPU} GPU(s) available, using {len(gpu_ids)} GPU(s): {gpu_ids}")
        print(f"Using device: {device}")
        if n_GPU > 0:
            print(f"GPU 0 name: {torch.cuda.get_device_name(0)}")
    else:
        n_GPU = 0
        gpu_ids = []
        device = torch.device('cpu')
        print("WARNING: CUDA not available, using CPU")

    batch_size = cfg.mode.batch_size
    
    # ------------ Create experiment name -----------
    experiment_name = f"spline_depth_{cfg.dataset.name}_R{args.rank}_C{args.ctrl_x}_BS{batch_size}_Lr{args.learning_rate}_{args.experiment_name}"
    if args.multi_scale:
        experiment_name = experiment_name.replace('spline_depth', 'spline_depth_ms')
    
    print(f"Experiment: {experiment_name}")
    
    # ------------ Create dataset -----------
    if cfg.dataset.name == 'batvisionv1':
        train_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_train)
        if cfg.mode.validation:
            val_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_val)
    elif cfg.dataset.name == 'batvisionv2':
        train_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_train) 
        if cfg.mode.validation:
            val_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_val) 
    else:
        raise Exception('Training can be done only on BV1 and BV2')

    print(f'Train Dataset of {len(train_set)} instances')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=cfg.mode.shuffle, 
                              num_workers=cfg.mode.num_threads) 

    if cfg.mode.validation:
        print(f'Validation Dataset of {len(val_set)} instances')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                                num_workers=cfg.mode.num_threads)

    # ---------- Load Model ----------
    model = define_spline_depth(
        cfg=cfg,
        input_nc=2,
        output_h=cfg.dataset.images_size,
        output_w=cfg.dataset.images_size,
        latent_dim=args.latent_dim,
        rank=args.rank,
        ctrl_x=args.ctrl_x,
        ctrl_y=args.ctrl_y,
        encoder_type=args.encoder_type,
        multi_scale=args.multi_scale,
        depth_activation=args.depth_activation,
        init_type='xavier',
        init_gain=0.02,
        gpu_ids=gpu_ids,
    )
    
    # Get actual model (handle DataParallel)
    model_unwrapped = model.module if hasattr(model, 'module') else model
    
    print(f'Model: SplineDepth')
    print(f'  - Parameters: {model_unwrapped.get_num_params():,}')
    print(f'  - Spline complexity: {model_unwrapped.get_spline_complexity()}')
    
    if len(gpu_ids) > 1:
        print(f'Using DataParallel on {len(gpu_ids)} GPUs: {gpu_ids}')
   
    # ---------- Criterion & Optimizers ----------
    max_depth = cfg.dataset.max_depth if cfg.dataset.max_depth else 30.0
    
    # Main loss function
    if cfg.mode.criterion == 'L1':
        criterion = nn.L1Loss().to(device)
        l1_criterion = None
        silog_criterion = None
        l1_weight = 0.0
        silog_weight = 0.0
        print(f"Using loss function: L1")
    elif cfg.mode.criterion == 'SIlog':
        lambda_scale = args.silog_lambda
        criterion = SIlogLoss(lambda_scale=lambda_scale).to(device)
        l1_criterion = None
        silog_criterion = None
        l1_weight = 0.0
        silog_weight = 0.0
        print(f"Using loss function: SIlog (lambda={lambda_scale})")
    elif cfg.mode.criterion == 'Combined':
        l1_weight = args.l1_weight
        silog_weight = args.silog_weight
        silog_lambda = args.silog_lambda
        l1_criterion = nn.L1Loss().to(device)
        silog_criterion = SIlogLoss(lambda_scale=silog_lambda).to(device)
        criterion = None
        print(f"Using loss function: Combined (L1={l1_weight}, SIlog={silog_weight})")
    
    # Spline regularization losses
    smooth_loss_fn = SplineSmoothLoss(weight=args.smooth_weight).to(device)
    sparsity_loss_fn = SplineSparsityLoss(weight=args.sparsity_weight).to(device)
    print(f"Spline regularization: smooth_weight={args.smooth_weight}, sparsity_weight={args.sparsity_weight}")
    
    # Optimizer
    learning_rate = cfg.mode.learning_rate
    if cfg.mode.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif cfg.mode.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif cfg.mode.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.mode.epochs, eta_min=1e-6)

    # ---------- Experiment Setup ----------
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment_name,
            mode=args.wandb_mode,
            config={
                'model': 'spline_depth',
                'dataset': cfg.dataset.name,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'optimizer': cfg.mode.optimizer,
                'criterion': cfg.mode.criterion,
                'epochs': cfg.mode.epochs,
                'rank': args.rank,
                'ctrl_x': args.ctrl_x,
                'ctrl_y': args.ctrl_y,
                'latent_dim': args.latent_dim,
                'encoder_type': args.encoder_type,
                'multi_scale': args.multi_scale,
                'depth_activation': args.depth_activation,
                'smooth_weight': args.smooth_weight,
                'sparsity_weight': args.sparsity_weight,
                'num_params': model_unwrapped.get_num_params(),
            },
            tags=[cfg.dataset.name, 'spline_depth', f'rank{args.rank}']
        )
        print(f"W&B initialized: project={args.wandb_project}, run={experiment_name}")
    
    # Create directories
    log_dir = "./logs/" + experiment_name + "/"
    results_dir = "./results/" + experiment_name + "/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save architecture info
    with open(os.path.join(log_dir, "architecture.txt"), "w") as f:
        f.write(f"Model: SplineDepth\n")
        f.write(f"Dataset: {cfg.dataset.name}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Optimizer: {cfg.mode.optimizer}\n")
        f.write(f"Criterion: {cfg.mode.criterion}\n\n")
        f.write(f"Spline Parameters:\n")
        f.write(f"  - Rank: {args.rank}\n")
        f.write(f"  - Control points (x): {args.ctrl_x}\n")
        f.write(f"  - Control points (y): {args.ctrl_y}\n")
        f.write(f"  - Latent dim: {args.latent_dim}\n")
        f.write(f"  - Encoder type: {args.encoder_type}\n")
        f.write(f"  - Multi-scale: {args.multi_scale}\n\n")
        f.write(f"Regularization:\n")
        f.write(f"  - Smooth weight: {args.smooth_weight}\n")
        f.write(f"  - Sparsity weight: {args.sparsity_weight}\n\n")
        f.write(f"Total parameters: {model_unwrapped.get_num_params():,}\n\n")
        f.write(str(model))

    # Load checkpoint if resuming
    if args.checkpoints is not None:
        load_epoch = args.checkpoints
        checkpoint_path = f'./checkpoints/{experiment_name}/checkpoint_{load_epoch}.pth'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        checkpoint_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from checkpoint epoch {load_epoch}")
    else:
        checkpoint_epoch = 1

    nb_epochs = cfg.mode.epochs

    for param_group in optimizer.param_groups:
        print(f"Learning rate: {param_group['lr']}")

    # ---------- Training Loop ----------
    train_iter = 0
    best_val_loss = float('inf')
    
    for epoch in range(checkpoint_epoch, nb_epochs + 1):
        t0 = time.time()
        
        batch_loss = []
        batch_loss_depth = []
        batch_loss_smooth = []
        batch_loss_sparse = []
        batch_loss_val = []

        # ------ Training ---------
        model.train()

        for i, (audio, gtdepth) in enumerate(train_loader):
            audio = audio.to(device)
            gtdepth = gtdepth.to(device)

            optimizer.zero_grad()

            # Forward pass with info (to get control points for regularization)
            if args.multi_scale:
                depth_pred, info = model_unwrapped.forward_with_info(audio) if not hasattr(model, 'module') else model.module.forward_with_info(audio)
                Px = torch.cat([info['Px_coarse'], info['Px_fine']], dim=1)
                Py = torch.cat([info['Py_coarse'], info['Py_fine']], dim=1)
            else:
                # Need to call forward_with_info on the unwrapped model
                if hasattr(model, 'module'):
                    depth_pred, info = model.module.forward_with_info(audio)
                else:
                    depth_pred, info = model.forward_with_info(audio)
                Px = info['Px']
                Py = info['Py']
            
            # Valid mask
            valid_mask = gtdepth > 0
            
            # Compute depth loss
            if cfg.dataset.depth_norm:
                depth_pred_denorm = depth_pred[valid_mask] * cfg.dataset.max_depth
                gtdepth_denorm = gtdepth[valid_mask] * cfg.dataset.max_depth
                
                if cfg.mode.criterion == 'Combined':
                    depth_loss = l1_weight * l1_criterion(depth_pred_denorm, gtdepth_denorm) + \
                                 silog_weight * silog_criterion(depth_pred_denorm, gtdepth_denorm)
                else:
                    depth_loss = criterion(depth_pred_denorm, gtdepth_denorm)
            else:
                if cfg.mode.criterion == 'Combined':
                    depth_loss = l1_weight * l1_criterion(depth_pred[valid_mask], gtdepth[valid_mask]) + \
                                 silog_weight * silog_criterion(depth_pred[valid_mask], gtdepth[valid_mask])
                else:
                    depth_loss = criterion(depth_pred[valid_mask], gtdepth[valid_mask])
            
            # Spline regularization
            smooth_loss = smooth_loss_fn(Px, Py)
            sparse_loss = sparsity_loss_fn(Px, Py)
            
            # Total loss
            loss = depth_loss + smooth_loss + sparse_loss
            
            batch_loss.append(loss.item())
            batch_loss_depth.append(depth_loss.item())
            batch_loss_smooth.append(smooth_loss.item())
            batch_loss_sparse.append(sparse_loss.item())
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_iter += 1
            
            # Debug first batch
            if epoch == 1 and i == 0:
                print(f"Debug - Audio shape: {audio.shape}, GT depth shape: {gtdepth.shape}")
                print(f"Debug - Pred depth range: [{depth_pred.min().item():.4f}, {depth_pred.max().item():.4f}]")
                print(f"Debug - Px shape: {Px.shape}, Py shape: {Py.shape}")
                print(f"Debug - Loss components: depth={depth_loss.item():.4f}, smooth={smooth_loss.item():.4f}, sparse={sparse_loss.item():.4f}")

        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - t0
        train_loss = np.mean(batch_loss)
        train_loss_depth = np.mean(batch_loss_depth)
        train_loss_smooth = np.mean(batch_loss_smooth)
        train_loss_sparse = np.mean(batch_loss_sparse)
        
        print(f'Epoch {epoch}: Loss={train_loss:.6f} (depth={train_loss_depth:.6f}, smooth={train_loss_smooth:.6f}, sparse={train_loss_sparse:.6f}), Time={epoch_time:.1f}s, LR={scheduler.get_last_lr()[0]:.6f}')
        
        # Log to wandb
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/loss_depth': train_loss_depth,
                'train/loss_smooth': train_loss_smooth,
                'train/loss_sparse': train_loss_sparse,
                'train/epoch_time': epoch_time,
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
                    
                    # Compute validation loss
                    if cfg.dataset.depth_norm:
                        depth_pred_val_denorm = depth_pred_val[valid_mask_val] * cfg.dataset.max_depth
                        gtdepth_val_denorm = gtdepth_val[valid_mask_val] * cfg.dataset.max_depth
                        
                        if cfg.mode.criterion == 'Combined':
                            loss_val = l1_weight * l1_criterion(depth_pred_val_denorm, gtdepth_val_denorm) + \
                                       silog_weight * silog_criterion(depth_pred_val_denorm, gtdepth_val_denorm)
                        else:
                            loss_val = criterion(depth_pred_val_denorm, gtdepth_val_denorm)
                    else:
                        if cfg.mode.criterion == 'Combined':
                            loss_val = l1_weight * l1_criterion(depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val]) + \
                                       silog_weight * silog_criterion(depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val])
                        else:
                            loss_val = criterion(depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val])
                    
                    batch_loss_val.append(loss_val.item())

                    # Store first batch for visualization
                    if batch_idx == 0:
                        if cfg.dataset.depth_norm:
                            val_preds.append(depth_pred_val * cfg.dataset.max_depth)
                            val_gts.append(gtdepth_val * cfg.dataset.max_depth)
                        else:
                            val_preds.append(depth_pred_val)
                            val_gts.append(gtdepth_val)

                    # Compute error metrics per sample
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
                        
                        # Clamp predictions
                        epsilon = 1e-3
                        pred_map = np.clip(pred_map, epsilon, cfg.dataset.max_depth)
                        gt_map = np.maximum(gt_map, 0.0)
                        
                        error_metrics = compute_errors(gt_map, pred_map, min_depth_threshold=0.0)
                        errors.append(error_metrics)

            mean_errors = np.array(errors).mean(0)
            val_loss = np.mean(batch_loss_val)
            abs_rel, rmse, delta1, delta2, delta3, log10, mae = mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4], mean_errors[5], mean_errors[6]
            
            print(f'Val - Loss: {val_loss:.6f}, RMSE: {rmse:.3f}, ABS_REL: {abs_rel:.3f}, Delta1: {delta1:.3f}')
            
            # Log to wandb
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
            
            # Save visualization
            if len(val_preds) > 0:
                pred_batch = torch.cat(val_preds, dim=0)
                gt_batch = torch.cat(val_gts, dim=0)
                vis_path = os.path.join(results_dir, f'epoch_{epoch:04d}_validation.png')
                save_batch_visualization(pred_batch, gt_batch, vis_path, epoch, 
                                         num_samples=min(4, pred_batch.shape[0]))
                print(f'Validation visualization saved to: {vis_path}')
                
                if WANDB_AVAILABLE and args.use_wandb:
                    log_dict['val/visualization'] = wandb.Image(vis_path, caption=f'Epoch {epoch}')
            
            if WANDB_AVAILABLE and args.use_wandb and log_dict:
                wandb.log(log_dict, step=epoch)
            
            # Save best model
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
                print(f'New best model saved (val_loss={val_loss:.6f})')

        # ------- Save Checkpoint ------------
        if epoch % cfg.mode.saving_checkpoints == 0:
            print('Save network')
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            path_check = './checkpoints/' + experiment_name + '/'
            os.makedirs(path_check, exist_ok=True)
            torch.save(state, f'{path_check}/checkpoint_{epoch}.pth')
            
            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log({'checkpoint_saved': epoch}, step=epoch)

    # Finish wandb
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()
        print("W&B run finished")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()



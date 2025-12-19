"""
Training Script for Base + Residual Depth Estimation

This script is independent from train.py and implements a novel approach:
    - Base Decoder: Learns coarse depth (room layout/structure)
    - Residual Decoder: Learns fine details (object details)
    - Custom Loss: 3-component loss (reconstruction + structural + sparsity)

Philosophy:
    Audio-to-depth is under-constrained. By decomposing the problem into
    "room layout" (base) and "details" (residual), we make it more tractable.
    This is analogous to Taylor series: f(x) ≈ f(a) + f'(a)(x-a)

Usage:
    python train_base_residual.py --dataset batvisionv2 --use_wandb --experiment_name base_res_exp1
"""

from dataloader.BatvisionV1_Dataset import BatvisionV1Dataset
from dataloader.BatvisionV2_Dataset import BatvisionV2Dataset
from models.base_residual_model import create_base_residual_model
from utils_base_residual_loss import BaseResidualLoss, AdaptiveBaseResidualLoss
from utils_criterion import compute_errors
from utils_visualization import save_batch_visualization

import time
import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

from config_loader import load_config
import wandb

WANDB_AVAILABLE = True


def save_component_visualization(base_batch, res_batch, final_batch, gt_batch, save_path, epoch):
    """
    Save visualization of base, residual, final, and GT depth
    
    Args:
        base_batch: [B, 1, H, W] - Base depth predictions
        res_batch: [B, 1, H, W] - Residual predictions
        final_batch: [B, 1, H, W] - Final depth predictions
        gt_batch: [B, 1, H, W] - Ground truth depth
        save_path: Path to save visualization
        epoch: Current epoch number
    """
    import matplotlib.pyplot as plt
    
    # Select first 4 samples
    num_samples = min(4, base_batch.shape[0])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Convert to numpy
        base = base_batch[i, 0].cpu().numpy()
        res = res_batch[i, 0].cpu().numpy()
        final = final_batch[i, 0].cpu().numpy()
        gt = gt_batch[i, 0].cpu().numpy()
        
        # Plot base depth
        im0 = axes[i, 0].imshow(base, cmap='viridis')
        axes[i, 0].set_title(f'Base (Layout)')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        # Plot residual (use different colormap to show +/-)
        im1 = axes[i, 1].imshow(res, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Residual (Details)')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Plot final depth
        im2 = axes[i, 2].imshow(final, cmap='viridis')
        axes[i, 2].set_title(f'Final (Base+Res)')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        # Plot GT
        im3 = axes[i, 3].imshow(gt, cmap='viridis')
        axes[i, 3].set_title(f'Ground Truth')
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)
    
    plt.suptitle(f'Epoch {epoch} - Base + Residual Decomposition', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train Base+Residual depth model on Batvision dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_base_residual.py --dataset batvisionv2
  
  # With W&B logging
  python train_base_residual.py --dataset batvisionv2 --use_wandb --experiment_name base_res_exp1
  
  # With adaptive loss (curriculum learning)
  python train_base_residual.py --use_adaptive_loss --warmup_epochs 20
  
  # Custom loss weights
  python train_base_residual.py --lambda_recon 1.0 --lambda_base 0.5 --lambda_sparse 0.1
        """
    )
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='batvisionv2',
                       choices=['batvisionv1', 'batvisionv2'])
    parser.add_argument('--audio_format', type=str, default=None,
                       choices=['spectrogram', 'mel_spectrogram', 'waveform'])
    
    # Model
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels in the network')
    parser.add_argument('--bilinear', action='store_true', default=True,
                       help='Use bilinear upsampling')
    
    # Loss
    parser.add_argument('--use_adaptive_loss', action='store_true', default=False,
                       help='Use adaptive loss weights (curriculum learning)')
    parser.add_argument('--lambda_recon', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    parser.add_argument('--lambda_base', type=float, default=1.2,
                       help='Weight for base structural loss')
    parser.add_argument('--lambda_sparse', type=float, default=0.05,
                       help='Weight for residual sparsity')
    parser.add_argument('--lowpass_kernel', type=int, default=16,
                       help='Kernel size for low-pass filtering (larger = coarser base)')
    parser.add_argument('--warmup_epochs', type=int, default=50,
                       help='Epochs for adaptive loss warmup')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', '--lr', type=float, default=None)
    parser.add_argument('--optimizer', type=str, default=None,
                       choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--epochs', type=int, default=None)
    
    # Validation
    parser.add_argument('--validation', type=lambda x: (str(x).lower() == 'true'), default=None)
    parser.add_argument('--validation_iter', type=int, default=None)
    
    # W&B
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='batvision-depth-estimation')
    parser.add_argument('--wandb_entity', type=str, default='branden')
    
    # Experiment
    parser.add_argument('--experiment_name', type=str, default='base_res_default')
    parser.add_argument('--checkpoints', type=int, default=None)
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(dataset_name=args.dataset, model_name='unet_baseline', mode='train', experiment_name=args.experiment_name)
    
    # Override config with arguments
    if args.batch_size is not None:
        cfg.mode.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.mode.learning_rate = args.learning_rate
    if args.optimizer is not None:
        cfg.mode.optimizer = args.optimizer
    if args.epochs is not None:
        cfg.mode.epochs = args.epochs
    if args.validation is not None:
        cfg.mode.validation = args.validation
    if args.validation_iter is not None:
        cfg.mode.validation_iter = args.validation_iter
    if args.audio_format is not None:
        cfg.dataset.audio_format = args.audio_format
    
    batch_size = cfg.mode.batch_size
    
    print("\n" + "="*60)
    print("Base + Residual Depth Estimation Training")
    print("="*60)
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Audio Format: {cfg.dataset.audio_format}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {cfg.mode.learning_rate}")
    print(f"Optimizer: {cfg.mode.optimizer}")
    print(f"Epochs: {cfg.mode.epochs}")
    print(f"Loss Type: {'Adaptive' if args.use_adaptive_loss else 'Fixed'}")
    print(f"Loss Weights: λ_recon={args.lambda_recon}, λ_base={args.lambda_base}, λ_sparse={args.lambda_sparse}")
    print("="*60 + "\n")
    
    # GPU setup
    if torch.cuda.is_available():
        n_GPU = torch.cuda.device_count()
        max_gpus = min(n_GPU, 4)
        gpu_ids = list(range(max_gpus))
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"{n_GPU} GPU(s) available, using {len(gpu_ids)}: {gpu_ids}")
    else:
        gpu_ids = []
        device = torch.device('cpu')
        print("WARNING: Using CPU")
    
    # Create datasets
    if cfg.dataset.name == 'batvisionv1':
        train_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_train)
        if cfg.mode.validation:
            val_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_val)
    else:
        train_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_train)
        if cfg.mode.validation:
            val_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_val)
    
    print(f'Train Dataset: {len(train_set)} instances')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=cfg.mode.shuffle, 
                             num_workers=cfg.mode.num_threads)
    
    if cfg.mode.validation:
        print(f'Validation Dataset: {len(val_set)} instances')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=cfg.mode.num_threads)
    
    # Create model
    print("\nCreating Base + Residual model...")
    model = create_base_residual_model(
        input_channels=2,  # Binaural audio
        base_channels=args.base_channels,
        bilinear=args.bilinear,
        output_size=cfg.dataset.images_size,
        max_depth=cfg.dataset.max_depth,
        gpu_ids=gpu_ids
    )
    
    # Print model info
    if hasattr(model, 'module'):
        params = model.module.get_parameters_count()
    else:
        params = model.get_parameters_count()
    print(f"Model Parameters:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Base Decoder: {params['base_decoder']:,}")
    print(f"  Residual Decoder: {params['residual_decoder']:,}")
    print(f"  Total: {params['total']:,}")
    
    # Create loss function
    print(f"\nCreating loss function...")
    if args.use_adaptive_loss:
        criterion = AdaptiveBaseResidualLoss(
            lambda_recon_init=args.lambda_recon * 0.5,
            lambda_base_init=args.lambda_base * 2.0,
            lambda_sparse=args.lambda_sparse,
            warmup_epochs=args.warmup_epochs,
            lowpass_kernel=args.lowpass_kernel
        ).to(device)
        print(f"Using Adaptive Loss (warmup: {args.warmup_epochs} epochs)")
    else:
        criterion = BaseResidualLoss(
            lambda_recon=args.lambda_recon,
            lambda_base=args.lambda_base,
            lambda_sparse=args.lambda_sparse,
            lowpass_kernel=args.lowpass_kernel
        ).to(device)
        print(f"Using Fixed Loss")
    
    # Create optimizer
    if cfg.mode.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.mode.learning_rate)
    elif cfg.mode.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.mode.learning_rate)
    elif cfg.mode.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.mode.learning_rate)
    
    # Experiment name
    experiment_name = f"base_residual_{cfg.dataset.name}_BS{batch_size}_Lr{cfg.mode.learning_rate}_{cfg.mode.optimizer}_{args.experiment_name}"
    
    # Initialize W&B
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment_name,
            config={
                'model': 'base_residual',
                'dataset': cfg.dataset.name,
                'batch_size': batch_size,
                'learning_rate': cfg.mode.learning_rate,
                'optimizer': cfg.mode.optimizer,
                'epochs': cfg.mode.epochs,
                'base_channels': args.base_channels,
                'adaptive_loss': args.use_adaptive_loss,
                'lambda_recon': args.lambda_recon,
                'lambda_base': args.lambda_base,
                'lambda_sparse': args.lambda_sparse,
                'lowpass_kernel': args.lowpass_kernel,
                'audio_format': cfg.dataset.audio_format,
            },
            tags=['base_residual', cfg.dataset.name, 'layout_learning']
        )
        print(f"W&B initialized: {experiment_name}")
    
    # Create directories
    log_dir = f"./logs/{experiment_name}/"
    results_dir = f"./results/{experiment_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load checkpoint if specified
    if args.checkpoints is not None:
        checkpoint_path = f'./checkpoints/{experiment_name}/checkpoint_{args.checkpoints}.pth'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        checkpoint_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {args.checkpoints}")
    else:
        checkpoint_epoch = 1
    
    nb_epochs = cfg.mode.epochs
    max_depth = cfg.dataset.max_depth
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    for epoch in range(checkpoint_epoch, nb_epochs + 1):
        t0 = time.time()
        
        # Update adaptive loss weights
        if args.use_adaptive_loss:
            criterion.set_epoch(epoch - 1)
            if epoch == 1 or epoch % 10 == 0:
                weights = criterion.get_current_weights()
                print(f"Epoch {epoch} loss weights: {weights}")
        
        # Training
        model.train()
        batch_losses = []
        loss_components = {'recon': [], 'base': [], 'sparse': []}
        
        for i, (audio, gtdepth) in enumerate(train_loader):
            audio = audio.to(device)
            gtdepth = gtdepth.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            base_depth, residual, _ = model(audio)
            
            # Gradient detachment strategy (curriculum learning)
            # Early: Learn together, Later: Fix base, refine residual
            if args.use_adaptive_loss and epoch > args.warmup_epochs:
                # Phase 2: Base is fixed, only residual learns details
                final_depth = base_depth.detach() + residual
            else:
                # Phase 1: Both learn together
                final_depth = base_depth + residual
            
            # Clamp final depth to valid range
            final_depth = torch.clamp(final_depth, 0, max_depth)
            
            # Compute loss
            valid_mask = gtdepth > 0
            loss, loss_dict = criterion(base_depth, residual, final_depth, gtdepth, valid_mask)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Log
            batch_losses.append(loss.item())
            loss_components['recon'].append(loss_dict['recon'])
            loss_components['base'].append(loss_dict['base'])
            loss_components['sparse'].append(loss_dict['sparse'])
        
        epoch_time = time.time() - t0
        train_loss = np.mean(batch_losses)
        
        print(f'Epoch {epoch}/{nb_epochs}: Loss={train_loss:.4f} '
              f'(recon={np.mean(loss_components["recon"]):.4f}, '
              f'base={np.mean(loss_components["base"]):.4f}, '
              f'sparse={np.mean(loss_components["sparse"]):.4f}) '
              f'Time={epoch_time:.1f}s')
        
        # Log to W&B
        if WANDB_AVAILABLE and args.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss_total': train_loss,
                'train/loss_recon': np.mean(loss_components['recon']),
                'train/loss_base': np.mean(loss_components['base']),
                'train/loss_sparse': np.mean(loss_components['sparse']),
                'train/epoch_time': epoch_time,
            }
            
            if args.use_adaptive_loss:
                weights = criterion.get_current_weights()
                log_dict.update({
                    'train/lambda_recon': weights['lambda_recon'],
                    'train/lambda_base': weights['lambda_base'],
                })
            
            wandb.log(log_dict, step=epoch)
        
        # Validation
        if cfg.mode.validation and epoch % cfg.mode.validation_iter == 0:
            model.eval()
            errors = []
            val_losses = []
            val_base_list = []
            val_res_list = []
            val_pred_list = []
            val_gt_list = []
            
            with torch.no_grad():
                for batch_idx, (audio_val, gtdepth_val) in enumerate(val_loader):
                    audio_val = audio_val.to(device)
                    gtdepth_val = gtdepth_val.to(device)
                    
                    # Forward
                    base_depth_val, residual_val, _ = model(audio_val)
                    
                    # Apply same detachment strategy as training
                    if args.use_adaptive_loss and epoch > args.warmup_epochs:
                        final_depth_val = base_depth_val.detach() + residual_val
                    else:
                        final_depth_val = base_depth_val + residual_val
                    
                    # Clamp final depth to valid range
                    final_depth_val = torch.clamp(final_depth_val, 0, max_depth)
                    
                    # Loss
                    valid_mask_val = gtdepth_val > 0
                    loss_val, _ = criterion(base_depth_val, residual_val, final_depth_val, 
                                           gtdepth_val, valid_mask_val)
                    val_losses.append(loss_val.item())
                    
                    # Store first batch for visualization
                    if batch_idx == 0:
                        val_base_list.append(base_depth_val)
                        val_res_list.append(residual_val)
                        val_pred_list.append(final_depth_val)
                        val_gt_list.append(gtdepth_val)
                    
                    # Compute metrics
                    for idx in range(final_depth_val.shape[0]):
                        gt_map = gtdepth_val[idx].cpu().numpy()
                        pred_map = final_depth_val[idx].cpu().numpy()
                        
                        if gt_map.ndim == 3:
                            gt_map = gt_map[0]
                        if pred_map.ndim == 3:
                            pred_map = pred_map[0]
                        
                        # Clip predictions
                        pred_map = np.clip(pred_map, 0.001, max_depth)
                        gt_map = np.maximum(gt_map, 0.0)
                        
                        error_metrics = compute_errors(gt_map, pred_map, min_depth_threshold=0.0)
                        errors.append(error_metrics)
            
            mean_errors = np.array(errors).mean(0)
            val_loss = np.mean(val_losses)
            abs_rel, rmse, delta1 = mean_errors[0], mean_errors[1], mean_errors[2]
            
            print(f'Val - Loss: {val_loss:.4f}, RMSE: {rmse:.3f}, ABS_REL: {abs_rel:.3f}, Delta1: {delta1:.3f}')
            
            # Visualization
            if len(val_pred_list) > 0:
                base_batch = torch.cat(val_base_list, dim=0)
                res_batch = torch.cat(val_res_list, dim=0)
                pred_batch = torch.cat(val_pred_list, dim=0)
                gt_batch = torch.cat(val_gt_list, dim=0)
                
                vis_path = os.path.join(results_dir, f'epoch_{epoch:04d}_decomposition.png')
                save_component_visualization(base_batch, res_batch, pred_batch, gt_batch, vis_path, epoch)
                print(f'Visualization saved: {vis_path}')
                
                if WANDB_AVAILABLE and args.use_wandb:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/rmse': rmse,
                        'val/abs_rel': abs_rel,
                        'val/delta1': delta1,
                        'val/decomposition': wandb.Image(vis_path)
                    }, step=epoch)
        
        # Save checkpoint
        if epoch % cfg.mode.saving_checkpoints == 0:
            checkpoint_dir = f'./checkpoints/{experiment_name}/'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'{checkpoint_dir}/checkpoint_{epoch}.pth')
            print(f'Checkpoint saved: epoch {epoch}')
    
    # Finish
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()



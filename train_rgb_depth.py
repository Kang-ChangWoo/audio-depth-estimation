"""
Training Script for RGB Depth Estimation

This script trains a CNN-based depth estimation model using RGB images.
The model architecture is compatible with binaural_attention_model for
future knowledge distillation.

Key Features:
    - Standard U-Net encoder-decoder architecture
    - RGB input (3 channels)
    - Feature sizes match audio model for distillation
    - Can serve as teacher model for audio-to-depth networks

Usage:
    # Standard training
    python train_rgb_depth.py --dataset batvisionv2 --batch_size 64 --use_wandb
    
    # Custom architecture
    python train_rgb_depth.py --base_channels 64 --learning_rate 0.0001
"""

from dataloader.BatvisionV1_Dataset import BatvisionV1Dataset
from dataloader.BatvisionV2_Dataset import BatvisionV2Dataset
from models.rgb_depth_model import create_rgb_depth_model
from utils_criterion import compute_errors
from utils_visualization import save_batch_visualization

import time
import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from config_loader import load_config
import wandb

WANDB_AVAILABLE = True


def create_depth_loss():
    """
    Create depth estimation loss function
    Combines L1 loss with edge-aware smoothness
    """
    class DepthLoss(nn.Module):
        def __init__(self, lambda_l1=1.0, lambda_smooth=0.1):
            super().__init__()
            self.lambda_l1 = lambda_l1
            self.lambda_smooth = lambda_smooth
        
        def forward(self, pred, target):
            """
            Args:
                pred: Predicted depth [B, 1, H, W]
                target: Ground truth depth [B, 1, H, W]
            
            Returns:
                loss: Total loss
                loss_dict: Dictionary of loss components
            """
            # L1 reconstruction loss
            l1_loss = torch.nn.functional.l1_loss(pred, target)
            
            # Edge-aware smoothness loss
            # Penalize depth gradients where image gradients are small
            pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
            pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
            
            smooth_loss = pred_dx.mean() + pred_dy.mean()
            
            # Total loss
            total_loss = (
                self.lambda_l1 * l1_loss +
                self.lambda_smooth * smooth_loss
            )
            
            loss_dict = {
                'l1': l1_loss.item(),
                'smooth': smooth_loss.item()
            }
            
            return total_loss, loss_dict
    
    return DepthLoss()


def main():
    parser = argparse.ArgumentParser(
        description='Train RGB depth estimation model on Batvision dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train_rgb_depth.py --dataset batvisionv2 --batch_size 64
    
    # With W&B logging
    python train_rgb_depth.py --dataset batvisionv2 --use_wandb
    
    # Custom learning rate
    python train_rgb_depth.py --learning_rate 0.0001 --batch_size 32
    
    # Resume from checkpoint
    python train_rgb_depth.py --checkpoints 50 --experiment_name my_rgb_model
        """
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='batvisionv2',
                        choices=['batvisionv1', 'batvisionv2'],
                        help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel count (64=~20M params, 32=~5M params)')
    parser.add_argument('--bilinear', action='store_true', default=True,
                        help='Use bilinear upsampling')
    
    # Loss arguments
    parser.add_argument('--lambda_l1', type=float, default=1.0,
                        help='Weight for L1 loss')
    parser.add_argument('--lambda_smooth', type=float, default=0.1,
                        help='Weight for smoothness loss')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--nb_epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['Adam', 'AdamW', 'SGD'],
                        help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'step'],
                        help='Learning rate scheduler')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoints', type=int, default=None,
                        help='Epoch to load checkpoint from')
    parser.add_argument('--save_frequency', type=int, default=2,
                        help='Save checkpoint every N epochs')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='batvision-depth-estimation',
                        help='W&B project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    cfg = load_config(dataset_name=args.dataset, model_name='unet_baseline', mode='train', experiment_name=args.experiment_name)
    
    # Create experiment name
    if args.experiment_name is None:
        args.experiment_name = (
            f"rgb_depth_{args.dataset}_"
            f"BS{args.batch_size}_Lr{args.learning_rate}_{args.optimizer}"
        )
    
    print("=" * 80)
    print(f"RGB Depth Estimation Training")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Base channels: {args.base_channels}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Initialize W&B
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Create datasets
    print("\n📊 Loading datasets...")
    if args.dataset == 'batvisionv1':
        train_dataset = BatvisionV1Dataset(
            cfg=cfg,
            annotation_file=cfg.dataset.annotation_file_train
        )
        val_dataset = BatvisionV1Dataset(
            cfg=cfg,
            annotation_file=cfg.dataset.annotation_file_val
        )
    else:  # batvisionv2
        train_dataset = BatvisionV2Dataset(
            cfg=cfg,
            annotation_file=cfg.dataset.annotation_file_train,
            use_image=True  # RGB images!
        )
        val_dataset = BatvisionV2Dataset(
            cfg=cfg,
            annotation_file=cfg.dataset.annotation_file_val,
            use_image=True
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = create_rgb_depth_model(
        base_channels=args.base_channels,
        bilinear=args.bilinear,
        output_size=cfg.dataset.images_size,
        max_depth=cfg.dataset.max_depth
    )
    model = model.to(args.device)
    
    # Create loss function
    criterion = create_depth_loss()
    criterion = criterion.to(args.device)
    print(f"✅ Using depth loss (λ_l1={args.lambda_l1}, λ_smooth={args.lambda_smooth})")
    
    # Create optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:  # SGD
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    
    # Create scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.nb_epochs
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.5
        )
    else:
        scheduler = None
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.checkpoints is not None:
        checkpoint_path = os.path.join(
            'checkpoints',
            args.experiment_name,
            f'epoch_{args.checkpoints:04d}.pth'
        )
        if os.path.exists(checkpoint_path):
            print(f"\n📂 Loading checkpoint from epoch {args.checkpoints}...")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = args.checkpoints
            print(f"✅ Resumed from epoch {start_epoch}")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    
    # Create directories
    log_dir = os.path.join('logs', args.experiment_name)
    checkpoint_dir = os.path.join('checkpoints', args.experiment_name)
    result_dir = os.path.join('results', args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # Save model architecture
    with open(os.path.join(log_dir, 'architecture.txt'), 'w') as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {model.get_num_params():,}\n")
    
    # Training loop
    print("\n🚀 Starting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.nb_epochs):
        epoch_start_time = time.time()
        
        # ==================== Training ====================
        model.train()
        train_loss = 0.0
        train_loss_dict = {}
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Handle different dataset formats
            if args.dataset == 'batvisionv2':
                # BatvisionV2 returns (audio, image, depth) or (image, depth) if use_image=True
                if len(batch_data) == 3:
                    _, image, depth_gt = batch_data
                else:
                    image, depth_gt = batch_data
            else:
                # BatvisionV1 returns (audio, depth)
                # Need to check if image is available
                image, depth_gt = batch_data
            
            image = image.to(args.device)
            depth_gt = depth_gt.to(args.device)
            
            # Forward pass
            depth_pred = model(image)
            
            # Compute loss
            loss, loss_dict = criterion(depth_pred, depth_gt)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item()
            for k, v in loss_dict.items():
                train_loss_dict[k] = train_loss_dict.get(k, 0.0) + v
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.nb_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # Average training losses
        train_loss /= len(train_loader)
        for k in train_loss_dict:
            train_loss_dict[k] /= len(train_loader)
        
        # ==================== Validation ====================
        model.eval()
        val_loss = 0.0
        val_loss_dict = {}
        val_errors = {'rmse': [], 'abs_rel': [], 'delta1': [], 'delta2': [], 'delta3': []}
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                # Handle different dataset formats
                if args.dataset == 'batvisionv2':
                    if len(batch_data) == 3:
                        _, image, depth_gt = batch_data
                    else:
                        image, depth_gt = batch_data
                else:
                    image, depth_gt = batch_data
                
                image = image.to(args.device)
                depth_gt = depth_gt.to(args.device)
                
                # Forward pass
                depth_pred = model(image)
                
                # Compute loss
                loss, loss_dict = criterion(depth_pred, depth_gt)
                
                # Accumulate losses
                val_loss += loss.item()
                for k, v in loss_dict.items():
                    val_loss_dict[k] = val_loss_dict.get(k, 0.0) + v
                
                # Compute depth errors
                abs_rel, rmse, delta1, delta2, delta3, log_10, mae = compute_errors(
                    depth_gt,
                    depth_pred
                )
                val_errors['abs_rel'].append(abs_rel)
                val_errors['rmse'].append(rmse)
                val_errors['delta1'].append(delta1)
                val_errors['delta2'].append(delta2)
                val_errors['delta3'].append(delta3)
        
        # Average validation metrics
        val_loss /= len(val_loader)
        for k in val_loss_dict:
            val_loss_dict[k] /= len(val_loader)
        for k in val_errors:
            val_errors[k] = np.mean(val_errors[k])
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print("=" * 80)
        print(f"Epoch [{epoch+1}/{args.nb_epochs}] - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val RMSE: {val_errors['rmse']:.4f} | "
              f"ABS_REL: {val_errors['abs_rel']:.4f} | "
              f"δ1: {val_errors['delta1']:.4f}")
        print("=" * 80)
        
        # Log to W&B
        if args.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/rmse': val_errors['rmse'],
                'val/abs_rel': val_errors['abs_rel'],
                'val/delta1': val_errors['delta1'],
                'val/delta2': val_errors['delta2'],
                'val/delta3': val_errors['delta3'],
                'lr': optimizer.param_groups[0]['lr']
            }
            # Add component losses
            for k, v in train_loss_dict.items():
                log_dict[f'train/{k}'] = v
            for k, v in val_loss_dict.items():
                log_dict[f'val/{k}'] = v
            
            wandb.log(log_dict)
        
        # Save checkpoint
        if (epoch + 1) % args.save_frequency == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'epoch_{epoch+1:04d}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_errors': val_errors
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_errors': val_errors
            }, best_model_path)
            print(f"⭐ Best model saved: {best_model_path}")
        
        # Save visualization
        if (epoch + 1) % args.save_frequency == 0:
            model.eval()
            with torch.no_grad():
                # Get a batch for visualization
                batch_data = next(iter(val_loader))
                if args.dataset == 'batvisionv2':
                    if len(batch_data) == 3:
                        _, image_vis, depth_gt_vis = batch_data
                    else:
                        image_vis, depth_gt_vis = batch_data
                else:
                    image_vis, depth_gt_vis = batch_data
                
                image_vis = image_vis.to(args.device)
                depth_gt_vis = depth_gt_vis.to(args.device)
                
                depth_pred_vis = model(image_vis)
                
                # Save visualization
                vis_path = os.path.join(
                    result_dir,
                    f'epoch_{epoch+1:04d}_prediction.png'
                )
                save_batch_visualization(
                    depth_pred_vis[:4],  # Predicted depths
                    depth_gt_vis[:4],     # Ground truth depths
                    vis_path,             # Save path
                    epoch + 1,            # Epoch number
                    num_samples=4
                )
                print(f"📊 Visualization saved: {vis_path}")
    
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved in: {result_dir}")
    print("=" * 80)
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()







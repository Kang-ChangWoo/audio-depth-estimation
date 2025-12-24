"""
Training Script for Binaural Attention Depth Estimation

This script implements a novel approach using cross-attention between
left and right audio channels to explicitly model binaural cues.

Key Innovation:
    - Separate encoders for left/right channels
    - Multi-scale cross-attention for correspondence
    - Explicit modeling of ITD (Inter-aural Time Difference)
    - Explicit modeling of ILD (Inter-aural Level Difference)

Loss Functions (NEW VERSION):
    - L1: Simple L1 loss for depth prediction
    - SIlog: Scale-Invariant Logarithmic loss (recommended for depth)
    - Combined: L1 + SIlog with adjustable weights

Usage:
    # Standard training with L1 loss
    python train_binaural_attention.py --dataset batvisionv2 --batch_size 64 --use_wandb
    
    # SIlog loss (recommended)
    python train_binaural_attention.py --dataset batvisionv2 --criterion SIlog --use_wandb
    
    # Combined loss
    python train_binaural_attention.py --dataset batvisionv2 --criterion Combined --l1_weight 0.5 --silog_weight 0.5
"""

from dataloader.BatvisionV1_Dataset import BatvisionV1Dataset
from dataloader.BatvisionV2_Dataset import BatvisionV2Dataset
from models.binaural_attention_model import create_binaural_attention_model
# OLD VERSION: Binaural-specific loss (recon+edge+smooth)
# from utils_binaural_attention_loss import create_binaural_loss
# NEW VERSION: Standard depth loss (L1/SIlog/Combined)
from utils_loss import SIlogLoss
from utils_criterion import compute_errors
from utils_visualization import save_batch_visualization

import time
import os
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config_loader import load_config
import wandb

WANDB_AVAILABLE = True


def main():
    parser = argparse.ArgumentParser(
        description='Train Binaural Attention depth model on Batvision dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training with L1 loss
    python train_binaural_attention.py --dataset batvisionv2 --batch_size 64
    
    # With SIlog loss (recommended for depth estimation)
    python train_binaural_attention.py --dataset batvisionv2 --criterion SIlog --use_wandb
    
    # Combined loss (L1 + SIlog)
    python train_binaural_attention.py --dataset batvisionv2 --criterion Combined --l1_weight 0.5 --silog_weight 0.5
    
    # Custom architecture
    python train_binaural_attention.py --base_channels 32 --attention_levels 2 3 4 5
    
    # Resume from checkpoint
    python train_binaural_attention.py --checkpoints 50 --experiment_name my_experiment
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
                        help='Base channel count (64=~40M params, 32=~10M params)')
    parser.add_argument('--bilinear', action='store_true', default=True,
                        help='Use bilinear upsampling')
    parser.add_argument('--attention_levels', type=int, nargs='+', default=[2, 3, 4, 5],
                        help='Encoder levels to apply attention (1-5)')
    
    # Loss arguments (NEW VERSION: L1/SIlog/Combined)
    parser.add_argument('--criterion', type=str, default='L1',
                        choices=['L1', 'SIlog', 'Combined'],
                        help='Loss function: L1, SIlog, or Combined')
    parser.add_argument('--use_silog', type=lambda x: (str(x).lower() == 'true'), default=None,
                        help='Enable/disable SIlog loss in Combined mode')
    parser.add_argument('--silog_lambda', type=float, default=0.5,
                        help='SIlog lambda parameter (default: 0.5)')
    parser.add_argument('--l1_weight', type=float, default=0.5,
                        help='L1 loss weight in Combined mode (default: 0.5)')
    parser.add_argument('--silog_weight', type=float, default=0.5,
                        help='SIlog loss weight in Combined mode (default: 0.5)')
    
    # OLD VERSION: Binaural-specific loss (recon+edge+smooth) - DEPRECATED
    # parser.add_argument('--use_adaptive_loss', action='store_true',
    #                     help='Use adaptive loss with curriculum learning')
    # parser.add_argument('--lambda_recon', type=float, default=1.0,
    #                     help='Weight for reconstruction loss')
    # parser.add_argument('--lambda_edge', type=float, default=0.2,
    #                     help='Weight for edge-aware loss')
    # parser.add_argument('--lambda_smooth', type=float, default=0.1,
    #                     help='Weight for smoothness loss')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=0.001,
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
            f"binaural_attn_{args.dataset}_"
            f"BS{args.batch_size}_Lr{args.learning_rate}_{args.optimizer}_"
            f"{args.criterion}"
        )
        # OLD VERSION: adaptive loss suffix
        # if args.use_adaptive_loss:
        #     args.experiment_name += "_adaptive"
    
    print("=" * 80)
    print(f"Binaural Attention Depth Estimation Training")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Base channels: {args.base_channels}")
    print(f"Attention levels: {args.attention_levels}")
    print(f"Loss criterion: {args.criterion}")
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
            use_image=False  # Audio only
        )
        val_dataset = BatvisionV2Dataset(
            cfg=cfg,
            annotation_file=cfg.dataset.annotation_file_val,
            use_image=False
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
    model = create_binaural_attention_model(
        base_channels=args.base_channels,
        bilinear=args.bilinear,
        output_size=cfg.dataset.images_size,
        max_depth=cfg.dataset.max_depth,
        attention_levels=args.attention_levels
    )
    model = model.to(args.device)
    
    # Create loss function (NEW VERSION: L1/SIlog/Combined)
    max_depth = cfg.dataset.max_depth if cfg.dataset.max_depth else 30.0
    
    if args.criterion == 'L1':
        criterion = nn.L1Loss().to(args.device)
        l1_criterion = None
        silog_criterion = None
        l1_weight = 0.0
        silog_weight = 0.0
        use_silog_loss = False
        print(f"✅ Using loss function: L1")
    elif args.criterion == 'SIlog':
        criterion = SIlogLoss(lambda_scale=args.silog_lambda).to(args.device)
        l1_criterion = None
        silog_criterion = None
        l1_weight = 0.0
        silog_weight = 0.0
        use_silog_loss = True
        print(f"✅ Using loss function: SIlog (lambda={args.silog_lambda})")
    elif args.criterion == 'Combined':
        # Combined loss: L1 + optional SIlog
        l1_weight = args.l1_weight
        silog_weight = args.silog_weight
        
        # Determine if SIlog should be used
        if args.use_silog is not None:
            use_silog_loss = args.use_silog
        elif silog_weight == 0.0:
            use_silog_loss = False
        else:
            use_silog_loss = True  # Default for Combined mode
        
        # Setup criterion based on use_silog_loss
        if not use_silog_loss:
            silog_weight = 0.0
            l1_weight = 1.0  # Use only L1
            l1_criterion = nn.L1Loss().to(args.device)
            silog_criterion = None
            criterion = None
            print(f"✅ Using loss function: L1 only (SIlog disabled)")
        else:
            l1_criterion = nn.L1Loss().to(args.device)
            silog_criterion = SIlogLoss(lambda_scale=args.silog_lambda).to(args.device)
            criterion = None  # Will compute manually
            print(f"✅ Using loss function: Combined (L1={l1_weight}, SIlog={silog_weight}, lambda={args.silog_lambda})")
    else:
        raise ValueError(f"Unknown criterion: {args.criterion}")
    
    # OLD VERSION: Binaural-specific loss (recon+edge+smooth) - COMMENTED OUT
    # if args.use_adaptive_loss:
    #     criterion = create_binaural_loss(
    #         loss_type='adaptive',
    #         warmup_epochs=20,
    #         total_epochs=args.nb_epochs
    #     )
    #     print("✅ Using adaptive loss with curriculum learning")
    # else:
    #     criterion = create_binaural_loss(
    #         loss_type='standard',
    #         lambda_recon=args.lambda_recon,
    #         lambda_edge=args.lambda_edge,
    #         lambda_smooth=args.lambda_smooth
    #     )
    #     print(f"✅ Using standard loss (λ_recon={args.lambda_recon}, "
    #           f"λ_edge={args.lambda_edge}, λ_smooth={args.lambda_smooth})")
    # 
    # # Move loss to device
    # criterion = criterion.to(args.device)
    
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
        
        for batch_idx, (audio, depth_gt) in enumerate(train_loader):
            audio = audio.to(args.device)
            depth_gt = depth_gt.to(args.device)
            
            # Forward pass
            depth_pred = model(audio)
            
            # Compute loss (NEW VERSION: L1/SIlog/Combined)
            valid_mask = depth_gt > 0  # Valid depth mask
            
            if cfg.dataset.depth_norm:
                # Denormalize to meters for loss computation
                depth_pred_denorm = depth_pred[valid_mask] * cfg.dataset.max_depth
                depth_gt_denorm = depth_gt[valid_mask] * cfg.dataset.max_depth
                
                # Compute loss based on criterion
                if args.criterion == 'Combined':
                    loss = l1_weight * l1_criterion(depth_pred_denorm, depth_gt_denorm)
                    if use_silog_loss:
                        loss += silog_weight * silog_criterion(depth_pred_denorm, depth_gt_denorm)
                else:
                    loss = criterion(depth_pred_denorm, depth_gt_denorm)
            else:
                if args.criterion == 'Combined':
                    loss = l1_weight * l1_criterion(depth_pred[valid_mask], depth_gt[valid_mask])
                    if use_silog_loss:
                        loss += silog_weight * silog_criterion(depth_pred[valid_mask], depth_gt[valid_mask])
                else:
                    loss = criterion(depth_pred[valid_mask], depth_gt[valid_mask])
            
            # OLD VERSION: Binaural-specific loss
            # if args.use_adaptive_loss:
            #     loss, loss_dict = criterion(depth_pred, depth_gt, epoch)
            # else:
            #     loss, loss_dict = criterion(depth_pred, depth_gt)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item()
            # OLD VERSION: accumulate loss_dict
            # for k, v in loss_dict.items():
            #     train_loss_dict[k] = train_loss_dict.get(k, 0.0) + v
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.nb_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # Average training losses
        train_loss /= len(train_loader)
        # OLD VERSION: average loss_dict components
        # for k in train_loss_dict:
        #     train_loss_dict[k] /= len(train_loader)
        
        # ==================== Validation ====================
        model.eval()
        val_loss = 0.0
        val_loss_dict = {}
        val_errors = {'rmse': [], 'abs_rel': [], 'delta1': [], 'delta2': [], 'delta3': []}
        
        with torch.no_grad():
            for batch_idx, (audio, depth_gt) in enumerate(val_loader):
                audio = audio.to(args.device)
                depth_gt = depth_gt.to(args.device)
                
                # Forward pass
                depth_pred = model(audio)
                
                # Compute loss (NEW VERSION: L1/SIlog/Combined)
                valid_mask = depth_gt > 0  # Valid depth mask
                
                if cfg.dataset.depth_norm:
                    # Denormalize to meters for loss computation
                    depth_pred_denorm = depth_pred[valid_mask] * cfg.dataset.max_depth
                    depth_gt_denorm = depth_gt[valid_mask] * cfg.dataset.max_depth
                    
                    # Compute loss based on criterion
                    if args.criterion == 'Combined':
                        loss = l1_weight * l1_criterion(depth_pred_denorm, depth_gt_denorm)
                        if use_silog_loss:
                            loss += silog_weight * silog_criterion(depth_pred_denorm, depth_gt_denorm)
                    else:
                        loss = criterion(depth_pred_denorm, depth_gt_denorm)
                else:
                    if args.criterion == 'Combined':
                        loss = l1_weight * l1_criterion(depth_pred[valid_mask], depth_gt[valid_mask])
                        if use_silog_loss:
                            loss += silog_weight * silog_criterion(depth_pred[valid_mask], depth_gt[valid_mask])
                    else:
                        loss = criterion(depth_pred[valid_mask], depth_gt[valid_mask])
                
                # OLD VERSION: Binaural-specific loss
                # if args.use_adaptive_loss:
                #     loss, loss_dict = criterion(depth_pred, depth_gt, epoch)
                # else:
                #     loss, loss_dict = criterion(depth_pred, depth_gt)
                
                # Accumulate losses
                val_loss += loss.item()
                # OLD VERSION: accumulate loss_dict
                # for k, v in loss_dict.items():
                #     val_loss_dict[k] = val_loss_dict.get(k, 0.0) + v
                
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
        # OLD VERSION: average loss_dict components
        # for k in val_loss_dict:
        #     val_loss_dict[k] /= len(val_loader)
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
            # OLD VERSION: Add component losses from binaural loss
            # for k, v in train_loss_dict.items():
            #     log_dict[f'train/{k}'] = v
            # for k, v in val_loss_dict.items():
            #     log_dict[f'val/{k}'] = v
            
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
                audio_vis, depth_gt_vis = next(iter(val_loader))
                audio_vis = audio_vis.to(args.device)
                depth_gt_vis = depth_gt_vis.to(args.device)
                
                depth_pred_vis = model(audio_vis)
                
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


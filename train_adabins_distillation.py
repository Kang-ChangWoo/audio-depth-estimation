"""
Training Script for AdaBins with Knowledge Distillation

Knowledge Transfer: RGB (Teacher) → Audio (Student)

Training Strategy:
    - Use paired RGB-Audio-Depth data
    - RGB teacher guides audio student through distillation losses
    - At inference, audio works independently

Usage:
    python train_adabins_distillation.py --dataset batvisionv2 --use_wandb
"""

from dataloader.BatvisionV1_Dataset import BatvisionV1Dataset
from dataloader.BatvisionV2_Dataset import BatvisionV2Dataset

from models.adabins_distillation_model import create_adabins_distillation_model
from utils_distillation_loss import DistillationLoss, AdaptiveDistillationLoss
from utils_criterion import compute_errors
from utils_visualization import save_batch_visualization

import time
import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from config_loader import load_config

# W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not available")


def visualize_distillation_batch(audio_img, rgb_img, gt_depth, audio_pred, rgb_pred, 
                                 audio_bins, rgb_bins, save_path, idx=0):
    """
    Visualize distillation results
    
    Shows:
    - Audio input
    - RGB input
    - GT depth
    - Audio prediction
    - RGB prediction (teacher)
    - Bin distributions
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Audio spectrogram
    audio_vis = audio_img[idx, 0].cpu().numpy()  # First channel
    axes[0, 0].imshow(audio_vis, cmap='viridis')
    axes[0, 0].set_title('Audio Input (Ch 0)')
    axes[0, 0].axis('off')
    
    # RGB image
    if rgb_img is not None:
        rgb_vis = rgb_img[idx].permute(1, 2, 0).cpu().numpy()
        rgb_vis = (rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min() + 1e-8)
        axes[0, 1].imshow(rgb_vis)
        axes[0, 1].set_title('RGB Input')
        axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'No RGB', ha='center', va='center')
        axes[0, 1].axis('off')
    
    # GT depth
    gt_vis = gt_depth[idx, 0].cpu().numpy()
    axes[0, 2].imshow(gt_vis, cmap='plasma', vmin=0, vmax=gt_vis.max())
    axes[0, 2].set_title(f'GT Depth (max={gt_vis.max():.1f}m)')
    axes[0, 2].axis('off')
    
    # Audio prediction
    audio_vis = audio_pred[idx, 0].cpu().numpy()
    axes[0, 3].imshow(audio_vis, cmap='plasma', vmin=0, vmax=gt_vis.max())
    axes[0, 3].set_title(f'Audio Pred (max={audio_vis.max():.1f}m)')
    axes[0, 3].axis('off')
    
    # RGB prediction (teacher)
    if rgb_pred is not None:
        rgb_vis = rgb_pred[idx, 0].cpu().numpy()
        axes[1, 0].imshow(rgb_vis, cmap='plasma', vmin=0, vmax=gt_vis.max())
        axes[1, 0].set_title(f'RGB Pred (Teacher, max={rgb_vis.max():.1f}m)')
        axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'No RGB Pred', ha='center', va='center')
        axes[1, 0].axis('off')
    
    # Error map (Audio)
    if gt_vis.max() > 0:
        error = np.abs(audio_vis - gt_vis)
        axes[1, 1].imshow(error, cmap='hot', vmin=0, vmax=5)
        axes[1, 1].set_title(f'Audio Error (MAE={error.mean():.2f}m)')
        axes[1, 1].axis('off')
    
    # Bin distributions
    audio_bins_vis = audio_bins[idx].cpu().numpy()
    axes[1, 2].bar(range(len(audio_bins_vis)), audio_bins_vis, alpha=0.7, label='Audio')
    if rgb_bins is not None:
        rgb_bins_vis = rgb_bins[idx].cpu().numpy()
        axes[1, 2].bar(range(len(rgb_bins_vis)), rgb_bins_vis, alpha=0.5, label='RGB')
    axes[1, 2].set_title('Bin Centers Distribution')
    axes[1, 2].set_xlabel('Bin Index')
    axes[1, 2].set_ylabel('Depth (m)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Histogram of GT depths vs predictions
    gt_flat = gt_vis[gt_vis > 0].flatten()
    audio_flat = audio_vis[gt_vis > 0].flatten()
    axes[1, 3].hist(gt_flat, bins=50, alpha=0.5, label='GT', density=True)
    axes[1, 3].hist(audio_flat, bins=50, alpha=0.5, label='Audio', density=True)
    if rgb_pred is not None and gt_vis.max() > 0:
        rgb_flat = rgb_vis[gt_vis > 0].flatten()
        axes[1, 3].hist(rgb_flat, bins=50, alpha=0.5, label='RGB', density=True)
    axes[1, 3].set_title('Depth Distribution')
    axes[1, 3].set_xlabel('Depth (m)')
    axes[1, 3].set_ylabel('Density')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train AdaBins with Knowledge Distillation (RGB → Audio)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with distillation
  python train_adabins_distillation.py --dataset batvisionv2 --use_wandb
  
  # With adaptive loss (curriculum learning)
  python train_adabins_distillation.py --dataset batvisionv2 --use_adaptive_loss
  
  # Freeze RGB teacher
  python train_adabins_distillation.py --freeze_rgb --temperature 6.0
  
  # Custom distillation weights
  python train_adabins_distillation.py --lambda_response 0.8 --lambda_feature 0.5
        """
    )
    
    # Dataset & Model
    parser.add_argument('--dataset', type=str, default='batvisionv2',
                       choices=['batvisionv1', 'batvisionv2'])
    parser.add_argument('--n_bins', type=int, default=128,
                       help='Number of adaptive bins')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base channel count')
    parser.add_argument('--max_depth', type=float, default=None,
                       help='Maximum depth value')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', '--lr', type=float, default=None)
    parser.add_argument('--nb_epochs', type=int, default=None)
    parser.add_argument('--optimizer', type=str, default='AdamW',
                       choices=['Adam', 'AdamW', 'SGD'])
    
    # Distillation
    parser.add_argument('--use_adaptive_loss', action='store_true', default=False,
                       help='Use adaptive distillation loss with curriculum learning')
    parser.add_argument('--freeze_rgb', action='store_true', default=False,
                       help='Freeze RGB teacher during training')
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='Temperature for soft targets (higher = softer)')
    parser.add_argument('--lambda_task', type=float, default=1.0,
                       help='Weight for task loss (audio depth vs GT)')
    parser.add_argument('--lambda_response', type=float, default=0.5,
                       help='Weight for response distillation')
    parser.add_argument('--lambda_feature', type=float, default=0.3,
                       help='Weight for feature distillation')
    parser.add_argument('--lambda_bin', type=float, default=0.2,
                       help='Weight for bin distribution matching')
    parser.add_argument('--lambda_sparse', type=float, default=0.1,
                       help='Weight for residual sparsity')
    
    # Checkpoint
    parser.add_argument('--checkpoints', type=int, default=0,
                       help='Epoch to resume from')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for saving')
    
    # W&B
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='batvision-depth-estimation')
    parser.add_argument('--wandb_entity', type=str, default='branden')
    
    # GPU
    parser.add_argument('--gpu_ids', type=str, default='0',
                       help='GPU IDs to use (comma-separated)')
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    cfg = load_config(dataset_name=args.dataset, mode='train', model_name='unet_baseline')
    
    # Override config with args
    if args.batch_size is not None:
        cfg.mode.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.mode.learning_rate = args.learning_rate
    if args.nb_epochs is not None:
        cfg.mode.epochs = args.nb_epochs
    if args.max_depth is not None:
        cfg.dataset.max_depth = args.max_depth
    
    batch_size = cfg.mode.batch_size
    nb_epochs = cfg.mode.epochs
    
    # Experiment name
    if args.experiment_name is None:
        args.experiment_name = f"adabins_distill_{cfg.dataset.name}_BS{batch_size}_Lr{cfg.mode.learning_rate}"
        if args.use_adaptive_loss:
            args.experiment_name += "_adaptive"
        if args.freeze_rgb:
            args.experiment_name += "_frozen"
    
    experiment_name = args.experiment_name
    
    # Create directories
    log_dir = f"./logs/{experiment_name}/"
    results_dir = f"./results/{experiment_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 60)
    print("AdaBins Knowledge Distillation Training")
    print("=" * 60)
    print(f"Dataset:        {cfg.dataset.name}")
    print(f"Batch Size:     {batch_size}")
    print(f"Learning Rate:  {cfg.mode.learning_rate}")
    print(f"Epochs:         {nb_epochs}")
    print(f"N Bins:         {args.n_bins}")
    print(f"Max Depth:      {cfg.dataset.max_depth}m")
    print(f"Temperature:    {args.temperature}")
    print(f"Freeze RGB:     {args.freeze_rgb}")
    print(f"Adaptive Loss:  {args.use_adaptive_loss}")
    print(f"Device:         {device}")
    print(f"Experiment:     {experiment_name}")
    print("=" * 60)
    
    # ==========================================
    # Dataset
    # ==========================================
    print("\nLoading datasets...")
    if cfg.dataset.name == 'batvisionv2':
        # Load audio dataset
        train_dataset_audio = BatvisionV2Dataset(
            cfg, 'train.csv',
            use_image=False  # Audio
        )
        val_dataset_audio = BatvisionV2Dataset(
            cfg, 'val.csv',
            use_image=False
        )
        # Load RGB dataset
        train_dataset_rgb = BatvisionV2Dataset(
            cfg, 'train.csv',
            use_image=True  # RGB
        )
        val_dataset_rgb = BatvisionV2Dataset(
            cfg, 'val.csv',
            use_image=True
        )
        # Create paired dataset
        from torch.utils.data import Dataset as TorchDataset
        class PairedDataset(TorchDataset):
            def __init__(self, audio_dataset, rgb_dataset):
                self.audio_dataset = audio_dataset
                self.rgb_dataset = rgb_dataset
                assert len(audio_dataset) == len(rgb_dataset)
            
            def __len__(self):
                return len(self.audio_dataset)
            
            def __getitem__(self, idx):
                audio, depth = self.audio_dataset[idx]
                rgb, _ = self.rgb_dataset[idx]
                return audio, rgb, depth
        
        train_dataset = PairedDataset(train_dataset_audio, train_dataset_rgb)
        val_dataset = PairedDataset(val_dataset_audio, val_dataset_rgb)
    elif cfg.dataset.name == 'batvisionv1':
        # BV1 doesn't have RGB, so only audio
        train_dataset = BatvisionV1Dataset(cfg, 'train.csv')
        val_dataset = BatvisionV1Dataset(cfg, 'val.csv')
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    
    # ==========================================
    # Model
    # ==========================================
    print("\nCreating model...")
    model = create_adabins_distillation_model(
        n_bins=args.n_bins,
        base_channels=args.base_channels,
        output_size=cfg.dataset.images_size,
        max_depth=cfg.dataset.max_depth,
        use_pretrained_rgb=False,  # TODO: Add pre-trained RGB support
        gpu_ids=gpu_ids
    )
    
    if args.freeze_rgb:
        model.freeze_rgb()
    
    # Print parameter counts
    params = model.get_parameters_count()
    print(f"Parameters:")
    print(f"  RGB Teacher:   {params['rgb_teacher']:,}")
    print(f"  Audio Student: {params['audio_student']:,}")
    print(f"  Total:         {params['total']:,}")
    
    # ==========================================
    # Loss & Optimizer
    # ==========================================
    if args.use_adaptive_loss:
        criterion = AdaptiveDistillationLoss(
            max_epochs=nb_epochs,
            temperature=args.temperature,
            lambda_sparse=args.lambda_sparse
        )
        print(f"\nUsing Adaptive Distillation Loss (curriculum learning)")
    else:
        criterion = DistillationLoss(
            lambda_task=args.lambda_task,
            lambda_response=args.lambda_response,
            lambda_feature=args.lambda_feature,
            lambda_bin=args.lambda_bin,
            lambda_sparse=args.lambda_sparse,
            temperature=args.temperature
        )
        print(f"\nUsing Standard Distillation Loss")
    
    # Optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.mode.learning_rate
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.mode.learning_rate,
            weight_decay=0.01
        )
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.mode.learning_rate,
            momentum=0.9
        )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=nb_epochs, eta_min=cfg.mode.learning_rate * 0.01
    )
    
    # ==========================================
    # W&B
    # ==========================================
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment_name,
            config={
                'model': 'adabins_distillation',
                'dataset': cfg.dataset.name,
                'batch_size': batch_size,
                'learning_rate': cfg.mode.learning_rate,
                'optimizer': args.optimizer,
                'epochs': nb_epochs,
                'n_bins': args.n_bins,
                'temperature': args.temperature,
                'freeze_rgb': args.freeze_rgb,
                'adaptive_loss': args.use_adaptive_loss,
            }
        )
        print(f"W&B initialized: {args.wandb_project}/{experiment_name}")
    
    # ==========================================
    # Training Loop
    # ==========================================
    print("\nStarting training...")
    best_rmse = float('inf')
    
    for epoch in range(1, nb_epochs + 1):
        model.train()
        
        # Update adaptive loss epoch
        if args.use_adaptive_loss:
            criterion.set_epoch(epoch)
        
        epoch_losses = []
        epoch_metrics = {'task': [], 'response': [], 'feature': [], 'bin': [], 'sparse': []}
        
        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch (depends on dataset)
            if cfg.dataset.name == 'batvisionv2':
                audio, rgb, gtdepth = batch
                audio = audio.to(device)
                rgb = rgb.to(device)
                gtdepth = gtdepth.to(device)
            else:  # batvisionv1 - audio only
                audio, gtdepth = batch
                audio = audio.to(device)
                gtdepth = gtdepth.to(device)
                rgb = None
            
            # Forward pass
            output = model(audio, rgb=rgb, mode='train')
            
            # Compute loss
            valid_mask = gtdepth > 0
            loss, loss_dict = criterion(output, gtdepth, valid_mask)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track losses
            epoch_losses.append(loss.item())
            for key in epoch_metrics:
                if key in loss_dict:
                    epoch_metrics[key].append(loss_dict[key])
        
        scheduler.step()
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
        
        print(f"\nEpoch {epoch}/{nb_epochs}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"    Task:     {avg_metrics['task']:.4f}")
        print(f"    Response: {avg_metrics['response']:.4f}")
        print(f"    Feature:  {avg_metrics['feature']:.4f}")
        print(f"    Sparse:   {avg_metrics['sparse']:.4f}")
        
        # ==========================================
        # Validation
        # ==========================================
        if epoch % 2 == 0 or epoch == nb_epochs:
            model.eval()
            val_metrics = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # Unpack batch (depends on dataset)
                    if cfg.dataset.name == 'batvisionv2':
                        audio, rgb, gtdepth = batch
                        audio = audio.to(device)
                        rgb = rgb.to(device)
                        gtdepth = gtdepth.to(device)
                    else:  # batvisionv1 - audio only
                        audio, gtdepth = batch
                        audio = audio.to(device)
                        gtdepth = gtdepth.to(device)
                        rgb = None
                    
                    # Forward (inference mode - audio only)
                    output = model(audio, rgb=None, mode='inference')
                    
                    # Compute metrics
                    pred_depth = output['audio']['final_depth']
                    valid_mask = gtdepth > 0
                    
                    if valid_mask.sum() > 0:
                        errors = compute_errors(
                            gtdepth[valid_mask].cpu().numpy(),
                            pred_depth[valid_mask].cpu().numpy()
                        )
                        # Convert tuple to dictionary
                        abs_rel, rmse, a1, a2, a3, log_10, mae = errors
                        errors_dict = {
                            'abs_rel': abs_rel,
                            'rmse': rmse,
                            'delta1': a1,
                            'delta2': a2,
                            'delta3': a3,
                            'log_10': log_10,
                            'mae': mae
                        }
                        val_metrics.append(errors_dict)
                    
                    # Visualize first batch
                    if batch_idx == 0:
                        save_path = os.path.join(results_dir, f'epoch_{epoch:04d}_distill.png')
                        visualize_distillation_batch(
                            audio, rgb, gtdepth,
                            output['audio']['final_depth'],
                            None,  # No RGB pred at inference
                            output['audio']['bin_centers'],
                            None,
                            save_path
                        )
            
            # Average metrics
            if len(val_metrics) > 0:
                avg_val_metrics = {}
                for key in val_metrics[0].keys():
                    avg_val_metrics[key] = np.mean([m[key] for m in val_metrics])
                
                print(f"  Validation:")
                print(f"    RMSE:    {avg_val_metrics['rmse']:.4f}")
                print(f"    ABS_REL: {avg_val_metrics['abs_rel']:.4f}")
                print(f"    DELTA1:  {avg_val_metrics['delta1']:.4f}")
            else:
                print(f"  Validation: No valid samples (skipping metrics)")
                avg_val_metrics = {'rmse': float('inf'), 'abs_rel': float('inf'), 'delta1': 0.0}
            
            # Save best model
            if avg_val_metrics['rmse'] < best_rmse and avg_val_metrics['rmse'] != float('inf'):
                best_rmse = avg_val_metrics['rmse']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_rmse': best_rmse,
                }, os.path.join(results_dir, 'best_model.pth'))
                print(f"    ✅ Best model saved (RMSE={best_rmse:.4f})")
            
            # W&B logging
            if WANDB_AVAILABLE and args.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': avg_loss,
                    'train/task': avg_metrics['task'],
                    'train/response': avg_metrics['response'],
                    'train/feature': avg_metrics['feature'],
                    'train/sparse': avg_metrics['sparse'],
                    'lr': optimizer.param_groups[0]['lr'],
                }
                # Only log validation metrics if available
                if len(val_metrics) > 0:
                    log_dict.update({
                        'val/rmse': avg_val_metrics['rmse'],
                        'val/abs_rel': avg_val_metrics['abs_rel'],
                        'val/delta1': avg_val_metrics['delta1'],
                    })
                wandb.log(log_dict)
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(results_dir, f'checkpoint_epoch_{epoch:04d}.pth'))
    
    print("\n" + "=" * 60)
    print(f"Training completed! Best RMSE: {best_rmse:.4f}")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)
    
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()


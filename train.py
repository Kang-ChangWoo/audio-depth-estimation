
from dataloader.BatvisionV1_Dataset import BatvisionV1Dataset
from dataloader.BatvisionV2_Dataset import BatvisionV2Dataset

from models.utils_models import *

from models.unetbaseline_model import *

from utils_criterion import compute_errors
from utils_visualization import save_batch_visualization
from utils_loss import SIlogLoss

import time
import os 
import numpy as np 
import math
import pickle

import torch
from torch.utils.data import DataLoader
import argparse

from config_loader import load_config
import wandb
WANDB_AVAILABLE = True


def main():
    parser = argparse.ArgumentParser(
        description='Train U-Net model on Batvision dataset for depth estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  python train.py --dataset batvisionv2 --use_wandb
  
  # Training with custom hyperparameters
  python train.py --dataset batvisionv2 --batch_size 128 --learning_rate 0.001 --criterion SIlog
  
  # Combined loss with custom weights (auto-detected, no --criterion needed)
  python train.py --l1_weight 0.8 --silog_weight 0.2
  
  # Disable SIlog (Combined mode auto-detected)
  python train.py --use_silog false
  
  # Resume from checkpoint
  python train.py --checkpoints 50 --experiment_name my_experiment
  
  # Image-based baseline (for comparison)
  python train.py --eval_img --max_depth 80.0
  
  # Track best model with W&B (saves best model based on validation RMSE)
  python train.py --use_wandb --save_best_model --best_metric rmse
  
  # Sequence holdout to check overfitting
  python train.py --sequence_holdout --holdout_test_seq Salle_Chevalier --holdout_eval_seq 3rd_Floor_Luxembourg
  
  # Full overfitting check with best model tracking
  python train.py --use_wandb --save_best_model --sequence_holdout --holdout_test_seq Salle_Chevalier
        """
    )
    
    # ========== Dataset & Model ==========
    group_data = parser.add_argument_group('Dataset & Model')
    group_data.add_argument('--dataset', type=str, default='batvisionv2', 
                           choices=['batvisionv1', 'batvisionv2'],
                           help='Dataset to use (default: batvisionv2)')
    group_data.add_argument('--audio_format', type=str, default=None, 
                           choices=['spectrogram', 'mel_spectrogram', 'waveform'],
                           help='Audio format (overrides config). Note: mel_spectrogram not supported for BV1')
    group_data.add_argument('--eval_img', action='store_true', default=False,
                           help='Use RGB camera images instead of audio (for baseline comparison)')
    group_data.add_argument('--max_depth', type=float, default=None,
                           help='Maximum depth value in meters (default: 30.0, use 80.0 for image-based)')
    group_data.add_argument('--sequence_holdout', action='store_true', default=False,
                           help='Enable sequence-level holdout: exclude one sequence for test, one for eval')
    group_data.add_argument('--holdout_test_seq', type=str, default=None,
                           help='Sequence name to hold out for testing (e.g., "Salle_Chevalier")')
    group_data.add_argument('--holdout_eval_seq', type=str, default=None,
                           help='Sequence name to hold out for evaluation (e.g., "3rd_Floor_Luxembourg")')
    
    # ========== Training Hyperparameters ==========
    group_train = parser.add_argument_group('Training Hyperparameters')
    group_train.add_argument('--batch_size', type=int, default=None,
                            help='Batch size (overrides config, paper default: 256)')
    group_train.add_argument('--learning_rate', '--lr', type=float, default=None,
                            help='Learning rate (overrides config, paper: 0.002 for BV2, 0.001 for BV1)')
    group_train.add_argument('--optimizer', type=str, default=None, 
                            choices=['Adam', 'AdamW', 'SGD'],
                            help='Optimizer (overrides config)')
    
    # ========== Loss Function ==========
    group_loss = parser.add_argument_group('Loss Function')
    group_loss.add_argument('--criterion', type=str, default=None, 
                           choices=['L1', 'SIlog', 'Combined'],
                           help='Loss function (optional): L1, SIlog, or Combined. '
                                'Auto-detected if loss weights are specified.')
    group_loss.add_argument('--use_silog', type=lambda x: (str(x).lower() == 'true'), default=None,
                           help='Enable/disable SIlog loss. If specified, auto-enables Combined mode. '
                                'Default: True if Combined, N/A otherwise')
    group_loss.add_argument('--silog_lambda', type=float, default=None,
                           help='SIlog lambda parameter (default: 0.5, controls scale-invariance)')
    group_loss.add_argument('--l1_weight', type=float, default=None,
                           help='L1 loss weight. If specified, auto-enables Combined mode (default: 0.5)')
    group_loss.add_argument('--silog_weight', type=float, default=None,
                           help='SIlog loss weight. If specified, auto-enables Combined mode (default: 0.5)')
    
    # ========== Validation & Logging ==========
    group_val = parser.add_argument_group('Validation & Logging')
    group_val.add_argument('--validation', type=lambda x: (str(x).lower() == 'true'), default=None,
                          help='Enable validation (True/False, overrides config)')
    group_val.add_argument('--validation_iter', type=int, default=None,
                          help='Validation frequency in epochs (overrides config)')
    group_val.add_argument('--use_wandb', action='store_true', default=False,
                          help='Enable Weights & Biases logging')
    group_val.add_argument('--save_best_model', action='store_true', default=False,
                          help='Save best model based on validation RMSE (requires --use_wandb)')
    group_val.add_argument('--best_metric', type=str, default='rmse',
                          choices=['rmse', 'abs_rel', 'delta1', 'mae', 'loss'],
                          help='Metric to use for best model selection (default: rmse, lower is better for all except delta1)')
    group_val.add_argument('--wandb_project', type=str, default='batvision-depth-estimation',
                          help='W&B project name')
    group_val.add_argument('--wandb_entity', type=str, default='branden',
                          help='W&B entity/team name (default: branden)')
    group_val.add_argument('--wandb_mode', type=str, default='online', 
                          choices=['online', 'offline', 'disabled'],
                          help='W&B logging mode')
    
    # ========== Experiment Management ==========
    group_exp = parser.add_argument_group('Experiment Management')
    group_exp.add_argument('--experiment_name', type=str, default='default',
                          help='Experiment name suffix (auto-generated name includes model/dataset info)')
    group_exp.add_argument('--checkpoints', type=int, default=None,
                          help='Checkpoint epoch to resume from (None to start from scratch)')
    
    args = parser.parse_args()
    
    # If running in wandb sweep, initialize wandb early to access config
    # wandb.init() will automatically load sweep config if we're in a sweep
    sweep_mode = False
    if WANDB_AVAILABLE and (args.use_wandb or os.environ.get('WANDB_SWEEP_ID')):
        try:
            # Initialize wandb early to get sweep config
            # If we're in a sweep, wandb.config will be populated automatically
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                mode='disabled' if not args.use_wandb else args.wandb_mode,
                reinit=True  # Allow reinitialization
            )
            
            # Check if we're in a sweep
            if wandb.run is not None and wandb.run.sweep_id is not None:
                sweep_mode = True
                sweep_config = wandb.config
                print(f"Running in wandb sweep: {wandb.run.sweep_id}")
                print(f"Sweep config: {dict(sweep_config)}")
                
                # Override args with sweep config values
                if hasattr(sweep_config, 'dataset') and sweep_config.dataset:
                    args.dataset = sweep_config.dataset
                if hasattr(sweep_config, 'batch_size') and sweep_config.batch_size:
                    args.batch_size = sweep_config.batch_size
                if hasattr(sweep_config, 'learning_rate') and sweep_config.learning_rate:
                    args.learning_rate = sweep_config.learning_rate
                if hasattr(sweep_config, 'criterion') and sweep_config.criterion:
                    args.criterion = sweep_config.criterion
                if hasattr(sweep_config, 'optimizer') and sweep_config.optimizer:
                    args.optimizer = sweep_config.optimizer
                if hasattr(sweep_config, 'silog_lambda') and sweep_config.silog_lambda is not None:
                    args.silog_lambda = sweep_config.silog_lambda
                if hasattr(sweep_config, 'l1_weight') and sweep_config.l1_weight is not None:
                    args.l1_weight = sweep_config.l1_weight
                if hasattr(sweep_config, 'silog_weight') and sweep_config.silog_weight is not None:
                    args.silog_weight = sweep_config.silog_weight
                if hasattr(sweep_config, 'audio_format') and sweep_config.audio_format:
                    args.audio_format = sweep_config.audio_format
                if hasattr(sweep_config, 'validation') and sweep_config.validation is not None:
                    args.validation = sweep_config.validation
                if hasattr(sweep_config, 'validation_iter') and sweep_config.validation_iter is not None:
                    args.validation_iter = sweep_config.validation_iter
                
                # Set experiment name from sweep
                if wandb.run.sweep_id:
                    args.experiment_name = f"sweep_{wandb.run.sweep_id}"
                
                # Enable wandb
                args.use_wandb = True
                
                # Don't finish wandb.init() here - we'll update the run later with full config
                # wandb.run will remain active and we'll update it in the later wandb.init() section
        except Exception as e:
            # Not in a sweep or wandb not properly configured, continue normally
            print(f"Note: Not running in wandb sweep or wandb init failed: {e}")
            if WANDB_AVAILABLE:
                try:
                    # Only finish if we're not in a sweep
                    if not sweep_mode:
                        wandb.finish()
                except:
                    pass
    
    # Load configuration
    cfg = load_config(dataset_name=args.dataset, mode='train', experiment_name=args.experiment_name)
    
    # Override checkpoint if provided
    if args.checkpoints is not None:
        cfg.mode.checkpoints = args.checkpoints
    
    # Override max_depth if provided (useful for image-based models)
    if args.max_depth is not None:
        cfg.dataset.max_depth = args.max_depth
        print(f"Max depth overridden to: {cfg.dataset.max_depth}m")
        if args.eval_img and args.max_depth > 30.0:
            print(f"Note: Using larger max_depth for image-based model to capture far objects")
    
    # Override batch size and learning rate based on paper settings if not provided
    if args.batch_size is not None:
        cfg.mode.batch_size = args.batch_size
    elif cfg.mode.batch_size != 256:
        # Paper uses batch size 256
        print(f"Note: Paper uses batch_size=256, current config has {cfg.mode.batch_size}")
    
    if args.learning_rate is not None:
        # Validate learning rate range (sanity check - should match sweep_config.yaml)
        if args.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {args.learning_rate}")
        if args.learning_rate > 0.1:
            raise ValueError(
                f"ERROR: Learning rate {args.learning_rate} exceeds safe maximum (0.1). "
                f"This will cause training instability and model divergence. "
                f"Expected range from sweep_config.yaml: 0.0001-0.01. "
                f"Please check sweep_config.yaml learning_rate settings."
            )
        if args.learning_rate > 0.01:
            print(f"⚠️  WARNING: Learning rate {args.learning_rate} exceeds sweep config max (0.01). "
                  f"This may indicate a sweep configuration issue.")
        cfg.mode.learning_rate = args.learning_rate
        print(f"Learning rate set to: {cfg.mode.learning_rate}")
    else:
        # Paper: BV2 uses 0.002, BV1 uses 0.001
        if args.dataset == 'batvisionv2' and cfg.mode.learning_rate != 0.002:
            print(f"Note: Paper uses learning_rate=0.002 for BV2, current config has {cfg.mode.learning_rate}")
        elif args.dataset == 'batvisionv1' and cfg.mode.learning_rate != 0.001:
            print(f"Note: Paper uses learning_rate=0.001 for BV1, current config has {cfg.mode.learning_rate}")
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")
    
    if cfg.mode.mode != 'train':
        raise Exception('This script is for training only. Please run test.py for evaluation')
    if cfg.model.name != 'unet_baseline':
        raise Exception('This script if for training on unet model only')

    # ------------ GPU config ------------
    # Check CUDA_VISIBLE_DEVICES environment variable
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible}")
    
    if torch.cuda.is_available():
        n_GPU = torch.cuda.device_count()
        # Get list of available GPU IDs (respects CUDA_VISIBLE_DEVICES)
        # Limit to 4 GPUs max to avoid peer mapping resource exhaustion
        max_gpus = min(n_GPU, 4)
        gpu_ids = list(range(max_gpus))
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"{n_GPU} GPU(s) available, using {len(gpu_ids)} GPU(s): {gpu_ids}")
        print(f"Using device: {device}")
        # Print GPU name for verification
        if n_GPU > 0:
            print(f"GPU 0 name: {torch.cuda.get_device_name(0)}")
        if n_GPU > 4:
            print(f"Note: Limited to 4 GPUs to avoid peer mapping issues. Use CUDA_VISIBLE_DEVICES to select specific GPUs.")
    else:
        n_GPU = 0
        gpu_ids = []
        device = torch.device('cpu')
        print("WARNING: CUDA not available, using CPU")
        print("This may indicate:")
        print("  1. PyTorch was not installed with CUDA support")
        print("  2. No GPU is available")
        print("  3. CUDA_VISIBLE_DEVICES is set incorrectly")

    batch_size = cfg.mode.batch_size
    
    # ------------ Create experiment name -----------
    experiment_name = cfg.model.generator + '_' +  cfg.dataset.name + '_' + 'BS' + str(cfg.mode.batch_size) + '_' + 'Lr' + str(cfg.mode.learning_rate) + '_' + cfg.mode.optimizer
    if args.eval_img:
        experiment_name += '_IMG'  # Mark experiments using images
    if args.max_depth is not None and args.max_depth != 30.0:
        experiment_name += f'_MD{int(args.max_depth)}'  # Mark non-standard max_depth
    
    # ------------ Setup sequence holdout -----------
    holdout_sequences = []
    if args.sequence_holdout:
        if args.holdout_test_seq:
            holdout_sequences.append(args.holdout_test_seq)
        if args.holdout_eval_seq:
            holdout_sequences.append(args.holdout_eval_seq)
        
        if len(holdout_sequences) == 0:
            raise ValueError("--sequence_holdout requires --holdout_test_seq and/or --holdout_eval_seq")
        
        experiment_name += '_holdout_' + '_'.join(holdout_sequences)
        print(f"\n{'='*60}")
        print(f"SEQUENCE HOLDOUT MODE ENABLED")
        print(f"{'='*60}")
        print(f"Held out sequences (excluded from training): {holdout_sequences}")
        print(f"These sequences will be used to check overfitting")
        print(f"{'='*60}\n")
    
    experiment_name += '_' + cfg.mode.experiment_name
    
    # ------------ Create dataset -----------
    
    # Prepare location blacklist for sequence holdout
    location_blacklist = holdout_sequences if args.sequence_holdout else None
        
    # Use corresponding dataset class
    if cfg.dataset.name == 'batvisionv1':
        if args.eval_img:
            raise ValueError("BatvisionV1 dataset does not support --eval_img. Use batvisionv2 instead.")
        train_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_train, location_blacklist=location_blacklist)
        if cfg.mode.validation:
            val_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_val, location_blacklist=location_blacklist)
    elif cfg.dataset.name == 'batvisionv2':
        train_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_train, location_blacklist=location_blacklist, use_image=args.eval_img) 
        if cfg.mode.validation:
            val_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_val, location_blacklist=location_blacklist, use_image=args.eval_img)
    else:
        raise Exception('Training can be done only on BV1 and BV2')

    print(f'Train Dataset of {len(train_set)} instances')
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=cfg.mode.shuffle, num_workers=cfg.mode.num_threads) 

    if cfg.mode.validation:
        print(f'Validation Dataset of {len(val_set)} instances')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=cfg.mode.shuffle, num_workers=cfg.mode.num_threads)
    
    # Create holdout test/eval loaders if specified
    holdout_test_loader = None
    holdout_eval_loader = None
    if args.sequence_holdout:
        if args.holdout_test_seq:
            if cfg.dataset.name == 'batvisionv1':
                # Create dataset with ONLY the test sequence
                holdout_test_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_train, location_blacklist=None)
                # Filter to only include the holdout_test_seq
                holdout_test_set.instances = holdout_test_set.instances[
                    holdout_test_set.instances['audio path'].str.contains(args.holdout_test_seq)
                ]
            else:
                holdout_test_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_train, location_blacklist=None, use_image=args.eval_img)
                holdout_test_set.instances = holdout_test_set.instances[
                    holdout_test_set.instances['audio path'].str.contains(args.holdout_test_seq)
                ]
            print(f'Holdout Test Set ({args.holdout_test_seq}): {len(holdout_test_set)} instances')
            holdout_test_loader = DataLoader(holdout_test_set, batch_size=batch_size, shuffle=False, num_workers=cfg.mode.num_threads)
        
        if args.holdout_eval_seq:
            if cfg.dataset.name == 'batvisionv1':
                holdout_eval_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_train, location_blacklist=None)
                holdout_eval_set.instances = holdout_eval_set.instances[
                    holdout_eval_set.instances['audio path'].str.contains(args.holdout_eval_seq)
                ]
            else:
                holdout_eval_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_train, location_blacklist=None, use_image=args.eval_img)
                holdout_eval_set.instances = holdout_eval_set.instances[
                    holdout_eval_set.instances['audio path'].str.contains(args.holdout_eval_seq)
                ]
            print(f'Holdout Eval Set ({args.holdout_eval_seq}): {len(holdout_eval_set)} instances')
            holdout_eval_loader = DataLoader(holdout_eval_set, batch_size=batch_size, shuffle=False, num_workers=cfg.mode.num_threads)

    # ---------- Load Model ----------
    # Set input channels: 3 for RGB images, 2 for binaural audio
    input_nc = 3 if args.eval_img else 2
    if args.eval_img:
        print("Using camera images (RGB, 3 channels) as input instead of audio")
    
    model = define_G(cfg, input_nc = input_nc, output_nc = 1, ngf = 64, netG = 'unet_256', norm = 'batch',
                                    use_dropout = False, init_type='normal', init_gain=0.02, gpu_ids = gpu_ids) # cfg.model.generator

    print('Model used:', cfg.model.generator)
    print(f'Input channels: {input_nc} ({"RGB image" if args.eval_img else "binaural audio"})')
    if len(gpu_ids) > 1:
        print(f'Using DataParallel on {len(gpu_ids)} GPUs: {gpu_ids}')
   
    # ---------- Criterion & Optimizers ----------
    max_depth = cfg.dataset.max_depth if cfg.dataset.max_depth else 30.0
    
    # Override config with command line arguments (for sweep)
    # Smart criterion inference: if loss weights or use_silog are specified, auto-set to Combined
    if args.criterion is not None:
        cfg.mode.criterion = args.criterion
    elif args.l1_weight is not None or args.silog_weight is not None or args.use_silog is not None:
        # User specified loss configuration -> automatically use Combined
        cfg.mode.criterion = 'Combined'
        print(f"Auto-detecting Combined loss mode (loss configuration specified)")
    
    if args.optimizer is not None:
        cfg.mode.optimizer = args.optimizer
    if args.silog_lambda is not None:
        cfg.mode.silog_lambda = args.silog_lambda
    if args.l1_weight is not None:
        cfg.mode.l1_weight = args.l1_weight
    if args.silog_weight is not None:
        cfg.mode.silog_weight = args.silog_weight
    if args.audio_format is not None:
        # Validate audio_format compatibility with dataset
        if args.dataset == 'batvisionv1' and args.audio_format == 'mel_spectrogram':
            raise ValueError(
                f"mel_spectrogram is not supported for batvisionv1. "
                f"Use 'spectrogram' or 'waveform' instead."
            )
        cfg.dataset.audio_format = args.audio_format
        print(f"Audio format overridden to: {args.audio_format}")
    
    # Loss function setup
    if cfg.mode.criterion == 'L1':
        criterion = nn.L1Loss().to(device)
        l1_criterion = None
        silog_criterion = None
        l1_weight = 0.0
        silog_weight = 0.0
        use_silog_loss = False
        print(f"Using loss function: L1")
    elif cfg.mode.criterion == 'SIlog':
        lambda_scale = getattr(cfg.mode, 'silog_lambda', 0.5)
        criterion = SIlogLoss(lambda_scale=lambda_scale).to(device)
        l1_criterion = None
        silog_criterion = None
        l1_weight = 0.0
        silog_weight = 0.0
        use_silog_loss = True
        print(f"Using loss function: SIlog (lambda={lambda_scale})")
    elif cfg.mode.criterion == 'Combined':
        # Combined loss: L1 + optional SIlog
        l1_weight = getattr(cfg.mode, 'l1_weight', 0.5)
        silog_weight = getattr(cfg.mode, 'silog_weight', 0.5)
        silog_lambda = getattr(cfg.mode, 'silog_lambda', 0.5)
        
        # Determine if SIlog should be used
        # Priority: 1) explicit args.use_silog, 2) silog_weight=0, 3) default True
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
            l1_criterion = nn.L1Loss().to(device)
            silog_criterion = None
            criterion = None
            print(f"Using loss function: L1 only (SIlog disabled)")
        else:
            l1_criterion = nn.L1Loss().to(device)
            silog_criterion = SIlogLoss(lambda_scale=silog_lambda).to(device)
            criterion = None  # Will compute manually
            print(f"Using loss function: Combined (L1={l1_weight}, SIlog={silog_weight}, lambda={silog_lambda})")
    else:
        raise ValueError(f"Unknown criterion: {cfg.mode.criterion}. "
                        f"Available: L1, SIlog, Combined")
    
    learning_rate = cfg.mode.learning_rate

    if cfg.mode.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif cfg.mode.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif cfg.mode.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # ---------- Experiment Setup ---------- 
    # Note: experiment_name is already defined above (before dataset creation)
    # This allows sequence-wise split mode to update it before wandb initialization

    # Initialize Weights & Biases
    if WANDB_AVAILABLE and args.use_wandb:
        # If we already initialized wandb for sweep (sweep_mode=True), don't re-init
        # Just update the run name and config
        if sweep_mode and wandb.run is not None:
            # We're in a sweep, update run name and add additional config
            wandb.run.name = experiment_name
            wandb.config.update({
                # Model config
                'model': cfg.model.generator,
                'max_depth': cfg.dataset.max_depth,
                'depth_norm': cfg.dataset.depth_norm,
                'images_size': cfg.dataset.images_size,
                'audio_format': cfg.dataset.audio_format,
                'use_image_input': args.eval_img,
                'input_channels': 3 if args.eval_img else 2,
                'epochs': cfg.mode.epochs,
                'num_threads': cfg.mode.num_threads,
                'shuffle': cfg.mode.shuffle,
                'validation': cfg.mode.validation,
                'validation_iter': cfg.mode.validation_iter if cfg.mode.validation else None,
                'num_gpus': len(gpu_ids),
                'gpu_ids': gpu_ids,
                'device': str(device),
                # Loss config
                'l1_weight': l1_weight,
                'silog_weight': silog_weight,
                'use_silog': use_silog_loss,
            })
            wandb.run.tags = [cfg.dataset.name, cfg.model.generator, cfg.mode.criterion, cfg.mode.optimizer]
            print(f"W&B sweep run updated: {experiment_name}")
        else:
            # Normal run (not in sweep) or sweep_mode was False, initialize normally
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=experiment_name,
                mode=args.wandb_mode,
                config={
                    # Model config
                    'model': cfg.model.generator,
                    'dataset': cfg.dataset.name,
                    'max_depth': cfg.dataset.max_depth,
                    'depth_norm': cfg.dataset.depth_norm,
                    'images_size': cfg.dataset.images_size,
                    'audio_format': cfg.dataset.audio_format,
                    'use_image_input': args.eval_img,
                    'input_channels': 3 if args.eval_img else 2,
                    
                    # Training config
                    'batch_size': cfg.mode.batch_size,
                    'learning_rate': cfg.mode.learning_rate,
                    'optimizer': cfg.mode.optimizer,
                    'criterion': cfg.mode.criterion,
                    'epochs': cfg.mode.epochs,
                    'num_threads': cfg.mode.num_threads,
                    'shuffle': cfg.mode.shuffle,
                    
                    # Loss config
                    'silog_lambda': getattr(cfg.mode, 'silog_lambda', 0.5) if cfg.mode.criterion in ['SIlog', 'Combined'] else None,
                    'l1_weight': l1_weight if cfg.mode.criterion == 'Combined' else None,
                    'silog_weight': silog_weight if cfg.mode.criterion == 'Combined' else None,
                    'use_silog': use_silog_loss,
                    
                    # Validation config
                    'validation': cfg.mode.validation,
                    'validation_iter': cfg.mode.validation_iter if cfg.mode.validation else None,
                    
                    # Holdout config
                    'sequence_holdout': args.sequence_holdout,
                    'holdout_test_seq': args.holdout_test_seq if args.sequence_holdout else None,
                    'holdout_eval_seq': args.holdout_eval_seq if args.sequence_holdout else None,
                    
                    # Best model tracking
                    'save_best_model': args.save_best_model,
                    'best_metric': args.best_metric if args.save_best_model else None,
                    
                    # System config
                    'num_gpus': len(gpu_ids),
                    'gpu_ids': gpu_ids,
                    'device': str(device),
                },
                tags=[cfg.dataset.name, cfg.model.generator, cfg.mode.criterion, cfg.mode.optimizer]
            )
            print(f"W&B initialized: project={args.wandb_project}, run={experiment_name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: --use_wandb specified but wandb not installed. Install with: pip install wandb")
    
    # Create logs and results directories
    log_dir = "./logs/" + experiment_name + "/"
    results_dir = "./results/" + experiment_name + "/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    file = open(os.path.join(log_dir, "architecture.txt"), "w")

    # dataloader parameters
    file.write("Dataset name: {}\n".format(cfg.dataset.name))
    file.write("Batch size: {}\n".format(batch_size))
    file.write("Image processing: {}\n".format(cfg.dataset.preprocess))
    file.write("Image resize: {}\n".format(cfg.dataset.images_size))
    file.write("Depth norm: {}\n".format(cfg.dataset.depth_norm))
    if args.eval_img:
        file.write("Input type: Camera RGB images (3 channels)\n")
    else:
        file.write("Audio type used for training: {}\n".format(cfg.dataset.audio_format))
    
    # parameters
    file.write("Learning rate: {}\n".format(cfg.mode.learning_rate))
    file.write("Optimize used : {}\n".format(cfg.mode.optimizer))

    # net architecture
    file.write("Generator: {}\n".format(cfg.model.generator))
    
    file.write(str(model))
    file.close()


    if cfg.mode.checkpoints is None:
        checkpoint_epoch=1
    else:
        load_epoch = cfg.mode.checkpoints
        checkpoint = torch.load('./checkpoints/' + experiment_name + '/checkpoint_' + str(load_epoch) + '.pth')
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_epoch = checkpoint["epoch"] + 1 

    nb_epochs = cfg.mode.epochs

    for param_group in optimizer.param_groups:
        print("Learning rate used: {}".format(param_group['lr']))

    # ------- Best model tracking -----------
    best_metric_value = float('inf')  # For metrics where lower is better
    best_epoch = 0
    if args.save_best_model:
        if not args.use_wandb:
            print("Warning: --save_best_model requires --use_wandb. Disabling best model saving.")
            args.save_best_model = False
        else:
            print(f"Best model tracking enabled using metric: {args.best_metric}")
            # For delta1, higher is better, so we track max instead
            if args.best_metric == 'delta1':
                best_metric_value = 0.0

    train_iter = 0
    for epoch in range(checkpoint_epoch, nb_epochs + 1):

        t0 = time.time()

        batch_loss = [] 
        batch_loss_val = [] 

        # ------ Training ---------
        model.train()  

        for i,(audio, gtdepth) in enumerate(train_loader):

            audio = audio.to(device)
            gtdepth = gtdepth.to(device)   

            # set parameter gradients to zero
            optimizer.zero_grad()

            # forward
            depth_pred = model(audio)
            
            # compute loss
            # Use > 0 to include all valid pixels (0.1m threshold was too restrictive)
            valid_mask = gtdepth != 0.0  # Include all valid pixels
            
            # PROBLEM 3 FIX: Calculate loss in denormalized space (actual meters)
            if cfg.dataset.depth_norm:
                # Denormalize to meters for loss computation
                # NOTE: Do NOT clamp here - clamp has zero gradient for out-of-range values
                # which prevents learning when model outputs negative/large values
                depth_pred_denorm = depth_pred[valid_mask] * cfg.dataset.max_depth
                gtdepth_denorm = gtdepth[valid_mask] * cfg.dataset.max_depth
                
                # Compute loss based on criterion
                if cfg.mode.criterion == 'Combined':
                    loss = l1_weight * l1_criterion(depth_pred_denorm, gtdepth_denorm)
                    if use_silog_loss:
                        loss += silog_weight * silog_criterion(depth_pred_denorm, gtdepth_denorm)
                else:
                    loss = criterion(depth_pred_denorm, gtdepth_denorm)
            else:
                if cfg.mode.criterion == 'Combined':
                    loss = l1_weight * l1_criterion(depth_pred[valid_mask], gtdepth[valid_mask])
                    if use_silog_loss:
                        loss += silog_weight * silog_criterion(depth_pred[valid_mask], gtdepth[valid_mask])
                else:
                    loss = criterion(depth_pred[valid_mask], gtdepth[valid_mask])
            
            batch_loss.append(loss.item()) 
            
            # optimize
            loss.backward()  # backward-pass
            
            # Check gradient before clipping
            if epoch == 1 and i == 0:
                total_norm = 0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                total_norm = total_norm ** (1. / 2)
                print(f"Debug - Gradient norm before clipping: {total_norm:.6f}, Parameters with grad: {param_count}")
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()  # update weights
            
            train_iter +=1
            
            # Debug: print first batch info occasionally
            if epoch == 1 and i == 0:
                print(f"Debug - Audio shape: {audio.shape}, GT depth shape: {gtdepth.shape}, Pred depth shape: {depth_pred.shape}")
                print(f"Debug - Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}], mean: {audio.mean().item():.6f}")
                print(f"Debug - GT depth range: [{gtdepth[valid_mask].min().item():.4f}, {gtdepth[valid_mask].max().item():.4f}]")
                print(f"Debug - Pred depth range (all): [{depth_pred.min().item():.6f}, {depth_pred.max().item():.6f}]")
                print(f"Debug - Pred depth range (valid): [{depth_pred[valid_mask].min().item():.6f}, {depth_pred[valid_mask].max().item():.6f}]")
                print(f"Debug - Pred mean: {depth_pred[valid_mask].mean().item():.6f}, GT mean: {gtdepth[valid_mask].mean().item():.6f}")
                print(f"Debug - Valid pixels: {valid_mask.sum().item()}/{valid_mask.numel()}")
                print(f"Debug - Loss: {loss.item():.6f}")
                
                # Check if model output is stuck at zero
                if depth_pred[valid_mask].max().item() < 1e-5:
                    print("WARNING: Model output is stuck near zero! Check model initialization and gradient flow.")
            
        epoch_time = time.time()-t0
        if len(batch_loss) > 0:
            train_loss = np.mean(batch_loss)
            print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Time: {epoch_time:.1f}s')
            
            # Log to wandb
            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/epoch_time': epoch_time,
                }, step=epoch)
        else:
            print(f'Epoch {epoch}: No valid training data, Time: {epoch_time:.1f}s')

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
                    
                    # Debug: check audio input in validation
                    if batch_idx == 0:
                        print(f"Val Debug - Audio input range: [{audio_val.min().item():.6f}, {audio_val.max().item():.6f}], mean: {audio_val.mean().item():.6f}")

                    depth_pred_val = model(audio_val)
                    
                    # Debug: check model output immediately after forward pass
                    if batch_idx == 0:
                        print(f"Val Debug - Model output (raw) range: [{depth_pred_val.min().item():.6f}, {depth_pred_val.max().item():.6f}], mean: {depth_pred_val.mean().item():.6f}")
                    
                    # Use > 0 to include all valid pixels (0.1m threshold was too restrictive)
                    valid_mask_val = gtdepth_val > 0  # Include all valid pixels
                    
                    # PROBLEM 3 FIX: Calculate loss in denormalized space (actual meters)
                    if cfg.dataset.depth_norm:
                        # Denormalize to meters for loss computation
                        # NOTE: Do NOT clamp - same as training for fair comparison
                        depth_pred_val_denorm = depth_pred_val[valid_mask_val] * cfg.dataset.max_depth
                        gtdepth_val_denorm = gtdepth_val[valid_mask_val] * cfg.dataset.max_depth
                        
                        # Compute loss (same as training)
                        if cfg.mode.criterion == 'Combined':
                            loss_val = l1_weight * l1_criterion(depth_pred_val_denorm, gtdepth_val_denorm)
                            if use_silog_loss:
                                loss_val += silog_weight * silog_criterion(depth_pred_val_denorm, gtdepth_val_denorm)
                        else:
                            loss_val = criterion(depth_pred_val_denorm, gtdepth_val_denorm)
                    else:
                        if cfg.mode.criterion == 'Combined':
                            loss_val = l1_weight * l1_criterion(depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val])
                            if use_silog_loss:
                                loss_val += silog_weight * silog_criterion(depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val])
                        else:
                            loss_val = criterion(depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val])
                    batch_loss_val.append(loss_val.item()) 

                    # Store first batch for visualization
                    if batch_idx == 0:
                        if cfg.dataset.depth_norm:
                            # Denormalize for visualization
                            val_preds.append(depth_pred_val * cfg.dataset.max_depth)
                            val_gts.append(gtdepth_val * cfg.dataset.max_depth)
                        else:
                            val_preds.append(depth_pred_val)
                            val_gts.append(gtdepth_val)

                    for idx in range(depth_pred_val.shape[0]):
                        # Get depth maps
                        gt_map = gtdepth_val[idx].cpu().numpy()
                        pred_map = depth_pred_val[idx].cpu().numpy()
                        
                        # Handle channel dimension
                        if gt_map.ndim == 3:
                            gt_map = gt_map[0]  # Remove channel dim
                        if pred_map.ndim == 3:
                            pred_map = pred_map[0]
                        
                        # Debug BEFORE denormalization
                        if batch_idx == 0 and idx == 0:
                            print(f"Val Debug (normalized) - GT range: [{gt_map.min():.6f}, {gt_map.max():.6f}], "
                                  f"Pred range: [{pred_map.min():.6f}, {pred_map.max():.6f}], "
                                  f"Pred mean: {pred_map.mean():.6f}, GT mean: {gt_map.mean():.6f}")
                            
                            # Check for negative predictions (indicates training instability)
                            if pred_map.min() < 0:
                                print(f"⚠️  WARNING: Model is outputting negative values (min: {pred_map.min():.6f})!")
                                print(f"   This may indicate:")
                                print(f"   1. Learning rate too high (current: {cfg.mode.learning_rate})")
                                print(f"   2. Model not properly trained")
                                print(f"   3. Gradient explosion")
                        
                        if cfg.dataset.depth_norm:
                            # Denormalize: multiply by max_depth to get true range
                            gt_map = gt_map * cfg.dataset.max_depth
                            pred_map = pred_map * cfg.dataset.max_depth
                        
                        # Debug BEFORE clipping to check for negative predictions
                        if batch_idx == 0 and idx == 0:
                            pred_map_before_clip = pred_map.copy()
                            gt_map_before_clip = gt_map.copy()
                            if pred_map_before_clip.min() < 0:
                                print(f"⚠️  WARNING: Model is outputting negative values (min: {pred_map_before_clip.min():.3f})!")
                                print(f"   These will be clipped to epsilon to pass compute_errors check")
                                print(f"   Learning rate: {cfg.mode.learning_rate} (check if too high)")
                        
                        # FIX: Apply clamping with epsilon to ensure pred > epsilon check in compute_errors passes
                        # Use same epsilon as utils_criterion.py: 1e-3 for meters, 1e-6 for normalized
                        epsilon = 1e-3 if cfg.dataset.depth_norm else 1e-6
                        pred_map = np.clip(pred_map, epsilon, cfg.dataset.max_depth)
                        gt_map = np.maximum(gt_map, 0.0)  # Ensure GT is non-negative
                        
                        # Debug AFTER denormalization and clipping
                        if batch_idx == 0 and idx == 0:
                            print(f"Val Debug (denormalized) - GT range: [{gt_map.min():.3f}, {gt_map.max():.3f}], "
                                  f"Pred range (before clip): [{pred_map_before_clip.min():.3f}, {pred_map_before_clip.max():.3f}], "
                                  f"Pred range (after clip): [{pred_map.min():.3f}, {pred_map.max():.3f}], "
                                  f"Valid GT pixels: {(gt_map > 0).sum()}/{gt_map.size}")
                        
                        # Use default threshold (0.0) to include all valid pixels
                        # compute_errors will filter: gt > min_depth_threshold, then pred > epsilon & gt > epsilon
                        # With clipping applied, negative predictions are already handled
                        error_metrics = compute_errors(gt_map, pred_map, min_depth_threshold=0.0)
                        errors.append(error_metrics)
	
                mean_errors = np.array(errors).mean(0)
                val_loss = np.mean(batch_loss_val)
                abs_rel, rmse, delta1, delta2, delta3, log10, mae = mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4], mean_errors[5], mean_errors[6]
                
                print(f'Val - Loss: {val_loss:.6f}, RMSE: {rmse:.3f}, ABS_REL: {abs_rel:.3f}, Log10: {log10:.3f}, Delta1: {delta1:.3f}, Delta2: {delta2:.3f}, Delta3: {delta3:.3f}')
                
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
                    save_batch_visualization(pred_batch, gt_batch, vis_path, epoch, num_samples=min(4, pred_batch.shape[0]))
                    print(f'Validation visualization saved to: {vis_path}')
                    
                    # Log image to wandb
                    if WANDB_AVAILABLE and args.use_wandb:
                        log_dict['val/visualization'] = wandb.Image(vis_path, caption=f'Epoch {epoch}')
                
                # Check if this is the best model
                if args.save_best_model:
                    current_metric = {
                        'rmse': rmse,
                        'abs_rel': abs_rel,
                        'delta1': delta1,
                        'mae': mae,
                        'loss': val_loss
                    }[args.best_metric]
                    
                    is_best = False
                    if args.best_metric == 'delta1':
                        # Higher is better for delta1
                        if current_metric > best_metric_value:
                            is_best = True
                            best_metric_value = current_metric
                            best_epoch = epoch
                    else:
                        # Lower is better for other metrics
                        if current_metric < best_metric_value:
                            is_best = True
                            best_metric_value = current_metric
                            best_epoch = epoch
                    
                    if is_best:
                        print(f"🎯 New best model! {args.best_metric}={best_metric_value:.4f} at epoch {epoch}")
                        # Save best model
                        best_model_path = './checkpoints/' + experiment_name + '/best_model.pth'
                        os.makedirs('./checkpoints/' + experiment_name, exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_metric': args.best_metric,
                            'best_metric_value': best_metric_value,
                        }, best_model_path)
                        print(f"Best model saved to: {best_model_path}")
                        
                        if WANDB_AVAILABLE and args.use_wandb:
                            log_dict['best_model_epoch'] = epoch
                            log_dict[f'best_{args.best_metric}'] = best_metric_value
                
                # Evaluate on holdout sequences if available
                if args.sequence_holdout and (holdout_test_loader or holdout_eval_loader):
                    model.eval()
                    
                    # Evaluate holdout test set
                    if holdout_test_loader:
                        print(f"\nEvaluating on holdout test sequence: {args.holdout_test_seq}")
                        holdout_test_errors = []
                        with torch.no_grad():
                            for audio_ho, gtdepth_ho in holdout_test_loader:
                                audio_ho = audio_ho.to(device)
                                gtdepth_ho = gtdepth_ho.to(device)
                                depth_pred_ho = model(audio_ho)
                                
                                for idx in range(depth_pred_ho.shape[0]):
                                    gt_map = gtdepth_ho[idx].cpu().numpy()
                                    pred_map = depth_pred_ho[idx].cpu().numpy()
                                    
                                    if gt_map.ndim == 3:
                                        gt_map = gt_map[0]
                                    if pred_map.ndim == 3:
                                        pred_map = pred_map[0]
                                    
                                    if cfg.dataset.depth_norm:
                                        gt_map = gt_map * cfg.dataset.max_depth
                                        pred_map = pred_map * cfg.dataset.max_depth
                                    
                                    epsilon = 1e-3 if cfg.dataset.depth_norm else 1e-6
                                    pred_map = np.clip(pred_map, epsilon, cfg.dataset.max_depth)
                                    gt_map = np.maximum(gt_map, 0.0)
                                    
                                    error_metrics = compute_errors(gt_map, pred_map, min_depth_threshold=0.0)
                                    holdout_test_errors.append(error_metrics)
                        
                        ho_test_mean_errors = np.array(holdout_test_errors).mean(0)
                        ho_test_abs_rel, ho_test_rmse, ho_test_delta1 = ho_test_mean_errors[0], ho_test_mean_errors[1], ho_test_mean_errors[2]
                        print(f"Holdout Test - RMSE: {ho_test_rmse:.3f}, ABS_REL: {ho_test_abs_rel:.3f}, Delta1: {ho_test_delta1:.3f}")
                        
                        if WANDB_AVAILABLE and args.use_wandb:
                            log_dict.update({
                                'holdout_test/abs_rel': ho_test_abs_rel,
                                'holdout_test/rmse': ho_test_rmse,
                                'holdout_test/delta1': ho_test_delta1,
                            })
                    
                    # Evaluate holdout eval set
                    if holdout_eval_loader:
                        print(f"Evaluating on holdout eval sequence: {args.holdout_eval_seq}")
                        holdout_eval_errors = []
                        with torch.no_grad():
                            for audio_ho, gtdepth_ho in holdout_eval_loader:
                                audio_ho = audio_ho.to(device)
                                gtdepth_ho = gtdepth_ho.to(device)
                                depth_pred_ho = model(audio_ho)
                                
                                for idx in range(depth_pred_ho.shape[0]):
                                    gt_map = gtdepth_ho[idx].cpu().numpy()
                                    pred_map = depth_pred_ho[idx].cpu().numpy()
                                    
                                    if gt_map.ndim == 3:
                                        gt_map = gt_map[0]
                                    if pred_map.ndim == 3:
                                        pred_map = pred_map[0]
                                    
                                    if cfg.dataset.depth_norm:
                                        gt_map = gt_map * cfg.dataset.max_depth
                                        pred_map = pred_map * cfg.dataset.max_depth
                                    
                                    epsilon = 1e-3 if cfg.dataset.depth_norm else 1e-6
                                    pred_map = np.clip(pred_map, epsilon, cfg.dataset.max_depth)
                                    gt_map = np.maximum(gt_map, 0.0)
                                    
                                    error_metrics = compute_errors(gt_map, pred_map, min_depth_threshold=0.0)
                                    holdout_eval_errors.append(error_metrics)
                        
                        ho_eval_mean_errors = np.array(holdout_eval_errors).mean(0)
                        ho_eval_abs_rel, ho_eval_rmse, ho_eval_delta1 = ho_eval_mean_errors[0], ho_eval_mean_errors[1], ho_eval_mean_errors[2]
                        print(f"Holdout Eval - RMSE: {ho_eval_rmse:.3f}, ABS_REL: {ho_eval_abs_rel:.3f}, Delta1: {ho_eval_delta1:.3f}")
                        
                        if WANDB_AVAILABLE and args.use_wandb:
                            log_dict.update({
                                'holdout_eval/abs_rel': ho_eval_abs_rel,
                                'holdout_eval/rmse': ho_eval_rmse,
                                'holdout_eval/delta1': ho_eval_delta1,
                            })
                
                # Log all validation metrics to wandb
                if WANDB_AVAILABLE and args.use_wandb and log_dict:
                    wandb.log(log_dict, step=epoch)

        # ------- Save ------------
        if epoch % cfg.mode.saving_checkpoints == 0:
            print('Save network')
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            path_check = './checkpoints/' + experiment_name + '/'
            isExist = os.path.exists(path_check)
            if not isExist:
                os.makedirs(path_check)
            torch.save(state, './checkpoints/' + experiment_name + '/checkpoint_' + str(epoch) + '.pth')
            
            # Log checkpoint to wandb
            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log({'checkpoint_saved': epoch}, step=epoch)

    # Finish wandb run
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()
        print("W&B run finished")

    
if __name__ == '__main__':
    main()
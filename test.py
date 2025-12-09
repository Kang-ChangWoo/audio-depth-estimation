
from dataloader.BatvisionV1_Dataset import BatvisionV1Dataset
from dataloader.BatvisionV2_Dataset import BatvisionV2Dataset

from models.utils_models import *

from models.unetbaseline_model import *

from utils_criterion import compute_errors
from utils_visualization import save_batch_visualization

import time
import os 
import numpy as np 
import math
import pickle

import torch
from torch.utils.data import DataLoader
import argparse

from config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description='Test U-Net model on Batvision dataset')
    parser.add_argument('--dataset', type=str, default='batvisionv2', choices=['batvisionv1', 'batvisionv2'],
                        help='Dataset to use')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment (must match training experiment). If not provided, will try to auto-detect from checkpoint path.')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Direct path to checkpoint file (e.g., ./checkpoints/exp_name/checkpoint_50.pth). Overrides experiment_name and checkpoints.')
    parser.add_argument('--checkpoints', type=int, default=50,
                        help='Checkpoint epoch to load (ignored if --checkpoint_path is provided)')
    parser.add_argument('--eval_on', type=str, default='test', choices=['test', 'val'],
                        help='Evaluate on test or val set')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Save visualization for all samples')
    parser.add_argument('--output_dir', type=str, default='./val/',
                        help='Output directory for visualizations (default: ./val/)')
    parser.add_argument('--vis_batch_size', type=int, default=4,
                        help='Number of samples per visualization image (default: 4)')
    args = parser.parse_args()
    
    # If checkpoint_path is provided, extract experiment_name from it
    if args.checkpoint_path is not None:
        # Extract experiment name from checkpoint path
        path_parts = args.checkpoint_path.split('/')
        # Find the directory name before checkpoint file
        checkpoint_file_idx = None
        for i, part in enumerate(path_parts):
            if part.startswith('checkpoint_') and part.endswith('.pth'):
                checkpoint_file_idx = i
                break
        
        if checkpoint_file_idx is not None and checkpoint_file_idx > 0:
            # Experiment name is the directory name before checkpoint file
            experiment_name_from_path = path_parts[checkpoint_file_idx - 1]
            if args.experiment_name is None:
                args.experiment_name = experiment_name_from_path
                print(f"Auto-detected experiment_name from checkpoint path: {args.experiment_name}")
    
    # Load configuration (experiment_name can be None if using checkpoint_path)
    experiment_name_for_config = args.experiment_name if args.experiment_name is not None else 'default'
    cfg = load_config(dataset_name=args.dataset, mode='test', experiment_name=experiment_name_for_config)
    
    # Override settings from command line
    if args.checkpoints is not None:
        cfg.mode.checkpoints = args.checkpoints
    cfg.mode.eval_on = args.eval_on

    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")
    
    if cfg.mode.mode != 'test':
        raise Exception('This script is for test only. Please run train.py for training')

    # ------------ GPU config ------------
    if torch.cuda.is_available():
        n_GPU = torch.cuda.device_count()
        # Get list of available GPU IDs (respects CUDA_VISIBLE_DEVICES)
        # Limit to 4 GPUs max to avoid peer mapping resource exhaustion
        max_gpus = min(n_GPU, 4)
        gpu_ids = list(range(max_gpus))
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"{n_GPU} GPU(s) available, using {len(gpu_ids)} GPU(s): {gpu_ids}")
        print(f"Using device: {device}")
        if n_GPU > 4:
            print(f"Note: Limited to 4 GPUs to avoid peer mapping issues. Use CUDA_VISIBLE_DEVICES to select specific GPUs.")
    else:
        n_GPU = 0
        gpu_ids = []
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    batch_size = cfg.mode.batch_size
    
    # ------------ Create dataset -----------
        
    # Use corresponding dataset class
    if cfg.dataset.name == 'batvisionv1':
        if cfg.mode.eval_on == 'val':
            eval_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_val)
        else:
            eval_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_test)
    elif cfg.dataset.name == 'batvisionv2':
        if cfg.mode.eval_on == 'val':
            eval_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_val) 
        else:
            eval_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_test) 
    else:
        raise Exception('Test can be done only on BV1 and BV2')

    print(f'Eval Dataset of {len(eval_set)} instances')

    eval_loader = DataLoader(eval_set, batch_size = batch_size, shuffle=False, num_workers=cfg.mode.num_threads) 

    # ---------- Load Model ----------
    
 
    model = define_G(cfg, input_nc = 2, output_nc = 1, ngf = 64, netG = cfg.model.generator, norm = 'batch',
                                    use_dropout = False, init_type='normal', init_gain=0.02, gpu_ids = gpu_ids)
    print('Network used:', cfg.model.generator)
    if len(gpu_ids) > 1:
        print(f'Using DataParallel on {len(gpu_ids)} GPUs: {gpu_ids}')
    
    if cfg.mode.criterion == 'L1':
        criterion = nn.L1Loss().to(device)
    
    # Load checkpoint
    if args.checkpoint_path is not None:
        # Direct checkpoint path provided
        checkpoint_path = args.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Extract experiment name and epoch from path for logging
        path_parts = checkpoint_path.split('/')
        checkpoint_file = None
        for part in path_parts:
            if part.startswith('checkpoint_') and part.endswith('.pth'):
                checkpoint_file = part
                break
        
        if checkpoint_file:
            load_epoch = int(checkpoint_file.replace('checkpoint_', '').replace('.pth', ''))
            # Experiment name is the directory name before checkpoint file
            checkpoint_idx = path_parts.index(checkpoint_file)
            if checkpoint_idx > 0:
                experiment_name_used = path_parts[checkpoint_idx - 1]
            else:
                experiment_name_used = 'unknown'
        else:
            load_epoch = 0
            experiment_name_used = 'unknown'
        
        print(f'Loading checkpoint from: {checkpoint_path}')
        print(f'Experiment: {experiment_name_used}, Epoch: {load_epoch}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f'✓ Checkpoint loaded successfully (epoch {load_epoch})')
        
    elif cfg.mode.checkpoints is None:
        raise AttributeError('In test mode, a checkpoint needs to be loaded. Provide --checkpoint_path or --checkpoints with --experiment_name.')
    else:
        # Use experiment_name from config or args
        if args.experiment_name is not None:
            experiment_name_used = args.experiment_name
        elif cfg.mode.experiment_name:
            experiment_name_used = cfg.mode.experiment_name
        else:
            # Try to auto-generate experiment name (same logic as train.py)
            experiment_name_used = (cfg.model.generator + '_' + cfg.dataset.name + '_' + 
                                   'BS' + str(cfg.mode.batch_size) + '_' + 
                                   'Lr' + str(cfg.mode.learning_rate) + '_' + 
                                   cfg.mode.optimizer + '_' + 
                                   (cfg.mode.experiment_name if cfg.mode.experiment_name else 'default'))
            print(f'⚠️  Warning: experiment_name not provided, auto-generated: {experiment_name_used}')
            print(f'   If this is incorrect, use --experiment_name or --checkpoint_path')
        
        load_epoch = cfg.mode.checkpoints
        checkpoint_path = './checkpoints/' + experiment_name_used + '/checkpoint_' + str(load_epoch) + '.pth'
        
        if not os.path.exists(checkpoint_path):
            # Try to find available checkpoints
            checkpoint_dir = './checkpoints/' + experiment_name_used
            if os.path.exists(checkpoint_dir):
                available = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
                available_epochs = sorted([int(f.replace('checkpoint_', '').replace('.pth', '')) for f in available])
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}\n"
                    f"Available checkpoints in {checkpoint_dir}: {available_epochs}\n"
                    f"Use --checkpoints with one of these epochs, or --checkpoint_path to specify exact file."
                )
            else:
                raise FileNotFoundError(
                    f"Checkpoint directory not found: {checkpoint_dir}\n"
                    f"Available experiments in ./checkpoints/: {os.listdir('./checkpoints/')[:10]}..."
                )
        
        print(f'Loading checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f'✓ Checkpoint loaded successfully (epoch {load_epoch})')
    

    # ------ Eval ---------
    model.eval()  # eval mode

    gt_imgs_to_save = []
    pred_imgs_to_save = []
    loss_list = []
    errors = []
    rmse_list = []
    abs_rel_list = []
    log10_list = []
    delta1_list = []
    delta2_list = []
    delta3_list = [] 
    mae_list = []
    
    # For visualization
    if args.visualize:
        vis_output_dir = os.path.join(args.output_dir, experiment_name_used if 'experiment_name_used' in dir() else 'unknown', cfg.mode.eval_on)
        os.makedirs(vis_output_dir, exist_ok=True)
        print(f"Visualization output directory: {vis_output_dir}")
        vis_batch_preds = []
        vis_batch_gts = []
        vis_batch_idx = 0
        sample_idx = 0

    with torch.no_grad():

        for batch_num, (audio, depthgt) in enumerate(eval_loader):

            audio = audio.to(device)
            depthgt = depthgt.to(device)        

            depth_pred = model(audio)

            loss_test = criterion(depth_pred[depthgt !=0], depthgt[depthgt !=0]) 
            loss_list.append(loss_test.cpu().item())

            for idx in range(depth_pred.shape[0]):
                # Get depth maps
                gt_map = depthgt[idx].detach().cpu().numpy()
                pred_map = depth_pred[idx].detach().cpu().numpy()
                
                # Handle channel dimension (remove if present)
                if gt_map.ndim == 3:
                    gt_map = gt_map[0]  # Remove channel dim
                if pred_map.ndim == 3:
                    pred_map = pred_map[0]
                
                # Store for saving (before denormalization)
                gt_imgs_to_save.append(gt_map)
                pred_imgs_to_save.append(pred_map)
                
                if cfg.dataset.depth_norm:
                    # Denormalize: multiply by max_depth to get true range
                    unscaledgt = gt_map * cfg.dataset.max_depth
                    unscaledpred = pred_map * cfg.dataset.max_depth
                    
                    # Clip negative values
                    unscaledgt = np.maximum(unscaledgt, 0)
                    unscaledpred = np.maximum(unscaledpred, 0)
                    
                    # Use default threshold (0.0) to include all valid pixels
                    abs_rel, rmse, a1, a2, a3, log_10, mae = compute_errors(unscaledgt, 
                        unscaledpred, min_depth_threshold=0.0)
                else:   
                    # Clip negative values even if not normalized
                    gt_map = np.maximum(gt_map, 0)
                    pred_map = np.maximum(pred_map, 0)
                    # Use default threshold (0.0) to include all valid pixels
                    abs_rel, rmse, a1, a2, a3, log_10, mae = compute_errors(gt_map, 
                            pred_map, min_depth_threshold=0.0)
                errors.append((abs_rel, rmse, a1, a2, a3, log_10, mae))
                # Append metrics for each sample, not just the last one in batch
                rmse_list.append(rmse)
                abs_rel_list.append(abs_rel)
                log10_list.append(log_10)
                delta1_list.append(a1)
                delta2_list.append(a2)
                delta3_list.append(a3)
                mae_list.append(mae)
                
                # Collect for visualization
                if args.visualize:
                    # Get denormalized depth for visualization
                    if cfg.dataset.depth_norm:
                        vis_pred = depth_pred[idx:idx+1] * cfg.dataset.max_depth
                        vis_gt = depthgt[idx:idx+1] * cfg.dataset.max_depth
                    else:
                        vis_pred = depth_pred[idx:idx+1]
                        vis_gt = depthgt[idx:idx+1]
                    
                    vis_batch_preds.append(vis_pred.cpu())
                    vis_batch_gts.append(vis_gt.cpu())
                    sample_idx += 1
                    
                    # Save visualization every vis_batch_size samples
                    if len(vis_batch_preds) >= args.vis_batch_size:
                        pred_batch = torch.cat(vis_batch_preds, dim=0)
                        gt_batch = torch.cat(vis_batch_gts, dim=0)
                        vis_path = os.path.join(vis_output_dir, f'batch_{vis_batch_idx:04d}_samples_{sample_idx-args.vis_batch_size:04d}-{sample_idx-1:04d}.png')
                        save_batch_visualization(pred_batch, gt_batch, vis_path, epoch=load_epoch, num_samples=args.vis_batch_size)
                        print(f"Saved visualization: {vis_path}")
                        vis_batch_preds = []
                        vis_batch_gts = []
                        vis_batch_idx += 1


        # Save remaining visualizations
        if args.visualize and len(vis_batch_preds) > 0:
            pred_batch = torch.cat(vis_batch_preds, dim=0)
            gt_batch = torch.cat(vis_batch_gts, dim=0)
            vis_path = os.path.join(vis_output_dir, f'batch_{vis_batch_idx:04d}_samples_{sample_idx-len(vis_batch_preds):04d}-{sample_idx-1:04d}.png')
            save_batch_visualization(pred_batch, gt_batch, vis_path, epoch=load_epoch, num_samples=len(vis_batch_preds))
            print(f"Saved visualization: {vis_path}")
            vis_batch_idx += 1

        mean_errors = np.array(errors).mean(0)	
        print('\n' + '='*50)
        print('Evaluation Results:')
        print('='*50)
        print('abs rel: {:.3f}'.format(mean_errors[0])) 
        print('RMSE: {:.3f}'.format(mean_errors[1])) 
        print('Delta1: {:.3f}'.format(mean_errors[2])) 
        print('Delta2: {:.3f}'.format(mean_errors[3])) 
        print('Delta3: {:.3f}'.format(mean_errors[4])) 
        print('Log10: {:.3f}'.format(mean_errors[5])) 
        print('MAE: {:.3f}'.format(mean_errors[6]))
        
        if args.visualize:
            print(f'\n✓ Visualizations saved to: {vis_output_dir}')
            print(f'  Total batches: {vis_batch_idx}')
            print(f'  Total samples: {sample_idx}') 
    
    # Save evaluation results as torch tensor dictionary
    stats_dict = {
        'loss': torch.tensor(loss_list),
        'abs_rel': torch.tensor(abs_rel_list),
        'rmse': torch.tensor(rmse_list),
        'log10': torch.tensor(log10_list),
        'delta1': torch.tensor(delta1_list),
        'delta2': torch.tensor(delta2_list),
        'delta3': torch.tensor(delta3_list),
        'mae': torch.tensor(mae_list),
        'gt_images': torch.tensor(np.array(gt_imgs_to_save)),
        'pred_imgs': torch.tensor(np.array(pred_imgs_to_save))
    }

    # Create output directory
    # Use experiment_name_used if available, otherwise fall back to cfg.mode.experiment_name
    exp_name_for_file = experiment_name_used if 'experiment_name_used' in locals() else (cfg.mode.experiment_name or 'unknown')
    
    if cfg.mode.eval_on == 'test':
        output_dir = os.path.join(cfg.mode.stat_dir, cfg.dataset.name, 'test')
        output_file = os.path.join(output_dir, f'stats_on_{cfg.dataset.name}_test_set_{exp_name_for_file}_epoch_{load_epoch}.pt')
    else:
        output_dir = os.path.join(cfg.mode.stat_dir, cfg.dataset.name, 'val')
        output_file = os.path.join(output_dir, f'stats_on_{cfg.dataset.name}_val_set_{exp_name_for_file}_epoch_{load_epoch}.pt')
    
    os.makedirs(output_dir, exist_ok=True)
    torch.save(stats_dict, output_file)
    print(f'Evaluation results saved to: {output_file}') 



if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("Exception happened during test")

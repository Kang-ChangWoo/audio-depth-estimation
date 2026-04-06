#!/usr/bin/env python3
"""Training script for AudioDepthFOA UNet."""

import argparse
import os
import time

import numpy as np
import torch

from utils.config import load_config
from utils.train_utils import build_model, build_criterion, compute_loss
from utils.visualization import save_batch_visualization
from utils.metrics import compute_errors
from data.dataloader import make_dataloader


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    gpu_ids = list(range(min(n_GPU, 4))) if n_GPU > 0 else []
    print(f"{n_GPU} {device} device(s)")

    batch_size = cfg.mode.batch_size

    # Dataset
    train_set, train_loader = make_dataloader(cfg, 'train', batch_size=batch_size)
    val_set, val_loader = make_dataloader(cfg, 'val', batch_size=batch_size)
    print(f'Train: {len(train_set)} samples, Val: {len(val_set)} samples')

    # Model
    model = build_model(cfg, gpu_ids)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {cfg.model.generator} ({total_params:.1f}M params)')

    # Loss & optimizer
    criterion, l1_criterion, silog_criterion, l1_weight, silog_weight, use_silog = build_criterion(cfg, device)
    print(f'Loss: {cfg.mode.criterion}')

    lr = cfg.mode.learning_rate
    if cfg.mode.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif cfg.mode.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Output directories
    project_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_name = (f"{cfg.model.generator}_{cfg.dataset.name}_BS{batch_size}_"
                       f"Lr{lr}_{cfg.mode.optimizer}_{cfg.mode.experiment_name}")
    ckpt_dir = os.path.join(project_dir, 'checkpoints', experiment_name)
    results_dir = os.path.join(project_dir, 'results', experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    print(f'Checkpoints: {ckpt_dir}')

    # Resume
    start_epoch = 1
    if cfg.mode.checkpoints is not None:
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{cfg.mode.checkpoints}.pth')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}')

    best_rmse = float('inf')
    best_abs_rel = float('inf')
    nb_epochs = cfg.mode.epochs

    for epoch in range(start_epoch, nb_epochs + 1):
        t0 = time.time()
        batch_loss = []

        # --- Train ---
        model.train()
        for audio, gtdepth in train_loader:
            audio, gtdepth = audio.to(device), gtdepth.to(device)
            optimizer.zero_grad()
            depth_pred = model(audio)

            valid_mask = gtdepth != 0.0
            loss = compute_loss(criterion, l1_criterion, silog_criterion,
                                l1_weight, silog_weight, use_silog,
                                depth_pred, gtdepth, valid_mask, cfg)

            batch_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_time = time.time() - t0
        train_loss = np.mean(batch_loss) if batch_loss else 0.0
        print(f'Epoch [{epoch}/{nb_epochs}] Loss: {train_loss:.4f} Time: {epoch_time:.1f}s')

        # --- Validation ---
        if cfg.mode.validation and epoch % cfg.mode.validation_iter == 0:
            model.eval()
            errors = []
            val_losses = []
            val_preds, val_gts = [], []

            with torch.no_grad():
                for batch_idx, (audio_v, gtdepth_v) in enumerate(val_loader):
                    audio_v, gtdepth_v = audio_v.to(device), gtdepth_v.to(device)
                    depth_pred_v = model(audio_v)

                    valid_mask_v = gtdepth_v != 0.0
                    lv = compute_loss(criterion, l1_criterion, silog_criterion,
                                      l1_weight, silog_weight, use_silog,
                                      depth_pred_v, gtdepth_v, valid_mask_v, cfg)
                    val_losses.append(lv.item())

                    if batch_idx == 0:
                        if cfg.dataset.depth_norm:
                            val_preds.append(depth_pred_v * cfg.dataset.max_depth)
                            val_gts.append(gtdepth_v * cfg.dataset.max_depth)
                        else:
                            val_preds.append(depth_pred_v)
                            val_gts.append(gtdepth_v)

                    for idx in range(depth_pred_v.shape[0]):
                        gt_map = gtdepth_v[idx].cpu().numpy()
                        pred_map = depth_pred_v[idx].cpu().numpy()
                        if gt_map.ndim == 3: gt_map = gt_map[0]
                        if pred_map.ndim == 3: pred_map = pred_map[0]
                        if cfg.dataset.depth_norm:
                            gt_map = gt_map * cfg.dataset.max_depth
                            pred_map = pred_map * cfg.dataset.max_depth
                        pred_map = np.clip(pred_map, 1e-3, cfg.dataset.max_depth)
                        gt_map = np.maximum(gt_map, 0.0)
                        errors.append(compute_errors(gt_map, pred_map))

            mean_errors = np.array(errors).mean(0)
            abs_rel, rmse = mean_errors[0], mean_errors[1]
            delta1, delta2, delta3 = mean_errors[2], mean_errors[3], mean_errors[4]
            val_loss = np.mean(val_losses)

            print(f'  Val Loss: {val_loss:.4f} | '
                  f'ABS_REL: {abs_rel:.4f} RMSE: {rmse:.4f} '
                  f'd1: {delta1:.4f} d2: {delta2:.4f} d3: {delta3:.4f}')

            if val_preds:
                pred_batch = torch.cat(val_preds, dim=0)
                gt_batch = torch.cat(val_gts, dim=0)
                vis_path = os.path.join(results_dir, f'epoch_{epoch:04d}_validation.png')
                save_batch_visualization(pred_batch, gt_batch, vis_path, epoch,
                                         num_samples=min(4, pred_batch.shape[0]))

            if rmse < best_rmse:
                best_rmse = rmse
                best_abs_rel = abs_rel
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_rmse': best_rmse,
                    'best_abs_rel': best_abs_rel,
                }, os.path.join(ckpt_dir, 'best_model.pth'))
                print(f'  >> Best model saved (RMSE: {best_rmse:.4f}, ABS_REL: {best_abs_rel:.4f})')

        if epoch % cfg.mode.saving_checkpoints == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(ckpt_dir, f'checkpoint_{epoch}.pth'))

    print(f'\nTraining complete. Best RMSE: {best_rmse:.4f}, Best ABS_REL: {best_abs_rel:.4f}')


def parse_args():
    p = argparse.ArgumentParser(description='Train AudioDepthFOA UNet')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--learning-rate', '--lr', type=float, default=None)
    p.add_argument('--optimizer', type=str, default=None, choices=['AdamW', 'Adam', 'SGD'])
    p.add_argument('--criterion', type=str, default=None, choices=['L1', 'SIlog', 'Combined'])
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--experiment-name', type=str, default='default')
    p.add_argument('--checkpoints', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(mode='train', experiment_name=args.experiment_name)

    if args.checkpoints is not None:
        cfg.mode.checkpoints = args.checkpoints
    if args.batch_size is not None:
        cfg.mode.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.mode.learning_rate = args.learning_rate
    if args.optimizer is not None:
        cfg.mode.optimizer = args.optimizer
    if args.criterion is not None:
        cfg.mode.criterion = args.criterion
    if args.epochs is not None:
        cfg.mode.epochs = args.epochs
    if args.num_workers is not None:
        cfg.mode.num_threads = args.num_workers

    print('=' * 60)
    print(f'AudioDepthFOA UNet — Training')
    print(f'Dataset: {cfg.dataset.name}')
    print(f'Batch size: {cfg.mode.batch_size}, LR: {cfg.mode.learning_rate}, '
          f'Optimizer: {cfg.mode.optimizer}')
    print('=' * 60)

    train(cfg)

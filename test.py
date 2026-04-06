#!/usr/bin/env python3
"""Testing/evaluation script for AudioDepthFOA UNet."""

import argparse
import os

import numpy as np
import torch

from utils.config import load_config
from utils.train_utils import build_model
from utils.test_utils import evaluate
from data.dataloader import make_dataloader


def test(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    gpu_ids = list(range(min(n_GPU, 4))) if n_GPU > 0 else []
    print(f"{n_GPU} {device} device(s)")

    batch_size = cfg.mode.batch_size
    eval_on = getattr(cfg.mode, 'eval_on', 'test')

    eval_set, eval_loader = make_dataloader(cfg, eval_on, batch_size=batch_size, shuffle=False)
    print(f'Eval [{eval_on}]: {len(eval_set)} samples')

    # Model
    model = build_model(cfg, gpu_ids)

    # Load checkpoint
    project_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_name = (f"{cfg.model.generator}_{cfg.dataset.name}_BS{cfg.mode.batch_size}_"
                       f"Lr{cfg.mode.learning_rate}_{cfg.mode.optimizer}_{cfg.mode.experiment_name}")
    ckpt_dir = os.path.join(project_dir, 'checkpoints', experiment_name)

    load_epoch = cfg.mode.checkpoints
    if load_epoch is None or str(load_epoch) == 'best':
        ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
    else:
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{load_epoch}.pth')

    if not os.path.exists(ckpt_path):
        if hasattr(cfg.mode, 'checkpoint_path') and cfg.mode.checkpoint_path:
            ckpt_path = cfg.mode.checkpoint_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f'Loading: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f'Loaded epoch {ckpt["epoch"]}')

    # Evaluate
    de = evaluate(model, eval_loader, eval_set, cfg, device)
    md = de.mean(0)

    print('\n' + '=' * 60)
    print('Test Results')
    print('=' * 60)
    print(f'ABS_REL: {md[0]:.4f}')
    print(f'RMSE:    {md[1]:.4f}')
    print(f'Delta1:  {md[2]:.4f}')
    print(f'Delta2:  {md[3]:.4f}')
    print(f'Delta3:  {md[4]:.4f}')
    print(f'Log10:   {md[5]:.4f}')
    print(f'MAE:     {md[6]:.4f}')
    print('=' * 60)

    # Save stats
    stats_dir = os.path.join(project_dir, 'eval', cfg.dataset.name, eval_on)
    os.makedirs(stats_dir, exist_ok=True)
    stats_dict = {
        'abs_rel': torch.tensor(de[:, 0]),
        'rmse': torch.tensor(de[:, 1]),
        'delta1': torch.tensor(de[:, 2]),
        'delta2': torch.tensor(de[:, 3]),
        'delta3': torch.tensor(de[:, 4]),
        'log10': torch.tensor(de[:, 5]),
        'mae': torch.tensor(de[:, 6]),
    }
    stats_path = os.path.join(stats_dir, f'stats_{cfg.mode.experiment_name}.pt')
    torch.save(stats_dict, stats_path)
    print(f'Statistics saved to: {stats_path}')


def parse_args():
    p = argparse.ArgumentParser(description='Test AudioDepthFOA UNet')
    p.add_argument('--eval-on', type=str, default='test', choices=['test', 'val'])
    p.add_argument('--experiment-name', type=str, default='default')
    p.add_argument('--checkpoints', type=str, default=None)
    p.add_argument('--checkpoint-path', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(mode='test', experiment_name=args.experiment_name)

    cfg.mode.eval_on = args.eval_on
    if args.checkpoints is not None:
        cfg.mode.checkpoints = args.checkpoints
    if args.checkpoint_path:
        cfg.mode.checkpoint_path = args.checkpoint_path

    print('=' * 60)
    print(f'AudioDepthFOA UNet — Testing')
    print(f'Dataset: {cfg.dataset.name}, Eval on: {args.eval_on}')
    print('=' * 60)

    test(cfg)

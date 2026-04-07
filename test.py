#!/usr/bin/env python3
"""Testing/evaluation script for audio-to-depth estimation."""

import argparse
import os

import numpy as np
import torch

from utils.config import load_config
from utils.train_utils import build_model, is_foa_model
from utils.test_utils import evaluate
from data.dataloader import make_dataloader


def test(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    gpu_ids = list(range(min(n_GPU, 4))) if n_GPU > 0 else []
    foa = is_foa_model(cfg)

    eval_on = getattr(cfg.mode, 'eval_on', 'test')
    eval_set, eval_loader = make_dataloader(cfg, eval_on,
                                            batch_size=cfg.mode.batch_size, shuffle=False)
    print(f'Eval [{eval_on}]: {len(eval_set)} samples')

    model = build_model(cfg, gpu_ids)

    # Load checkpoint
    project_dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = (f"{cfg.model.generator}_{cfg.dataset.name}_BS{cfg.mode.batch_size}_"
                f"Lr{cfg.mode.learning_rate}_{cfg.mode.optimizer}_{cfg.mode.experiment_name}")
    ckpt_dir = os.path.join(project_dir, 'checkpoints', exp_name)

    load_epoch = cfg.mode.checkpoints
    if load_epoch is None or str(load_epoch) == 'best':
        ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
    else:
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{load_epoch}.pth')

    if not os.path.exists(ckpt_path):
        cp = getattr(cfg.mode, 'checkpoint_path', None)
        if cp:
            ckpt_path = cp
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f'Loading: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError:
        sd = {(k[len('module.'):] if k.startswith('module.') else k): v
              for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(sd)
    print(f'Loaded epoch {ckpt["epoch"]}')

    # Evaluate
    de, foa_err = evaluate(model, eval_loader, eval_set, cfg, device)
    md = de.mean(0)

    print('\n' + '=' * 60)
    labels = ['ABS_REL', 'RMSE', 'Delta1', 'Delta2', 'Delta3', 'Log10', 'MAE']
    for i, lbl in enumerate(labels):
        print(f'{lbl:>8s}: {md[i]:.4f}')

    if foa and foa_err:
        print('-' * 30)
        print(f'  FOA_L1: {np.mean([e["foa_l1"] for e in foa_err]):.4f}')
        print(f' FOA_COS: {np.mean([e["foa_cosine"] for e in foa_err]):.4f}')
        print(f' FOA_DIR: {np.mean([e["foa_dir_cosine"] for e in foa_err]):.4f}')
    print('=' * 60)

    # Save stats
    stats_dir = os.path.join(project_dir, 'eval', cfg.dataset.name, eval_on)
    os.makedirs(stats_dir, exist_ok=True)
    stats = {lbl.lower(): torch.tensor(de[:, i]) for i, lbl in enumerate(labels)}
    if foa and foa_err:
        for k in ('foa_l1', 'foa_cosine', 'foa_dir_cosine'):
            stats[k] = torch.tensor([e[k] for e in foa_err])
    stats_path = os.path.join(stats_dir, f'stats_{cfg.mode.experiment_name}.pt')
    torch.save(stats, stats_path)
    print(f'Saved: {stats_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='foa', help='Config name (baseline, foa)')
    p.add_argument('--eval-on', type=str, default='test', choices=['test', 'val'])
    p.add_argument('--experiment-name', type=str, default='default')
    p.add_argument('--checkpoints', type=str, default=None)
    p.add_argument('--checkpoint-path', type=str, default=None)
    args = p.parse_args()

    cfg = load_config(config_name=args.config, mode='test',
                      experiment_name=args.experiment_name)
    cfg.mode.eval_on = args.eval_on
    if args.checkpoints is not None: cfg.mode.checkpoints = args.checkpoints
    if args.checkpoint_path: cfg.mode.checkpoint_path = args.checkpoint_path

    print('=' * 60)
    print(f'Model: {cfg.model.name}  Eval: {args.eval_on}')
    print('=' * 60)
    test(cfg)

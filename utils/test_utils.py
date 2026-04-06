"""Testing/evaluation utilities."""

import os
import numpy as np
import torch

from .metrics import compute_errors


def evaluate(model, eval_loader, eval_set, cfg, device):
    """Run evaluation and return error array.

    Args:
        model: trained model (already in eval mode)
        eval_loader: DataLoader for evaluation split
        eval_set: dataset object (for length)
        cfg: config namespace
        device: torch device
    Returns:
        np.ndarray of shape (N, 7) with per-sample errors
    """
    model.eval()
    errors = []
    batch_size = cfg.mode.batch_size

    with torch.no_grad():
        for batch_idx, (audio, depthgt) in enumerate(eval_loader):
            audio, depthgt = audio.to(device), depthgt.to(device)
            depth_pred = model(audio)

            for idx in range(depth_pred.shape[0]):
                gt_map = depthgt[idx].cpu().numpy()
                pred_map = depth_pred[idx].cpu().numpy()
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

                errors.append(compute_errors(gt_map, pred_map))

            if (batch_idx + 1) % 10 == 0:
                total = min((batch_idx + 1) * batch_size, len(eval_set))
                print(f'Processed {batch_idx + 1}/{len(eval_loader)} batches ({total} samples)')

    return np.array(errors)

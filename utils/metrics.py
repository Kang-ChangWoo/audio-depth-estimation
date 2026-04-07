"""Evaluation metrics for depth estimation."""

import numpy as np
import torch


def compute_errors(gt, pred):
    """Depth error metrics between predicted and ground truth.

    Returns:
        (abs_rel, rmse, a1, a2, a3, log_10, mae)
    """
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()

    mask = gt > 0
    pred, gt = pred[mask], gt[mask]
    if len(pred) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    log_10 = np.abs(np.log10(gt + 1e-8) - np.log10(pred + 1e-8)).mean()
    mae = np.abs(gt - pred).mean()

    for v in [rmse, a1, a2, a3, abs_rel, log_10, mae]:
        if v != v:
            v = 0.0
    return abs_rel, rmse, a1, a2, a3, log_10, mae


def compute_foa_errors(gt_foa, pred_foa):
    """FOA evaluation metrics (guided channels only)."""
    foa_l1 = np.abs(gt_foa - pred_foa).mean()
    dot = np.dot(gt_foa, pred_foa)
    foa_cosine = dot / (np.linalg.norm(gt_foa) + 1e-8) / (np.linalg.norm(pred_foa) + 1e-8)
    gt_dir, pred_dir = gt_foa[1:], pred_foa[1:]
    foa_dir_cosine = np.dot(gt_dir, pred_dir) / (np.linalg.norm(gt_dir) + 1e-8) / (np.linalg.norm(pred_dir) + 1e-8)
    return {'foa_l1': float(foa_l1), 'foa_cosine': float(foa_cosine), 'foa_dir_cosine': float(foa_dir_cosine)}

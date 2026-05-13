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


def _erp_cos_lat_grid(H):
    """ERP per-row cos(latitude). Top row = +π/2, bottom row = −π/2."""
    i = np.arange(H, dtype=np.float64)
    lat = (np.pi / 2.0) - np.pi * (i + 0.5) / H
    return np.cos(lat)                                       # (H,)


def compute_errors_sphere(gt, pred):
    """cos(lat)-weighted ERP depth metrics.

    Inputs are 2-D ERP maps (H, W) in metres; pixels with gt > 0 are
    valid. Each pixel gets weight cos(lat_row) so polar pixels (which
    are oversampled in equirectangular projection) do not dominate the
    average. Returns the same 7 metrics as ``compute_errors`` in the
    same order, sphere-weighted.
    """
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if gt.ndim != 2 or pred.ndim != 2:
        raise ValueError(
            f"compute_errors_sphere expects 2-D (H,W); got "
            f"gt={gt.shape}, pred={pred.shape}")
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt={gt.shape} pred={pred.shape}")

    H = gt.shape[0]
    cos_lat = _erp_cos_lat_grid(H).reshape(H, 1)            # (H,1)
    w_full = np.broadcast_to(cos_lat, gt.shape)             # (H,W)
    mask = gt > 0
    g = gt[mask]
    p = pred[mask]
    w = w_full[mask].astype(np.float64)
    wsum = w.sum()
    if g.size == 0 or wsum <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    thresh = np.maximum(g / p, p / g)
    a1 = float((w * (thresh < 1.25)).sum() / wsum)
    a2 = float((w * (thresh < 1.25 ** 2)).sum() / wsum)
    a3 = float((w * (thresh < 1.25 ** 3)).sum() / wsum)
    rmse = float(np.sqrt((w * (g - p) ** 2).sum() / wsum))
    abs_rel = float((w * np.abs(g - p) / g).sum() / wsum)
    log_10 = float((w * np.abs(
        np.log10(g + 1e-8) - np.log10(p + 1e-8))).sum() / wsum)
    mae = float((w * np.abs(g - p)).sum() / wsum)
    return abs_rel, rmse, a1, a2, a3, log_10, mae


def compute_foa_errors(gt_foa, pred_foa):
    """FOA evaluation metrics (guided channels only)."""
    foa_l1 = np.abs(gt_foa - pred_foa).mean()
    dot = np.dot(gt_foa, pred_foa)
    foa_cosine = dot / (np.linalg.norm(gt_foa) + 1e-8) / (np.linalg.norm(pred_foa) + 1e-8)
    gt_dir, pred_dir = gt_foa[1:], pred_foa[1:]
    foa_dir_cosine = np.dot(gt_dir, pred_dir) / (np.linalg.norm(gt_dir) + 1e-8) / (np.linalg.norm(pred_dir) + 1e-8)
    return {'foa_l1': float(foa_l1), 'foa_cosine': float(foa_cosine), 'foa_dir_cosine': float(foa_dir_cosine)}

"""Testing/evaluation utilities."""

import numpy as np
import torch

from .metrics import compute_errors, compute_foa_errors
from .train_utils import is_foa_model, is_echodiffusion_model


def evaluate(model, eval_loader, eval_set, cfg, device):
    """Run evaluation. Returns (depth_errors, foa_errors_list_or_None)."""
    model.eval()
    depth_errors = []
    foa = is_foa_model(cfg)
    echodiff = is_echodiffusion_model(cfg)
    foa_errors = [] if foa else None
    batch_size = cfg.mode.batch_size

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if foa:
                audio, depthgt, gt_foa_batch, _ = batch
                gt_foa_batch = gt_foa_batch.to(device)
                audio, depthgt = audio.to(device), depthgt.to(device)
                raw_out = model(audio)
                depth_pred = raw_out["pred_depth"]
                pred_foa_batch = raw_out["pred_foa"]
            elif echodiff:
                audio, depthgt, waveform = batch
                audio, depthgt = audio.to(device), depthgt.to(device)
                waveform = waveform.to(device)
                depth_pred = model(audio, waveform)
            else:
                audio, depthgt = batch[0], batch[1]
                audio, depthgt = audio.to(device), depthgt.to(device)
                depth_pred = model(audio)

            for idx in range(depth_pred.shape[0]):
                gt_map = depthgt[idx, 0].cpu().numpy()
                pred_map = depth_pred[idx, 0].cpu().numpy()
                if cfg.dataset.depth_norm:
                    gt_map = gt_map * cfg.dataset.max_depth
                    pred_map = pred_map * cfg.dataset.max_depth
                pred_map = np.clip(pred_map, 1e-3, cfg.dataset.max_depth)
                gt_map = np.maximum(gt_map, 0.0)
                depth_errors.append(compute_errors(gt_map, pred_map))

                if foa_errors is not None:
                    foa_errors.append(compute_foa_errors(
                        gt_foa_batch[idx].cpu().numpy(),
                        pred_foa_batch[idx].cpu().numpy()))

            if (batch_idx + 1) % 10 == 0:
                total = min((batch_idx + 1) * batch_size, len(eval_set))
                print(f'  {batch_idx + 1}/{len(eval_loader)} ({total} samples)')

    return np.array(depth_errors), foa_errors

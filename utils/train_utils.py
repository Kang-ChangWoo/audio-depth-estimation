"""Training utilities: model builder, criterion builder, loss computation."""

import torch
import torch.nn as nn

from models import define_G, SIlogLoss


def build_model(cfg, gpu_ids):
    """Build UNet model."""
    return define_G(cfg, input_nc=2, output_nc=1, ngf=64,
                    netG=cfg.model.generator, norm='batch',
                    use_dropout=False, init_type='normal',
                    init_gain=0.02, gpu_ids=gpu_ids)


def build_criterion(cfg, device):
    """Build loss function(s) based on config.

    Returns:
        (criterion, l1_criterion, silog_criterion, l1_weight, silog_weight, use_silog)
    """
    criterion_name = cfg.mode.criterion

    if criterion_name == 'L1':
        return nn.L1Loss().to(device), None, None, 0.0, 0.0, False
    elif criterion_name == 'SIlog':
        lambda_scale = getattr(cfg.mode, 'silog_lambda', 0.5)
        return SIlogLoss(lambda_scale=lambda_scale).to(device), None, None, 0.0, 0.0, True
    elif criterion_name == 'Combined':
        l1_weight = getattr(cfg.mode, 'l1_weight', 0.5)
        silog_weight = getattr(cfg.mode, 'silog_weight', 0.5)
        silog_lambda = getattr(cfg.mode, 'silog_lambda', 0.5)
        l1_criterion = nn.L1Loss().to(device)
        silog_criterion = SIlogLoss(lambda_scale=silog_lambda).to(device)
        return None, l1_criterion, silog_criterion, l1_weight, silog_weight, True
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}. Available: L1, SIlog, Combined")


def compute_loss(criterion, l1_criterion, silog_criterion, l1_weight, silog_weight,
                 use_silog, pred, gt, valid_mask, cfg):
    """Compute loss given the criterion setup."""
    if cfg.dataset.depth_norm:
        pred_d = pred[valid_mask] * cfg.dataset.max_depth
        gt_d = gt[valid_mask] * cfg.dataset.max_depth
    else:
        pred_d = pred[valid_mask]
        gt_d = gt[valid_mask]

    if criterion is not None:
        return criterion(pred_d, gt_d)
    else:
        loss = l1_weight * l1_criterion(pred_d, gt_d)
        if use_silog:
            loss += silog_weight * silog_criterion(pred_d, gt_d)
        return loss

"""Training utilities: model/criterion builders and helpers."""

import torch
import torch.nn as nn

from models import (
    define_G, AudioDepthFOAGenerator,
    DepthLoss, FOAGuidedLoss, SHHistogramAlignmentLoss, AudioDepthFOALoss,
)


def is_foa_model(cfg):
    return getattr(cfg.model, 'name', 'unet_baseline') == 'audio_depth_foa'


def build_model(cfg, gpu_ids):
    """Build model based on config."""
    if is_foa_model(cfg):
        num_downs = 7 if cfg.model.generator == 'unet_128' else 8
        net = AudioDepthFOAGenerator(
            cfg, input_nc=2, output_nc=1, num_downs=num_downs, ngf=64,
            use_dropout=False,
            proj_dim=getattr(cfg.model, 'proj_dim', 128),
            foa_dim=getattr(cfg.model, 'foa_dim', 4),
            sh_order=getattr(cfg.model, 'sh_order', 5),
            scale_shift_hidden=getattr(cfg.model, 'scale_shift_hidden', 256),
            scale_shift_layers=getattr(cfg.model, 'scale_shift_layers', 4),
            H_erp=int(cfg.dataset.images_size[0]),
            W_erp=int(cfg.dataset.images_size[1]),
        )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net
    else:
        return define_G(cfg, input_nc=2, output_nc=1, ngf=64,
                        netG=cfg.model.generator, norm='batch',
                        use_dropout=False, init_type='normal',
                        init_gain=0.02, gpu_ids=gpu_ids)


def build_depth_criterion(cfg, device):
    """Build the shared DepthLoss (BerHu + SILog by default)."""
    return DepthLoss(
        use_berhu=getattr(cfg.mode, 'use_berhu', True),
        use_silog=getattr(cfg.mode, 'use_silog', True),
        w_berhu=getattr(cfg.mode, 'w_berhu', 1.0),
        w_silog=getattr(cfg.mode, 'w_silog', 0.5),
    ).to(device)


def build_criterion(cfg, device):
    """Build full criterion.

    Returns:
        DepthLoss for baseline, AudioDepthFOALoss for FOA.
    """
    depth_criterion = build_depth_criterion(cfg, device)

    if is_foa_model(cfg):
        foa_criterion = FOAGuidedLoss(
            use_cosine=getattr(cfg.model, 'foa_use_cosine', True),
            cosine_weight=getattr(cfg.model, 'foa_cosine_weight', 0.1),
        ).to(device)

        hist_weight = getattr(cfg.model, 'hist_weight', 0.1)
        hist_criterion = SHHistogramAlignmentLoss().to(device) if hist_weight > 0 else None

        return AudioDepthFOALoss(
            depth_criterion=depth_criterion,
            foa_criterion=foa_criterion,
            depth_weight=getattr(cfg.model, 'depth_weight', 1.0),
            foa_weight=getattr(cfg.model, 'foa_weight', 0.1),
            hist_criterion=hist_criterion,
            hist_weight=hist_weight,
            latent_reg_weight=getattr(cfg.model, 'latent_reg_weight', 0.0),
        ).to(device)
    else:
        return depth_criterion


def get_base_model(model):
    """Unwrap DataParallel if needed."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def compute_gt_depth_sh(model, gt_depth):
    """Compute ground-truth depth SH projection."""
    base = get_base_model(model)
    with torch.no_grad():
        coeffs = base.project_depth_to_sh(gt_depth)
        sh_map = base.reconstruct_from_coeffs(coeffs)
    return sh_map, coeffs


def set_sh_branch_frozen(model, frozen):
    """Freeze/unfreeze the SH branch parameters."""
    base = get_base_model(model)
    for name in ('audio_proj', 'foa_head', 'hoa_head', 'scale_shift'):
        module = getattr(base, name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = not frozen

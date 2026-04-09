"""Training utilities: model/criterion builders and helpers."""

import torch
import torch.nn as nn

from models import (
    define_G, AudioDepthFOAGenerator, EchoDiffusion, EchoNet,
    BatVisionUNet, PretrainedViT, PretrainedResNet, AudioDepthViT,
    FOACrossAttnGenerator, FOAFeatBankGenerator,
    FOAMultiScaleAttnGenerator, FOAChannelAttnGenerator,
    DepthLoss, FOAGuidedLoss, SHHistogramAlignmentLoss, AudioDepthFOALoss,
)

_FOA_VARIANT_CLASSES = {
    'foa_crossattn': FOACrossAttnGenerator,
    'foa_featbank': FOAFeatBankGenerator,
    'foa_msattn': FOAMultiScaleAttnGenerator,
    'foa_channelattn': FOAChannelAttnGenerator,
}


def is_foa_model(cfg):
    return getattr(cfg.model, 'name', 'unet_baseline') == 'audio_depth_foa'


def is_foa_variant_model(cfg):
    """Check if config specifies an FOA variant model (crossattn, featbank, msattn, channelattn)."""
    foa_variants = ('foa_crossattn', 'foa_featbank', 'foa_msattn', 'foa_channelattn')
    return getattr(cfg.model, 'name', '') in foa_variants


def is_echodiffusion_model(cfg):
    return getattr(cfg.model, 'name', '') == 'echodiffusion'


def is_echonet_model(cfg):
    return getattr(cfg.model, 'name', '') == 'echonet'


def is_batvision_model(cfg):
    return getattr(cfg.model, 'name', '') == 'batvision'


def is_vit_model(cfg):
    return getattr(cfg.model, 'name', '') == 'vit_baseline'


def build_model(cfg, gpu_ids):
    """Build model based on config."""
    model_name = getattr(cfg.model, 'name', 'unet_baseline')

    if model_name == 'audio_depth_foa' or model_name in _FOA_VARIANT_CLASSES:
        cls = _FOA_VARIANT_CLASSES.get(model_name, AudioDepthFOAGenerator)
        num_downs = 7 if cfg.model.generator == 'unet_128' else 8
        net = cls(
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
    elif model_name == 'echonet':
        net = EchoNet(
            cfg,
            input_nc=2,
            output_nc=1,
            conv1x1_dim=getattr(cfg.model, 'conv1x1_dim', 8),
            bottleneck_dim=getattr(cfg.model, 'bottleneck_dim', 512),
        )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net
    elif model_name == 'batvision':
        num_downs = 7 if cfg.model.generator == 'unet_128' else 8
        net = BatVisionUNet(
            cfg,
            input_nc=2,
            output_nc=1,
            num_downs=num_downs,
            ngf=getattr(cfg.model, 'ngf', 64),
            use_dropout=getattr(cfg.model, 'use_dropout', False),
        )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net
    elif model_name == 'echodiffusion':
        net = EchoDiffusion(
            max_depth=getattr(cfg.model, 'max_depth', cfg.dataset.max_depth),
            embed_dim=getattr(cfg.model, 'embed_dim', 192),
            emb_dim=getattr(cfg.model, 'emb_dim', 768),
        )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net
    elif model_name == 'vit_baseline':
        img_size = tuple(int(s) for s in cfg.dataset.images_size)
        net = AudioDepthViT(
            cfg,
            img_size=img_size,
            patch_size=getattr(cfg.model, 'patch_size', 16),
            in_chans=2,
            embed_dim=getattr(cfg.model, 'embed_dim', 768),
            depth=getattr(cfg.model, 'depth', 12),
            num_heads=getattr(cfg.model, 'num_heads', 12),
            mlp_ratio=getattr(cfg.model, 'mlp_ratio', 4.0),
            drop_rate=getattr(cfg.model, 'drop_rate', 0.0),
            attn_drop_rate=getattr(cfg.model, 'attn_drop_rate', 0.0),
        )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net
    elif model_name == 'pretrained_vit':
        net = PretrainedViT(
            cfg,
            input_nc=2,
            pretrained=getattr(cfg.model, 'pretrained', True),
            freeze_encoder=getattr(cfg.model, 'freeze_encoder', False),
        )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net
    elif model_name == 'pretrained_resnet':
        net = PretrainedResNet(
            cfg,
            input_nc=2,
            pretrained=getattr(cfg.model, 'pretrained', True),
            freeze_encoder=getattr(cfg.model, 'freeze_encoder', False),
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

    if is_foa_model(cfg) or is_foa_variant_model(cfg):
        foa_criterion = FOAGuidedLoss(
            use_cosine=getattr(cfg.model, 'foa_use_cosine', True),
            cosine_weight=getattr(cfg.model, 'foa_cosine_weight', 0.1),
        ).to(device)

        hist_weight = getattr(cfg.model, 'hist_weight', 0.1)
        hist_criterion = SHHistogramAlignmentLoss().to(device) if hist_weight > 0 else None

        kl_weight = 0.0
        if is_foa_variant_model(cfg):
            kl_weight = getattr(cfg.model, 'kl_weight', 0.01)

        return AudioDepthFOALoss(
            depth_criterion=depth_criterion,
            foa_criterion=foa_criterion,
            depth_weight=getattr(cfg.model, 'depth_weight', 1.0),
            foa_weight=getattr(cfg.model, 'foa_weight', 0.1),
            hist_criterion=hist_criterion,
            hist_weight=hist_weight,
            latent_reg_weight=getattr(cfg.model, 'latent_reg_weight', 0.0),
            kl_weight=kl_weight,
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

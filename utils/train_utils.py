"""Training utilities: model/criterion builders and helpers.

Phase E1 (2026-05-13): trimmed to the keep-list models only. The N1/N2/N3/
N4/N9/renew/foa_0415/foa_variant factory tables were moved to
models/deprecated/ and their dispatch branches removed from build_model().
Predicate helpers (is_*_model) that referenced moved classes are pruned;
only the predicates for keep-list configs (audio_depth_foa, echodiffusion,
echonet, batvision, vit_baseline, echorange, pretrained_*) remain.

train.py still imports the legacy predicates (is_foa_0415_model,
is_n2_model, etc.) at module load time; we therefore keep stub predicates
that always return False so train.py imports cleanly. The actual
train-step branches inside train.py are dead code for keep-list configs.
"""

import torch
import torch.nn as nn

from models import (
    define_G, AudioDepthFOAGenerator, EchoDiffusion, EchoDiffusionAmbi, EchoDiffusionAmbiSH, EchoNet,
    EchoDiffusionSHSide, EchoDiffusionSHSidePlus,
    BatVisionUNet, PretrainedViT, PretrainedViTFOA, PretrainedResNet, AudioDepthViT,
    PretrainedViTFOAV2, PretrainedViTFOAV3, PretrainedViTFOAV4, PretrainedViTFOAV5,
    PretrainedViTFOAV6EAttn, PretrainedViTFOAV6MSSH, PretrainedViTFOAV6OracleNC3,
    EchoRangeDepth,
    DepthLoss, FOAGuidedLoss, SHHistogramAlignmentLoss, AudioDepthFOALoss,
)


# pretrained_vit_foa routing.
#
# After E1, the auxiliary-SH-head set covers only the deprecated training-step
# branches that the kept ViT-FOA variants share with each other. Any deprecated
# model name listed here is harmless: build_model() will not dispatch to it
# (no factory branch exists), but predicate functions still need to recognise
# the kept names so train.py's loss/branch selection works.
_PVITFOA_AUX_SH_NAMES = {
    'pretrained_vit_foa',
    'pretrained_vit_foa_v3',
    'pretrained_vit_foa_v4',
    'pretrained_vit_foa_v5',
    'pretrained_vit_foa_v6_eattn',
    'pretrained_vit_foa_v6_mssh',
    # n2_0427 — EchoDiffusion + SH side-prior (concise/Plus); kept for
    # echodiff_sh_side_plus.yaml.
    'echodiff_sh_side',
    'echodiff_sh_side_plus',
    # echodiffusion_ambi family — bin-gated FOA conditioning paths.
    'echodiffusion_ambi',
    'echodiffusion_ambi_sh',
}
_PVITFOA_HIST_NAMES = {
    'pretrained_vit_foa_v2',
}

_PVITFOA_CLASSES = {
    'pretrained_vit_foa':    PretrainedViTFOA,
    'pretrained_vit_foa_v2': PretrainedViTFOAV2,
    'pretrained_vit_foa_v3': PretrainedViTFOAV3,
    'pretrained_vit_foa_v4': PretrainedViTFOAV4,
    'pretrained_vit_foa_v5': PretrainedViTFOAV5,
    'pretrained_vit_foa_v6_eattn':      PretrainedViTFOAV6EAttn,
    'pretrained_vit_foa_v6_mssh':       PretrainedViTFOAV6MSSH,
    'pretrained_vit_foa_v6_oracle_nc3': PretrainedViTFOAV6OracleNC3,
}

# ViT-backbone oracle: input assembly via cat(audio, energy_map) but
# built through the _PVITFOA_CLASSES branch in build_model (ViT-specific kwargs).
_PVITFOA_ORACLE_CLASSES = {
    'pretrained_vit_foa_v6_oracle_nc3': PretrainedViTFOAV6OracleNC3,
}


def is_foa_0415_model(cfg):
    """Whether cfg selects an aux-SH-head ViT-FOA variant or echodiffusion_ambi.

    Post-E1: the UNet-based foa_0415_v{1..5} family was moved to
    models/deprecated/. This predicate now only matches the ViT-FOA v1/v3/v4/v5
    pretrains plus the echodiffusion_ambi / echodiff_sh_side(_plus) routes.
    Same train/test routing: forward returns ``{pred_depth, pred_sh, ...}`` and
    the training loss is ``L_depth + lambda_sh * L1(pred_sh[:, :4], gt_foa)``.
    """
    return getattr(cfg.model, 'name', '') in _PVITFOA_AUX_SH_NAMES


def is_emap_temporal_model(cfg):
    """Post-E1: energy-map temporal models removed (deprecated). Always False."""
    return False


def is_foa_oracle_model(cfg):
    """Whether cfg selects a Group-D oracle model (GT energy/FOA at input).

    Post-E1 the UNet oracle family was moved to deprecated. Only the ViT
    oracle (_PVITFOA_ORACLE_CLASSES) remains.
    """
    return getattr(cfg.model, 'name', '') in _PVITFOA_ORACLE_CLASSES


def is_n2_model(cfg):
    """Post-E1: N2 temporal-FOA models removed (deprecated). Always False."""
    return False


def is_n3_0425_model(cfg):
    """Post-E1: n3_0425 moved to deprecated. Always False."""
    return False


def is_echorange_model(cfg):
    """echorange — EchoDiffusion backbone with a switchable depth head.

    Routes via _train_step_echorange. Forward signature is the same as
    EchoDiffusion (audio_spec, audio_wave) but returns a dict that always
    contains 'pred_depth'. With depth_head_type='scalar' the model is
    behaviourally identical to plain echodiffusion.
    """
    return getattr(cfg.model, 'name', '') == 'echorange'


def is_foa_model(cfg):
    return getattr(cfg.model, 'name', 'unet_baseline') == 'audio_depth_foa'


def is_foa_variant_model(cfg):
    """Post-E1: legacy UNet FOA variants moved; only pretrained_vit_foa_v2 remains."""
    return getattr(cfg.model, 'name', '') in _PVITFOA_HIST_NAMES


def is_foa_v2_js_model(cfg):
    """Post-E1: foa_v2_js moved; only pretrained_vit_foa_v2 routes via the js path."""
    return getattr(cfg.model, 'name', '') in _PVITFOA_HIST_NAMES


def is_foa_v2_js_rgb_model(cfg):
    """Post-E1: foa_v2_js_rgb moved to deprecated. Always False."""
    return False


def is_echodiffusion_model(cfg):
    return getattr(cfg.model, 'name', '') == 'echodiffusion'


def is_echonet_model(cfg):
    return getattr(cfg.model, 'name', '') == 'echonet'


def is_batvision_model(cfg):
    return getattr(cfg.model, 'name', '') == 'batvision'


def is_vit_model(cfg):
    return getattr(cfg.model, 'name', '') == 'vit_baseline'


def build_oracle_teacher(ckpt_path, cfg, device, input_nc=3):
    """Post-E1: UNet FOAOracleGenerator moved to deprecated. This function is
    retained as a no-op stub for import-compatibility; calling it raises a
    clear error so the dependency is obvious if a deprecated config is run.
    """
    raise RuntimeError(
        "build_oracle_teacher requires models.deprecated.foa_oracle_nc3; "
        "the UNet oracle family was archived in Phase E1. Restore "
        "models/deprecated/foa_0415_v1.py and friends, or re-add the "
        "factory branch, before using oracle distillation.")


def build_model(cfg, gpu_ids):
    """Build model based on config (keep-list configs only after E1)."""
    model_name = getattr(cfg.model, 'name', 'unet_baseline')

    if model_name == 'audio_depth_foa':
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
    elif model_name == 'echorange':
        # EchoDiffusion backbone + switchable scalar/range/hazard depth head.
        # depth_head_type='scalar' (default) reproduces echodiffusion exactly.
        net = EchoRangeDepth(
            max_depth=getattr(cfg.model, 'max_depth', cfg.dataset.max_depth),
            embed_dim=getattr(cfg.model, 'embed_dim', 192),
            emb_dim=getattr(cfg.model, 'emb_dim', 768),
            depth_head_type=str(getattr(cfg.model, 'depth_head_type', 'scalar')),
            range_num_bins=int(getattr(cfg.model, 'range_num_bins', 64)),
            range_bin_spacing=str(getattr(cfg.model, 'range_bin_spacing', 'log')),
            range_min_depth=float(getattr(cfg.model, 'range_min_depth', 0.1)),
            range_max_depth=float(getattr(cfg.model, 'range_max_depth', 20.0)),
            range_output_mode=str(getattr(cfg.model, 'range_output_mode', 'expectation')),
            hazard_bias_init=float(getattr(cfg.model, 'hazard_bias_init', -4.6)),
        )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net
    elif model_name == 'echodiffusion_ambi':
        kwargs = dict(
            max_depth=getattr(cfg.model, 'max_depth', cfg.dataset.max_depth),
            embed_dim=getattr(cfg.model, 'embed_dim', 192),
            emb_dim=getattr(cfg.model, 'emb_dim', 768),
            K=int(getattr(cfg.model, 'K', 8)),
            foa_mode=str(getattr(cfg.model, 'foa_mode', 'condition')),
            gate_init=float(getattr(cfg.model, 'gate_init', 2.0)),
            use_cide=bool(getattr(cfg.model, 'use_cide', False)),
        )
        if hasattr(cfg.model, 'gate_mask'):
            kwargs['gate_mask'] = cfg.model.gate_mask
        net = EchoDiffusionAmbi(cfg, **kwargs)
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net
    elif model_name == 'echodiff_sh_side':
        net = EchoDiffusionSHSide(
            cfg,
            max_depth=getattr(cfg.model, 'max_depth', cfg.dataset.max_depth),
            embed_dim=getattr(cfg.model, 'embed_dim', 192),
            K=int(getattr(cfg.model, 'K', 8)),
            rep_hidden=int(getattr(cfg.model, 'rep_hidden', 512)),
            side_fusion=bool(getattr(cfg.model, 'side_fusion', True)),
            oracle_mode=bool(getattr(cfg.model, 'oracle_mode', False)),
        )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net
    elif model_name == 'echodiff_sh_side_plus':
        net = EchoDiffusionSHSidePlus(
            cfg,
            max_depth=getattr(cfg.model, 'max_depth', cfg.dataset.max_depth),
            embed_dim=getattr(cfg.model, 'embed_dim', 192),
            K=int(getattr(cfg.model, 'K', 8)),
            rep_hidden=int(getattr(cfg.model, 'rep_hidden', 512)),
            unet_channels=int(getattr(cfg.model, 'unet_channels', 64)),
            decoder_channels=list(getattr(cfg.model, 'decoder_channels',
                                          [256, 128, 64])),
            side_fusion=bool(getattr(cfg.model, 'side_fusion', True)),
            oracle_mode=bool(getattr(cfg.model, 'oracle_mode', False)),
            oracle_gate_mode=str(getattr(cfg.model, 'oracle_gate_mode', 'ones')),
        )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net
    elif model_name == 'echodiffusion_ambi_sh':
        net = EchoDiffusionAmbiSH(
            cfg,
            max_depth=getattr(cfg.model, 'max_depth', cfg.dataset.max_depth),
            embed_dim=getattr(cfg.model, 'embed_dim', 192),
            emb_dim=getattr(cfg.model, 'emb_dim', 768),
            K=int(getattr(cfg.model, 'K', 8)),
            sh_order=int(getattr(cfg.model, 'sh_order', 5)),
            sh_hidden=int(getattr(cfg.model, 'sh_hidden', 256)),
            unet_channels=int(getattr(cfg.model, 'unet_channels', 64)),
            decoder_channels=list(getattr(cfg.model, 'decoder_channels',
                                          [256, 128, 64])),
            use_cide=bool(getattr(cfg.model, 'use_cide', False)),
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
    elif model_name in _PVITFOA_CLASSES:
        cls = _PVITFOA_CLASSES[model_name]
        default_input_nc = 3 if model_name in _PVITFOA_ORACLE_CLASSES else 2
        kwargs = dict(
            input_nc=int(getattr(cfg.model, 'input_nc', default_input_nc)),
            pretrained=getattr(cfg.model, 'pretrained', True),
            freeze_encoder=getattr(cfg.model, 'freeze_encoder', False),
        )
        for k in ('sh_dim', 'head_hidden', 'n_early', 'taps',
                  'n_slots', 'num_heads',
                  'proj_dim', 'foa_dim', 'sh_order',
                  'scale_shift_hidden', 'scale_shift_layers'):
            if hasattr(cfg.model, k):
                kwargs[k] = getattr(cfg.model, k)
        if model_name == 'pretrained_vit_foa_v2':
            kwargs.setdefault('H_erp', int(cfg.dataset.images_size[0]))
            kwargs.setdefault('W_erp', int(cfg.dataset.images_size[1]))
        net = cls(cfg, **kwargs)
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

    if is_foa_v2_js_rgb_model(cfg):
        return depth_criterion

    if is_foa_oracle_model(cfg):
        return depth_criterion

    if is_foa_0415_model(cfg):
        # Variant 1 loss: depth uses BerHu + SILog; SH L1 is applied inline in
        # the train step (see train.py::_train_step_foa_0415). No FOA/hist heads.
        return depth_criterion

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
        # foa_v2 does not use KL loss
        if getattr(cfg.model, 'name', '') == 'foa_v2':
            kl_weight = 0.0

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


def compute_gt_energy_sh(model, gt_energy):
    """Compute SH projection of the ambisonic energy map.

    Used by foa_v2_js to align the predicted SH branch directly against the
    actual ambisonic-derived directional energy distribution, rather than
    against an SH projection of the depth map.
    """
    base = get_base_model(model)
    with torch.no_grad():
        coeffs = base.project_depth_to_sh(gt_energy)
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

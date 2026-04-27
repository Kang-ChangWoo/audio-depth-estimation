"""Training utilities: model/criterion builders and helpers."""

import torch
import torch.nn as nn

from models import (
    define_G, AudioDepthFOAGenerator, EchoDiffusion, EchoDiffusionAmbi, EchoDiffusionAmbiSH, EchoNet,
    EchoDiffusionSHSide, EchoDiffusionSHSidePlus,
    BatVisionUNet, PretrainedViT, PretrainedViTFOA, PretrainedResNet, AudioDepthViT,
    PretrainedViTFOAV2, PretrainedViTFOAV3, PretrainedViTFOAV4, PretrainedViTFOAV5,
    PretrainedViTFOAV6EAttn, PretrainedViTFOAV6MSSH, PretrainedViTFOAV6OracleNC3,
    FOACrossAttnGenerator, FOAFeatBankGenerator,
    FOAMultiScaleAttnGenerator, FOAChannelAttnGenerator,
    FOAv2Generator, FOAv2JSGenerator,
    FOA0415V1Generator, FOA0415V2Generator, FOA0415V3Generator,
    FOA0415V4Generator, FOA0415V5Generator,
    N3FiLMGenerator, N3MultiScaleSHGenerator,
    N3EnergyAttnGenerator, N3TemporalWindowGenerator,
    FOAOracleGenerator,
    N3FiLMEnergyAttnGenerator, N3MSSHEnergyAttnGenerator,
    EmapUNetGenerator, EmapUNetTemporalGenerator,
    EmapViTGenerator, EmapViTTemporalGenerator,
    N2TemapInputGenerator,
    N2TemporalEnergyGenerator, N2DualEncGenerator,
    N2FOASTFTFiLMGenerator, N2TemporalRMSFiLMGenerator,
    N2TBinCrossAttnGenerator,
    PVitN1TemapInput, PVitN1TemapRMSFiLM,
    PVitN1TemapEAttn, PVitN1TemapMSSH,
    RenewSingleNet, RenewDPTOnlyNet,
    N9_0424Net,
    N4_0425Net,
    N3_0425Net,
    N9_0425Net,
    N9_0426Net,
    DepthLoss, FOAGuidedLoss, SHHistogramAlignmentLoss, AudioDepthFOALoss,
)

# pretrained_vit_foa family routing:
#   v1, v3, v4, v5 -> aux SH head only (_train_step_foa_0415 path)
#   v2             -> full histogram alignment via energy-map SH
#                      (_train_step_js path: same train step as foa_v2_js)
_PVITFOA_AUX_SH_NAMES = {
    'pretrained_vit_foa',
    'pretrained_vit_foa_v3',
    'pretrained_vit_foa_v4',
    'pretrained_vit_foa_v5',
    # v6 — n3_0419 ports (exp216-217). Oracle (exp219) routes via
    # _PVITFOA_ORACLE_CLASSES below, not through this set.
    'pretrained_vit_foa_v6_eattn',
    'pretrained_vit_foa_v6_mssh',
    # Renew — dual-ViT with sound-field bottleneck. Returns pred_depth +
    # pred_sh + pred_energy; foa_0415 train step handles all three
    # losses (depth, SH-L1, energy-L1 when lambda_energy > 0).
    'renew_single',
    'renew_dpt_only',
    # n9_0424 — Implicit Sound-Field Projection Fusion (2026-04-24).
    # Forward returns pred_depth + rep_pred (B, 8, 4) + pred_sh (B, 4).
    # Routes via _train_step_foa_0415 which picks up rep_gt from the
    # 5-tuple batch (use_distance_bins=True) and applies weighted_rep_loss.
    'n9_0424',
    # n4_0425 — UNet + oracle bin-gated FOA conditioning (2026-04-25).
    # Forward consumes rep_gt as kwarg, returns pred_depth + gate (K,) +
    # rep_pred (B, K, 4) + pred_sh (B, 4). Routes via _train_step_foa_0415
    # which (a) passes rep_gt to forward(), and (b) applies the optional
    # sparsity loss `lambda_sparsity * gate.mean()` when 'gate' is present.
    'n4_0425',
    # n9_0425 — same UNet+gate as n4_0425 but rep comes from a pre-trained
    # n3_0425 (binaural→FOA) instead of dataset's oracle rep_gt. Forward
    # signature is model(audio); inner n3 produces rep_pred. Routes via
    # _train_step_foa_0415 same as n4_0425. lambda_sh > 0 fine-tunes n3
    # toward rep_gt (only meaningful when cfg.model.freeze_n3=False).
    'n9_0425',
    # n9_0426 — n9_0425 cascade but the outer UNet is replaced by a
    # pretrained ViT-B/16 or ResNet-50 hourglass that consumes
    # concat(binaural, gated_em) as 3ch input. Same forward dict so it
    # routes through _train_step_foa_0415 unchanged.
    'n9_0426',
    # echodiffusion_ambi — EchoDiffusion backbone (ASPP+ASFF + diffusion-UNet
    # feature extractor) with bin-gated FOA conditioning replacing CIDE.
    # Two foa_modes (input concat / cross-attention condition). Returns the
    # n4_0425-style {pred_depth, gate, rep_pred, pred_sh} dict.
    'echodiffusion_ambi',
    # echodiffusion_ambi_sh — EchoDiffusion + audio→SH coefficient prediction
    # → real SH-basis renderer → coarse layout → gated residual fusion at
    # decoder output. Returns {pred_depth, sh_coeff, sh_layout, rep_pred,
    # pred_sh}; rep_pred=rep_gt so the legacy weighted_rep_loss is identity.
    'echodiffusion_ambi_sh',
    # n2_0427 — EchoDiffusion + SH side-prior (concise/Plus). The SH path is
    # a side feature, never a bottleneck. Forward returns {pred_depth,
    # pred_energy, rep_pred, rep_used, pred_sh, gate}; routes through
    # _train_step_foa_0415 which picks up rep_gt from the 5-tuple batch.
    'echodiff_sh_side',
    'echodiff_sh_side_plus',
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
    'renew_single':                     RenewSingleNet,
    'renew_dpt_only':                   RenewDPTOnlyNet,
    'n9_0424':                          N9_0424Net,
    'n4_0425':                          N4_0425Net,
    # n3_0425 — binaural-only FOA prediction (see docs/experiment_plan_predict_FOA.md).
    # Built here for kwarg propagation (input_nc, K, ngf), but routed via
    # is_n3_0425_model() to its own train step (not foa_0415).
    'n3_0425':                          N3_0425Net,
    'n9_0425':                          N9_0425Net,
    'n9_0426':                          N9_0426Net,
}

# ViT-backbone oracle: same train/val path as the UNet foa_oracle family
# (input assembly via cat(audio, energy_map)) but built through the
# _PVITFOA_CLASSES branch in build_model (ViT-specific kwargs).
_PVITFOA_ORACLE_CLASSES = {
    'pretrained_vit_foa_v6_oracle_nc3': PretrainedViTFOAV6OracleNC3,
}

_FOA_VARIANT_CLASSES = {
    'foa_crossattn': FOACrossAttnGenerator,
    'foa_featbank': FOAFeatBankGenerator,
    'foa_msattn': FOAMultiScaleAttnGenerator,
    'foa_channelattn': FOAChannelAttnGenerator,
    'foa_v2': FOAv2Generator,
    'foa_v2_js': FOAv2JSGenerator,
    'foa_v2_js_rgb': FOAv2JSGenerator,
}


_FOA_0415_CLASSES = {
    'foa_0415_v1': FOA0415V1Generator,
    'foa_0415_v2': FOA0415V2Generator,
    'foa_0415_v3': FOA0415V3Generator,
    'foa_0415_v4': FOA0415V4Generator,
    'foa_0415_v5': FOA0415V5Generator,
    # N3 Group C variants (same train path: pred_depth + pred_sh)
    'n3_film':            N3FiLMGenerator,
    'n3_multiscale_sh':   N3MultiScaleSHGenerator,
    'n3_energy_attn':     N3EnergyAttnGenerator,
    'n3_temporal_window': N3TemporalWindowGenerator,
    # N3 0419 hybrids (report_d: exp212-214; same train path but models also
    # emit pred_energy — picked up by _train_step_foa_0415 if
    # cfg.model.lambda_energy > 0)
    'n3_film_energy_attn': N3FiLMEnergyAttnGenerator,
    'n3_mssh_energy_attn': N3MSSHEnergyAttnGenerator,
}

_FOA_ORACLE_CLASSES = {
    'foa_oracle': FOAOracleGenerator,
    # Energy-map-only (exp247-252): single-channel energy map as input
    'n3_emap_unet_repeat': EmapUNetGenerator,
    'n3_emap_unet_conv':   EmapUNetGenerator,
    'n3_emap_unet_edge':   EmapUNetGenerator,
    'n3_emap_vit_repeat':  EmapViTGenerator,
    'n3_emap_vit_conv':    EmapViTGenerator,
    'n3_emap_vit_edge':    EmapViTGenerator,
}

_EMAP_TEMPORAL_CLASSES = {
    # Temporal energy maps (exp253-256): 3ch from N2 dataset
    'n3_emap_unet_temporal':    EmapUNetTemporalGenerator,
    'n3_emap_vit_temporal':     EmapViTTemporalGenerator,
    'n3_emap_unet_temporal_ov': EmapUNetTemporalGenerator,
    'n3_emap_vit_temporal_ov':  EmapViTTemporalGenerator,
}

_N2_CLASSES = {
    # E1: temporal energy maps as extra input channels
    'n2_temap_input':     N2TemapInputGenerator,
    # E2: FOA spec concat (reuse v1 with input_nc=6)
    'n2_6ch_input':       FOA0415V1Generator,
    # E3: temporal RMS FiLM
    'n2_temporal_rms_film': N2TemporalRMSFiLMGenerator,
    # E4: temporal energy spatial attention
    'n2_temporal_energy':  N2TemporalEnergyGenerator,
    # E5: FOA STFT FiLM
    'n2_foa_stft_film':    N2FOASTFTFiLMGenerator,
    # E6: dual encoder
    'n2_dual_enc':         N2DualEncGenerator,
    # E7: supervision-only (reuse v1 with sh_dim=12)
    'n2_temporal_rms':    FOA0415V1Generator,
    # E8: temporal bin cross-attention
    'n2_tbin_crossattn':   N2TBinCrossAttnGenerator,
    # N1 (2026-04-20): ViT x temporal hybrids. Same dataset/train/test
    # routing as N2 (7-tuple batch, _train_step_n2 path, n2-branch in
    # val/test). Model-wise they are ViT classes; data-wise they are N2.
    'pvit_n1_temap_input':    PVitN1TemapInput,
    'pvit_n1_temap_rms_film': PVitN1TemapRMSFiLM,
    'pvit_n1_temap_eattn':    PVitN1TemapEAttn,
    'pvit_n1_temap_mssh':     PVitN1TemapMSSH,
}

# Subset of _N2_CLASSES that are ViT-backed and need the ViT build path
# (pretrained=True, freeze_encoder=..., no UNet num_downs/ngf).
_N1_PVIT_NAMES = {
    'pvit_n1_temap_input',
    'pvit_n1_temap_rms_film',
    'pvit_n1_temap_eattn',
    'pvit_n1_temap_mssh',
}


def is_foa_0415_model(cfg):
    """Whether cfg selects a Variant-1-style (aux-SH-head only) model.

    Covers the UNet-based foa_0415_v{1..5} family and the ViT-based
    pretrained_vit_foa v1/v3/v4/v5 variants. All share the same train/test
    routing path: forward returns ``{pred_depth, pred_sh}`` and the training
    loss is ``L_depth + lambda_sh * L1(pred_sh[:, :4], gt_foa)``.
    """
    name = getattr(cfg.model, 'name', '')
    return name in _FOA_0415_CLASSES or name in _PVITFOA_AUX_SH_NAMES


def is_emap_temporal_model(cfg):
    """Whether cfg selects an energy-map temporal model (3ch temporal energies)."""
    return getattr(cfg.model, 'name', '') in _EMAP_TEMPORAL_CLASSES


def is_foa_oracle_model(cfg):
    """Whether cfg selects a Group-D oracle model (GT energy/FOA at input).

    Covers UNet oracle (_FOA_ORACLE_CLASSES) and ViT oracle
    (_PVITFOA_ORACLE_CLASSES). Both route through _train_step_oracle; the
    build path differs (UNet vs ViT kwargs) and is handled in build_model.
    """
    name = getattr(cfg.model, 'name', '')
    return name in _FOA_ORACLE_CLASSES or name in _PVITFOA_ORACLE_CLASSES


def is_n2_model(cfg):
    """Whether cfg selects an N2 temporal-FOA model."""
    return getattr(cfg.model, 'name', '') in _N2_CLASSES


def is_n3_0425_model(cfg):
    """n3_0425 — binaural-only FOA prediction (no depth target).

    Routes via _train_step_n3_0425 in train.py: forward returns
    ``{pred_rep: (B, K, 4)}`` and the training loss is L1 on coefficients
    plus an optional cosine direction term (eigen variants only).
    """
    return getattr(cfg.model, 'name', '') == 'n3_0425'


def is_foa_model(cfg):
    return getattr(cfg.model, 'name', 'unet_baseline') == 'audio_depth_foa'


def is_foa_variant_model(cfg):
    """Whether cfg specifies an FOA variant (full FOA/hist alignment loss path).

    Includes the legacy UNet-based variants (crossattn, featbank, msattn,
    channelattn, foa_v2, foa_v2_js) and pretrained_vit_foa_v2 (the ViT
    variant with energy-map histogram alignment).
    """
    name = getattr(cfg.model, 'name', '')
    foa_variants = ('foa_crossattn', 'foa_featbank', 'foa_msattn',
                    'foa_channelattn', 'foa_v2', 'foa_v2_js')
    return name in foa_variants or name in _PVITFOA_HIST_NAMES


def is_foa_v2_js_model(cfg):
    """Whether cfg routes through the energy-map (js) training step.

    Applies to the legacy foa_v2_js and to pretrained_vit_foa_v2.
    """
    name = getattr(cfg.model, 'name', '')
    return name == 'foa_v2_js' or name in _PVITFOA_HIST_NAMES


def is_foa_v2_js_rgb_model(cfg):
    """Whether cfg routes through the RGB teacher-guided training step."""
    return getattr(cfg.model, 'name', '') == 'foa_v2_js_rgb'


def is_echodiffusion_model(cfg):
    return getattr(cfg.model, 'name', '') == 'echodiffusion'


def is_echonet_model(cfg):
    return getattr(cfg.model, 'name', '') == 'echonet'


def is_batvision_model(cfg):
    return getattr(cfg.model, 'name', '') == 'batvision'


def is_vit_model(cfg):
    return getattr(cfg.model, 'name', '') == 'vit_baseline'


def build_oracle_teacher(ckpt_path, cfg, device, input_nc=3):
    """Build a frozen oracle teacher (foa_oracle_nc3) from a checkpoint.

    Used for report_d exp215 (oracle distillation): the student is a
    deployable N3 variant trained on binaural audio alone; the teacher is
    a previously-trained oracle that receives GT energy map as input.
    Teacher output guides the student depth/SH predictions via an L1 KD
    loss (see _train_step_foa_0415 ``teacher`` kwarg).
    """
    import copy
    teacher_cfg = copy.deepcopy(cfg)
    teacher_cfg.model.name = 'foa_oracle'
    teacher_cfg.model.input_nc = input_nc
    num_downs = 7 if teacher_cfg.model.generator == 'unet_128' else 8
    teacher = FOAOracleGenerator(
        teacher_cfg, input_nc=input_nc, output_nc=1, num_downs=num_downs, ngf=64,
        use_dropout=False,
        sh_dim=getattr(teacher_cfg.model, 'sh_dim', FOAOracleGenerator.DEFAULT_SH_DIM),
        head_hidden=getattr(teacher_cfg.model, 'head_hidden',
                            FOAOracleGenerator.DEFAULT_HEAD_HIDDEN),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['state_dict']
    sd_clean = {(k[len('module.'):] if k.startswith('module.') else k): v
                for k, v in sd.items()}
    teacher.load_state_dict(sd_clean, strict=False)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def build_model(cfg, gpu_ids):
    """Build model based on config."""
    model_name = getattr(cfg.model, 'name', 'unet_baseline')

    if model_name in _N2_CLASSES:
        cls = _N2_CLASSES[model_name]
        if model_name in _N1_PVIT_NAMES:
            # ViT-backed N2 variant: pretrained + freeze_encoder kwargs,
            # no UNet num_downs/ngf. input_nc defaults per model (5 for
            # concat variants, 2 for rms_film).
            default_nc = 2 if model_name == 'pvit_n1_temap_rms_film' else 5
            kwargs = dict(
                input_nc=int(getattr(cfg.model, 'input_nc', default_nc)),
                pretrained=getattr(cfg.model, 'pretrained', True),
                freeze_encoder=getattr(cfg.model, 'freeze_encoder', False),
                sh_dim=getattr(cfg.model, 'sh_dim', cls.DEFAULT_SH_DIM),
                head_hidden=getattr(cfg.model, 'head_hidden',
                                    cls.DEFAULT_HEAD_HIDDEN),
            )
            if hasattr(cfg.model, 'n_early'):
                kwargs['n_early'] = int(cfg.model.n_early)
            if hasattr(cfg.model, 'taps'):
                kwargs['taps'] = list(cfg.model.taps)
            net = cls(cfg, **kwargs)
        else:
            input_nc = int(getattr(cfg.model, 'input_nc', 2))
            num_downs = 7 if cfg.model.generator == 'unet_128' else 8
            net = cls(
                cfg, input_nc=input_nc, output_nc=1,
                num_downs=num_downs, ngf=64, use_dropout=False,
                sh_dim=getattr(cfg.model, 'sh_dim', cls.DEFAULT_SH_DIM),
                head_hidden=getattr(cfg.model, 'head_hidden',
                                    cls.DEFAULT_HEAD_HIDDEN),
            )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net

    if model_name in _FOA_0415_CLASSES:
        cls = _FOA_0415_CLASSES[model_name]
        num_downs = 7 if cfg.model.generator == 'unet_128' else 8
        net = cls(
            cfg, input_nc=2, output_nc=1, num_downs=num_downs, ngf=64,
            use_dropout=False,
            sh_dim=getattr(cfg.model, 'sh_dim', cls.DEFAULT_SH_DIM),
            head_hidden=getattr(cfg.model, 'head_hidden', cls.DEFAULT_HEAD_HIDDEN),
        )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net

    if model_name in _EMAP_TEMPORAL_CLASSES:
        cls = _EMAP_TEMPORAL_CLASSES[model_name]
        num_downs = 7 if cfg.model.generator == 'unet_128' else 8
        kwargs = dict(
            cfg=cfg, input_nc=3, output_nc=1, num_downs=num_downs, ngf=64,
            use_dropout=False,
            sh_dim=getattr(cfg.model, 'sh_dim', cls.DEFAULT_SH_DIM),
            head_hidden=getattr(cfg.model, 'head_hidden', cls.DEFAULT_HEAD_HIDDEN),
        )
        if 'vit' in model_name:
            kwargs['pretrained'] = getattr(cfg.model, 'pretrained', True)
            kwargs['freeze_encoder'] = getattr(cfg.model, 'freeze_encoder', False)
        net = cls(**kwargs)
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net

    if model_name in _FOA_ORACLE_CLASSES:
        cls = _FOA_ORACLE_CLASSES[model_name]
        input_nc = int(getattr(cfg.model, 'input_nc', 3))
        num_downs = 7 if cfg.model.generator == 'unet_128' else 8
        net = cls(
            cfg, input_nc=input_nc, output_nc=1, num_downs=num_downs, ngf=64,
            use_dropout=False,
            sh_dim=getattr(cfg.model, 'sh_dim', cls.DEFAULT_SH_DIM),
            head_hidden=getattr(cfg.model, 'head_hidden', cls.DEFAULT_HEAD_HIDDEN),
        )
        if len(gpu_ids) > 0:
            assert torch.cuda.is_available()
            net = net.to(gpu_ids[0])
            net = nn.DataParallel(net, gpu_ids)
        return net

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
        # Hyperparameters passed only if set on cfg.model (variants accept
        # different kwargs; unknown kwargs are swallowed by **_unused).
        for k in ('sh_dim', 'head_hidden', 'n_early', 'taps',
                  'n_slots', 'num_heads',
                  'proj_dim', 'foa_dim', 'sh_order',
                  'scale_shift_hidden', 'scale_shift_layers',
                  # renew_single knobs
                  'sf_mode', 'teacher_ratio', 'fusion_ch', 'attn_heads',
                  'disable_fusion',
                  # n9_0424 knobs
                  'K', 'D', 'res_scale_init', 'gate_learnable',
                  'enable_fusion', 'enable_refinement',
                  # n4_0425 knobs
                  'ngf', 'gate_init', 'gate_mask',
                  # n9_0425 specific (cascade from pre-trained n3_0425)
                  'n3_checkpoint', 'freeze_n3', 'n3_ngf',
                  # n9_0426 specific (pretrained outer hourglass)
                  'backbone', 'freeze_backbone'):
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

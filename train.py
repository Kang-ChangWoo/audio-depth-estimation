#!/usr/bin/env python3
"""Training script for audio-to-depth estimation."""

import argparse
import os
import time

import numpy as np
import torch
# import wandb

from utils.config import load_config
from utils.train_utils import (
    build_model, build_criterion, build_oracle_teacher,
    is_foa_model, is_foa_variant_model,
    is_echodiffusion_model, is_foa_v2_js_model, is_foa_0415_model,
    is_foa_v2_js_rgb_model, is_foa_oracle_model, is_n2_model, is_emap_temporal_model,
    is_n3_0425_model,
    compute_gt_depth_sh, compute_gt_energy_sh, set_sh_branch_frozen,
)
import torch.nn.functional as F
from utils.visualization import save_batch_visualization
from utils.metrics import compute_errors
from data.dataloader import make_dataloader


# ── helpers ──────────────────────────────────────────────────

def _train_step_baseline(model, batch, criterion, optimizer, cfg, device):
    audio, gtdepth = batch[0], batch[1]
    audio, gtdepth = audio.to(device), gtdepth.to(device)
    optimizer.zero_grad()
    pred = model(audio)
    loss = criterion(pred, gtdepth)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {'total': loss.item(), 'depth': loss.item()}


def _train_step_echodiffusion(model, batch, criterion, optimizer, cfg, device):
    audio, gtdepth, waveform = batch
    audio, gtdepth, waveform = audio.to(device), gtdepth.to(device), waveform.to(device)
    optimizer.zero_grad()
    pred = model(audio, waveform)
    loss = criterion(pred, gtdepth)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {'total': loss.item(), 'depth': loss.item()}


def _sh_l1_loss(pred_sh, gt_sh):
    """L1 on the overlapping SH dims. GT FOA is 4-dim; pred may be wider."""
    n = min(pred_sh.shape[-1], gt_sh.shape[-1])
    return F.l1_loss(pred_sh[..., :n], gt_sh[..., :n])


def _train_step_foa_0415(model, batch, criterion, optimizer, cfg, device,
                         teacher=None):
    """Variant 1 (Auxiliary SH Head Only) training step.

    Loss:  L = depth_weight * L_depth + lambda_sh * L_sh
                + (optional) lambda_energy * L1(pred_energy, gt_energy_map)
                + (optional) lambda_energy_map * energy_map_loss(rep, basis)
                + (optional) lambda_kd * [L1(depth_s, depth_t) + L1(sh_s, sh_t)]

    Energy-supervision branch (report_d exp210/211/212/213): active when
    ``cfg.model.lambda_energy > 0`` AND the model output carries a
    ``pred_energy`` key. The batch's 4th element (GT energy map) is the target.

    rep-prediction branch (n9_0424, 2026-04-24): when the model output
    carries ``rep_pred`` AND the batch is the 5-tuple produced by a
    ``use_distance_bins=True`` dataset (5th element = rep_gt ∈ ℝ^{B×8×4}),
    ``lambda_sh`` weights a per-bin weighted SmoothL1 on rep_pred INSTEAD
    of the legacy pred_sh L1. Additionally, ``lambda_energy_map > 0``
    turns on the projection-consistency loss.

    Oracle-distillation branch (report_d exp215): active when ``teacher`` is
    provided. The teacher (foa_oracle_nc3) is fed ``cat(audio, gt_energy)`` and
    is frozen. Student is pulled toward teacher's pred_depth / pred_sh.
    """
    # Supports the legacy 4-tuple, the n9 5-tuple (rep_gt), and the
    # echodiffusion_ambi+CIDE 6-tuple (rep_gt + waveform).
    waveform = None
    if len(batch) == 6:
        audio, gtdepth, gt_foa, energy_map, rep_gt, waveform = batch
        rep_gt = rep_gt.to(device)
        waveform = waveform.to(device)
    elif len(batch) == 5:
        audio, gtdepth, gt_foa, energy_map, rep_gt = batch
        rep_gt = rep_gt.to(device)
    else:
        audio, gtdepth, gt_foa, energy_map = batch
        rep_gt = None
    audio = audio.to(device)
    gtdepth = gtdepth.to(device)
    gt_foa = gt_foa.to(device)
    energy_map_dev = energy_map.to(device)

    optimizer.zero_grad()
    # n4_0425 consumes rep_gt + energy_map as oracle forward kwargs; models
    # with **_unused (n9_0424, vit_foa_*) silently ignore the extras.
    # echodiffusion_ambi (use_cide=True) consumes audio_wave for its CIDE
    # branch; other models ignore via **_unused.
    if rep_gt is not None:
        out = model(audio, rep_gt=rep_gt, energy_map=energy_map_dev,
                    audio_wave=waveform)
    else:
        out = model(audio)
    pred_depth = out["pred_depth"]

    depth_loss = criterion(pred_depth, gtdepth)
    lambda_sh = float(getattr(cfg.model, 'lambda_sh', 0.1))
    depth_weight = float(getattr(cfg.model, 'depth_weight', 1.0))
    loss = depth_weight * depth_loss

    # --- Auxiliary FOA loss: n9_0424 rep-prediction path OR legacy pred_sh.
    if 'rep_pred' in out:
        # n9_0424 requires rep_gt. Never fall back to pred_sh here —
        # pred_sh = rep_pred[:,0,:] is the first-bin dominant FOA (near
        # echoes only), while gt_foa is the whole-IR channel RMS. L1-ing
        # them silently would be incorrect training. Raise loudly so the
        # config error is caught before wasting a 40-epoch run.
        if rep_gt is None:
            raise ValueError(
                "Model emits 'rep_pred' but rep_gt is missing from the "
                "batch. Set cfg.dataset.use_distance_bins=True in the "
                "config so the dataset returns the 5-tuple "
                "(audio, depth, foa_target, energy_map, rep_gt).")
        from models.n9_0424.losses import weighted_rep_loss
        sh_loss = weighted_rep_loss(out['rep_pred'], rep_gt)
    else:
        sh_loss = _sh_l1_loss(out['pred_sh'], gt_foa)
    loss = loss + lambda_sh * sh_loss

    # --- Optional projection-consistency loss on rep_pred (n9_0424 Stage D).
    lambda_energy_map = float(getattr(cfg.model, 'lambda_energy_map', 0.0))
    energy_map_loss_val = 0.0
    if (lambda_energy_map > 0
            and 'rep_pred' in out and rep_gt is not None):
        from models.n9_0424.losses import energy_map_loss as _n9_emap_loss
        base_model = model.module if hasattr(model, 'module') else model
        basis = base_model.fusion.basis         # (4, grid_h, grid_w)
        emap_loss = _n9_emap_loss(out['rep_pred'], rep_gt, basis)
        loss = loss + lambda_energy_map * emap_loss
        energy_map_loss_val = emap_loss.item()

    # --- Optional sparsity loss on the bin gate (n4_0425) ---
    # Encourages the learnable bin gate g ∈ R^K to be sparse.
    # See docs/experiment_plan_bin_selection.md §3.
    lambda_sparsity = float(getattr(cfg.model, 'lambda_sparsity', 0.0))
    sparsity_loss_val = 0.0
    if lambda_sparsity > 0 and 'gate' in out:
        sparsity_loss = out['gate'].mean()
        loss = loss + lambda_sparsity * sparsity_loss
        sparsity_loss_val = sparsity_loss.item()

    # --- Optional energy supervision (per-pixel L1) ---
    lambda_energy = float(getattr(cfg.model, 'lambda_energy', 0.0))
    energy_loss_val = 0.0
    em_aligned = None          # cached for the KL branch below
    if lambda_energy > 0 and isinstance(out, dict) and 'pred_energy' in out:
        pred_energy = out['pred_energy']
        em = energy_map_dev
        if pred_energy.shape[-2:] != em.shape[-2:]:
            em = F.interpolate(em, size=pred_energy.shape[-2:],
                               mode='bilinear', align_corners=False)
        em_aligned = em
        energy_loss = F.l1_loss(pred_energy, em)
        loss = loss + lambda_energy * energy_loss
        energy_loss_val = energy_loss.item()

    # --- Optional histogram-KL on the energy distribution ---
    # NOT peak-normalized — uses SUM-normalize so each flattened map is a
    # proper probability distribution (sum=1). This matters when frames
    # have different absolute magnitudes: peak-norm would clamp every
    # frame to max=1 and bias the loss toward non-peak regions, whereas
    # sum-norm preserves per-pixel relative weight regardless of
    # absolute amplitude. Resulting KL compares SHAPE only.
    #     KL(gt || pred) = Σ p_i · (log p_i - log q_i)
    lambda_kl_energy = float(getattr(cfg.model, 'lambda_kl_energy', 0.0))
    kl_energy_val = 0.0
    if lambda_kl_energy > 0 and isinstance(out, dict) and 'pred_energy' in out:
        pred_energy = out['pred_energy']
        if em_aligned is None:
            em = energy_map_dev
            if pred_energy.shape[-2:] != em.shape[-2:]:
                em = F.interpolate(em, size=pred_energy.shape[-2:],
                                   mode='bilinear', align_corners=False)
            em_aligned = em
        _eps = 1e-8
        p = em_aligned.flatten(1).clamp_min(0)       # (B, HW) raw
        q = pred_energy.flatten(1).clamp_min(0)
        p = p / (p.sum(dim=1, keepdim=True) + _eps)  # sum-normalize
        q = q / (q.sum(dim=1, keepdim=True) + _eps)
        kl_energy = (p * (p.clamp_min(_eps).log()
                          - q.clamp_min(_eps).log())).sum(dim=1).mean()
        loss = loss + lambda_kl_energy * kl_energy
        kl_energy_val = kl_energy.item()

    # --- Optional oracle distillation (exp215) ---
    lambda_kd = float(getattr(cfg.model, 'lambda_kd', 0.0))
    kd_loss_val = 0.0
    if teacher is not None and lambda_kd > 0:
        with torch.no_grad():
            t_input = torch.cat([audio, energy_map_dev], dim=1)
            t_out = teacher(t_input)
        kd_depth = F.l1_loss(pred_depth, t_out['pred_depth'])
        kd_sh = F.l1_loss(pred_sh[:, :4], t_out['pred_sh'][:, :4])
        kd_loss = kd_depth + kd_sh
        loss = loss + lambda_kd * kd_loss
        kd_loss_val = kd_loss.item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {'total': loss.item(), 'depth': depth_loss.item(),
            'foa': sh_loss.item(), 'energy': energy_loss_val,
            'kl_energy': kl_energy_val, 'kd': kd_loss_val,
            'energy_map': energy_map_loss_val,
            'sparsity': sparsity_loss_val}


def _train_step_n2(model, batch, criterion, optimizer, cfg, device):
    """N2 training step — temporal FOA decomposition experiments.

    Batch is a 7-tuple from dataset_n2:
        (audio, gtdepth, foa_target, energy_map, foa_spec, temporal_rms, temporal_energies)
    Each model variant selects the inputs it needs.
    """
    audio, gtdepth, gt_foa, _energy, foa_spec, temporal_rms, temporal_energies = batch
    audio = audio.to(device)
    gtdepth = gtdepth.to(device)
    model_name = cfg.model.name

    optimizer.zero_grad()

    if model_name in ('n2_temap_input',
                      'pvit_n1_temap_input',
                      'pvit_n1_temap_eattn',
                      'pvit_n1_temap_mssh'):
        temporal_energies = temporal_energies.to(device)
        out = model(torch.cat([audio, temporal_energies], dim=1))
        gt_sh = temporal_rms.to(device)
    elif model_name == 'n2_6ch_input':
        foa_spec = foa_spec.to(device)
        out = model(torch.cat([audio, foa_spec], dim=1))
        gt_sh = gt_foa.to(device)
    elif model_name == 'n2_temporal_rms':
        out = model(audio)
        gt_sh = temporal_rms.to(device)
    elif model_name == 'n2_temporal_energy':
        temporal_energies = temporal_energies.to(device)
        out = model(audio)
        gt_sh = gt_foa.to(device)
    elif model_name == 'n2_dual_enc':
        foa_spec = foa_spec.to(device)
        out = model(audio, foa_spec)
        gt_sh = gt_foa.to(device)
    elif model_name == 'n2_foa_stft_film':
        foa_spec = foa_spec.to(device)
        out = model(audio, foa_spec)
        gt_sh = gt_foa.to(device)
    elif model_name in ('n2_temporal_rms_film', 'pvit_n1_temap_rms_film'):
        temporal_rms_d = temporal_rms.to(device)
        out = model(audio, temporal_rms_d)
        gt_sh = temporal_rms_d
    elif model_name == 'n2_tbin_crossattn':
        temporal_energies = temporal_energies.to(device)
        out = model(audio, temporal_energies=temporal_energies)
        gt_sh = gt_foa.to(device)
    elif model_name in ('n3_emap_unet_temporal', 'n3_emap_vit_temporal',
                        'n3_emap_unet_temporal_ov', 'n3_emap_vit_temporal_ov'):
        temporal_energies = temporal_energies.to(device)
        out = model(temporal_energies)
        gt_sh = gt_foa.to(device)
    else:
        out = model(audio)
        gt_sh = gt_foa.to(device)

    pred_depth = out["pred_depth"]
    pred_sh = out["pred_sh"]

    depth_loss = criterion(pred_depth, gtdepth)
    sh_loss = _sh_l1_loss(pred_sh, gt_sh)

    lambda_sh = float(getattr(cfg.model, 'lambda_sh', 0.1))
    depth_weight = float(getattr(cfg.model, 'depth_weight', 1.0))
    loss = depth_weight * depth_loss + lambda_sh * sh_loss

    if (model_name in ('n2_temporal_energy', 'pvit_n1_temap_eattn')
            and 'pred_temporal_energies' in out):
        lambda_energy = float(getattr(cfg.model, 'lambda_energy', 0.1))
        lambda_bins = getattr(cfg.model, 'lambda_bins', None)
        pred_te = out['pred_temporal_energies']
        if temporal_energies.device != pred_te.device:
            temporal_energies = temporal_energies.to(pred_te.device)
        for k in range(pred_te.shape[1]):
            lw = float(lambda_bins[k]) if lambda_bins is not None else lambda_energy
            loss = loss + lw * F.l1_loss(
                pred_te[:, k:k+1],
                F.interpolate(temporal_energies[:, k:k+1],
                              size=pred_te.shape[2:], mode='bilinear',
                              align_corners=False))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {'total': loss.item(), 'depth': depth_loss.item(), 'foa': sh_loss.item()}


def _train_step_oracle(model, batch, criterion, optimizer, cfg, device):
    """Oracle (Group D) training step — GT energy/FOA concatenated at input.

    Assembles the model input based on ``cfg.model.input_nc``:
        input_nc=3  -> cat(binaural, energy_map)   [D1]
        input_nc=1  -> energy_map only              [D5]
        input_nc=6  -> cat(binaural, foa_spec)      [D2, future]
    Otherwise identical to ``_train_step_foa_0415``.
    """
    audio, gtdepth, gt_foa, energy_map = batch
    audio = audio.to(device)
    gtdepth = gtdepth.to(device)
    gt_foa = gt_foa.to(device)
    energy_map = energy_map.to(device)

    input_nc = int(getattr(cfg.model, 'input_nc', 3))
    if input_nc == 3:
        model_input = torch.cat([audio, energy_map], dim=1)
    elif input_nc == 1:
        model_input = energy_map
    else:
        # Fallback: pass whatever channels are available
        model_input = torch.cat([audio, energy_map], dim=1)

    optimizer.zero_grad()
    out = model(model_input)
    pred_depth = out["pred_depth"]
    pred_sh = out["pred_sh"]

    depth_loss = criterion(pred_depth, gtdepth)
    sh_loss = _sh_l1_loss(pred_sh, gt_foa)

    lambda_sh = float(getattr(cfg.model, 'lambda_sh', 0.1))
    depth_weight = float(getattr(cfg.model, 'depth_weight', 1.0))
    loss = depth_weight * depth_loss + lambda_sh * sh_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {'total': loss.item(), 'depth': depth_loss.item(), 'foa': sh_loss.item()}


def _train_step_foa(model, batch, criterion, optimizer, cfg, device,
                    use_hist, foa_frozen):
    audio, gtdepth, gt_foa, _ = batch
    audio, gtdepth, gt_foa = audio.to(device), gtdepth.to(device), gt_foa.to(device)
    optimizer.zero_grad()
    outputs = model(audio, return_hist_maps=use_hist)

    if foa_frozen:
        loss = criterion.depth_criterion(outputs["pred_depth"], gtdepth) * criterion.depth_weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return {'total': loss.item(), 'depth': loss.item() / criterion.depth_weight}

    gt_dsh, gt_dsh_c = (compute_gt_depth_sh(model, gtdepth) if use_hist
                        else (None, None))
    ld = criterion(outputs, gtdepth, gt_foa,
                   gt_depth_sh=gt_dsh, gt_depth_sh_coeffs=gt_dsh_c)
    # KL loss is already included in ld["total"] by AudioDepthFOALoss (weighted by kl_weight).
    # Add FOA-depth gradient consistency loss if present (foa_v2).
    if "foa_depth_consistency" in outputs:
        consistency = outputs["foa_depth_consistency"].mean()  # DataParallel gathers scalars into vector
        foa_consistency_weight = getattr(cfg.model, 'foa_consistency_weight', 0.05)
        ld["total"] = ld["total"] + foa_consistency_weight * consistency
        ld["consistency"] = consistency
    ld["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {k: v.item() for k, v in ld.items()}


def _train_step_js(model, batch, criterion, optimizer, cfg, device,
                   use_hist, foa_frozen):
    """Training step for foa_v2_js: uses the ambisonic energy map directly.

    Unlike _train_step_foa, which projects the depth map into SH space as the
    histogram alignment target, this step uses the actual ambisonic-derived
    directional energy map (4th element of the batch). This provides a more
    direct supervision signal grounded in the recorded sound field rather than
    a depth-derived proxy.
    """
    audio, gtdepth, gt_foa, gt_energy = batch
    audio = audio.to(device)
    gtdepth = gtdepth.to(device)
    gt_foa = gt_foa.to(device)
    gt_energy = gt_energy.to(device)
    optimizer.zero_grad()
    outputs = model(audio, return_hist_maps=use_hist)

    if foa_frozen:
        loss = criterion.depth_criterion(outputs["pred_depth"], gtdepth) * criterion.depth_weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return {'total': loss.item(), 'depth': loss.item() / criterion.depth_weight}

    # Project the ambisonic energy map into SH space and use it as the
    # histogram alignment target (instead of the depth-derived projection).
    gt_esh, gt_esh_c = (compute_gt_energy_sh(model, gt_energy) if use_hist
                        else (None, None))
    ld = criterion(outputs, gtdepth, gt_foa,
                   gt_depth_sh=gt_esh, gt_depth_sh_coeffs=gt_esh_c)
    # FOA-depth gradient consistency (inherited from foa_v2 forward path)
    if "foa_depth_consistency" in outputs:
        consistency = outputs["foa_depth_consistency"].mean()
        foa_consistency_weight = getattr(cfg.model, 'foa_consistency_weight', 0.05)
        ld["total"] = ld["total"] + foa_consistency_weight * consistency
        ld["consistency"] = consistency
    ld["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {k: v.item() for k, v in ld.items()}


def _train_step_js_rgb(model, batch, criterion, optimizer, cfg, device):
    """Training step for foa_v2_js_rgb: RGB teacher-guided with feature alignment.

    Losses: (1) depth (BerHu+SILog), (2) FOA L1 on SH coeffs, (3) L2
    alignment between audio_feat and rgb_feat (detached teacher).
    """
    audio, gtdepth, gt_foa, _energy, rgb = batch
    audio = audio.to(device)
    gtdepth = gtdepth.to(device)
    gt_foa = gt_foa.to(device)
    rgb = rgb.to(device)

    optimizer.zero_grad()
    outputs = model(audio, rgb, mode='train')

    pred_depth = outputs["pred_depth"]
    if pred_depth.shape != gtdepth.shape:
        pred_depth = F.interpolate(pred_depth, size=gtdepth.shape[2:],
                                   mode='nearest')
    depth_loss = criterion(pred_depth, gtdepth)
    foa_loss = F.l1_loss(outputs["pred_foa"], gt_foa)
    align_loss = F.mse_loss(outputs["audio_feat"], outputs["rgb_feat"].detach())

    dw = float(getattr(cfg.model, 'depth_weight', 1.0))
    fw = float(getattr(cfg.model, 'foa_weight', 0.1))
    aw = float(getattr(cfg.model, 'align_weight', 0.1))

    loss = dw * depth_loss + fw * foa_loss + aw * align_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {'total': loss.item(), 'depth': depth_loss.item(),
            'foa': foa_loss.item(), 'align': align_loss.item()}


def _train_step_n3_0425(model, batch, criterion, optimizer, cfg, device):
    """n3_0425 — binaural-only FOA representation prediction (no depth).

    Loss:  L = L1(pred_rep, rep_gt) + lambda_dir * (1 - cos(pred_rep, rep_gt))

    The cosine term is applied per (B, K) over the 4-channel axis. For
    rep_kind='rms' (channel-wise energy, no direction), cosine is skipped.
    Requires the 5-tuple batch produced by use_distance_bins=True.
    """
    if len(batch) != 5:
        raise RuntimeError(
            "n3_0425 requires the 5-tuple batch (audio, depth, foa_target, "
            "energy_map, rep_gt). Set cfg.dataset.use_distance_bins=true.")
    audio = batch[0].to(device)
    rep_gt = batch[4].to(device)              # (B, K, 4)

    optimizer.zero_grad()
    out = model(audio)
    pred = out['pred_rep']                    # (B, K, 4)

    coef_loss = F.l1_loss(pred, rep_gt)

    rep_kind = str(getattr(cfg.dataset, 'rep_kind', 'eigen')).lower()
    lambda_dir = float(getattr(cfg.model, 'lambda_dir', 0.1))
    if rep_kind == 'eigen' and lambda_dir > 0:
        cos = F.cosine_similarity(pred, rep_gt, dim=-1)   # (B, K)
        dir_loss = (1.0 - cos).mean()
    else:
        dir_loss = torch.tensor(0.0, device=device)

    loss = coef_loss + lambda_dir * dir_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {'total': loss.item(), 'coef': coef_loss.item(),
            'dir': dir_loss.item() if isinstance(dir_loss, torch.Tensor) else 0.0}


def _val_metrics(model, val_loader, criterion, cfg, device, foa, echodiff,
                 use_hist, foa_frozen, js=False, foa0415=False, js_rgb=False,
                 foa_oracle=False, n2=False, n3_0425=False):
    model.eval()
    errors, val_losses = [], []
    vis_pred, vis_gt = None, None

    # Heartbeat: prints the current batch index every ``_HEARTBEAT`` steps
    # so a hang leaves a last-known-location in the log. flush=True so the
    # line lands in the file even under block-buffered redirect. Tuned to
    # ~20 lines per val pass (2 951 samples / bs=16 ≈ 185 batches → every
    # 50 steps = 4 prints; bs=1 at test time → 2 951 batches → every 200).
    _N_val = len(val_loader)
    _HEARTBEAT = max(50, _N_val // 20)
    _val_t0 = time.time()
    print(f"  [val] starting ({_N_val} batches)", flush=True)

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi > 0 and (bi % _HEARTBEAT == 0):
                print(f"  [val] {bi}/{_N_val} "
                      f"({time.time() - _val_t0:.0f}s)", flush=True)
            if n3_0425:
                # FOA prediction: no depth metrics. Stuff coef_l1 into both
                # abs_rel and rmse slots so the existing best-model selection
                # (0.7*rmse + 0.3*abs_rel) selects the lowest-coef-L1 model.
                # cosine similarity goes into the delta1 slot.
                audio_v = batch[0].to(device)
                rep_gt_v = batch[4].to(device)
                out = model(audio_v)
                pred_v = out['pred_rep']
                coef_l1 = F.l1_loss(pred_v, rep_gt_v).item()
                cos_v = F.cosine_similarity(
                    pred_v, rep_gt_v, dim=-1).mean().item()
                val_losses.append(coef_l1)
                # errors row: [abs_rel, rmse, d1, d2, d3, log10, mae]
                errors.append(
                    [coef_l1, coef_l1, cos_v, 0.0, 0.0, 0.0, 0.0])
                continue
            if n2:
                audio_v = batch[0].to(device)
                gtdepth = batch[1].to(device)
                foa_spec_v = batch[4].to(device) if len(batch) > 4 else None
                trms_v = batch[5].to(device) if len(batch) > 5 else None
                tenergy_v = batch[6].to(device) if len(batch) > 6 else None
                mn = cfg.model.name
                if mn == 'n2_6ch_input':
                    out = model(torch.cat([audio_v, foa_spec_v], dim=1))
                elif mn in ('n2_dual_enc', 'n2_foa_stft_film'):
                    out = model(audio_v, foa_spec_v)
                elif mn in ('n2_temporal_rms_film', 'pvit_n1_temap_rms_film'):
                    out = model(audio_v, trms_v)
                elif mn in ('n2_temap_input',
                            'pvit_n1_temap_input',
                            'pvit_n1_temap_eattn',
                            'pvit_n1_temap_mssh'):
                    out = model(torch.cat([audio_v, tenergy_v], dim=1))
                elif mn == 'n2_tbin_crossattn':
                    out = model(audio_v, temporal_energies=tenergy_v)
                elif mn in ('n3_emap_unet_temporal', 'n3_emap_vit_temporal',
                            'n3_emap_unet_temporal_ov', 'n3_emap_vit_temporal_ov'):
                    out = model(tenergy_v)
                else:
                    out = model(audio_v)
                depth_pred = out["pred_depth"]
                lv = criterion(depth_pred, gtdepth)
            elif js_rgb:
                audio, gtdepth, gt_foa_v, _, _rgb = batch
                audio, gtdepth = audio.to(device), gtdepth.to(device)
                gt_foa_v = gt_foa_v.to(device)
                out = model(audio, mode='val')
                depth_pred = out["pred_depth"]
                if depth_pred.shape != gtdepth.shape:
                    depth_pred = F.interpolate(depth_pred, size=gtdepth.shape[2:],
                                               mode='nearest')
                dw = float(getattr(cfg.model, 'depth_weight', 1.0))
                fw = float(getattr(cfg.model, 'foa_weight', 0.1))
                lv = dw * criterion(depth_pred, gtdepth) + fw * F.l1_loss(out["pred_foa"], gt_foa_v)
            elif foa_oracle:
                audio, gtdepth, gt_foa_v, energy_map_v = batch
                audio = audio.to(device)
                gtdepth = gtdepth.to(device)
                gt_foa_v = gt_foa_v.to(device)
                energy_map_v = energy_map_v.to(device)
                input_nc = int(getattr(cfg.model, 'input_nc', 3))
                if input_nc == 3:
                    model_input = torch.cat([audio, energy_map_v], dim=1)
                elif input_nc == 1:
                    model_input = energy_map_v
                else:
                    model_input = torch.cat([audio, energy_map_v], dim=1)
                out = model(model_input)
                depth_pred = out["pred_depth"]
                depth_loss_v = criterion(depth_pred, gtdepth)
                sh_loss_v = _sh_l1_loss(out["pred_sh"], gt_foa_v)
                lambda_sh_v = float(getattr(cfg.model, 'lambda_sh', 0.1))
                depth_w_v = float(getattr(cfg.model, 'depth_weight', 1.0))
                lv = depth_w_v * depth_loss_v + lambda_sh_v * sh_loss_v
            elif foa0415:
                # 4-tuple or 5-tuple (n9_0424/n4_0425 add rep_gt as batch[4]).
                audio, gtdepth, gt_foa_v = batch[0], batch[1], batch[2]
                audio, gtdepth = audio.to(device), gtdepth.to(device)
                gt_foa_v = gt_foa_v.to(device)
                # Pass rep_gt and energy_map to forward so n4_0425's oracle
                # paths are fed at val time too. Other models with **_unused
                # ignore the extras.
                rep_gt_v = batch[4].to(device) if len(batch) >= 5 else None
                em_v = batch[3].to(device) if len(batch) >= 4 else None
                wave_v = batch[5].to(device) if len(batch) >= 6 else None
                if rep_gt_v is not None:
                    out = model(audio, rep_gt=rep_gt_v, energy_map=em_v,
                                audio_wave=wave_v)
                else:
                    out = model(audio)
                depth_pred = out["pred_depth"]
                depth_loss_v = criterion(depth_pred, gtdepth)
                if 'rep_pred' in out and rep_gt_v is not None:
                    from models.n9_0424.losses import weighted_rep_loss
                    sh_loss_v = weighted_rep_loss(out['rep_pred'], rep_gt_v)
                else:
                    sh_loss_v = _sh_l1_loss(out["pred_sh"], gt_foa_v)
                lambda_sh_v = float(getattr(cfg.model, 'lambda_sh', 0.1))
                depth_w_v = float(getattr(cfg.model, 'depth_weight', 1.0))
                lv = depth_w_v * depth_loss_v + lambda_sh_v * sh_loss_v
            elif js:
                audio, gtdepth, gt_foa_v, gt_energy_v = batch
                audio = audio.to(device)
                gtdepth = gtdepth.to(device)
                gt_foa_v = gt_foa_v.to(device)
                gt_energy_v = gt_energy_v.to(device)
                out = model(audio, return_hist_maps=use_hist)
                depth_pred = out["pred_depth"]
                if foa_frozen:
                    lv = criterion.depth_criterion(depth_pred, gtdepth) * criterion.depth_weight
                else:
                    gt_esh, gt_esh_c = (compute_gt_energy_sh(model, gt_energy_v) if use_hist
                                        else (None, None))
                    lv = criterion(out, gtdepth, gt_foa_v,
                                   gt_depth_sh=gt_esh, gt_depth_sh_coeffs=gt_esh_c)["total"]
            elif foa:
                audio, gtdepth, gt_foa_v, _ = batch
                audio, gtdepth = audio.to(device), gtdepth.to(device)
                gt_foa_v = gt_foa_v.to(device)
                out = model(audio, return_hist_maps=use_hist)
                depth_pred = out["pred_depth"]
                if foa_frozen:
                    lv = criterion.depth_criterion(depth_pred, gtdepth) * criterion.depth_weight
                else:
                    gt_dsh, gt_dsh_c = (compute_gt_depth_sh(model, gtdepth) if use_hist
                                        else (None, None))
                    lv = criterion(out, gtdepth, gt_foa_v,
                                   gt_depth_sh=gt_dsh, gt_depth_sh_coeffs=gt_dsh_c)["total"]
            elif echodiff:
                audio, gtdepth, waveform = batch
                audio, gtdepth = audio.to(device), gtdepth.to(device)
                waveform = waveform.to(device)
                depth_pred = model(audio, waveform)
                lv = criterion(depth_pred, gtdepth)
            else:
                audio, gtdepth = batch[0], batch[1]
                audio, gtdepth = audio.to(device), gtdepth.to(device)
                depth_pred = model(audio)
                lv = criterion(depth_pred, gtdepth)

            val_losses.append(lv.item())

            if bi == 0:
                s = cfg.dataset.max_depth if cfg.dataset.depth_norm else 1.0
                vis_pred = depth_pred * s
                vis_gt = gtdepth * s

            for idx in range(depth_pred.shape[0]):
                gt_map = gtdepth[idx, 0].cpu().numpy()
                pred_map = depth_pred[idx, 0].cpu().numpy()
                if cfg.dataset.depth_norm:
                    gt_map *= cfg.dataset.max_depth
                    pred_map *= cfg.dataset.max_depth
                pred_map = np.clip(pred_map, 1e-3, cfg.dataset.max_depth)
                gt_map = np.maximum(gt_map, 0.0)
                errors.append(compute_errors(gt_map, pred_map))

    me = np.array(errors).mean(0)
    print(f"  [val] done ({_N_val} batches, {time.time() - _val_t0:.0f}s)",
          flush=True)
    return {
        'val_loss': np.mean(val_losses),
        'abs_rel': me[0], 'rmse': me[1],
        'delta1': me[2], 'delta2': me[3], 'delta3': me[4],
        'log10': me[5], 'mae': me[6],
    }, vis_pred, vis_gt


# ── main ─────────────────────────────────────────────────────

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    gpu_ids = list(range(min(n_GPU, 4))) if n_GPU > 0 else []
    emap_temporal = is_emap_temporal_model(cfg)
    n2 = is_n2_model(cfg) or emap_temporal
    n3_0425 = is_n3_0425_model(cfg)
    foa_oracle = is_foa_oracle_model(cfg)
    foa0415 = is_foa_0415_model(cfg)
    js_rgb = is_foa_v2_js_rgb_model(cfg)
    foa = (is_foa_model(cfg) or is_foa_variant_model(cfg)) and not foa0415 and not js_rgb
    echodiff = is_echodiffusion_model(cfg)
    js = is_foa_v2_js_model(cfg) and not js_rgb

    train_set, train_loader = make_dataloader(cfg, 'train', batch_size=cfg.mode.batch_size)
    val_set, val_loader = make_dataloader(cfg, 'val', batch_size=cfg.mode.batch_size)
    print(f'Train: {len(train_set)}, Val: {len(val_set)}')

    model = build_model(cfg, gpu_ids)
    criterion = build_criterion(cfg, device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {cfg.model.name} ({total_params:.1f}M params)')

    lr = cfg.mode.learning_rate
    opt_name = cfg.mode.optimizer
    optimizer = (torch.optim.AdamW if opt_name == 'AdamW' else
                 torch.optim.Adam if opt_name == 'Adam' else
                 torch.optim.SGD)(model.parameters(), lr=lr)

    # --- Optional warmup + cosine LR schedule (Tier 2a). ---
    # Per-step LambdaLR, active only when cfg.mode.lr_schedule == 'cosine'
    # or cfg.mode.lr_warmup_epochs > 0. Safe no-op otherwise — existing
    # experiments see constant LR exactly as before.
    _use_lr_sched = (getattr(cfg.mode, 'lr_schedule', None) == 'cosine'
                     or int(getattr(cfg.mode, 'lr_warmup_epochs', 0)) > 0)
    if _use_lr_sched:
        import math as _math
        _warmup_ep  = int(getattr(cfg.mode, 'lr_warmup_epochs', 1))
        _steps_per_epoch = max(1, len(train_loader))
        _warmup_steps = _warmup_ep * _steps_per_epoch
        _total_steps  = cfg.mode.epochs * _steps_per_epoch
        def _lr_lambda(step):
            if step < _warmup_steps:
                return step / max(1, _warmup_steps)
            t = (step - _warmup_steps) / max(1, _total_steps - _warmup_steps)
            return 0.5 * (1.0 + _math.cos(_math.pi * t))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        print(f'  [LR] cosine schedule enabled  '
              f'warmup={_warmup_ep}ep ({_warmup_steps} steps), '
              f'total={_total_steps} steps')
    else:
        scheduler = None

    project_dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = (f"{cfg.model.generator}_{cfg.dataset.name}_BS{cfg.mode.batch_size}_"
                f"Lr{lr}_{opt_name}_{cfg.mode.experiment_name}")
    ckpt_dir = os.path.join(project_dir, 'checkpoints', exp_name)
    results_dir = os.path.join(project_dir, 'results', exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # W&B
    wb_cfg = {
        'experiment_name': exp_name, 'model': cfg.model.name,
        'dataset': cfg.dataset.name, 'optimizer': opt_name,
        'lr': lr, 'batch_size': cfg.mode.batch_size,
        'epochs': cfg.mode.epochs, 'params_M': total_params,
    }
    if foa:
        wb_cfg.update({k: getattr(cfg.model, k, None)
                       for k in ('depth_weight', 'foa_weight', 'hist_weight',
                                 'sh_order', 'proj_dim', 'foa_freeze_epochs')})
    # wandb.init(project='neurips_audio_depth', name=exp_name,
            #    config=wb_cfg, tags=[cfg.model.name, cfg.dataset.name])

    # Resume
    start_epoch = 1
    if cfg.mode.checkpoints is not None:
        ckpt = torch.load(os.path.join(ckpt_dir, f'checkpoint_{cfg.mode.checkpoints}.pth'),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}')

    foa_freeze = getattr(cfg.model, 'foa_freeze_epochs', 0) if foa else 0
    use_hist_align = foa and getattr(cfg.model, 'hist_weight', 0) > 0
    best_rmse, best_abs_rel = float('inf'), float('inf')
    best_score = float('inf')  # weighted: 0.7*rmse + 0.3*abs_rel

    # Build optional oracle-distillation teacher for foa_0415 family (exp215).
    teacher = None
    teacher_ckpt = getattr(cfg.model, 'teacher_ckpt', None)
    if foa0415 and teacher_ckpt:
        teacher = build_oracle_teacher(teacher_ckpt, cfg, device)
        lambda_kd = float(getattr(cfg.model, 'lambda_kd', 0.0))
        print(f'  [Distill] Teacher loaded from {teacher_ckpt}  (lambda_kd={lambda_kd})')

    # --- Tier 2b: renew-family ViT freeze curriculum ----------------
    # Freeze both Spec and Sound-field ViT encoders for the first
    # `cfg.model.renew_freeze_epochs` epochs, then unfreeze. No-op for
    # every other experiment (guarded by model.name).
    _renew_freeze_ep = int(getattr(cfg.model, 'renew_freeze_epochs', 0))
    _is_renew = (getattr(cfg.model, 'name', '') == 'renew_single'
                 and _renew_freeze_ep > 0)

    def _set_renew_vits_frozen(net, frozen: bool):
        base = net.module if isinstance(net, torch.nn.DataParallel) else net
        for mod in (base.vit_spec, base.vit_sf):
            for p in mod.parameters():
                p.requires_grad_(not frozen)

    for epoch in range(start_epoch, cfg.mode.epochs + 1):
        foa_frozen = foa and foa_freeze > 0 and epoch <= foa_freeze
        if foa and foa_freeze > 0:
            set_sh_branch_frozen(model, foa_frozen)
            if epoch == 1:
                print(f'  [Warmup] SH branch frozen for {foa_freeze} epochs')
            elif epoch == foa_freeze + 1:
                print(f'  [Warmup done] SH branch unfrozen')

        if _is_renew:
            in_freeze = epoch <= _renew_freeze_ep
            if epoch == 1:
                _set_renew_vits_frozen(model, True)
                print(f'  [Renew] both ViT encoders frozen for '
                      f'{_renew_freeze_ep} epochs (heads/neck/decoder trainable)')
            elif epoch == _renew_freeze_ep + 1:
                _set_renew_vits_frozen(model, False)
                print(f'  [Renew] ViT encoders unfrozen at epoch {epoch}')

        use_hist = use_hist_align and not foa_frozen
        t0 = time.time()
        accum = {'total': [], 'depth': [], 'foa': [], 'hist': [], 'kl': [], 'consistency': [], 'align': [], 'energy': [], 'kd': []}

        model.train()
        # Train-loop heartbeat — like the val one, so a silent stall in
        # training (e.g. dataloader pipe stuck, CUDA OOM retry, model
        # forward hang on a specific sample) leaves a last-known step
        # index in the log. ~5 prints/epoch at typical N_train.
        _N_train = len(train_loader)
        _TRAIN_HEARTBEAT = max(200, _N_train // 5)
        _train_t0 = time.time()
        for _bi, batch in enumerate(train_loader):
            if _bi > 0 and (_bi % _TRAIN_HEARTBEAT == 0):
                print(f"  [train] epoch {epoch} step {_bi}/{_N_train} "
                      f"({time.time() - _train_t0:.0f}s)", flush=True)
            if n3_0425:
                s = _train_step_n3_0425(model, batch, criterion, optimizer,
                                        cfg, device)
            elif n2:
                s = _train_step_n2(model, batch, criterion, optimizer,
                                   cfg, device)
            elif foa_oracle:
                s = _train_step_oracle(model, batch, criterion, optimizer,
                                       cfg, device)
            elif js_rgb:
                s = _train_step_js_rgb(model, batch, criterion, optimizer,
                                       cfg, device)
            elif foa0415:
                s = _train_step_foa_0415(model, batch, criterion, optimizer,
                                         cfg, device, teacher=teacher)
            elif js:
                s = _train_step_js(model, batch, criterion, optimizer,
                                   cfg, device, use_hist, foa_frozen)
            elif foa:
                s = _train_step_foa(model, batch, criterion, optimizer,
                                    cfg, device, use_hist, foa_frozen)
            elif echodiff:
                s = _train_step_echodiffusion(model, batch, criterion, optimizer,
                                              cfg, device)
            else:
                s = _train_step_baseline(model, batch, criterion, optimizer,
                                         cfg, device)
            for k, v in s.items():
                if k in accum:
                    accum[k].append(v)
            # Per-step LR schedule (Tier 2a). No-op if scheduler is None.
            if scheduler is not None:
                scheduler.step()

        dt = time.time() - t0
        log = {'epoch': epoch, 'train/loss': np.mean(accum['total'])}
        for k in ('depth', 'foa', 'hist', 'kl', 'consistency', 'align'):
            if accum[k]:
                log[f'train/{k}'] = np.mean(accum[k])

        parts = [f"Epoch [{epoch}/{cfg.mode.epochs}] L:{log['train/loss']:.4f}"]
        if accum['depth']: parts.append(f"D:{np.mean(accum['depth']):.4f}")
        if accum['foa']:   parts.append(f"F:{np.mean(accum['foa']):.4f}")
        if accum['hist']:  parts.append(f"H:{np.mean(accum['hist']):.4f}")
        if accum['kl']:    parts.append(f"KL:{np.mean(accum['kl']):.4f}")
        if accum['consistency']: parts.append(f"CON:{np.mean(accum['consistency']):.4f}")
        if accum['align']: parts.append(f"ALN:{np.mean(accum['align']):.4f}")
        parts.append(f"{dt:.0f}s")
        print(' '.join(parts))

        # Validation
        if cfg.mode.validation and epoch % cfg.mode.validation_iter == 0:
            vm, vis_p, vis_g = _val_metrics(
                model, val_loader, criterion, cfg, device, foa, echodiff,
                use_hist, foa_frozen, js=js, foa0415=foa0415, js_rgb=js_rgb,
                foa_oracle=foa_oracle, n2=n2, n3_0425=n3_0425)
            print(f"  Val L:{vm['val_loss']:.4f} ABS:{vm['abs_rel']:.4f} "
                  f"RMSE:{vm['rmse']:.4f} d1:{vm['delta1']:.4f}")

            for k, v in vm.items():
                log[f'val/{k}'] = v

            if vis_p is not None:
                vis_path = os.path.join(results_dir, f'epoch_{epoch:04d}_val.png')
                save_batch_visualization(vis_p, vis_g, vis_path, epoch,
                                         num_samples=min(4, vis_p.shape[0]))

            score = 0.7 * vm['rmse'] + 0.3 * vm['abs_rel']
            if score < best_score:
                best_score = score
                best_rmse, best_abs_rel = vm['rmse'], vm['abs_rel']
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_rmse': best_rmse, 'best_abs_rel': best_abs_rel,
                            'best_score': best_score},
                           os.path.join(ckpt_dir, 'best_model.pth'))
                print(f"  >> Best (score:{best_score:.4f} RMSE:{best_rmse:.4f} ABS:{best_abs_rel:.4f})")
                log.update({'best/score': best_score, 'best/rmse': best_rmse,
                            'best/abs_rel': best_abs_rel, 'best/epoch': epoch})

        # wandb.log(log)

        if epoch % cfg.mode.saving_checkpoints == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(ckpt_dir, f'checkpoint_{epoch}.pth'))

    print(f'\nDone. Best score:{best_score:.4f} RMSE:{best_rmse:.4f} ABS:{best_abs_rel:.4f}')
    # wandb.finish()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='foa', help='Config name (baseline, foa)')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--optimizer', type=str, default=None, choices=['AdamW', 'Adam', 'SGD'])
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--experiment-name', type=str, default='default')
    p.add_argument('--checkpoints', type=str, default=None)
    p.add_argument('--foa-freeze-epochs', type=int, default=None)
    p.add_argument('--depth-weight', type=float, default=None)
    p.add_argument('--foa-weight', type=float, default=None)
    p.add_argument('--hist-weight', type=float, default=None)
    p.add_argument('--kl-weight', type=float, default=None)
    p.add_argument('--foa-consistency-weight', type=float, default=None)
    p.add_argument('--align-weight', type=float, default=None,
                   help='foa_v2_js_rgb: weight on the L2 feature alignment loss.')
    p.add_argument('--lambda-sh', type=float, default=None,
                   help='foa_0415: weight on the auxiliary SH L1 loss.')
    p.add_argument('--lambda-energy', type=float, default=None,
                   help='foa_0415 (report_d): weight on L1(pred_energy, gt_energy_map). '
                        'Active only when model emits pred_energy (n3_energy_attn + hybrids).')
    p.add_argument('--teacher-ckpt', type=str, default=None,
                   help='Oracle distillation (report_d exp215): path to teacher '
                        'best_model.pth (expects foa_oracle_nc3 checkpoint).')
    p.add_argument('--lambda-kd', type=float, default=None,
                   help='Oracle distillation weight; paired with --teacher-ckpt.')
    p.add_argument('--sh-dim', type=int, default=None,
                   help='foa_0415: override SH head output dimensionality.')
    p.add_argument('--head-hidden', type=int, default=None,
                   help='foa_0415: override SH head hidden width.')
    p.add_argument('--depth-dir', type=str, default=None,
                   help=('Override cfg.dataset.depth_dir: load GT depth '
                         'from root_dir/scene/<depth-dir>/ instead of '
                         "{depth_type}_depth/. E.g. 'erp_depth_radial' "
                         "to train against radial-distance GT."))
    p.add_argument('--dataset-dir', type=str, default=None,
                   help=('Override cfg.dataset.dataset_dir: dataset root '
                         'path, for running on a server where the data '
                         'lives elsewhere than the yaml default.'))
    p.add_argument('--rotate-canonical', action='store_true',
                   help='Rotate FOA into a canonical listener frame (dataset_rotated.py).')
    p.add_argument('--lambda-sparsity', type=float, default=None,
                   help='n4_0425: weight on sigmoid(gate).mean() sparsity loss.')
    p.add_argument('--gate-mask', type=str, default=None,
                   help=('n4_0425: comma-separated K floats fixing the bin gate '
                         '(non-learnable). E.g. "1,1,1,1,1,1,1,0" drops bin 7. '
                         '"0,0,1,0,0,0,0,0" enables only bin 2.'))
    p.add_argument('--rep-kind', type=str, default=None,
                   choices=['eigen', 'rms'],
                   help='n3_0425: FOA target type — "eigen" or "rms".')
    p.add_argument('--rep-K', type=int, default=None,
                   help='n3_0425: number of temporal bins (1, 8, or 100).')
    p.add_argument('--lambda-dir', type=float, default=None,
                   help='n3_0425: weight on the cosine direction loss.')
    p.add_argument('--ngf', type=int, default=None,
                   help='n3_0425/n4_0425: base UNet width (8*ngf bottleneck).')
    p.add_argument('--n3-checkpoint', type=str, default=None,
                   help='n9_0425: path to a pre-trained n3_0425 best_model.pth.')
    p.add_argument('--freeze-n3', dest='freeze_n3',
                   action='store_true', default=None,
                   help='n9_0425: freeze the inner n3 (default per cfg).')
    p.add_argument('--no-freeze-n3', dest='freeze_n3', action='store_false',
                   help='n9_0425: fine-tune the inner n3 along with the UNet.')
    p.add_argument('--backbone', type=str, default=None,
                   choices=['vit', 'resnet'],
                   help='n9_0426: outer hourglass backbone (vit or resnet).')
    p.add_argument('--freeze-backbone', dest='freeze_backbone',
                   action='store_true', default=None,
                   help='n9_0426: freeze the pretrained outer backbone encoder.')
    p.add_argument('--no-freeze-backbone', dest='freeze_backbone',
                   action='store_false',
                   help='n9_0426: fine-tune the pretrained outer backbone encoder.')
    p.add_argument('--foa-mode', type=str, default=None,
                   choices=['input', 'condition'],
                   help='echodiffusion_ambi: how to inject FOA — channel-concat '
                        'input or cross-attention condition.')
    p.add_argument('--side-fusion', dest='side_fusion',
                   action='store_true', default=None,
                   help='echodiff_sh_side(_plus): enable SH side-prior fusion.')
    p.add_argument('--no-side-fusion', dest='side_fusion',
                   action='store_false',
                   help='echodiff_sh_side(_plus): pure binaural baseline (SH path off).')
    p.add_argument('--oracle-mode', dest='oracle_mode',
                   action='store_true', default=None,
                   help='echodiff_sh_side(_plus): use rep_gt instead of rep_pred for the side prior.')
    p.add_argument('--no-oracle-mode', dest='oracle_mode',
                   action='store_false',
                   help='echodiff_sh_side(_plus): use rep_pred (default training mode).')
    p.add_argument('--oracle-gate-mode', type=str, default=None,
                   choices=['ones', 'pred'],
                   help='echodiff_sh_side_plus: oracle gate replacement strategy '
                        '("ones" = all-ones, "pred" = use predicted gate).')
    args = p.parse_args()

    cfg = load_config(config_name=args.config, mode='train',
                      experiment_name=args.experiment_name)

    if args.checkpoints is not None: cfg.mode.checkpoints = args.checkpoints
    if args.batch_size is not None:  cfg.mode.batch_size = args.batch_size
    if args.lr is not None:          cfg.mode.learning_rate = args.lr
    if args.optimizer is not None:   cfg.mode.optimizer = args.optimizer
    if args.epochs is not None:      cfg.mode.epochs = args.epochs
    if args.num_workers is not None: cfg.mode.num_threads = args.num_workers
    if args.foa_freeze_epochs is not None: cfg.model.foa_freeze_epochs = args.foa_freeze_epochs
    if args.depth_weight is not None: cfg.model.depth_weight = args.depth_weight
    if args.foa_weight is not None:   cfg.model.foa_weight = args.foa_weight
    if args.hist_weight is not None:  cfg.model.hist_weight = args.hist_weight
    if args.kl_weight is not None:    cfg.model.kl_weight = args.kl_weight
    if args.foa_consistency_weight is not None: cfg.model.foa_consistency_weight = args.foa_consistency_weight
    if args.align_weight is not None: cfg.model.align_weight = args.align_weight
    if args.lambda_sh is not None:   cfg.model.lambda_sh = args.lambda_sh
    if args.lambda_energy is not None: cfg.model.lambda_energy = args.lambda_energy
    if args.teacher_ckpt is not None: cfg.model.teacher_ckpt = args.teacher_ckpt
    if args.lambda_kd is not None:   cfg.model.lambda_kd = args.lambda_kd
    if args.sh_dim is not None:      cfg.model.sh_dim = args.sh_dim
    if args.head_hidden is not None: cfg.model.head_hidden = args.head_hidden
    if args.rotate_canonical: cfg.dataset.rotate_canonical = True
    if args.depth_dir is not None:
        cfg.dataset.depth_dir = args.depth_dir
    if args.dataset_dir is not None:
        cfg.dataset.dataset_dir = args.dataset_dir
    if args.lambda_sparsity is not None:
        cfg.model.lambda_sparsity = args.lambda_sparsity
    if args.gate_mask is not None:
        cfg.model.gate_mask = [float(v) for v in args.gate_mask.split(',')]
    if args.rep_kind is not None:   cfg.dataset.rep_kind = args.rep_kind
    if args.rep_K is not None:
        cfg.dataset.rep_K = args.rep_K
        # Mirror onto cfg.model.K so build_model picks up the right head size
        # when the model class consumes 'K' from cfg.model.
        cfg.model.K = args.rep_K
    if args.lambda_dir is not None: cfg.model.lambda_dir = args.lambda_dir
    if args.ngf is not None:        cfg.model.ngf = args.ngf
    if args.n3_checkpoint is not None:
        cfg.model.n3_checkpoint = args.n3_checkpoint
    if args.freeze_n3 is not None:
        cfg.model.freeze_n3 = bool(args.freeze_n3)
    if args.backbone is not None:
        cfg.model.backbone = args.backbone
    if args.freeze_backbone is not None:
        cfg.model.freeze_backbone = bool(args.freeze_backbone)
    if args.foa_mode is not None:
        cfg.model.foa_mode = args.foa_mode
    if args.side_fusion is not None:
        cfg.model.side_fusion = bool(args.side_fusion)
    if args.oracle_mode is not None:
        cfg.model.oracle_mode = bool(args.oracle_mode)
    if args.oracle_gate_mode is not None:
        cfg.model.oracle_gate_mode = args.oracle_gate_mode

    print('=' * 60)
    print(f'Model: {cfg.model.name}  Dataset: {cfg.dataset.name}')
    print(f'BS: {cfg.mode.batch_size}  LR: {cfg.mode.learning_rate}  '
          f'Opt: {cfg.mode.optimizer}')
    print('=' * 60)
    train(cfg)

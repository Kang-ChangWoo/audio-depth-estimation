#!/usr/bin/env python3
"""Training script for audio-to-depth estimation."""

import argparse
import math
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
    is_n3_0425_model, is_echorange_model,
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


def _hazard_diagnostics(raw_out, gt_norm, cfg, prefix='[haz]'):
    """One-shot hazard-head diagnostics on a single (val) batch.

    Prints alpha distribution, bg_weight stats, argmax-bin histogram,
    rendered-weight entropy, and depth-bin sliced error so we can see
    *why* a hazard cell is failing without re-running with viz dumps.

    Cheap by construction (one batch per val pass, all torch ops).
    """
    if raw_out is None or 'hazard_alpha' not in raw_out:
        return
    a   = raw_out['hazard_alpha'].detach().float()      # (B, R, h, w)
    bg  = raw_out['hazard_bg_weight'].detach().float()  # (B, 1, h, w)
    w   = raw_out['hazard_weights'].detach().float()    # (B, R, h, w)
    R   = a.shape[1]

    # torch.quantile has a CUDA hard cap (~2^24 elements); a_flat at
    # B=32,R=32,H=W=256 is ~67M and overflows. Subsample for diagnostics.
    def _quantile_safe(t, qs, max_n=4_000_000):
        t = t.reshape(-1)
        if t.numel() > max_n:
            idx = torch.randint(0, t.numel(), (max_n,), device=t.device)
            t = t[idx]
        return torch.quantile(t, qs)

    a_flat = a.reshape(-1)
    qs = torch.tensor([0.5, 0.9, 0.95, 0.99], device=a.device)
    a_q = _quantile_safe(a_flat, qs).cpu().tolist()
    fracs = [(a > t).float().mean().item() for t in (0.5, 0.9, 0.95, 0.99)]

    bg_flat = bg.reshape(-1)
    bg_q = _quantile_safe(
        bg_flat,
        torch.tensor([0.1, 0.5, 0.9], device=bg.device),
    ).cpu().tolist()

    argmax = w.argmax(dim=1).reshape(-1)
    hist = torch.bincount(argmax, minlength=R).cpu().tolist()
    # Compact histogram print: fold to ≤12 buckets so logs stay readable.
    fold = max(1, R // 12)
    folded = [sum(hist[i:i+fold]) for i in range(0, R, fold)]
    total = max(1, sum(folded))
    folded_pct = [f"{100.0*v/total:.1f}" for v in folded]

    ent = raw_out.get('range_entropy')
    ent_str = ""
    if ent is not None:
        e_flat = ent.detach().float().reshape(-1)
        e_q = _quantile_safe(
            e_flat,
            torch.tensor([0.1, 0.5, 0.9], device=e_flat.device),
        ).cpu().tolist()
        ent_str = (f"  ent: p10={e_q[0]:.3f} p50={e_q[1]:.3f} "
                   f"p90={e_q[2]:.3f}")

    print(f"{prefix} alpha: mean={a_flat.mean().item():.4f} "
          f"p50={a_q[0]:.4f} p90={a_q[1]:.4f} p95={a_q[2]:.4f} p99={a_q[3]:.4f}",
          flush=True)
    print(f"{prefix} frac_alpha: >.5={fracs[0]:.3f} >.9={fracs[1]:.3f} "
          f">.95={fracs[2]:.3f} >.99={fracs[3]:.3f}", flush=True)
    print(f"{prefix} bg_weight: p10={bg_q[0]:.3f} p50={bg_q[1]:.3f} "
          f"p90={bg_q[2]:.3f}{ent_str}", flush=True)
    print(f"{prefix} argmax_bin_hist (folded {R}→{len(folded)}, %): "
          f"[{', '.join(folded_pct)}]", flush=True)

    # ── Depth-bin sliced ABS_REL on this batch (raw, no clip) ────────
    # gt_norm is (B,1,H,W) in [0,1]. Lift back to metres for slicing.
    md = float(cfg.dataset.max_depth)
    pred_m = raw_out['pred_depth'].detach().float()    # already metres
    if pred_m.shape[-2:] != gt_norm.shape[-2:]:
        # rare: hazard pred upsampled inside head, but be safe.
        pred_m = F.interpolate(pred_m, size=gt_norm.shape[-2:], mode='nearest')
    gt_m = gt_norm.detach().float() * md
    diff = (pred_m - gt_m).abs()
    bands = [(0.1, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 9.8), (9.8, md + 1e-3)]
    parts = []
    for lo, hi in bands:
        m = (gt_m >= lo) & (gt_m < hi)
        if m.any():
            ar  = (diff[m] / gt_m[m].clamp(min=1e-3)).mean().item()
            rms = (diff[m].pow(2).mean().sqrt()).item()
            parts.append(f"[{lo:g},{hi:g}) AR={ar:.3f} RMSE={rms:.3f}")
        else:
            parts.append(f"[{lo:g},{hi:g}) -")
    print(f"{prefix} sliced: " + "  ".join(parts), flush=True)


def _train_step_echorange(model, batch, criterion, optimizer, cfg, device,
                          epoch: int = 1):
    """EchoRange — scalar baseline OR distribution / hazard head.

    Three head modes share this train step (selected by
    ``cfg.model.depth_head_type``):

      scalar : L = BerHu + SILog (criterion as usual on out['pred_depth']).
      range  : L = λ_NLL · soft_range_nll(logits, gt, bins, σ)
                 + λ_BerHu · BerHu(pred_depth, gt) + λ_SILog · SILog(...)
                 + λ_ent · mean(range_entropy)
      hazard : L = λ_hit_eff · L_hit + λ_free_eff · L_free
                 + λ_BerHu · BerHu(pred_depth, gt) + λ_SILog · SILog(...)
               where (λ_hit_eff, λ_free_eff) follow a 2-stage warmup:
                 epoch ≤ hazard_warmup_epochs : (λ_hit_warm, λ_free_warm)
                 epoch >  hazard_warmup_epochs: (λ_hit,      λ_free)

    Ablation flags (cfg.model.*):
      disable_hit_loss   — force λ_hit = 0   (free-only).
      disable_free_loss  — force λ_free = 0  (hit-only).
      hazard_depth_only  — force λ_hit = λ_free = 0; train the renderer
                           through BerHu/SILog alone.

    Either way, ``out['pred_depth']`` is what downstream metrics use.
    """
    from models.bin_based import (
        soft_range_nll_loss, hazard_supervision_loss,
        rendered_event_nll, survival_loss, soft_hit_bce_loss,
        hazard_free_loss, soft_quantile_depth, spherical_sh_loss,
    )
    from models.losses import BerHuLoss, SILogLoss

    audio, gtdepth, waveform = batch
    audio = audio.to(device)
    gtdepth = gtdepth.to(device)
    waveform = waveform.to(device)

    optimizer.zero_grad()
    out = model(audio, waveform)
    pred_depth = out['pred_depth']

    if 'range_logits' not in out:
        # scalar path — exact echodiffusion baseline.
        loss = criterion(pred_depth, gtdepth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return {'total': loss.item(), 'depth': loss.item()}

    # ── Common preamble for range AND hazard heads ─────────────────────
    head_type = str(getattr(cfg.model, 'depth_head_type', 'range'))

    # NB: dataset.depth_norm=true normalises GT to [0, 1]; both range and
    # hazard heads operate in raw metres on the bin grid. Rescale GT.
    if getattr(cfg.dataset, 'depth_norm', False):
        gtdepth_m = gtdepth * float(cfg.dataset.max_depth)
    else:
        gtdepth_m = gtdepth

    range_bins = (model.module.range_head.range_bins
                  if hasattr(model, 'module')
                  else model.range_head.range_bins)
    logits = out['range_logits']

    # Match GT to logits resolution for per-bin losses (downsampling GT
    # is cheaper than upsampling Br-channel logits to (256, 512)).
    if gtdepth_m.shape[-2:] != logits.shape[-2:]:
        gt_for_nll = F.interpolate(
            gtdepth_m, size=logits.shape[-2:], mode='nearest')
    else:
        gt_for_nll = gtdepth_m

    # ERP-aware extras (shared between range and hazard):
    #   erp_far_mask       : exclude pixels at GT >= range_max_depth (ceiling
    #                        saturation in radial ERP) so they don't pile up
    #                        on the last bin and skew the distribution.
    #   erp_cos_lat_weight : weight per-pixel loss by cos(latitude) so
    #                        polar pixels (oversampled in ERP) don't dominate.
    far_mask_flag = bool(getattr(cfg.model, 'erp_far_mask', False))
    cos_lat_flag  = bool(getattr(cfg.model, 'erp_cos_lat_weight', False))

    far_valid = None
    if far_mask_flag:
        r_max_m = float(getattr(cfg.model, 'range_max_depth', cfg.dataset.max_depth))
        far_valid = (gt_for_nll > 0) & (gt_for_nll < r_max_m)

    pix_w = None
    if cos_lat_flag:
        H = gt_for_nll.shape[-2]
        # latitude per row centre, top-row = +π/2, bottom-row = -π/2.
        lat = (math.pi / 2.0) - math.pi * (torch.arange(H, device=device) + 0.5) / H
        pix_w = torch.cos(lat).clamp(min=1e-3)         # (H,)

    # ── Cylindrical bin-axis support (round 5 patch) ───────────────────
    # The bin grid can represent radial depth (default), horizontal
    # ρ_xy = D·cos(lat), or vertical |z| = D·|sin(lat)|. This is a
    # bin-axis-only swap (analysis doc level (1)): GT is internally
    # projected onto the chosen axis for the per-bin NLL/event loss, and
    # the head's pred_depth is projected back to radial before BerHu /
    # SILog / SH so the eval pipeline (radial RMSE) stays consistent.
    range_bin_axis = str(getattr(cfg.model, 'range_bin_axis', 'radial'))
    cyl_min_factor = float(
        getattr(cfg.model, 'cyl_min_axis_factor', 0.15))
    if range_bin_axis == 'radial':
        axis_factor_orig = None
        axis_factor_nll = None
    else:
        H_orig = gtdepth.shape[-2]
        H_nll = logits.shape[-2]
        # Pixel-centre latitude. Top row = +π/2, bottom row = −π/2.
        lat_orig = (math.pi / 2.0) - math.pi * (
            torch.arange(H_orig, device=device).float() + 0.5) / H_orig
        lat_nll = (math.pi / 2.0) - math.pi * (
            torch.arange(H_nll, device=device).float() + 0.5) / H_nll
        if range_bin_axis == 'horizontal':
            f_orig = torch.cos(lat_orig).clamp(min=cyl_min_factor)
            f_nll = torch.cos(lat_nll).clamp(min=cyl_min_factor)
        elif range_bin_axis == 'z':
            f_orig = torch.sin(lat_orig).abs().clamp(min=cyl_min_factor)
            f_nll = torch.sin(lat_nll).abs().clamp(min=cyl_min_factor)
        else:
            raise ValueError(
                f"range_bin_axis must be one of 'radial', 'horizontal', "
                f"'z'; got {range_bin_axis!r}")
        axis_factor_orig = f_orig.view(1, 1, -1, 1)         # (1,1,H,1)
        axis_factor_nll = f_nll.view(1, 1, -1, 1)
        # Project GT (radial metres) → bin-axis metres for NLL.
        gt_for_nll = gt_for_nll * axis_factor_nll
        # Polar / equatorial pixels with axis_factor at the clamp floor
        # have ambiguous bin-axis meaning. Mask them out from the NLL too.
        polar_mask_nll = (axis_factor_nll > cyl_min_factor + 1e-6)
        if far_valid is None:
            far_valid = polar_mask_nll.expand_as(gt_for_nll)
        else:
            far_valid = far_valid & polar_mask_nll.expand_as(far_valid)

    # ── Per-head loss term ─────────────────────────────────────────────
    lam_b   = float(getattr(cfg.model, 'lambda_berhu', 1.0))
    lam_s   = float(getattr(cfg.model, 'lambda_silog', 1.0))
    lam_ent = float(getattr(cfg.model, 'lambda_entropy_smooth', 0.0))

    if head_type == 'range':
        sigma = float(getattr(cfg.model, 'range_soft_label_sigma', 0.08))
        nll = soft_range_nll_loss(
            logits, gt_for_nll, range_bins,
            valid_mask=far_valid, sigma=sigma, weights=pix_w)
        lam_nll = float(getattr(cfg.model, 'lambda_range_nll', 1.0))
        head_loss = lam_nll * nll
        head_metrics = {'range_nll': float(nll.item())}

        # ── Optional differentiable soft-quantile auxiliary loss (round 5).
        # Hard quantile / median cuts gradient flow from depth loss to
        # logits; soft_quantile_depth keeps it alive. Used to imitate the
        # round-2 exp907 *median* gain in a gradient-friendly way without
        # giving up the expectation-trained anchor.
        lam_sq = float(getattr(cfg.model, 'lambda_soft_quantile', 0.0))
        if lam_sq > 0.0:
            sq_q = float(getattr(cfg.model, 'soft_quantile_q', 0.5))
            sq_tau = float(getattr(cfg.model, 'soft_quantile_tau', 0.05))
            pred_q_axis = soft_quantile_depth(
                logits=logits, range_bins=range_bins,
                q=sq_q, tau=sq_tau)                          # (B,1,h,w) bin-axis
            # Resample to GT resolution and project bin-axis → radial.
            if pred_q_axis.shape[-2:] != gtdepth.shape[-2:]:
                pred_q_axis = F.interpolate(
                    pred_q_axis, size=gtdepth.shape[-2:], mode='nearest')
            if axis_factor_orig is not None:
                pred_q_radial = pred_q_axis / axis_factor_orig
            else:
                pred_q_radial = pred_q_axis
            # Match the BerHu/SILog scale convention.
            if getattr(cfg.dataset, 'depth_norm', False):
                pred_q_norm = (pred_q_radial /
                               float(cfg.dataset.max_depth)).clamp(min=1e-6)
                gt_q_norm = gtdepth
            else:
                pred_q_norm = pred_q_radial
                gt_q_norm = gtdepth
            q_berhu = BerHuLoss().to(device)(pred_q_norm, gt_q_norm)
            q_silog = SILogLoss().to(device)(pred_q_norm, gt_q_norm)
            q_loss = q_berhu + q_silog
            head_loss = head_loss + lam_sq * q_loss
            head_metrics.update({
                'soft_quantile_loss': float(q_loss.item()),
                'soft_quantile_q': sq_q,
                'soft_quantile_tau': sq_tau,
            })
    elif head_type == 'hazard':
        # ── Smooth aux-weight ramp (replaces round-3's discontinuous jump).
        # progress = min(1, epoch / ramp_epochs). At epoch ≤ ramp_epochs the
        # primary aux weight scales from 1/ramp_epochs up to its target;
        # afterwards it stays at the target. The round-3 epoch-3→4 jump
        # (0.3 → 0.5 in λ_hit) was identified as the failure trigger.
        ramp_ep = max(1, int(getattr(cfg.model, 'hazard_warmup_epochs', 3)))
        ramp_progress = min(1.0, float(epoch) / float(ramp_ep))

        lam_aux_target  = float(getattr(cfg.model, 'lambda_hit',  0.10))
        lam_free_target = float(getattr(cfg.model, 'lambda_free', 0.05))
        lam_aux  = ramp_progress * lam_aux_target
        lam_free = ramp_progress * lam_free_target

        # Ablation flags (compose freely with mode below).
        if bool(getattr(cfg.model, 'hazard_depth_only', False)):
            lam_aux = lam_free = 0.0
        if bool(getattr(cfg.model, 'disable_hit_loss', False)):
            lam_aux = 0.0
        if bool(getattr(cfg.model, 'disable_free_loss', False)):
            lam_free = 0.0

        # ── Mode dispatch — round 4 introduces three new aux losses on top
        # of the round-3 raw α-hit BCE. ``--hazard-aux-mode`` selects which.
        #   raw_hit    : original BCE(α, target=1) on hit bins (round 3, kept
        #                for failure-signature reproduction).
        #   event_nll  : NLL of rendered first-hit weight w_j vs soft Gaussian
        #                target q_j over log-bin space (round 4 main).
        #   survival   : BCE on cumulative survival S_j = P(D > r_j).
        #   soft_hit   : BCE(α, target=0.75) on hit bins — saturation guard.
        aux_mode = str(getattr(cfg.model, 'hazard_aux_mode', 'raw_hit'))
        far_thresh = float(getattr(cfg.model, 'hazard_far_thresh', 9.8))
        sigma_bins = float(getattr(cfg.model, 'hazard_event_sigma_bins', 1.0))
        tau_bins   = float(getattr(cfg.model, 'hazard_survival_tau_bins', 1.0))
        soft_hit_t = float(getattr(cfg.model, 'hazard_soft_hit_target', 0.75))
        log_delta_cfg = getattr(cfg.model, 'hazard_log_delta', None)
        log_delta = float(log_delta_cfg) if log_delta_cfg is not None else None

        # The far-censored constraint is enforced *inside* the new losses
        # (event/surv/soft_hit) via far_thresh, so we don't compose it
        # with the existing far_valid (which uses range_max_depth=10.0).
        # For raw_hit we keep the round-3 behaviour (far_valid as-is).
        primary_loss = logits.sum() * 0.0
        primary_name = 'aux'
        if lam_aux > 0.0:
            if aux_mode == 'raw_hit':
                haz = hazard_supervision_loss(
                    logits, gt_for_nll, range_bins,
                    log_delta=log_delta,
                    valid_mask=far_valid, weights=pix_w,
                    use_hit=True, use_free=False,
                )
                primary_loss = haz['hit']
                primary_name = 'hit_raw'
            elif aux_mode == 'event_nll':
                primary_loss = rendered_event_nll(
                    logits, gt_for_nll, range_bins,
                    valid_mask=far_valid, weights=pix_w,
                    sigma_bins=sigma_bins, far_thresh=far_thresh,
                )
                primary_name = 'event_nll'
            elif aux_mode == 'survival':
                primary_loss = survival_loss(
                    logits, gt_for_nll, range_bins,
                    valid_mask=far_valid, weights=pix_w,
                    tau_bins=tau_bins, far_thresh=far_thresh,
                )
                primary_name = 'survival'
            elif aux_mode == 'soft_hit':
                primary_loss = soft_hit_bce_loss(
                    logits, gt_for_nll, range_bins,
                    log_delta=log_delta, soft_target=soft_hit_t,
                    valid_mask=far_valid, weights=pix_w,
                    far_thresh=far_thresh,
                )
                primary_name = 'hit_soft'
            else:
                raise ValueError(
                    f"Unknown hazard_aux_mode: {aux_mode!r}. "
                    f"Expected one of: raw_hit, event_nll, survival, soft_hit.")

        # Free loss is shared across raw_hit / event_nll / soft_hit. Survival
        # already encodes free-space behaviour through its cumulative target,
        # so combining with explicit free is redundant — skip in that mode.
        l_free = logits.sum() * 0.0
        if lam_free > 0.0 and aux_mode != 'survival':
            l_free = hazard_free_loss(
                logits, gt_for_nll, range_bins,
                log_delta=log_delta,
                valid_mask=far_valid, weights=pix_w,
                far_thresh=far_thresh,
            )

        head_loss = lam_aux * primary_loss + lam_free * l_free
        head_metrics = {
            f'hazard_{primary_name}': float(primary_loss.item()),
            'hazard_free':            float(l_free.item()),
            'hazard_lam_aux':         lam_aux,
            'hazard_lam_free':        lam_free,
            'hazard_aux_mode':        aux_mode,
        }
    else:
        raise ValueError(
            f"Unknown depth_head_type for echorange: {head_type!r}. "
            f"Expected 'scalar', 'range', or 'hazard'.")

    # ── BerHu/SILog on rendered depth (shared, both heads) ─────────────
    # Both range and hazard heads emit depth in metres. If the dataset is
    # normalised, scale pred down to [0, 1] so the losses live on the same
    # numerical scale as gt.
    #
    # For cylindrical bin-axis configs, the head outputs depth in the
    # bin-axis (ρ_xy or |z|). Project back to radial for BerHu/SILog/SH so
    # eval (which measures radial RMSE) sees a radial prediction.
    if axis_factor_orig is not None:
        if pred_depth.shape[-2:] == axis_factor_orig.shape[-2:]:
            pred_radial = pred_depth / axis_factor_orig
        else:
            scale = F.interpolate(
                axis_factor_orig.expand(1, 1, -1, pred_depth.shape[-1]),
                size=pred_depth.shape[-2:], mode='nearest')
            pred_radial = pred_depth / scale
    else:
        pred_radial = pred_depth

    if getattr(cfg.dataset, 'depth_norm', False):
        pred_norm = (pred_radial / float(cfg.dataset.max_depth)).clamp(min=1e-6)
        gt_norm = gtdepth
    else:
        pred_norm = pred_radial
        gt_norm = gtdepth

    # Mask polar pixels out of BerHu/SILog under cylindrical mode by
    # zeroing GT there — BerHuLoss/SILogLoss skip pixels where gt ≤ 0.
    if axis_factor_orig is not None:
        polar_keep = (axis_factor_orig > cyl_min_factor + 1e-6)
        if polar_keep.shape[-2:] != gt_norm.shape[-2:]:
            polar_keep = F.interpolate(
                polar_keep.float().expand(1, 1, -1, gt_norm.shape[-1]),
                size=gt_norm.shape[-2:], mode='nearest').bool()
        gt_norm = torch.where(polar_keep, gt_norm, torch.zeros_like(gt_norm))

    berhu = BerHuLoss().to(device)(pred_norm, gt_norm)
    silog = SILogLoss().to(device)(pred_norm, gt_norm)

    loss = head_loss + lam_b * berhu + lam_s * silog
    ent_val = 0.0
    if lam_ent > 0 and 'range_entropy' in out:
        ent = out['range_entropy'].mean()
        loss = loss + lam_ent * ent
        ent_val = float(ent.item())

    # ── Spherical-harmonic auxiliary loss (round 5) ────────────────────
    # Match low-order SH coefficients of pred and gt depth fields on the
    # ERP sphere. Both inputs must be in metres, radial. Caller toggles
    # via cfg.model.lambda_spherical_sh > 0.
    sh_val = 0.0
    lam_sh = float(getattr(cfg.model, 'lambda_spherical_sh', 0.0))
    if lam_sh > 0.0:
        if getattr(cfg.dataset, 'depth_norm', False):
            pred_m_for_sh = pred_radial
            gt_m_for_sh = gtdepth * float(cfg.dataset.max_depth)
        else:
            pred_m_for_sh = pred_radial
            gt_m_for_sh = gtdepth
        sh_l = spherical_sh_loss(
            pred_depth_m=pred_m_for_sh,
            gt_depth_m=gt_m_for_sh,
            L=int(getattr(cfg.model, 'spherical_sh_order', 2)),
            use_log_depth=bool(
                getattr(cfg.model, 'spherical_sh_log_depth', True)),
        )
        loss = loss + lam_sh * sh_l
        sh_val = float(sh_l.item())

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    metrics = {
        'total': float(loss.item()),
        'depth': float(berhu.item() + silog.item()),
        'berhu': float(berhu.item()),
        'silog': float(silog.item()),
        'entropy': ent_val,
        'spherical_sh': sh_val,
    }
    metrics.update(head_metrics)
    return metrics


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
                 foa_oracle=False, n2=False, n3_0425=False, echorange=False):
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
            elif echorange:
                audio, gtdepth, waveform = batch
                audio, gtdepth = audio.to(device), gtdepth.to(device)
                waveform = waveform.to(device)
                out = model(audio, waveform)
                depth_pred = out["pred_depth"]
                # Hazard diagnostics on the first val batch only — cheap
                # one-shot dump of α / bg / argmax / sliced metrics.
                if (bi == 0
                        and getattr(cfg.model, 'depth_head_type', 'scalar')
                            == 'hazard'
                        and 'hazard_alpha' in out):
                    _hazard_diagnostics(out, gtdepth, cfg)
                # Range/hazard heads emit metres; the scalar head, after
                # training against normalised GT, ends up numerically in
                # [0,1]. The downstream metric path (lines below) multiplies
                # pred by max_depth under depth_norm=true, so for range/
                # hazard heads we scale to normalised here so all heads
                # share the same metric path.
                if (getattr(cfg.dataset, 'depth_norm', False)
                        and getattr(cfg.model, 'depth_head_type',
                                    'scalar') in ('range', 'hazard')):
                    depth_pred = depth_pred / float(cfg.dataset.max_depth)
                if getattr(cfg.dataset, 'depth_norm', False):
                    pred_for_crit = depth_pred.clamp(min=1e-6)
                else:
                    pred_for_crit = depth_pred
                lv = criterion(pred_for_crit, gtdepth)
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
    echorange = is_echorange_model(cfg)
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
    best_delta1 = -float('inf')  # delta1 is "higher is better"
    # Per-metric best epochs (round 5: 4-best checkpoints).
    best_epoch = {'score': -1, 'rmse': -1, 'absrel': -1, 'delta1': -1}

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
            elif echorange:
                s = _train_step_echorange(model, batch, criterion, optimizer,
                                          cfg, device, epoch=epoch)
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
                foa_oracle=foa_oracle, n2=n2, n3_0425=n3_0425,
                echorange=echorange)
            print(f"  Val L:{vm['val_loss']:.4f} ABS:{vm['abs_rel']:.4f} "
                  f"RMSE:{vm['rmse']:.4f} d1:{vm['delta1']:.4f}")

            for k, v in vm.items():
                log[f'val/{k}'] = v

            if vis_p is not None:
                vis_path = os.path.join(results_dir, f'epoch_{epoch:04d}_val.png')
                save_batch_visualization(vis_p, vis_g, vis_path, epoch,
                                         num_samples=min(4, vis_p.shape[0]))

            # ── Round 5: per-metric 4-best checkpoint save ─────────────
            # `best_model.pth` (legacy filename) is kept as the same payload
            # that wins the composite score, so existing test scripts that
            # search for `best_model.pth` continue to work without changes.
            score = 0.7 * vm['rmse'] + 0.3 * vm['abs_rel']
            ckpt_payload = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            if score < best_score:
                best_score = score
                best_rmse_at_score = vm['rmse']
                best_absrel_at_score = vm['abs_rel']
                best_epoch['score'] = epoch
                payload = {**ckpt_payload,
                           'best_score': best_score,
                           'best_rmse_at_score': best_rmse_at_score,
                           'best_absrel_at_score': best_absrel_at_score}
                torch.save(payload, os.path.join(ckpt_dir, 'best_score.pth'))
                # Legacy alias — same payload, same path.
                torch.save(payload, os.path.join(ckpt_dir, 'best_model.pth'))
                # Track the running best_rmse / best_abs_rel scalars under
                # the round-5 score winner for the final print line.
                best_rmse, best_abs_rel = vm['rmse'], vm['abs_rel']
                print(f"  >> Best score:{best_score:.4f} "
                      f"RMSE:{best_rmse:.4f} ABS:{best_abs_rel:.4f}")
                log.update({'best/score': best_score,
                            'best/rmse_at_score': best_rmse,
                            'best/absrel_at_score': best_abs_rel,
                            'best/score_epoch': epoch})

            if vm['rmse'] < best_rmse:
                # Pure-RMSE winner can be different from the composite best.
                best_rmse = vm['rmse']
                best_epoch['rmse'] = epoch
                torch.save({**ckpt_payload, 'best_rmse': best_rmse,
                            'metric': 'rmse'},
                           os.path.join(ckpt_dir, 'best_rmse.pth'))
                print(f"  >> Best RMSE:{best_rmse:.4f}")
                log.update({'best/rmse': best_rmse,
                            'best/rmse_epoch': epoch})

            if vm['abs_rel'] < best_abs_rel:
                best_abs_rel = vm['abs_rel']
                best_epoch['absrel'] = epoch
                torch.save({**ckpt_payload, 'best_absrel': best_abs_rel,
                            'metric': 'absrel'},
                           os.path.join(ckpt_dir, 'best_absrel.pth'))
                print(f"  >> Best ABS_REL:{best_abs_rel:.4f}")
                log.update({'best/absrel': best_abs_rel,
                            'best/absrel_epoch': epoch})

            if vm['delta1'] > best_delta1:
                best_delta1 = vm['delta1']
                best_epoch['delta1'] = epoch
                torch.save({**ckpt_payload, 'best_delta1': best_delta1,
                            'metric': 'delta1'},
                           os.path.join(ckpt_dir, 'best_delta1.pth'))
                print(f"  >> Best Delta1:{best_delta1:.4f}")
                log.update({'best/delta1': best_delta1,
                            'best/delta1_epoch': epoch})

        # wandb.log(log)

        if epoch % cfg.mode.saving_checkpoints == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(ckpt_dir, f'checkpoint_{epoch}.pth'))

    print(
        f'\nDone. '
        f'Best score:{best_score:.4f}@ep{best_epoch["score"]}  '
        f'RMSE:{best_rmse:.4f}@ep{best_epoch["rmse"]}  '
        f'ABS:{best_abs_rel:.4f}@ep{best_epoch["absrel"]}  '
        f'D1:{best_delta1:.4f}@ep{best_epoch["delta1"]}')
    # wandb.finish()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='foa', help='Config name (baseline, foa)')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--lr-schedule', type=str, default=None,
                   choices=['cosine'],
                   help='Round-4: per-step LR schedule. cosine = '
                        'linear warmup → cosine decay. Default off '
                        '(constant LR — preserves prior round behaviour).')
    p.add_argument('--lr-warmup-epochs', type=int, default=None,
                   help='Round-4: warmup epochs for cosine schedule. '
                        'Default 1 when --lr-schedule=cosine. Pass 0 to '
                        'disable warmup but keep cosine decay.')
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
    # ── echorange (bin-based depth head) ──
    p.add_argument('--depth-head-type', type=str, default=None,
                   choices=['scalar', 'range', 'hazard'],
                   help='echorange: scalar (=baseline), range (soft-bin NLL), '
                        'or hazard (first-hit rendering).')
    p.add_argument('--range-num-bins', type=int, default=None)
    p.add_argument('--range-bin-spacing', type=str, default=None,
                   choices=['log', 'linear'])
    p.add_argument('--range-min-depth', type=float, default=None)
    p.add_argument('--range-max-depth', type=float, default=None)
    p.add_argument('--range-soft-label-sigma', type=float, default=None,
                   help='sigma of the log-Gaussian soft label (default 0.08).')
    p.add_argument('--range-output-mode', type=str, default=None,
                   choices=['expectation', 'median', 'map', 'quantile',
                            'temperature_expectation'],
                   help='RangeDepthHead.output_mode. Round-5 patch added '
                        'map/quantile/temperature_expectation; defaults to '
                        'expectation. Hard quantile/median cut gradient — '
                        'use them at eval, prefer expectation at train.')
    p.add_argument('--range-output-quantile', type=float, default=None,
                   help='q ∈ (0,1) used when range_output_mode=quantile. '
                        'Default 0.5 (median).')
    p.add_argument('--range-output-temperature', type=float, default=None,
                   help='Softmax T > 0 used by '
                        'range_output_mode=temperature_expectation. T=1 is '
                        'the regular expectation; T<1 sharpens; T>1 smears.')
    p.add_argument('--lambda-range-nll', type=float, default=None)
    p.add_argument('--lambda-berhu', type=float, default=None)
    p.add_argument('--lambda-silog', type=float, default=None)
    p.add_argument('--lambda-entropy-smooth', type=float, default=None)
    # ── Round-5 differentiable soft-quantile auxiliary -----------------
    p.add_argument('--lambda-soft-quantile', type=float, default=None,
                   help='Round-5: weight on soft-quantile (BerHu+SILog) '
                        'auxiliary loss. 0 disables (default).')
    p.add_argument('--soft-quantile-q', type=float, default=None,
                   help='Round-5: target quantile for soft-quantile aux '
                        '(default 0.5).')
    p.add_argument('--soft-quantile-tau', type=float, default=None,
                   help='Round-5: softness τ > 0 for soft-quantile aux '
                        '(default 0.05). Smaller = closer to hard quantile.')
    # ── Round-5 SH spherical auxiliary ---------------------------------
    p.add_argument('--lambda-spherical-sh', type=float, default=None,
                   help='Round-5: weight on low-order SH coefficient '
                        'matching loss. 0 disables (default).')
    p.add_argument('--spherical-sh-order', type=int, default=None,
                   choices=[0, 1, 2, 3, 4],
                   help='Round-5: SH max order L. Default 2.')
    p.add_argument('--spherical-sh-log-depth',
                   dest='spherical_sh_log_depth',
                   action='store_true', default=None,
                   help='Round-5: project SH on log(depth) instead of '
                        'linear depth (default true). Compresses dynamic '
                        'range for layout matching.')
    p.add_argument('--no-spherical-sh-log-depth',
                   dest='spherical_sh_log_depth', action='store_false')
    # ── Round-5 cylindrical bin axis -----------------------------------
    p.add_argument('--range-bin-axis', type=str, default=None,
                   choices=['radial', 'horizontal', 'z'],
                   help='Round-5: bin axis interpretation. radial (default) '
                        '= GT radial depth. horizontal = ρ_xy = D·cos(lat). '
                        'z = |z| = D·|sin(lat)|. Pred is projected back to '
                        'radial for BerHu/SILog/SH so eval stays radial.')
    p.add_argument('--cyl-min-axis-factor', type=float, default=None,
                   help='Round-5: min axis_factor (cos lat or |sin lat|) '
                        'before clamping. Pixels with smaller factors are '
                        'masked from all losses. Default 0.15 (~|lat|<81°).')
    # ── Round-5 eval-time representative override (consumed by test.py) -
    p.add_argument('--range-eval-mode', type=str, default=None,
                   choices=['default', 'expectation', 'map',
                            'q25', 'q35', 'q45', 'q50', 'q55', 'q65', 'q75',
                            'temp05', 'temp075', 'temp15'],
                   help='Eval-only: re-decode pred_depth from range_logits '
                        'using the chosen representative. default leaves '
                        "the head's training-time mode untouched.")
    p.add_argument('--checkpoint-tag', type=str, default=None,
                   choices=['score', 'absrel', 'rmse', 'delta1'],
                   help='Eval-only: pick best_<tag>.pth automatically. '
                        'Default score (= legacy best_model.pth).')
    # Hazard head only ----------------------------------------------------
    p.add_argument('--hazard-bias-init', type=float, default=None,
                   help='Initial bias of the hazard head final 1×1 conv. '
                        'sigmoid(bias) is the starting alpha. Default −4.6 '
                        '(α≈0.01).')
    p.add_argument('--hazard-warmup-epochs', type=int, default=None,
                   help='Epochs (inclusive) using the warmup hit/free weights. '
                        'After this, the post-warmup weights take over.')
    p.add_argument('--lambda-hit-warmup', type=float, default=None,
                   help='λ_hit during warmup (default 0.3).')
    p.add_argument('--lambda-free-warmup', type=float, default=None,
                   help='λ_free during warmup (default 0.05).')
    p.add_argument('--lambda-hit', type=float, default=None,
                   help='λ_hit after warmup (default 0.5).')
    p.add_argument('--lambda-free', type=float, default=None,
                   help='λ_free after warmup (default 0.1).')
    p.add_argument('--hazard-log-delta', type=float, default=None,
                   help='Half-width of the hit window in log-depth space. '
                        'Default = 0.5 × log(bin_ratio) → adjacent hit '
                        'windows tile the bin grid without overlap.')
    # Round-4 hazard rescue ----------------------------------------------
    p.add_argument('--hazard-aux-mode', type=str, default=None,
                   choices=['raw_hit', 'event_nll', 'survival', 'soft_hit'],
                   help='Round-4 dispatch for the primary hazard auxiliary '
                        'loss. raw_hit = round-3 BCE(α,1). event_nll = NLL '
                        'on rendered first-hit weight. survival = BCE on '
                        'cumulative S_j = P(D > r_j). soft_hit = BCE(α, '
                        'soft_target) on hit bins.')
    p.add_argument('--hazard-far-thresh', type=float, default=None,
                   help='Metres. GT depth ≥ this is treated as censored '
                        '(excluded from event/hit/soft_hit; in survival '
                        'mode used only as "we know D > r_j for r_j < '
                        'far_thresh"). Default 9.8.')
    p.add_argument('--hazard-event-sigma-bins', type=float, default=None,
                   help='σ of the event_nll soft target, in units of bin-'
                        'log-spacing (default 1.0).')
    p.add_argument('--hazard-survival-tau-bins', type=float, default=None,
                   help='τ of the survival GT sigmoid, in units of bin-log-'
                        'spacing (default 1.0).')
    p.add_argument('--hazard-soft-hit-target', type=float, default=None,
                   help='soft-hit α target (default 0.75 → logit ≈ 1.10).')
    p.add_argument('--disable-hit-loss', dest='disable_hit_loss',
                   action='store_true', default=None,
                   help='Hazard ablation: drop L_hit (free-only).')
    p.add_argument('--disable-free-loss', dest='disable_free_loss',
                   action='store_true', default=None,
                   help='Hazard ablation: drop L_free (hit-only).')
    p.add_argument('--hazard-depth-only', dest='hazard_depth_only',
                   action='store_true', default=None,
                   help='Hazard ablation: train the renderer with BerHu/'
                        'SILog only, no per-bin hit/free supervision.')
    # ── ERP-aware loss extras (echorange + ERP depth) ──
    p.add_argument('--erp-cos-lat-weight', dest='erp_cos_lat_weight',
                   action='store_true', default=None,
                   help='Weight per-pixel range NLL by cos(latitude) so '
                        'polar pixels (oversampled in ERP) do not dominate.')
    p.add_argument('--no-erp-cos-lat-weight', dest='erp_cos_lat_weight',
                   action='store_false')
    p.add_argument('--erp-far-mask', dest='erp_far_mask',
                   action='store_true', default=None,
                   help='Exclude pixels with GT >= range_max_depth from '
                        'the range NLL (avoids last-bin saturation spike).')
    p.add_argument('--no-erp-far-mask', dest='erp_far_mask',
                   action='store_false')
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
    if args.lr_schedule is not None: cfg.mode.lr_schedule = args.lr_schedule
    if args.lr_warmup_epochs is not None:
        cfg.mode.lr_warmup_epochs = args.lr_warmup_epochs
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
    # echorange overrides (covers scalar / range / hazard heads)
    for _k in ('depth_head_type', 'range_num_bins', 'range_bin_spacing',
               'range_min_depth', 'range_max_depth', 'range_soft_label_sigma',
               'range_output_mode', 'lambda_range_nll', 'lambda_berhu',
               'lambda_silog', 'lambda_entropy_smooth',
               'erp_cos_lat_weight', 'erp_far_mask',
               'hazard_bias_init', 'hazard_warmup_epochs',
               'lambda_hit_warmup', 'lambda_free_warmup',
               'lambda_hit', 'lambda_free', 'hazard_log_delta',
               'disable_hit_loss', 'disable_free_loss', 'hazard_depth_only',
               # Round-4 ----------------------------------------------
               'hazard_aux_mode', 'hazard_far_thresh',
               'hazard_event_sigma_bins', 'hazard_survival_tau_bins',
               'hazard_soft_hit_target',
               # Round-5 ----------------------------------------------
               'range_output_quantile', 'range_output_temperature',
               'lambda_soft_quantile', 'soft_quantile_q', 'soft_quantile_tau',
               'lambda_spherical_sh', 'spherical_sh_order',
               'spherical_sh_log_depth',
               'range_bin_axis', 'cyl_min_axis_factor',
               'range_eval_mode', 'checkpoint_tag'):
        _v = getattr(args, _k, None)
        if _v is not None:
            setattr(cfg.model, _k, _v)
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

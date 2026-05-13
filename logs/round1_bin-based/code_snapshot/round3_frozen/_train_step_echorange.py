"""Round 3/5 frozen excerpt — train.py:141..516 (`_train_step_echorange`).

This file is a verbatim snapshot of the echorange training step that drives
all R4 (hazard rescue) and R5 (range posterior + soft-quantile + SH +
cylindrical) experiments captured in `logs/n9_0427_test/`.

It is NOT a runnable module on its own — `train.py` imports it inline. The
snapshot exists for reviewers to read the loss-composition logic outside the
~1500-line train.py.

Three head modes share this step (cfg.model.depth_head_type):
    scalar  → exact echodiffusion baseline (BerHu + SILog).
    range   → λ_NLL · soft_range_nll + λ_BerHu · BerHu + λ_SILog · SILog
              + (R5) λ_sq · soft-quantile-BerHu/SILog
              + (R5) λ_sh · spherical SH coeff matching
              + (R5) range_bin_axis ∈ {radial, horizontal, z}
    hazard  → λ_aux · primary_loss(raw_hit | event_nll | survival | soft_hit)
              + λ_free · hazard_free_loss
              + λ_BerHu · BerHu + λ_SILog · SILog
              with smooth aux-weight ramp progress = min(1, epoch / ramp_ep).

Round-5 additions over round-3 are explicitly tagged in comments.
"""
import math
import torch
import torch.nn.functional as F


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
    # ρ_xy = D·cos(lat), or vertical |z| = D·|sin(lat)|. Bin-axis-only
    # swap: GT is internally projected onto the chosen axis for the
    # per-bin NLL/event loss, and the head's pred_depth is projected back
    # to radial before BerHu / SILog / SH so the eval pipeline (radial
    # RMSE) stays consistent.
    range_bin_axis = str(getattr(cfg.model, 'range_bin_axis', 'radial'))
    cyl_min_factor = float(
        getattr(cfg.model, 'cyl_min_axis_factor', 0.15))
    if range_bin_axis == 'radial':
        axis_factor_orig = None
        axis_factor_nll = None
    else:
        H_orig = gtdepth.shape[-2]
        H_nll = logits.shape[-2]
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
        gt_for_nll = gt_for_nll * axis_factor_nll
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
                q=sq_q, tau=sq_tau)                          # (B,1,h,w)
            if pred_q_axis.shape[-2:] != gtdepth.shape[-2:]:
                pred_q_axis = F.interpolate(
                    pred_q_axis, size=gtdepth.shape[-2:], mode='nearest')
            if axis_factor_orig is not None:
                pred_q_radial = pred_q_axis / axis_factor_orig
            else:
                pred_q_radial = pred_q_axis
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

        if bool(getattr(cfg.model, 'hazard_depth_only', False)):
            lam_aux = lam_free = 0.0
        if bool(getattr(cfg.model, 'disable_hit_loss', False)):
            lam_aux = 0.0
        if bool(getattr(cfg.model, 'disable_free_loss', False)):
            lam_free = 0.0

        # Mode dispatch (round 4): ``--hazard-aux-mode`` ∈
        #   raw_hit    : original BCE(α, target=1) on hit bins.
        #   event_nll  : NLL of rendered first-hit weight w_j vs Gaussian q_j.
        #   survival   : BCE on cumulative survival S_j = P(D > r_j).
        #   soft_hit   : BCE(α, target=0.75) — saturation guard.
        aux_mode = str(getattr(cfg.model, 'hazard_aux_mode', 'raw_hit'))
        far_thresh = float(getattr(cfg.model, 'hazard_far_thresh', 9.8))
        sigma_bins = float(getattr(cfg.model, 'hazard_event_sigma_bins', 1.0))
        tau_bins   = float(getattr(cfg.model, 'hazard_survival_tau_bins', 1.0))
        soft_hit_t = float(getattr(cfg.model, 'hazard_soft_hit_target', 0.75))
        log_delta_cfg = getattr(cfg.model, 'hazard_log_delta', None)
        log_delta = float(log_delta_cfg) if log_delta_cfg is not None else None

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
                    f"Unknown hazard_aux_mode: {aux_mode!r}.")

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
            f"Unknown depth_head_type for echorange: {head_type!r}.")

    # ── BerHu/SILog on rendered depth (shared, both heads) ─────────────
    # Project back to radial under cylindrical mode so eval (radial RMSE)
    # sees a radial prediction.
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

    # Mask polar pixels under cylindrical mode by zeroing GT there —
    # BerHuLoss/SILogLoss skip pixels where gt ≤ 0.
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

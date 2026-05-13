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
        hazard_free_loss,
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
        head_metrics = {'range_nll': nll.item()}
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
    if getattr(cfg.dataset, 'depth_norm', False):
        pred_norm = (pred_depth / float(cfg.dataset.max_depth)).clamp(min=1e-6)
        gt_norm = gtdepth
    else:
        pred_norm = pred_depth
        gt_norm = gtdepth

    berhu = BerHuLoss().to(device)(pred_norm, gt_norm)
    silog = SILogLoss().to(device)(pred_norm, gt_norm)

    loss = head_loss + lam_b * berhu + lam_s * silog
    ent_val = 0.0
    if lam_ent > 0 and 'range_entropy' in out:
        ent = out['range_entropy'].mean()
        loss = loss + lam_ent * ent
        ent_val = float(ent.item())

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    metrics = {
        'total': float(loss.item()),
        'depth': float(berhu.item() + silog.item()),
        'berhu': float(berhu.item()),
        'silog': float(silog.item()),
        'entropy': ent_val,
    }
    metrics.update(head_metrics)
    return metrics

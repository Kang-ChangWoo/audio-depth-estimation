"""Extracted from train.py @ commit-equivalent of 2026-04-28.

Echorange training step — handles scalar (= echodiffusion baseline)
and range (soft-bin NLL + expectation depth) head modes.
"""

def _train_step_echorange(model, batch, criterion, optimizer, cfg, device):
    """EchoRange — scalar baseline OR range distribution head.

    Scalar mode (depth_head_type='scalar'):
        L = BerHu + SILog (criterion as usual on out['pred_depth']).
    Range mode (depth_head_type='range'):
        L = lambda_range_nll * soft_range_nll_loss(logits, gt, bins, sigma)
          + lambda_berhu * BerHu(out['pred_depth'], gt)
          + lambda_silog * SILog(out['pred_depth'], gt)
          + lambda_entropy_smooth * mean(range_entropy)   (>0 → favours peakier dists)
    Either way, ``out['pred_depth']`` is what downstream metrics use.
    """
    from models.bin_based import soft_range_nll_loss
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

    # Range path — components computed inline so we don't depend on the
    # exact split inside `criterion`.
    # NB: dataset.depth_norm=true normalises GT to [0, 1]; the range head
    # works in raw metres. Rescale GT to metres for the NLL.
    if getattr(cfg.dataset, 'depth_norm', False):
        gtdepth_m = gtdepth * float(cfg.dataset.max_depth)
        pred_for_metric = pred_depth                          # already in metres? scalar head was sigmoid*max_depth too
    else:
        gtdepth_m = gtdepth
        pred_for_metric = pred_depth

    range_bins = (model.module.range_head.range_bins
                  if hasattr(model, 'module')
                  else model.range_head.range_bins)
    sigma = float(getattr(cfg.model, 'range_soft_label_sigma', 0.08))

    # Decoder may output at a different spatial size than the GT — match
    # GT to logits res for the NLL (downsampling GT is cheaper than
    # upsampling 64-ch logits to (256, 512)).
    logits = out['range_logits']
    if gtdepth_m.shape[-2:] != logits.shape[-2:]:
        gt_for_nll = F.interpolate(
            gtdepth_m, size=logits.shape[-2:], mode='nearest')
    else:
        gt_for_nll = gtdepth_m

    # ── ERP-aware extras ──
    # erp_far_mask     : exclude pixels at GT >= range_max_depth (ceiling
    #                    saturation in radial ERP) so they don't pile up
    #                    on the last bin and skew the distribution.
    # erp_cos_lat_weight : weight per-pixel loss by cos(latitude) so
    #                    polar pixels (oversampled in equirectangular
    #                    projection) don't dominate the mean.
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

    nll = soft_range_nll_loss(
        logits, gt_for_nll, range_bins,
        valid_mask=far_valid, sigma=sigma, weights=pix_w)

    # BerHu/SILog on the expected depth — push the GT back to whatever
    # range pred_depth lives in. Range head output is in metres; if the
    # criterion was set up for normalised depth, scale pred down to [0,1].
    if getattr(cfg.dataset, 'depth_norm', False):
        pred_norm = (pred_depth / float(cfg.dataset.max_depth)).clamp(min=1e-6)
        gt_norm = gtdepth
    else:
        pred_norm = pred_depth
        gt_norm = gtdepth

    berhu = BerHuLoss().to(device)(pred_norm, gt_norm)
    silog = SILogLoss().to(device)(pred_norm, gt_norm)

    lam_nll  = float(getattr(cfg.model, 'lambda_range_nll', 1.0))
    lam_b    = float(getattr(cfg.model, 'lambda_berhu', 1.0))
    lam_s    = float(getattr(cfg.model, 'lambda_silog', 1.0))
    lam_ent  = float(getattr(cfg.model, 'lambda_entropy_smooth', 0.0))

    loss = lam_nll * nll + lam_b * berhu + lam_s * silog
    ent_val = 0.0
    if lam_ent > 0:
        ent = out['range_entropy'].mean()
        loss = loss + lam_ent * ent
        ent_val = ent.item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {
        'total': loss.item(), 'depth': berhu.item() + silog.item(),
        'range_nll': nll.item(), 'berhu': berhu.item(),
        'silog': silog.item(), 'entropy': ent_val,
    }

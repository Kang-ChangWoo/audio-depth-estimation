"""Testing/evaluation utilities."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_errors, compute_errors_sphere, compute_foa_errors
from .train_utils import (
    is_foa_model, is_foa_variant_model, is_echodiffusion_model,
    is_foa_v2_js_model, is_foa_0415_model, is_foa_v2_js_rgb_model,
    is_foa_oracle_model, is_n2_model, is_emap_temporal_model,
    is_n3_0425_model, is_echorange_model,
)


# ---------------------------------------------------------------------------
# Round-5 eval-time helpers
# ---------------------------------------------------------------------------

# Mapping for --range-eval-mode → (point_estimate_mode, q, temperature).
# The mode names q25/q35/.../q75 and temp05/temp075/temp15 keep CLI args
# valid Python identifiers and also match the cell-script vocabulary.
_RANGE_EVAL_PRESETS = {
    'expectation':  ('expectation',             0.5, 1.0),
    'map':          ('map',                     0.5, 1.0),
    'q25':          ('quantile',                0.25, 1.0),
    'q35':          ('quantile',                0.35, 1.0),
    'q45':          ('quantile',                0.45, 1.0),
    'q50':          ('quantile',                0.50, 1.0),
    'q55':          ('quantile',                0.55, 1.0),
    'q65':          ('quantile',                0.65, 1.0),
    'q75':          ('quantile',                0.75, 1.0),
    'temp05':       ('temperature_expectation', 0.5, 0.5),
    'temp075':      ('temperature_expectation', 0.5, 0.75),
    'temp15':       ('temperature_expectation', 0.5, 1.5),
}


def _override_range_pred_depth(raw_out, eval_mode):
    """Re-decode pred_depth from range_logits using the chosen representative.

    Returns the new (B, 1, h, w) prediction in the head's bin-axis units
    (caller is responsible for any cylindrical → radial projection / depth
    norm scaling that the regular path applies).

    For hazard heads (which don't expose a softmax distribution over bins
    in the standard sense) we fall back to the raw pred_depth — eval_mode
    overrides only apply when range_logits + range_bins are both present
    AND the head is a softmax range head.
    """
    if 'range_logits' not in raw_out or 'range_bins' not in raw_out:
        return raw_out['pred_depth']
    if eval_mode == 'default':
        return raw_out['pred_depth']
    if eval_mode not in _RANGE_EVAL_PRESETS:
        raise ValueError(
            f"unknown range_eval_mode {eval_mode!r}; "
            f"expected one of: default, {sorted(_RANGE_EVAL_PRESETS.keys())}")

    # Local import so test_utils doesn't pull torch-heavy modules at top.
    from models.bin_based.range_head import range_point_estimate

    mode, q, T = _RANGE_EVAL_PRESETS[eval_mode]
    pred, _ = range_point_estimate(
        logits=raw_out['range_logits'],
        range_bins=raw_out['range_bins'],
        mode=mode, q=q, temperature=T,
    )
    # The head's normal forward upsamples pred_depth to the input ERP
    # resolution; range_logits stays at decoder resolution. Match the
    # output resolution so the downstream metric gets the right shape.
    target_hw = raw_out['pred_depth'].shape[-2:]
    if pred.shape[-2:] != target_hw:
        pred = F.interpolate(pred, size=target_hw, mode='nearest')
    return pred


def _project_pred_to_radial(pred_depth, cfg, device):
    """Bin-axis → radial projection for cylindrical configs.

    pred_depth: (B, 1, H, W) in bin-axis units.
    Returns (B, 1, H, W) in radial metres; for radial configs returns
    pred_depth unchanged (no allocation).
    """
    bin_axis = str(getattr(cfg.model, 'range_bin_axis', 'radial'))
    if bin_axis == 'radial':
        return pred_depth
    H = pred_depth.shape[-2]
    cyl_min = float(getattr(cfg.model, 'cyl_min_axis_factor', 0.15))
    lat = (math.pi / 2.0) - math.pi * (
        torch.arange(H, device=device).float() + 0.5) / H
    if bin_axis == 'horizontal':
        f = torch.cos(lat).clamp(min=cyl_min)
    elif bin_axis == 'z':
        f = torch.sin(lat).abs().clamp(min=cyl_min)
    else:
        raise ValueError(f"range_bin_axis must be radial / horizontal / z; "
                         f"got {bin_axis!r}")
    return pred_depth / f.view(1, 1, -1, 1)


def evaluate(model, eval_loader, eval_set, cfg, device):
    """Run evaluation. Returns (depth_errors, foa_errors_list_or_None).

    For n3_0425 (FOA-prediction-only), depth_errors carries the FOA metrics
    so the existing test.py print path / bulk-script grep keep working:
      ABS_REL ← mean L1(pred_rep, rep_gt)
      RMSE    ← mean sqrt(MSE) per-(batch, bin, channel)
      Delta1  ← mean cosine(pred_rep, rep_gt)  (eigen) or NaN (rms)
    """
    model.eval()
    depth_errors = []
    foa0415 = is_foa_0415_model(cfg)
    js_rgb = is_foa_v2_js_rgb_model(cfg)
    oracle = is_foa_oracle_model(cfg)
    emap_temporal = is_emap_temporal_model(cfg)
    n2 = is_n2_model(cfg) or emap_temporal
    n3_0425 = is_n3_0425_model(cfg)
    foa = (is_foa_model(cfg) or is_foa_variant_model(cfg)) and not foa0415 and not js_rgb
    echodiff = is_echodiffusion_model(cfg)
    echorange = is_echorange_model(cfg)
    js = is_foa_v2_js_model(cfg) and not js_rgb
    foa_errors = [] if (foa or foa0415 or js_rgb or oracle or n2) else None
    batch_size = cfg.mode.batch_size

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if n3_0425:
                # FOA prediction (no depth). Stuff [coef_l1, rmse, cos, …]
                # into depth_errors so the existing print/save path emits
                # the bulk-script-friendly ABS_REL / RMSE / Delta1 labels.
                if len(batch) != 5:
                    raise RuntimeError(
                        "n3_0425 evaluation requires the 5-tuple batch from "
                        "use_distance_bins=True dataset.")
                audio = batch[0].to(device)
                rep_gt = batch[4].to(device)            # (B, K, 4)
                raw_out = model(audio)
                pred_rep = raw_out['pred_rep']           # (B, K, 4)
                # Per-sample metrics: L1, RMSE, mean cosine over (K, 4).
                diff = (pred_rep - rep_gt)
                l1_per = diff.abs().mean(dim=(1, 2))     # (B,)
                rmse_per = diff.pow(2).mean(dim=(1, 2)).sqrt()
                rep_kind = str(getattr(cfg.dataset, 'rep_kind', 'eigen')).lower()
                if rep_kind == 'eigen':
                    cos_per = torch.nn.functional.cosine_similarity(
                        pred_rep, rep_gt, dim=-1).mean(dim=1)  # (B,)
                else:
                    cos_per = torch.full_like(l1_per, float('nan'))
                for b in range(pred_rep.shape[0]):
                    # n3_0425 is FOA-only — sphere-weighted metrics are
                    # not meaningful here, so pad the trailing 7 slots
                    # with zeros to keep the (N, 14) row shape uniform
                    # with the depth-metric branch below.
                    depth_errors.append(np.array([
                        l1_per[b].item(),       # ABS_REL slot ← L1
                        rmse_per[b].item(),     # RMSE slot ← RMSE
                        cos_per[b].item(),      # Delta1 slot ← cosine
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    ]))
                if (batch_idx + 1) % 10 == 0:
                    total = min((batch_idx + 1) * batch_size, len(eval_set))
                    print(f'  {batch_idx + 1}/{len(eval_loader)} ({total} samples)')
                continue
            if js_rgb:
                audio, depthgt, gt_foa_batch, _, _rgb = batch
                gt_foa_batch = gt_foa_batch.to(device)
                audio, depthgt = audio.to(device), depthgt.to(device)
                raw_out = model(audio, mode='test')
                depth_pred = raw_out["pred_depth"]
                if depth_pred.shape != depthgt.shape:
                    import torch.nn.functional as _F
                    depth_pred = _F.interpolate(depth_pred, size=depthgt.shape[2:],
                                                mode='nearest')
                pred_foa_batch = raw_out["pred_foa"]
            elif foa0415:
                # Batch shapes (depending on dataset flags):
                #   4-tuple — legacy foa_0415 (use_ambisonic only)
                #   5-tuple — n9_0424 / n4_0425  (+ use_distance_bins)
                #   6-tuple — echodiffusion_ambi(+sh) CIDE path
                #             (+ use_waveform — adds raw waveform last)
                audio, depthgt = batch[0].to(device), batch[1].to(device)
                if len(batch) >= 5:
                    # n9_0424 per-bin path. pred_sh = rep_pred[:, 0] is a
                    # legacy compat stub sign-canonicalized per-bin, while
                    # batch[2] (gt_foa) is sign-canonicalized once on the
                    # full IR — the two conventions disagree, yielding a
                    # systematically negative FOA_DIR. Compare bin-0 of
                    # rep_pred against bin-0 of rep_gt instead: same sign
                    # convention, matches the model's training target.
                    rep_gt = batch[4].to(device)
                    em = batch[3].to(device)
                    # Pass rep_gt + energy_map for n4_0425's oracle paths,
                    # plus audio_wave for the echodiffusion_ambi(+sh) CIDE
                    # branch when the dataset emits the 6-tuple. Models
                    # with **_unused (n9_0424, vit_foa_*) silently ignore.
                    if len(batch) >= 6:
                        waveform = batch[5].to(device)
                        raw_out = model(audio, rep_gt=rep_gt, energy_map=em,
                                        audio_wave=waveform)
                    else:
                        raw_out = model(audio, rep_gt=rep_gt, energy_map=em)
                    depth_pred = raw_out["pred_depth"]
                    gt_foa_batch = rep_gt[:, 0, :]
                    pred_foa_batch = raw_out["rep_pred"][:, 0, :]
                else:
                    gt_foa_batch = batch[2].to(device)
                    raw_out = model(audio)
                    depth_pred = raw_out["pred_depth"]
                    pred_foa_batch = raw_out["pred_sh"][:, :4]
            elif oracle:
                audio, depthgt, gt_foa_batch, energy_map = batch
                audio, depthgt = audio.to(device), depthgt.to(device)
                gt_foa_batch = gt_foa_batch.to(device)
                energy_map = energy_map.to(device)
                input_nc = int(getattr(cfg.model, 'input_nc', 3))
                if input_nc == 1:
                    model_input = energy_map
                else:
                    model_input = torch.cat([audio, energy_map], dim=1)
                raw_out = model(model_input)
                depth_pred = raw_out["pred_depth"]
                pred_foa_batch = raw_out["pred_sh"][:, :4]
            elif n2:
                audio, depthgt, gt_foa_batch, _energy, foa_spec, temporal_rms, temporal_energies = batch
                audio, depthgt = audio.to(device), depthgt.to(device)
                gt_foa_batch = gt_foa_batch.to(device)
                model_name = cfg.model.name
                if model_name in ('n2_temap_input',
                                  'pvit_n1_temap_input',
                                  'pvit_n1_temap_eattn',
                                  'pvit_n1_temap_mssh'):
                    temporal_energies = temporal_energies.to(device)
                    raw_out = model(torch.cat([audio, temporal_energies], dim=1))
                elif model_name == 'n2_6ch_input':
                    foa_spec = foa_spec.to(device)
                    raw_out = model(torch.cat([audio, foa_spec], dim=1))
                elif model_name in ('n2_dual_enc', 'n2_foa_stft_film'):
                    foa_spec = foa_spec.to(device)
                    raw_out = model(audio, foa_spec)
                elif model_name in ('n2_temporal_rms_film',
                                    'pvit_n1_temap_rms_film'):
                    temporal_rms_d = temporal_rms.to(device)
                    raw_out = model(audio, temporal_rms_d)
                elif model_name == 'n2_tbin_crossattn':
                    temporal_energies = temporal_energies.to(device)
                    raw_out = model(audio, temporal_energies=temporal_energies)
                elif model_name in ('n3_emap_unet_temporal', 'n3_emap_vit_temporal',
                                    'n3_emap_unet_temporal_ov', 'n3_emap_vit_temporal_ov'):
                    temporal_energies = temporal_energies.to(device)
                    raw_out = model(temporal_energies)
                else:
                    raw_out = model(audio)
                depth_pred = raw_out["pred_depth"]
                pred_foa_batch = raw_out["pred_sh"][:, :4]
            elif js:
                audio, depthgt, gt_foa_batch, _gt_energy = batch
                gt_foa_batch = gt_foa_batch.to(device)
                audio, depthgt = audio.to(device), depthgt.to(device)
                raw_out = model(audio)
                depth_pred = raw_out["pred_depth"]
                pred_foa_batch = raw_out["pred_foa"]
            elif foa:
                audio, depthgt, gt_foa_batch, _ = batch
                gt_foa_batch = gt_foa_batch.to(device)
                audio, depthgt = audio.to(device), depthgt.to(device)
                raw_out = model(audio)
                depth_pred = raw_out["pred_depth"]
                pred_foa_batch = raw_out["pred_foa"]
            elif echorange:
                audio, depthgt, waveform = batch
                audio, depthgt = audio.to(device), depthgt.to(device)
                waveform = waveform.to(device)
                raw_out = model(audio, waveform)

                head_type = getattr(cfg.model, 'depth_head_type', 'scalar')
                eval_mode = str(
                    getattr(cfg.model, 'range_eval_mode', 'default'))

                # Round-5: optional eval-time representative override on
                # the range head (re-decode pred_depth from range_logits).
                # Only applies when both head_type='range' AND eval_mode is
                # not 'default'; hazard / scalar heads use raw pred_depth.
                if head_type == 'range' and eval_mode != 'default':
                    depth_pred = _override_range_pred_depth(raw_out, eval_mode)
                else:
                    depth_pred = raw_out["pred_depth"]

                # Round-5: cylindrical bin-axis → radial projection so
                # depth_pred lives in radial metres before depth_norm
                # scaling and metric computation.
                if head_type in ('range', 'hazard'):
                    depth_pred = _project_pred_to_radial(
                        depth_pred, cfg, device)

                # Range/hazard heads emit metres; the scalar head, after
                # training against normalised GT, ends up numerically in
                # [0,1]. The downstream metric path multiplies pred by
                # max_depth under depth_norm=true, so for range/hazard
                # heads we scale back to normalised [0,1] first to keep
                # the units sane.
                if (cfg.dataset.depth_norm
                        and head_type in ('range', 'hazard')):
                    depth_pred = depth_pred / float(cfg.dataset.max_depth)
            elif echodiff:
                audio, depthgt, waveform = batch
                audio, depthgt = audio.to(device), depthgt.to(device)
                waveform = waveform.to(device)
                depth_pred = model(audio, waveform)
            else:
                audio, depthgt = batch[0], batch[1]
                audio, depthgt = audio.to(device), depthgt.to(device)
                depth_pred = model(audio)

            for idx in range(depth_pred.shape[0]):
                gt_map = depthgt[idx, 0].cpu().numpy()
                pred_map = depth_pred[idx, 0].cpu().numpy()
                if cfg.dataset.depth_norm:
                    gt_map = gt_map * cfg.dataset.max_depth
                    pred_map = pred_map * cfg.dataset.max_depth
                pred_map = np.clip(pred_map, 1e-3, cfg.dataset.max_depth)
                gt_map = np.maximum(gt_map, 0.0)
                # Round-4: append 14 columns = (7 uniform) + (7 sphere
                # cos-lat-weighted). Downstream test.py reads md[:7] for
                # the uniform block (preserving the grep contract) and
                # md[7:] for the sphere block.
                u = compute_errors(gt_map, pred_map)
                s = compute_errors_sphere(gt_map, pred_map)
                depth_errors.append(tuple(u) + tuple(s))

                if foa_errors is not None:
                    foa_errors.append(compute_foa_errors(
                        gt_foa_batch[idx].cpu().numpy(),
                        pred_foa_batch[idx].cpu().numpy()))

            if (batch_idx + 1) % 10 == 0:
                total = min((batch_idx + 1) * batch_size, len(eval_set))
                print(f'  {batch_idx + 1}/{len(eval_loader)} ({total} samples)')

    return np.array(depth_errors), foa_errors

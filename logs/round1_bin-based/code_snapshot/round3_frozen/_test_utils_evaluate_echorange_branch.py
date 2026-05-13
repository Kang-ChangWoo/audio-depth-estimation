"""Round 5 frozen excerpt — utils/test_utils.py.

Echorange evaluation branch. Includes:
 - `_RANGE_EVAL_PRESETS`     : --range-eval-mode → (mode, q, T) lookup.
 - `_override_range_pred_depth` : re-decode pred_depth from range_logits at
                                  test time using the chosen representative.
 - `_project_pred_to_radial` : bin-axis (horizontal/z) → radial projection
                               for cylindrical configs.

Inserted into `evaluate(...)` in the `elif echorange:` branch.
"""
import math

import torch
import torch.nn.functional as F


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
    """Re-decode pred_depth from range_logits with the chosen representative.

    For hazard heads (no normal softmax distribution) we fall back to the raw
    pred_depth. eval_mode overrides only apply when range_logits + range_bins
    are both present AND the head is a softmax range head.
    """
    if 'range_logits' not in raw_out or 'range_bins' not in raw_out:
        return raw_out['pred_depth']
    if eval_mode == 'default':
        return raw_out['pred_depth']
    if eval_mode not in _RANGE_EVAL_PRESETS:
        raise ValueError(
            f"unknown range_eval_mode {eval_mode!r}; "
            f"expected one of: default, {sorted(_RANGE_EVAL_PRESETS.keys())}")

    from models.bin_based.range_head import range_point_estimate

    mode, q, T = _RANGE_EVAL_PRESETS[eval_mode]
    pred, _ = range_point_estimate(
        logits=raw_out['range_logits'],
        range_bins=raw_out['range_bins'],
        mode=mode, q=q, temperature=T,
    )
    target_hw = raw_out['pred_depth'].shape[-2:]
    if pred.shape[-2:] != target_hw:
        pred = F.interpolate(pred, size=target_hw, mode='nearest')
    return pred


def _project_pred_to_radial(pred_depth, cfg, device):
    """Bin-axis → radial projection for cylindrical configs.

    pred_depth: (B, 1, H, W) in bin-axis units.
    Returns radial metres; for radial configs returns pred_depth unchanged.
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


# ────────────────────────────────────────────────────────────────────────
# Inserted in evaluate(...):
#
#     elif echorange:
#         audio, depthgt, waveform = batch
#         audio, depthgt = audio.to(device), depthgt.to(device)
#         waveform = waveform.to(device)
#         raw_out = model(audio, waveform)
#
#         head_type = getattr(cfg.model, 'depth_head_type', 'scalar')
#         eval_mode = str(getattr(cfg.model, 'range_eval_mode', 'default'))
#
#         if head_type == 'range' and eval_mode != 'default':
#             depth_pred = _override_range_pred_depth(raw_out, eval_mode)
#         else:
#             depth_pred = raw_out["pred_depth"]
#
#         if head_type in ('range', 'hazard'):
#             depth_pred = _project_pred_to_radial(depth_pred, cfg, device)
# ────────────────────────────────────────────────────────────────────────

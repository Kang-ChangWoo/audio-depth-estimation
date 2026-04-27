"""Testing/evaluation utilities."""

import numpy as np
import torch

from .metrics import compute_errors, compute_foa_errors
from .train_utils import (
    is_foa_model, is_foa_variant_model, is_echodiffusion_model,
    is_foa_v2_js_model, is_foa_0415_model, is_foa_v2_js_rgb_model,
    is_foa_oracle_model, is_n2_model, is_emap_temporal_model,
    is_n3_0425_model,
)


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
                    depth_errors.append(np.array([
                        l1_per[b].item(),       # ABS_REL slot ← L1
                        rmse_per[b].item(),     # RMSE slot ← RMSE
                        cos_per[b].item(),      # Delta1 slot ← cosine
                        0.0, 0.0, 0.0, 0.0,
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
                depth_errors.append(compute_errors(gt_map, pred_map))

                if foa_errors is not None:
                    foa_errors.append(compute_foa_errors(
                        gt_foa_batch[idx].cpu().numpy(),
                        pred_foa_batch[idx].cpu().numpy()))

            if (batch_idx + 1) % 10 == 0:
                total = min((batch_idx + 1) * batch_size, len(eval_set))
                print(f'  {batch_idx + 1}/{len(eval_loader)} ({total} samples)')

    return np.array(depth_errors), foa_errors

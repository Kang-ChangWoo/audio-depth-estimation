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

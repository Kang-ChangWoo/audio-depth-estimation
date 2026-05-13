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
                depth_pred = raw_out["pred_depth"]
                # Range/hazard heads emit metres; the scalar head, after
                # training against normalised GT, ends up numerically in
                # [0,1]. The downstream metric path multiplies pred by
                # max_depth under depth_norm=true, so for range/hazard
                # heads we scale back to normalised [0,1] first to keep
                # the units sane.
                if (cfg.dataset.depth_norm
                        and getattr(cfg.model, 'depth_head_type',
                                    'scalar') in ('range', 'hazard')):
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
                depth_errors.append(compute_errors(gt_map, pred_map))

                if foa_errors is not None:
                    foa_errors.append(compute_foa_errors(
                        gt_foa_batch[idx].cpu().numpy(),
                        pred_foa_batch[idx].cpu().numpy()))

            if (batch_idx + 1) % 10 == 0:
                total = min((batch_idx + 1) * batch_size, len(eval_set))
                print(f'  {batch_idx + 1}/{len(eval_loader)} ({total} samples)')

    return np.array(depth_errors), foa_errors

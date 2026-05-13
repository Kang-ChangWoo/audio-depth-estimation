"""Extracted from train.py val pass @ 2026-04-28.

Range-head pred is in metres; downstream metric path multiplies by
max_depth under depth_norm=true, so range pred is normalised here
so both head modes share the same metric path.
"""

elif echorange:
                audio, gtdepth, waveform = batch
                audio, gtdepth = audio.to(device), gtdepth.to(device)
                waveform = waveform.to(device)
                out = model(audio, waveform)
                depth_pred = out["pred_depth"]
                # Range head emits metres; the scalar head, after training
                # against normalised GT, ends up numerically in [0,1].
                # The downstream metric path (lines below) multiplies pred
                # by max_depth under depth_norm=true, so for range head we
                # scale to normalised here so both heads share the path.
                if (getattr(cfg.dataset, 'depth_norm', False)
                        and getattr(cfg.model, 'depth_head_type',
                                    'scalar') == 'range'):
                    depth_pred = depth_pred / float(cfg.dataset.max_depth)
                if getattr(cfg.dataset, 'depth_norm', False):
                    pred_for_crit = depth_pred.clamp(min=1e-6)
                else:
                    pred_for_crit = depth_pred
                lv = criterion(pred_for_crit, gtdepth)

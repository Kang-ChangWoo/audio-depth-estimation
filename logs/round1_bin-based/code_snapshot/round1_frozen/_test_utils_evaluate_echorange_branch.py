            elif echorange:
                audio, depthgt, waveform = batch
                audio, depthgt = audio.to(device), depthgt.to(device)
                waveform = waveform.to(device)
                raw_out = model(audio, waveform)
                depth_pred = raw_out["pred_depth"]
                # Range head emits metres; the scalar head, after training
                # against normalised GT, ends up numerically in [0,1].
                # The downstream metric path multiplies pred by max_depth
                # under depth_norm=true, so for range head we must scale
                # back to normalised [0,1] first to keep the units sane.
                if (cfg.dataset.depth_norm
                        and getattr(cfg.model, 'depth_head_type',
                                    'scalar') == 'range'):
                    depth_pred = depth_pred / float(cfg.dataset.max_depth)
            elif echodiff:
                audio, depthgt, waveform = batch
                audio, depthgt = audio.to(device), depthgt.to(device)
                waveform = waveform.to(device)

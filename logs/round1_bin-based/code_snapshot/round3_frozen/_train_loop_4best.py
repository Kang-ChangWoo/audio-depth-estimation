"""Round 5 frozen excerpt — train.py:1270..1488.

Per-epoch validation + 4-best checkpoint save (best_score / best_rmse /
best_absrel / best_delta1). best_model.pth is kept as a legacy alias of
best_score.pth so existing test pipelines continue to work.

Composite score is hard-coded as 0.7·RMSE + 0.3·ABS_REL.
"""
import os
import time

import numpy as np
import torch


def _train_loop_excerpt(model, train_loader, val_loader, optimizer, scheduler,
                        criterion, cfg, device, ckpt_dir, results_dir,
                        echorange, echodiff, foa, ...):
    # ↓↓↓ Verbatim from train.py:1277..1488 ↓↓↓

    best_rmse, best_abs_rel = float('inf'), float('inf')
    best_score = float('inf')               # weighted: 0.7*rmse + 0.3*abs_rel
    best_delta1 = -float('inf')
    best_epoch = {'score': -1, 'rmse': -1, 'absrel': -1, 'delta1': -1}

    for epoch in range(start_epoch, cfg.mode.epochs + 1):
        # ... (training inner loop omitted; calls _train_step_echorange) ...

        if cfg.mode.validation and epoch % cfg.mode.validation_iter == 0:
            vm, vis_p, vis_g = _val_metrics(
                model, val_loader, criterion, cfg, device, foa, echodiff,
                use_hist, foa_frozen, js=js, foa0415=foa0415, js_rgb=js_rgb,
                foa_oracle=foa_oracle, n2=n2, n3_0425=n3_0425,
                echorange=echorange)
            print(f"  Val L:{vm['val_loss']:.4f} ABS:{vm['abs_rel']:.4f} "
                  f"RMSE:{vm['rmse']:.4f} d1:{vm['delta1']:.4f}")

            # ── Round 5: per-metric 4-best checkpoint save ─────────────
            score = 0.7 * vm['rmse'] + 0.3 * vm['abs_rel']
            ckpt_payload = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            if score < best_score:
                best_score = score
                best_rmse_at_score = vm['rmse']
                best_absrel_at_score = vm['abs_rel']
                best_epoch['score'] = epoch
                payload = {**ckpt_payload,
                           'best_score': best_score,
                           'best_rmse_at_score': best_rmse_at_score,
                           'best_absrel_at_score': best_absrel_at_score}
                torch.save(payload, os.path.join(ckpt_dir, 'best_score.pth'))
                # Legacy alias — same payload, same path.
                torch.save(payload, os.path.join(ckpt_dir, 'best_model.pth'))
                best_rmse, best_abs_rel = vm['rmse'], vm['abs_rel']

            if vm['rmse'] < best_rmse:
                best_rmse = vm['rmse']
                best_epoch['rmse'] = epoch
                torch.save({**ckpt_payload, 'best_rmse': best_rmse,
                            'metric': 'rmse'},
                           os.path.join(ckpt_dir, 'best_rmse.pth'))

            if vm['abs_rel'] < best_abs_rel:
                best_abs_rel = vm['abs_rel']
                best_epoch['absrel'] = epoch
                torch.save({**ckpt_payload, 'best_absrel': best_abs_rel,
                            'metric': 'absrel'},
                           os.path.join(ckpt_dir, 'best_absrel.pth'))

            if vm['delta1'] > best_delta1:
                best_delta1 = vm['delta1']
                best_epoch['delta1'] = epoch
                torch.save({**ckpt_payload, 'best_delta1': best_delta1,
                            'metric': 'delta1'},
                           os.path.join(ckpt_dir, 'best_delta1.pth'))

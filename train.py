#!/usr/bin/env python3
"""Training script for audio-to-depth estimation."""

import argparse
import os
import time

import numpy as np
import torch
# import wandb

from utils.config import load_config
from utils.train_utils import (
    build_model, build_criterion, is_foa_model, is_foa_variant_model,
    is_echodiffusion_model, is_foa_v2_js_model,
    compute_gt_depth_sh, compute_gt_energy_sh, set_sh_branch_frozen,
)
from utils.visualization import save_batch_visualization
from utils.metrics import compute_errors
from data.dataloader import make_dataloader


# ── helpers ──────────────────────────────────────────────────

def _train_step_baseline(model, batch, criterion, optimizer, cfg, device):
    audio, gtdepth = batch[0], batch[1]
    audio, gtdepth = audio.to(device), gtdepth.to(device)
    optimizer.zero_grad()
    pred = model(audio)
    loss = criterion(pred, gtdepth)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {'total': loss.item(), 'depth': loss.item()}


def _train_step_echodiffusion(model, batch, criterion, optimizer, cfg, device):
    audio, gtdepth, waveform = batch
    audio, gtdepth, waveform = audio.to(device), gtdepth.to(device), waveform.to(device)
    optimizer.zero_grad()
    pred = model(audio, waveform)
    loss = criterion(pred, gtdepth)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {'total': loss.item(), 'depth': loss.item()}


def _train_step_foa(model, batch, criterion, optimizer, cfg, device,
                    use_hist, foa_frozen):
    audio, gtdepth, gt_foa, _ = batch
    audio, gtdepth, gt_foa = audio.to(device), gtdepth.to(device), gt_foa.to(device)
    optimizer.zero_grad()
    outputs = model(audio, return_hist_maps=use_hist)

    if foa_frozen:
        loss = criterion.depth_criterion(outputs["pred_depth"], gtdepth) * criterion.depth_weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return {'total': loss.item(), 'depth': loss.item() / criterion.depth_weight}

    gt_dsh, gt_dsh_c = (compute_gt_depth_sh(model, gtdepth) if use_hist
                        else (None, None))
    ld = criterion(outputs, gtdepth, gt_foa,
                   gt_depth_sh=gt_dsh, gt_depth_sh_coeffs=gt_dsh_c)
    # KL loss is already included in ld["total"] by AudioDepthFOALoss (weighted by kl_weight).
    # Add FOA-depth gradient consistency loss if present (foa_v2).
    if "foa_depth_consistency" in outputs:
        consistency = outputs["foa_depth_consistency"].mean()  # DataParallel gathers scalars into vector
        foa_consistency_weight = getattr(cfg.model, 'foa_consistency_weight', 0.05)
        ld["total"] = ld["total"] + foa_consistency_weight * consistency
        ld["consistency"] = consistency
    ld["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {k: v.item() for k, v in ld.items()}


def _train_step_js(model, batch, criterion, optimizer, cfg, device,
                   use_hist, foa_frozen):
    """Training step for foa_v2_js: uses the ambisonic energy map directly.

    Unlike _train_step_foa, which projects the depth map into SH space as the
    histogram alignment target, this step uses the actual ambisonic-derived
    directional energy map (4th element of the batch). This provides a more
    direct supervision signal grounded in the recorded sound field rather than
    a depth-derived proxy.
    """
    audio, gtdepth, gt_foa, gt_energy = batch
    audio = audio.to(device)
    gtdepth = gtdepth.to(device)
    gt_foa = gt_foa.to(device)
    gt_energy = gt_energy.to(device)
    optimizer.zero_grad()
    outputs = model(audio, return_hist_maps=use_hist)

    if foa_frozen:
        loss = criterion.depth_criterion(outputs["pred_depth"], gtdepth) * criterion.depth_weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return {'total': loss.item(), 'depth': loss.item() / criterion.depth_weight}

    # Project the ambisonic energy map into SH space and use it as the
    # histogram alignment target (instead of the depth-derived projection).
    gt_esh, gt_esh_c = (compute_gt_energy_sh(model, gt_energy) if use_hist
                        else (None, None))
    ld = criterion(outputs, gtdepth, gt_foa,
                   gt_depth_sh=gt_esh, gt_depth_sh_coeffs=gt_esh_c)
    # FOA-depth gradient consistency (inherited from foa_v2 forward path)
    if "foa_depth_consistency" in outputs:
        consistency = outputs["foa_depth_consistency"].mean()
        foa_consistency_weight = getattr(cfg.model, 'foa_consistency_weight', 0.05)
        ld["total"] = ld["total"] + foa_consistency_weight * consistency
        ld["consistency"] = consistency
    ld["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {k: v.item() for k, v in ld.items()}


def _val_metrics(model, val_loader, criterion, cfg, device, foa, echodiff,
                 use_hist, foa_frozen, js=False):
    model.eval()
    errors, val_losses = [], []
    vis_pred, vis_gt = None, None

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if js:
                audio, gtdepth, gt_foa_v, gt_energy_v = batch
                audio = audio.to(device)
                gtdepth = gtdepth.to(device)
                gt_foa_v = gt_foa_v.to(device)
                gt_energy_v = gt_energy_v.to(device)
                out = model(audio, return_hist_maps=use_hist)
                depth_pred = out["pred_depth"]
                if foa_frozen:
                    lv = criterion.depth_criterion(depth_pred, gtdepth) * criterion.depth_weight
                else:
                    gt_esh, gt_esh_c = (compute_gt_energy_sh(model, gt_energy_v) if use_hist
                                        else (None, None))
                    lv = criterion(out, gtdepth, gt_foa_v,
                                   gt_depth_sh=gt_esh, gt_depth_sh_coeffs=gt_esh_c)["total"]
            elif foa:
                audio, gtdepth, gt_foa_v, _ = batch
                audio, gtdepth = audio.to(device), gtdepth.to(device)
                gt_foa_v = gt_foa_v.to(device)
                out = model(audio, return_hist_maps=use_hist)
                depth_pred = out["pred_depth"]
                if foa_frozen:
                    lv = criterion.depth_criterion(depth_pred, gtdepth) * criterion.depth_weight
                else:
                    gt_dsh, gt_dsh_c = (compute_gt_depth_sh(model, gtdepth) if use_hist
                                        else (None, None))
                    lv = criterion(out, gtdepth, gt_foa_v,
                                   gt_depth_sh=gt_dsh, gt_depth_sh_coeffs=gt_dsh_c)["total"]
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
    return {
        'val_loss': np.mean(val_losses),
        'abs_rel': me[0], 'rmse': me[1],
        'delta1': me[2], 'delta2': me[3], 'delta3': me[4],
        'log10': me[5], 'mae': me[6],
    }, vis_pred, vis_gt


# ── main ─────────────────────────────────────────────────────

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    gpu_ids = list(range(min(n_GPU, 4))) if n_GPU > 0 else []
    foa = is_foa_model(cfg) or is_foa_variant_model(cfg)
    echodiff = is_echodiffusion_model(cfg)
    js = is_foa_v2_js_model(cfg)

    train_set, train_loader = make_dataloader(cfg, 'train', batch_size=cfg.mode.batch_size)
    val_set, val_loader = make_dataloader(cfg, 'val', batch_size=cfg.mode.batch_size)
    print(f'Train: {len(train_set)}, Val: {len(val_set)}')

    model = build_model(cfg, gpu_ids)
    criterion = build_criterion(cfg, device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {cfg.model.name} ({total_params:.1f}M params)')

    lr = cfg.mode.learning_rate
    opt_name = cfg.mode.optimizer
    optimizer = (torch.optim.AdamW if opt_name == 'AdamW' else
                 torch.optim.Adam if opt_name == 'Adam' else
                 torch.optim.SGD)(model.parameters(), lr=lr)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = (f"{cfg.model.generator}_{cfg.dataset.name}_BS{cfg.mode.batch_size}_"
                f"Lr{lr}_{opt_name}_{cfg.mode.experiment_name}")
    ckpt_dir = os.path.join(project_dir, 'checkpoints', exp_name)
    results_dir = os.path.join(project_dir, 'results', exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # W&B
    wb_cfg = {
        'experiment_name': exp_name, 'model': cfg.model.name,
        'dataset': cfg.dataset.name, 'optimizer': opt_name,
        'lr': lr, 'batch_size': cfg.mode.batch_size,
        'epochs': cfg.mode.epochs, 'params_M': total_params,
    }
    if foa:
        wb_cfg.update({k: getattr(cfg.model, k, None)
                       for k in ('depth_weight', 'foa_weight', 'hist_weight',
                                 'sh_order', 'proj_dim', 'foa_freeze_epochs')})
    # wandb.init(project='neurips_audio_depth', name=exp_name,
            #    config=wb_cfg, tags=[cfg.model.name, cfg.dataset.name])

    # Resume
    start_epoch = 1
    if cfg.mode.checkpoints is not None:
        ckpt = torch.load(os.path.join(ckpt_dir, f'checkpoint_{cfg.mode.checkpoints}.pth'),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}')

    foa_freeze = getattr(cfg.model, 'foa_freeze_epochs', 0) if foa else 0
    use_hist_align = foa and getattr(cfg.model, 'hist_weight', 0) > 0
    best_rmse, best_abs_rel = float('inf'), float('inf')
    best_score = float('inf')  # weighted: 0.7*rmse + 0.3*abs_rel

    for epoch in range(start_epoch, cfg.mode.epochs + 1):
        foa_frozen = foa and foa_freeze > 0 and epoch <= foa_freeze
        if foa and foa_freeze > 0:
            set_sh_branch_frozen(model, foa_frozen)
            if epoch == 1:
                print(f'  [Warmup] SH branch frozen for {foa_freeze} epochs')
            elif epoch == foa_freeze + 1:
                print(f'  [Warmup done] SH branch unfrozen')

        use_hist = use_hist_align and not foa_frozen
        t0 = time.time()
        accum = {'total': [], 'depth': [], 'foa': [], 'hist': [], 'kl': [], 'consistency': []}

        model.train()
        for batch in train_loader:
            if js:
                s = _train_step_js(model, batch, criterion, optimizer,
                                   cfg, device, use_hist, foa_frozen)
            elif foa:
                s = _train_step_foa(model, batch, criterion, optimizer,
                                    cfg, device, use_hist, foa_frozen)
            elif echodiff:
                s = _train_step_echodiffusion(model, batch, criterion, optimizer,
                                              cfg, device)
            else:
                s = _train_step_baseline(model, batch, criterion, optimizer,
                                         cfg, device)
            for k, v in s.items():
                if k in accum:
                    accum[k].append(v)

        dt = time.time() - t0
        log = {'epoch': epoch, 'train/loss': np.mean(accum['total'])}
        for k in ('depth', 'foa', 'hist', 'kl', 'consistency'):
            if accum[k]:
                log[f'train/{k}'] = np.mean(accum[k])

        parts = [f"Epoch [{epoch}/{cfg.mode.epochs}] L:{log['train/loss']:.4f}"]
        if accum['depth']: parts.append(f"D:{np.mean(accum['depth']):.4f}")
        if accum['foa']:   parts.append(f"F:{np.mean(accum['foa']):.4f}")
        if accum['hist']:  parts.append(f"H:{np.mean(accum['hist']):.4f}")
        if accum['kl']:    parts.append(f"KL:{np.mean(accum['kl']):.4f}")
        if accum['consistency']: parts.append(f"CON:{np.mean(accum['consistency']):.4f}")
        parts.append(f"{dt:.0f}s")
        print(' '.join(parts))

        # Validation
        if cfg.mode.validation and epoch % cfg.mode.validation_iter == 0:
            vm, vis_p, vis_g = _val_metrics(
                model, val_loader, criterion, cfg, device, foa, echodiff,
                use_hist, foa_frozen, js=js)
            print(f"  Val L:{vm['val_loss']:.4f} ABS:{vm['abs_rel']:.4f} "
                  f"RMSE:{vm['rmse']:.4f} d1:{vm['delta1']:.4f}")

            for k, v in vm.items():
                log[f'val/{k}'] = v

            if vis_p is not None:
                vis_path = os.path.join(results_dir, f'epoch_{epoch:04d}_val.png')
                save_batch_visualization(vis_p, vis_g, vis_path, epoch,
                                         num_samples=min(4, vis_p.shape[0]))

            score = 0.7 * vm['rmse'] + 0.3 * vm['abs_rel']
            if score < best_score:
                best_score = score
                best_rmse, best_abs_rel = vm['rmse'], vm['abs_rel']
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_rmse': best_rmse, 'best_abs_rel': best_abs_rel,
                            'best_score': best_score},
                           os.path.join(ckpt_dir, 'best_model.pth'))
                print(f"  >> Best (score:{best_score:.4f} RMSE:{best_rmse:.4f} ABS:{best_abs_rel:.4f})")
                log.update({'best/score': best_score, 'best/rmse': best_rmse,
                            'best/abs_rel': best_abs_rel, 'best/epoch': epoch})

        # wandb.log(log)

        if epoch % cfg.mode.saving_checkpoints == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(ckpt_dir, f'checkpoint_{epoch}.pth'))

    print(f'\nDone. Best score:{best_score:.4f} RMSE:{best_rmse:.4f} ABS:{best_abs_rel:.4f}')
    # wandb.finish()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='foa', help='Config name (baseline, foa)')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--optimizer', type=str, default=None, choices=['AdamW', 'Adam', 'SGD'])
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--experiment-name', type=str, default='default')
    p.add_argument('--checkpoints', type=str, default=None)
    p.add_argument('--foa-freeze-epochs', type=int, default=None)
    p.add_argument('--depth-weight', type=float, default=None)
    p.add_argument('--foa-weight', type=float, default=None)
    p.add_argument('--hist-weight', type=float, default=None)
    p.add_argument('--kl-weight', type=float, default=None)
    p.add_argument('--foa-consistency-weight', type=float, default=None)
    p.add_argument('--rotate-canonical', action='store_true',
                   help='Rotate FOA into a canonical listener frame (dataset_rotated.py).')
    args = p.parse_args()

    cfg = load_config(config_name=args.config, mode='train',
                      experiment_name=args.experiment_name)

    if args.checkpoints is not None: cfg.mode.checkpoints = args.checkpoints
    if args.batch_size is not None:  cfg.mode.batch_size = args.batch_size
    if args.lr is not None:          cfg.mode.learning_rate = args.lr
    if args.optimizer is not None:   cfg.mode.optimizer = args.optimizer
    if args.epochs is not None:      cfg.mode.epochs = args.epochs
    if args.num_workers is not None: cfg.mode.num_threads = args.num_workers
    if args.foa_freeze_epochs is not None: cfg.model.foa_freeze_epochs = args.foa_freeze_epochs
    if args.depth_weight is not None: cfg.model.depth_weight = args.depth_weight
    if args.foa_weight is not None:   cfg.model.foa_weight = args.foa_weight
    if args.hist_weight is not None:  cfg.model.hist_weight = args.hist_weight
    if args.kl_weight is not None:    cfg.model.kl_weight = args.kl_weight
    if args.foa_consistency_weight is not None: cfg.model.foa_consistency_weight = args.foa_consistency_weight
    if args.rotate_canonical: cfg.dataset.rotate_canonical = True

    print('=' * 60)
    print(f'Model: {cfg.model.name}  Dataset: {cfg.dataset.name}')
    print(f'BS: {cfg.mode.batch_size}  LR: {cfg.mode.learning_rate}  '
          f'Opt: {cfg.mode.optimizer}')
    print('=' * 60)
    train(cfg)

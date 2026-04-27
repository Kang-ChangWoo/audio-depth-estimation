#!/usr/bin/env python3
"""Testing/evaluation script for audio-to-depth estimation."""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from utils.config import load_config
from utils.train_utils import build_model, is_foa_model, is_foa_variant_model, is_echodiffusion_model, is_foa_0415_model, is_foa_v2_js_rgb_model, is_foa_oracle_model, is_n2_model
from utils.test_utils import evaluate
from utils.visualization import save_test_visualizations
from data.dataloader import make_dataloader


def test(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    gpu_ids = list(range(min(n_GPU, 4))) if n_GPU > 0 else []
    foa0415 = is_foa_0415_model(cfg)
    js_rgb = is_foa_v2_js_rgb_model(cfg)
    oracle = is_foa_oracle_model(cfg)
    n2 = is_n2_model(cfg)
    foa = (is_foa_model(cfg) or is_foa_variant_model(cfg)) or foa0415 or js_rgb or oracle or n2
    echodiff = is_echodiffusion_model(cfg)

    eval_on = getattr(cfg.mode, 'eval_on', 'test')
    eval_set, eval_loader = make_dataloader(cfg, eval_on,
                                            batch_size=cfg.mode.batch_size, shuffle=False)
    print(f'Eval [{eval_on}]: {len(eval_set)} samples')

    model = build_model(cfg, gpu_ids)

    # Load checkpoint
    project_dir = os.path.dirname(os.path.abspath(__file__))

    ckpt_path = getattr(cfg.mode, 'checkpoint_path', None)
    if not ckpt_path:
        # Construct from config (use defaults if lr/optimizer not in test config)
        lr = getattr(cfg.mode, 'learning_rate', 0.001)
        opt = getattr(cfg.mode, 'optimizer', 'AdamW')
        exp_name = (f"{cfg.model.generator}_{cfg.dataset.name}_BS{cfg.mode.batch_size}_"
                    f"Lr{lr}_{opt}_{cfg.mode.experiment_name}")
        ckpt_dir = os.path.join(project_dir, 'checkpoints', exp_name)

        load_epoch = cfg.mode.checkpoints
        if load_epoch is None or str(load_epoch) == 'best':
            ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
        else:
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{load_epoch}.pth')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f'Loading: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        # strict=False with original keys (handles frozen components like Wav2Vec2)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            # If too many missing, try stripping 'module.' prefix
            sd_clean = {(k[len('module.'):] if k.startswith('module.') else k): v
                        for k, v in sd.items()}
            m2, u2 = model.load_state_dict(sd_clean, strict=False)
            if len(m2) < len(missing):
                missing, unexpected = m2, u2
            else:
                # Re-load with original keys (fewer missing)
                model.load_state_dict(sd, strict=False)
        if missing:
            print(f'  Loaded (strict=False): {len(missing)} missing, {len(unexpected)} unexpected keys')
    print(f'Loaded epoch {ckpt["epoch"]}')

    # Evaluate
    de, foa_err = evaluate(model, eval_loader, eval_set, cfg, device)
    md = de.mean(0)

    print('\n' + '=' * 60)
    labels = ['ABS_REL', 'RMSE', 'Delta1', 'Delta2', 'Delta3', 'Log10', 'MAE']
    for i, lbl in enumerate(labels):
        print(f'{lbl:>8s}: {md[i]:.4f}')

    if foa and foa_err:
        print('-' * 30)
        print(f'  FOA_L1: {np.mean([e["foa_l1"] for e in foa_err]):.4f}')
        print(f' FOA_COS: {np.mean([e["foa_cosine"] for e in foa_err]):.4f}')
        print(f' FOA_DIR: {np.mean([e["foa_dir_cosine"] for e in foa_err]):.4f}')
    print('=' * 60)

    # Determine results directory from checkpoint path
    ckpt_dir_name = os.path.basename(os.path.dirname(ckpt_path))
    results_dir = os.path.join(project_dir, 'results', ckpt_dir_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save stats
    stats_dir = os.path.join(project_dir, 'eval', cfg.dataset.name, eval_on)
    os.makedirs(stats_dir, exist_ok=True)
    stats = {lbl.lower(): torch.tensor(de[:, i]) for i, lbl in enumerate(labels)}
    if foa and foa_err:
        for k in ('foa_l1', 'foa_cosine', 'foa_dir_cosine'):
            stats[k] = torch.tensor([e[k] for e in foa_err])
    stats_path = os.path.join(stats_dir, f'stats_{cfg.mode.experiment_name}.pt')
    torch.save(stats, stats_path)
    print(f'Saved stats: {stats_path}')

    # Generate per-scene visualizations (merged + separate gt/pred)
    vis_per_scene = getattr(cfg.mode, 'vis_per_scene', 100)
    print(f'Generating visualizations ({vis_per_scene} per scene)...')

    # Group sample indices by scene
    scene_indices = {}
    for i, (scene, _) in enumerate(eval_set.samples):
        scene_indices.setdefault(scene, []).append(i)

    model.eval()
    vis_count = 0
    s = cfg.dataset.max_depth if cfg.dataset.depth_norm else 1.0

    for scene, indices in scene_indices.items():
        scene_dir = os.path.join(results_dir, 'test', scene)
        sel = indices[:vis_per_scene]

        with torch.no_grad():
            for j, sample_idx in enumerate(sel):
                batch = eval_set[sample_idx]
                if js_rgb:
                    audio, depthgt, _foa, _energy, _rgb = batch
                    audio = audio.unsqueeze(0).to(device)
                    depthgt = depthgt.unsqueeze(0).to(device)
                    raw_out = model(audio, mode='test')
                    depth_pred = raw_out["pred_depth"]
                    if depth_pred.shape != depthgt.shape:
                        depth_pred = F.interpolate(
                            depth_pred, size=depthgt.shape[2:],
                            mode='nearest')
                elif oracle:
                    audio, depthgt, gt_foa_s, energy_map = batch
                    audio = audio.unsqueeze(0).to(device)
                    depthgt = depthgt.unsqueeze(0).to(device)
                    energy_map = energy_map.unsqueeze(0).to(device)
                    input_nc = int(getattr(cfg.model, 'input_nc', 3))
                    if input_nc == 1:
                        model_input = energy_map
                    else:
                        model_input = torch.cat([audio, energy_map], dim=1)
                    raw_out = model(model_input)
                    depth_pred = raw_out["pred_depth"]
                elif n2:
                    audio, depthgt, gt_foa_s, _energy, foa_spec, temporal_rms, temporal_energies = batch
                    audio = audio.unsqueeze(0).to(device)
                    depthgt = depthgt.unsqueeze(0).to(device)
                    model_name = cfg.model.name
                    if model_name in ('n2_temap_input',
                                      'pvit_n1_temap_input',
                                      'pvit_n1_temap_eattn',
                                      'pvit_n1_temap_mssh'):
                        temporal_energies = temporal_energies.unsqueeze(0).to(device)
                        raw_out = model(torch.cat([audio, temporal_energies], dim=1))
                    elif model_name == 'n2_6ch_input':
                        foa_spec = foa_spec.unsqueeze(0).to(device)
                        raw_out = model(torch.cat([audio, foa_spec], dim=1))
                    elif model_name == 'n2_dual_enc':
                        foa_spec = foa_spec.unsqueeze(0).to(device)
                        raw_out = model(audio, foa_spec)
                    elif model_name == 'n2_foa_stft_film':
                        foa_spec = foa_spec.unsqueeze(0).to(device)
                        raw_out = model(audio, foa_spec)
                    elif model_name in ('n2_temporal_rms_film',
                                        'pvit_n1_temap_rms_film'):
                        temporal_rms = temporal_rms.unsqueeze(0).to(device)
                        raw_out = model(audio, temporal_rms)
                    elif model_name == 'n2_tbin_crossattn':
                        temporal_energies = temporal_energies.unsqueeze(0).to(device)
                        raw_out = model(audio, temporal_energies=temporal_energies)
                    else:
                        raw_out = model(audio)
                    depth_pred = raw_out["pred_depth"]
                elif foa:
                    # Accept 4-tuple (legacy foa/foa0415) or 5-tuple
                    # (n9_0424 adds rep_gt at index 4). Viz only needs depth.
                    audio, depthgt = batch[0], batch[1]
                    audio = audio.unsqueeze(0).to(device)
                    depthgt = depthgt.unsqueeze(0).to(device)
                    raw_out = model(audio)
                    depth_pred = raw_out["pred_depth"]
                elif echodiff:
                    audio, depthgt, waveform = batch
                    audio = audio.unsqueeze(0).to(device)
                    depthgt = depthgt.unsqueeze(0).to(device)
                    waveform = waveform.unsqueeze(0).to(device)
                    depth_pred = model(audio, waveform)
                else:
                    audio, depthgt = batch[0], batch[1]
                    audio = audio.unsqueeze(0).to(device)
                    depthgt = depthgt.unsqueeze(0).to(device)
                    depth_pred = model(audio)

                save_test_visualizations(
                    depth_pred * s, depthgt * s,
                    scene_dir, f'vis_{j:03d}', epoch=ckpt["epoch"])
                vis_count += 1

        print(f'  {scene}: {len(sel)} images -> merged/ gt_depth/ pred_depth/')

    print(f'Saved {vis_count} samples ({vis_count*3} files) to {results_dir}/test/')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='foa', help='Config name (baseline, foa)')
    p.add_argument('--eval-on', type=str, default='test', choices=['test', 'val'])
    p.add_argument('--experiment-name', type=str, default='default')
    p.add_argument('--checkpoints', type=str, default=None)
    p.add_argument('--checkpoint-path', type=str, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--num-workers', type=int, default=None,
                   help='Override cfg.test.num_threads (dataloader workers).')
    p.add_argument('--vis-per-scene', type=int, default=100,
                   help='Number of visualizations per scene (default: 100)')
    p.add_argument('--rotate-canonical', action='store_true',
                   help='Rotate FOA into a canonical listener frame (dataset_rotated.py).')
    p.add_argument('--depth-dir', type=str, default=None,
                   help=('Override cfg.dataset.depth_dir: load GT depth from '
                         'root_dir/scene/<depth-dir>/ instead of '
                         "{depth_type}_depth/. E.g. 'erp_depth_radial' to "
                         "evaluate against radial-distance GT."))
    p.add_argument('--dataset-dir', type=str, default=None,
                   help=('Override cfg.dataset.dataset_dir: dataset root '
                         'path, for running on a server where the data '
                         'lives elsewhere than the yaml default.'))
    p.add_argument('--sh-dim', type=int, default=None,
                   help='Override cfg.model.sh_dim (must match the checkpoint).')
    p.add_argument('--depth-weight', type=float, default=None)
    p.add_argument('--lambda-sh', type=float, default=None)
    p.add_argument('--lambda-energy', type=float, default=None)
    p.add_argument('--lambda-kd', type=float, default=None)
    p.add_argument('--foa-freeze-epochs', type=int, default=None)
    p.add_argument('--teacher-ckpt', type=str, default=None)
    p.add_argument('--gate-mask', type=str, default=None,
                   help=('n4_0425: comma-separated K floats fixing the gate '
                         'at test time (matched to the trained checkpoint).'))
    p.add_argument('--rep-kind', type=str, default=None,
                   choices=['eigen', 'rms'],
                   help='n3_0425: FOA target type at test time.')
    p.add_argument('--rep-K', type=int, default=None,
                   help='n3_0425: number of temporal bins (must match checkpoint).')
    p.add_argument('--ngf', type=int, default=None,
                   help='n3_0425/n4_0425: base UNet width (must match checkpoint).')
    p.add_argument('--backbone', type=str, default=None,
                   choices=['vit', 'resnet'],
                   help='n9_0426: outer hourglass backbone (must match checkpoint).')
    p.add_argument('--foa-mode', type=str, default=None,
                   choices=['input', 'condition'],
                   help='echodiffusion_ambi: FOA injection mode (must match checkpoint).')
    p.add_argument('--side-fusion', dest='side_fusion',
                   action='store_true', default=None)
    p.add_argument('--no-side-fusion', dest='side_fusion', action='store_false')
    p.add_argument('--oracle-mode', dest='oracle_mode',
                   action='store_true', default=None)
    p.add_argument('--no-oracle-mode', dest='oracle_mode', action='store_false')
    p.add_argument('--oracle-gate-mode', type=str, default=None,
                   choices=['ones', 'pred'])
    args, _unknown = p.parse_known_args()

    cfg = load_config(config_name=args.config, mode='test',
                      experiment_name=args.experiment_name)
    cfg.mode.eval_on = args.eval_on
    if args.checkpoints is not None: cfg.mode.checkpoints = args.checkpoints
    if args.checkpoint_path: cfg.mode.checkpoint_path = args.checkpoint_path
    if args.batch_size is not None: cfg.mode.batch_size = args.batch_size
    if args.num_workers is not None: cfg.mode.num_threads = args.num_workers
    cfg.mode.vis_per_scene = args.vis_per_scene
    if args.rotate_canonical: cfg.dataset.rotate_canonical = True
    if args.depth_dir is not None:
        cfg.dataset.depth_dir = args.depth_dir
    if args.dataset_dir is not None:
        cfg.dataset.dataset_dir = args.dataset_dir
    if args.sh_dim is not None: cfg.model.sh_dim = args.sh_dim
    if args.depth_weight is not None: cfg.model.depth_weight = args.depth_weight
    if args.lambda_sh is not None: cfg.model.lambda_sh = args.lambda_sh
    if args.lambda_energy is not None: cfg.model.lambda_energy = args.lambda_energy
    if args.gate_mask is not None:
        cfg.model.gate_mask = [float(v) for v in args.gate_mask.split(',')]
    if args.rep_kind is not None:   cfg.dataset.rep_kind = args.rep_kind
    if args.rep_K is not None:
        cfg.dataset.rep_K = args.rep_K
        cfg.model.K = args.rep_K
    if args.ngf is not None:        cfg.model.ngf = args.ngf
    if args.backbone is not None:   cfg.model.backbone = args.backbone
    if args.foa_mode is not None:   cfg.model.foa_mode = args.foa_mode
    if args.side_fusion is not None:      cfg.model.side_fusion = bool(args.side_fusion)
    if args.oracle_mode is not None:      cfg.model.oracle_mode = bool(args.oracle_mode)
    if args.oracle_gate_mode is not None: cfg.model.oracle_gate_mode = args.oracle_gate_mode

    print('=' * 60)
    print(f'Model: {cfg.model.name}  Eval: {args.eval_on}')
    print('=' * 60)
    test(cfg)

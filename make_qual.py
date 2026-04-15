#!/usr/bin/env python3
"""Generate qualitative comparison figures for the paper.

For each test scene (100 samples per scene), creates:
  qual/{scene}/qual_NNN.png — single-row: GT RGB | GT Depth | ResNet | ViT | Echo-Net | BatVision | EchoDiffusion | Ours

Loads each model once, caches all predictions, then composes figures.
"""

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = '/root/storage/implementation/shared_audio/baseline'
DATASET_DIR = '/root/storage/matterport3d_0303renew'
QUAL_DIR = os.path.join(PROJECT_DIR, 'qual')

sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from utils.config import load_config
from utils.train_utils import build_model, is_foa_model, is_foa_variant_model, is_echodiffusion_model
from data.dataset import SoundSpacesDataset

# ── Table methods: (config, exp_name, ckpt_dir, display_name) ──
METHODS = [
    ('pretrain_resnet', 'exp60_resnet_lr3e4_bs32',
     'pretrained_resnet_soundspaces_BS32_Lr0.0003_AdamW_exp60_resnet_lr3e4_bs32',
     'ResNet-50'),
    ('pretrain_vit', 'exp64_vit_lr1e4_bs8',
     'pretrained_vit_soundspaces_BS8_Lr0.0001_AdamW_exp64_vit_lr1e4_bs8',
     'ViT-B/16'),
    ('echonet', 'exp69_echonet_lr1e3_bs16',
     'echonet_soundspaces_BS16_Lr0.001_AdamW_exp69_echonet_lr1e3_bs16',
     'Echo-Net'),
    ('baseline', 'exp05_baseline_lr5e4_bs16',
     'unet_256_soundspaces_BS16_Lr0.0005_AdamW_exp05_baseline_lr5e4_bs16',
     'BatVision'),
    ('echodiffusion', 'exp121_echodiff_wav2vec_lr1e4_bs16',
     'echodiffusion_soundspaces_BS16_Lr0.0001_AdamW_exp121_echodiff_wav2vec_lr1e4_bs16',
     'EchoDiffusion'),
    ('foa', 'exp49_foa_lr1e3_dw1.0_fw0.1_hw0.05',
     'unet_256_soundspaces_BS32_Lr0.001_AdamW_exp49_foa_lr1e3_dw1.0_fw0.1_hw0.05',
     'Ours'),
]


def load_model_and_predict(config_name, ckpt_dir, dataset, sample_indices, device):
    """Load a model, run inference on selected samples, return raw depth maps."""
    cfg = load_config(config_name, 'test', 'default')
    foa = is_foa_model(cfg) or is_foa_variant_model(cfg)
    echodiff = is_echodiffusion_model(cfg)

    model = build_model(cfg, [0])

    ckpt_path = os.path.join(PROJECT_DIR, 'checkpoints', ckpt_dir, 'best_model.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['state_dict']
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)
    model.eval()

    max_depth = cfg.dataset.max_depth
    depth_norm = cfg.dataset.depth_norm
    predictions = {}

    with torch.no_grad():
        for idx in sample_indices:
            batch = dataset[idx]
            if foa:
                audio = batch[0].unsqueeze(0).to(device)
                out = model(audio)
                pred = out['pred_depth']
            elif echodiff:
                audio = batch[0].unsqueeze(0).to(device)
                waveform = batch[2].unsqueeze(0).to(device)
                pred = model(audio, waveform)
            else:
                audio = batch[0].unsqueeze(0).to(device)
                pred = model(audio)

            pred_np = pred[0, 0].cpu().numpy()
            if depth_norm:
                pred_np = pred_np * max_depth
            pred_np = np.clip(pred_np, 0, max_depth)
            predictions[idx] = pred_np

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    return predictions


def load_gt_rgb(scene, sample_idx, h=256, w=512):
    rgb_path = os.path.join(DATASET_DIR, scene, 'erp_rgb', f'erp_{sample_idx}.png')
    if os.path.exists(rgb_path):
        img = Image.open(rgb_path).convert('RGB').resize((w, h), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0
    return None


def load_gt_depth(scene, sample_idx, h=256, w=512, max_depth=10.0):
    depth_path = os.path.join(DATASET_DIR, scene, 'erp_depth', f'erp_depth_{sample_idx}.npy')
    if os.path.exists(depth_path):
        d = np.load(depth_path).astype(np.float32)
        d = np.nan_to_num(d); d[d < 0] = 0; d[d > max_depth] = max_depth
        d_img = Image.fromarray(d).resize((w, h), Image.NEAREST)
        return np.array(d_img, dtype=np.float32)
    return None


def render_depth(depth, vmin=0, vmax=10.0, cmap='Spectral'):
    """Render depth map to RGB numpy array using colormap."""
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    colored = cm(norm(depth))[:, :, :3]  # drop alpha
    return (colored * 255).astype(np.uint8)


def make_qual_row(gt_rgb, gt_depth, method_depths, method_names, save_path, scene, sample_idx):
    """Create GT RGB | GT Depth | Method1 | ... | Ours as a single image."""
    h, w = gt_depth.shape
    valid = gt_depth[gt_depth > 0]
    vmax = valid.max() if len(valid) > 0 else 10.0

    # Render all depth maps with same colormap range
    gt_depth_rgb = render_depth(gt_depth, vmax=vmax)
    gt_rgb_uint8 = (gt_rgb * 255).astype(np.uint8)

    method_depth_rgbs = []
    for d in method_depths:
        method_depth_rgbs.append(render_depth(d, vmax=vmax))

    # Column labels
    labels = ['GT RGB', 'GT Depth'] + method_names
    n_cols = len(labels)

    # Create figure
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.0))

    axes[0].imshow(gt_rgb_uint8)
    axes[0].set_title('GT RGB', fontsize=9, pad=2)
    axes[0].axis('off')

    axes[1].imshow(gt_depth_rgb)
    axes[1].set_title('GT Depth', fontsize=9, pad=2)
    axes[1].axis('off')

    for i, (drgb, name) in enumerate(zip(method_depth_rgbs, method_names)):
        ax = axes[2 + i]
        ax.imshow(drgb)
        fw = 'bold' if name == 'Ours' else 'normal'
        ax.set_title(name, fontsize=9, fontweight=fw, pad=2)
        ax.axis('off')

    plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.92, bottom=0.01)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-vis', type=int, default=100, help='Images per scene')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    os.makedirs(QUAL_DIR, exist_ok=True)

    # Build dataset (use baseline config for non-FOA loading, FOA separately)
    print('Building test dataset...')
    cfg_base = load_config('baseline', 'test', 'default')
    ds_base = SoundSpacesDataset(cfg_base, 'test')

    cfg_foa = load_config('foa', 'test', 'default')
    ds_foa = SoundSpacesDataset(cfg_foa, 'test')

    cfg_echo = load_config('echodiffusion', 'test', 'default')
    ds_echo = SoundSpacesDataset(cfg_echo, 'test')

    # Group samples by scene
    scene_map = {}  # scene -> [(global_idx, sample_str_idx), ...]
    for i, (scene, sidx) in enumerate(ds_base.samples):
        scene_map.setdefault(scene, []).append((i, sidx))

    # Select sample indices per scene
    scenes = sorted(scene_map.keys())
    selected = {}  # scene -> [(global_idx, sample_str_idx), ...]
    for scene in scenes:
        selected[scene] = scene_map[scene][:args.num_vis]

    all_global_indices = []
    for scene in scenes:
        all_global_indices.extend([x[0] for x in selected[scene]])
    print(f'Total samples to process: {len(all_global_indices)} across {len(scenes)} scenes')

    # Run each model and cache predictions
    all_predictions = {}  # method_name -> {global_idx: depth_array}

    for config_name, exp_name, ckpt_dir, display_name in METHODS:
        print(f'\n[{display_name}] Loading {config_name}/{exp_name}...')

        # Pick correct dataset
        cfg = load_config(config_name, 'test', 'default')
        foa = is_foa_model(cfg) or is_foa_variant_model(cfg)
        echodiff = is_echodiffusion_model(cfg)

        if foa:
            ds = ds_foa
        elif echodiff:
            ds = ds_echo
        else:
            ds = ds_base

        preds = load_model_and_predict(
            config_name, ckpt_dir, ds, all_global_indices, device)
        all_predictions[display_name] = preds
        print(f'  [{display_name}] {len(preds)} predictions cached')

    # Compose qualitative figures
    print('\nComposing qualitative comparison figures...')
    method_names = [m[3] for m in METHODS]
    total_saved = 0

    for scene in scenes:
        scene_dir = os.path.join(QUAL_DIR, scene)
        os.makedirs(scene_dir, exist_ok=True)

        for j, (global_idx, sample_str_idx) in enumerate(selected[scene]):
            gt_rgb = load_gt_rgb(scene, sample_str_idx)
            gt_depth = load_gt_depth(scene, sample_str_idx)
            if gt_rgb is None or gt_depth is None:
                continue

            method_depths = []
            skip = False
            for name in method_names:
                d = all_predictions[name].get(global_idx)
                if d is None:
                    print(f'  SKIP {scene}/vis_{j:03d}: missing {name}')
                    skip = True
                    break
                method_depths.append(d)
            if skip:
                continue

            save_path = os.path.join(scene_dir, f'qual_{j:03d}.png')
            make_qual_row(gt_rgb, gt_depth, method_depths, method_names,
                          save_path, scene, sample_str_idx)
            total_saved += 1

        print(f'  {scene}: {len(selected[scene])} images')

    print(f'\nDone! {total_saved} qualitative figures saved to {QUAL_DIR}')


if __name__ == '__main__':
    main()

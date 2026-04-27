#!/bin/bash
# ============================================================
# vis_all.sh — dump pred_depth and gt_depth as .npy files for the
# full test set of exp202 (n2_temap_input) and exp219 (pvit oracle).
#
# For each experiment, loads best_model.pth from
#   checkpoints/<full_run_dir>/best_model.pth
# and writes the npy pairs alongside the matching results/<full_run_dir>
# that test.py already populates:
#
#   results/<full_run_dir>/vis_npy/<scene>/<sample_idx>_pred.npy
#   results/<full_run_dir>/vis_npy/<scene>/<sample_idx>_gt.npy
#
# <full_run_dir> is the auto-generated name like
#   unet_256_soundspaces_BS128_Lr0.0005_AdamW_exp202_n2_temap_lr5e4_lsh0.1
# and is derived from the checkpoint's parent directory.
#
# Depth maps are saved in METERS (un-normalized, clipped to
# [1e-3, max_depth] for pred, >= 0 for gt) — matching the convention
# used by compute_errors in utils/metrics.py.
#
# Usage:
#   bash scripts/vis_all.sh                 # default GPU 0
#   CUDA_VISIBLE_DEVICES=3 bash scripts/vis_all.sh
# ============================================================
set -euo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
_REQUIRED="torch torchvision torchaudio timm scipy numpy yaml einops"
_MISSING=$("$PYTHON" - <<EOF 2>&1
import importlib
miss=[]
for m in "$_REQUIRED".split():
    try: importlib.import_module(m)
    except Exception as e: miss.append(f"{m}:{e.__class__.__name__}")
print(" ".join(miss) if miss else "OK")
EOF
)
if [ "$_MISSING" != "OK" ]; then
    echo "ERROR: $PYTHON missing: $_MISSING" >&2
    echo "       Known-good: /opt/conda/bin/python3" >&2
    exit 1
fi
echo "Using PYTHON=$PYTHON"

# "<EXP>  <CONFIG>" — config drives load_config(); EXP is the experiment
# name stamped into the checkpoint dir name.
EXPS=(
    "exp121_echodiff_wav2vec_lr1e4_bs16   echodiffusion"
    # "exp202_n2_temap_lr5e4_lsh0.1         n2_temap_input"
    # "exp219_pvit_oracle_nc3_lr1e4_lsh0.1  pvitfoa_v3_oracle_nc3"
)

DEFAULT_GPU="${CUDA_VISIBLE_DEVICES:-0}"

for line in "${EXPS[@]}"; do
    read -r EXP CONFIG <<< "$line"
    CKPT=$(find "$PROJECT_DIR/checkpoints" -name "best_model.pth" \
                -path "*${EXP}*" 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "SKIP $EXP — no best_model.pth found"
        continue
    fi
    # full_run_dir = parent directory of the checkpoint, e.g.
    #   unet_256_soundspaces_BS128_Lr0.0005_AdamW_exp202_n2_temap_lr5e4_lsh0.1
    RUN_DIR_NAME=$(basename "$(dirname "$CKPT")")
    OUT_DIR="$PROJECT_DIR/results/$RUN_DIR_NAME/vis_npy"
    mkdir -p "$OUT_DIR"
    echo "============================================================"
    echo "EXP=$EXP  CONFIG=$CONFIG"
    echo "  ckpt: $CKPT"
    echo "  out : $OUT_DIR"
    echo "============================================================"

    EXP="$EXP" CONFIG="$CONFIG" CKPT="$CKPT" OUT_DIR="$OUT_DIR" \
    CUDA_VISIBLE_DEVICES="$DEFAULT_GPU" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$PYTHON" - <<'PYEOF'
import os, sys
sys.path.insert(0, '/root/storage/implementation/shared_audio/baseline')
os.chdir('/root/storage/implementation/shared_audio/baseline')

EXP = os.environ['EXP']
CONFIG = os.environ['CONFIG']
CKPT = os.environ['CKPT']
OUT_DIR = os.environ['OUT_DIR']

import numpy as np
import torch
from utils.config import load_config
from utils.train_utils import (
    build_model, is_foa_oracle_model, is_n2_model, is_echodiffusion_model,
)
from data.dataloader import make_dataloader

cfg = load_config(config_name=CONFIG, mode='test', experiment_name=EXP)
cfg.mode.batch_size = 1
cfg.dataset.rotate_canonical = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
gpu_ids = list(range(min(n_gpu, 1)))

eval_set, eval_loader = make_dataloader(cfg, 'test', batch_size=1, shuffle=False)
samples = eval_set.samples
print(f'  {len(eval_set)} test samples')

model = build_model(cfg, gpu_ids)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
sd = ckpt['state_dict']
try:
    model.load_state_dict(sd)
except RuntimeError:
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        sd_clean = {(k[len('module.'):] if k.startswith('module.') else k): v
                    for k, v in sd.items()}
        model.load_state_dict(sd_clean, strict=False)
model.eval()
print(f'  loaded epoch {ckpt["epoch"]}')

oracle = is_foa_oracle_model(cfg)
n2 = is_n2_model(cfg)
echodiff = is_echodiffusion_model(cfg)
max_depth = float(cfg.dataset.max_depth)
depth_norm = bool(cfg.dataset.depth_norm)

saved = 0
with torch.no_grad():
    for i, batch in enumerate(eval_loader):
        scene, sample_idx = samples[i]

        if oracle:
            audio, depthgt, _gt_foa, energy_map = batch
            audio = audio.to(device); depthgt = depthgt.to(device)
            energy_map = energy_map.to(device)
            input_nc = int(getattr(cfg.model, 'input_nc', 3))
            x = energy_map if input_nc == 1 else torch.cat([audio, energy_map], dim=1)
            out = model(x)
        elif n2:
            audio, depthgt, _gt_foa, _em, foa_spec, trms, tenergies = batch
            audio = audio.to(device); depthgt = depthgt.to(device)
            mn = cfg.model.name
            if mn in ('n2_temap_input', 'pvit_n1_temap_input',
                      'pvit_n1_temap_eattn', 'pvit_n1_temap_mssh'):
                tenergies = tenergies.to(device)
                out = model(torch.cat([audio, tenergies], dim=1))
            elif mn == 'n2_6ch_input':
                out = model(torch.cat([audio, foa_spec.to(device)], dim=1))
            elif mn in ('n2_dual_enc', 'n2_foa_stft_film'):
                out = model(audio, foa_spec.to(device))
            elif mn in ('n2_temporal_rms_film', 'pvit_n1_temap_rms_film'):
                out = model(audio, trms.to(device))
            elif mn == 'n2_tbin_crossattn':
                out = model(audio, temporal_energies=tenergies.to(device))
            else:
                out = model(audio)
        elif echodiff:
            # EchoDiffusion returns a depth tensor directly (not a dict).
            # Batch is a 3-tuple (audio, depth, waveform).
            audio, depthgt, waveform = batch
            audio = audio.to(device); depthgt = depthgt.to(device)
            waveform = waveform.to(device)
            out = {"pred_depth": model(audio, waveform)}
        else:
            audio = batch[0].to(device); depthgt = batch[1].to(device)
            out = model(audio)
            if not isinstance(out, dict):
                out = {"pred_depth": out}

        pred = out['pred_depth'][0, 0].detach().cpu().numpy()
        gt = depthgt[0, 0].detach().cpu().numpy()
        if depth_norm:
            pred = pred * max_depth
            gt = gt * max_depth
        pred = np.clip(pred, 1e-3, max_depth).astype(np.float32)
        gt = np.maximum(gt, 0.0).astype(np.float32)

        scene_dir = os.path.join(OUT_DIR, scene)
        os.makedirs(scene_dir, exist_ok=True)
        np.save(os.path.join(scene_dir, f'{sample_idx}_pred.npy'), pred)
        np.save(os.path.join(scene_dir, f'{sample_idx}_gt.npy'),   gt)
        saved += 1
        if saved % 200 == 0:
            print(f'  {saved}/{len(eval_set)}')

print(f'  saved {saved} (pred,gt) pairs to {OUT_DIR}')
PYEOF

done

echo ""
echo "============================================================"
echo "vis_all.sh done — $(date)"
echo "Output root: $OUT_ROOT"
echo "============================================================"

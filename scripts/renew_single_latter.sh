#!/bin/bash
# ============================================================
# renew_single_latter.sh — radial-depth experiment driver.
#
# Current active block (2026-04-24):
#   exp390–410 — n9_0424 ablation sweep (21 experiments, 5 variants).
#   Model: N9_0424Net  (models/n9_0424/)  Architecture: binaural spec-ViT
#   encoder + ImplicitSoundFieldProjectionFusion at F12 + DPT decoder.
#   Each variant validates ONE architectural claim:
#       A. baseline      (no rep supervision, no fusion)  → reference
#       B. aux-only      (rep supervised, no fusion)      → rep learnable?
#       C. proj fusion   (full: rep + fusion)             → main hypothesis
#       D. + energy map  (Variant C + proj consistency)   → basis helps?
#       E. fixed gate    (Variant C with gate frozen)     → prior enough?
#
# Previous blocks (now DONE — kept as comments for traceability):
#   exp301–304 — renew_single on ERP depth      (all DONE)
#   exp305–308 — renew_single/v2/dpt_only on
#                radial depth                   (all DONE 40/40, ckpt saved)
#
# Per-experiment spec format (5 whitespace-separated fields):
#     "CONFIG  EXP_NAME  LR  BS  LSH"
#   LSH is --lambda-sh on the CLI (overrides cfg.model.lambda_sh).
#
# Usage:
#   bash scripts/renew_single_latter.sh
#   GPU=4,5,6,7 bash scripts/renew_single_latter.sh
#   EXTRA="--lambda-energy 0.3" bash scripts/renew_single_latter.sh
# ============================================================
set -euo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

# --- Interpreter preflight ---------------------------------------------
PYTHON="${PYTHON:-/opt/conda/bin/python3}"
_REQUIRED="torch torchvision torchaudio timm scipy numpy yaml einops"
_MISSING=$("$PYTHON" - <<EOF 2>&1
import importlib
miss = []
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

# --- Hard GPU check -----------------------------------------------------
_GPU_OK=$(CUDA_VISIBLE_DEVICES="${GPU:-0,1,2,3}" "$PYTHON" - <<'PYEOF' 2>&1
import os
try:
    import torch
    if not torch.cuda.is_available():
        print(f"NO_CUDA (visible={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    else:
        print(f"OK {torch.cuda.device_count()}")
except Exception as e:
    print(f"ERROR {e.__class__.__name__}: {e}")
PYEOF
)
if [[ "$_GPU_OK" != OK\ * ]]; then
    echo "ERROR: CUDA not usable for this process." >&2
    echo "       detail: $_GPU_OK" >&2
    echo "       unset CUDA_VISIBLE_DEVICES and retry, or check nvidia-smi." >&2
    exit 2
fi
echo "CUDA preflight: $_GPU_OK"

# --- Config ------------------------------------------------------------
# ---- ERP depth (DONE) ----
# exp301: full renew (DONE 40/40)
# CONFIG="renew_single"
# EXP="exp301_renew_single_lr1e4_lsh0.1"

# exp302: full renew v2 (DONE 40/40)
# CONFIG="renew_single_v2"
# EXP="exp302_renew_v2_lr1e4_lsh0.3"

# exp303: DPT-only ablation, keep KL loss (DONE 40/40)
# CONFIG="renew_dpt_only"
# EXP="exp303_dpt_only_lr1e4_lsh0.3"

# exp304: DPT-only ablation + no KL loss (DONE 40/40)
# CONFIG="renew_dpt_only_nokl"
# EXP="exp304_dpt_only_nokl_lr1e4_lsh0.3"

# ---- Radial depth — exp305-308 (DONE 40/40, kept for traceability) ----
# "renew_single_radial         exp305_renew_single_radial_lr1e4_lsh0.1     0.1"
# "renew_single_v2_radial      exp306_renew_v2_radial_lr1e4_lsh0.3         0.3"
# "renew_dpt_only_radial       exp307_dpt_only_radial_lr1e4_lsh0.3         0.3"
# "renew_dpt_only_nokl_radial  exp308_dpt_only_nokl_radial_lr1e4_lsh0.3    0.3"

# ---- Radial depth — exp390-410: n9_0424 ablation sweep (21 exps) ----
# Format: "CONFIG EXP LR BS LSH".
#
# Each block below targets one ablation variant from the spec §13. Within
# a block, LR / BS / lambda_sh vary so we can check sensitivity to those
# knobs under that variant's architecture. The config file fixes the
# architecture (enable_fusion, gate_learnable, lambda_energy_map, etc.);
# LR, BS, and --lambda-sh are passed on the CLI.
ALL_RADIAL=(
    # --- Variant C: projection fusion (FULL / main architecture) — DONE 40/40 + tested ---
    # "n9_0424_proj_fusion_radial  exp396_n9C_lr1e4_lsh005       0.0001  64   0.05"
    # "n9_0424_proj_fusion_radial  exp397_n9C_lr1e4_lsh01        0.0001  64   0.1"
    # "n9_0424_proj_fusion_radial  exp398_n9C_lr5e5_lsh005       0.00005 64   0.05"
    # "n9_0424_proj_fusion_radial  exp399_n9C_lr3e4_lsh005       0.0003  64   0.05"
    # "n9_0424_proj_fusion_radial  exp400_n9C_lr1e4_lsh005_bs128 0.0001 128   0.05"
    # "n9_0424_proj_fusion_radial  exp401_n9C_lr1e4_lsh005_bs32  0.0001  32   0.05"

    # --- Variant A: baseline (no rep, no fusion) ---
    # "n9_0424_baseline_radial     exp390_n9A_lr1e4              0.0001  64   0.0"   # DONE (train-skip + tested on BS8 ckpt)
    # "n9_0424_baseline_radial     exp391_n9A_lr5e5              0.00005 64   0.0"   # DONE 40/40 + tested

    # --- Variant B: aux-only (rep supervised, no fusion) ---
    # "n9_0424_aux_only_radial     exp392_n9B_lr1e4_lsh005       0.0001  64   0.05"  # DONE 40/40 + tested
    "n9_0424_aux_only_radial     exp393_n9B_lr1e4_lsh01        0.0001  64   0.1"
    "n9_0424_aux_only_radial     exp394_n9B_lr5e5_lsh005       0.00005 64   0.05"
    "n9_0424_aux_only_radial     exp395_n9B_lr3e4_lsh005       0.0003  64   0.05"

    # --- Variant D: + energy map loss (lambda_energy_map=0.25 in config) ---
    "n9_0424_energy_map_radial   exp402_n9D_lr1e4_lsh005       0.0001  64   0.05"
    "n9_0424_energy_map_radial   exp403_n9D_lr5e5_lsh005       0.00005 64   0.05"
    "n9_0424_energy_map_radial   exp404_n9D_lr3e4_lsh005       0.0003  64   0.05"
    "n9_0424_energy_map_radial   exp405_n9D_lr1e4_lsh01        0.0001  64   0.1"
    "n9_0424_energy_map_radial   exp406_n9D_lr1e4_lsh005_bs128 0.0001 128   0.05"

    # --- Variant E: fixed bin gate ---
    "n9_0424_fixed_gate_radial   exp407_n9E_lr1e4_lsh005       0.0001  64   0.05"
    "n9_0424_fixed_gate_radial   exp408_n9E_lr5e5_lsh005       0.00005 64   0.05"
    "n9_0424_fixed_gate_radial   exp409_n9E_lr1e4_lsh01        0.0001  64   0.1"
    "n9_0424_fixed_gate_radial   exp410_n9E_lr3e4_lsh005       0.0003  64   0.05"
)

GPU="${GPU:-0,1,2,3}"
EPOCHS="${EPOCHS:-40}"
WORKERS="${WORKERS:-8}"
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303_renew}"
EXTRA=(${EXTRA:-})

TRAIN_LOG_DIR="$PROJECT_DIR/logs/renew_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/renew_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: DATASET_DIR not found: $DATASET_DIR" >&2
    exit 3
fi

echo "============================================================"
echo "renew_single_latter — ${#ALL_RADIAL[@]} radial experiments (n9_0424 ablation: exp390-410)"
echo "  GPU=$GPU  epochs=$EPOCHS  workers=$WORKERS"
echo "  dataset_dir=$DATASET_DIR"
echo "  $(date)"
echo "============================================================"

for SPEC in "${ALL_RADIAL[@]}"; do
    CONFIG=$(echo "$SPEC" | awk '{print $1}')
    EXP=$(echo    "$SPEC" | awk '{print $2}')
    LR=$(echo     "$SPEC" | awk '{print $3}')
    BS=$(echo     "$SPEC" | awk '{print $4}')
    LSH=$(echo    "$SPEC" | awk '{print $5}')

    echo ""
    echo "[$(date +%H:%M:%S)] TRAIN $EXP (config=$CONFIG lr=$LR bs=$BS lsh=$LSH)"

    TRAIN_EC=0
    CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    "$PYTHON" -u train.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --dataset-dir "$DATASET_DIR" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BS" \
        --num-workers "$WORKERS" \
        --lambda-sh "$LSH" \
        --rotate-canonical \
        "${EXTRA[@]}" \
        > "$TRAIN_LOG_DIR/${EXP}.log" 2>&1 || TRAIN_EC=$?

    if [ "$TRAIN_EC" -ne 0 ] && [ "$TRAIN_EC" -ne 141 ]; then
        echo "  TRAIN_FAILED (exit=$TRAIN_EC). Tail:"
        tail -5 "$TRAIN_LOG_DIR/${EXP}.log"
        continue
    fi
    DONE_LINE=$(grep "^Done\." "$TRAIN_LOG_DIR/${EXP}.log" | tail -1)
    echo "  TRAIN_DONE — $DONE_LINE"

    CKPT_PATH=$(find checkpoints/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
    if [ -z "$CKPT_PATH" ]; then
        echo "  TEST_SKIP — no checkpoint found for $EXP"
        continue
    fi

    echo "[$(date +%H:%M:%S)] TEST  $EXP"
    echo "  ckpt=$CKPT_PATH"
    TEST_EC=0
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
        "$PYTHON" -u test.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --dataset-dir "$DATASET_DIR" \
        --checkpoint-path "$CKPT_PATH" \
        --eval-on test \
        --batch-size 1 \
        --vis-per-scene 100 \
        --rotate-canonical \
        "${EXTRA[@]}" \
        > "$TEST_LOG_DIR/${EXP}_test.log" 2>&1 || TEST_EC=$?

    if [ "$TEST_EC" -ne 0 ]; then
        echo "  TEST_FAILED (exit=$TEST_EC)"
    else
        ABS=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        RMSE=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        D1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        echo "  TEST_OK  ABS=$ABS  RMSE=$RMSE  D1=$D1"
    fi
done

echo ""
echo "============================================================"
echo "renew_single_latter finished — $(date)"
echo "============================================================"

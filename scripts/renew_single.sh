#!/bin/bash
# ============================================================
# renew_single.sh — single train + test run for the renew dual-ViT
# sound-field bottleneck model. Matches the n1_bulk.sh safety
# conventions (explicit PYTHON, preflight module check, captured exit
# codes, test arguments forwarded so sh_dim etc. propagate).
#
# Model:     RenewSingleNet      (models/renew/renew_single.py)
# Config:    config/renew_single.yaml
# Exp name:  exp301_renew_single_lr1e4_lsh0.1
#
# Usage:
#   bash scripts/renew_single.sh                  # uses GPUs 0,1 by default
#   GPU=3,4 bash scripts/renew_single.sh          # override GPUs
#   EXTRA="--lambda-energy 0.3" bash scripts/renew_single.sh
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
# Dual-ViT at 183 M params is unusable on CPU — a single forward pass at
# (2, 2, 256, 512) takes tens of minutes. Fail fast (not silent) if the
# process about to be launched cannot see a CUDA device.
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
    echo "       troubleshooting:" >&2
    echo "         nvidia-smi          # confirm driver sees GPUs" >&2
    echo "         echo \$CUDA_VISIBLE_DEVICES   # make sure not set to '' or '-1'" >&2
    echo "         unset CUDA_VISIBLE_DEVICES && bash scripts/renew_single.sh" >&2
    exit 2
fi
echo "CUDA preflight: $_GPU_OK"

# --- Config ------------------------------------------------------------
CONFIG="renew_single"
EXP="${EXP:-exp301_renew_single_lr1e4_lsh0.1}"
# 4-GPU default. train.py auto-wraps in DataParallel over all visible
# devices (capped at 4). Override via env, e.g. GPU=4,5,6,7.
GPU="${GPU:-0,1,2,3}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-0.0001}"
# 183 M params × DP replicas + ~1 GB activations per sample at 256×512;
# BS=16 ≈ 4/GPU fits comfortably on 24 GB+ cards. Drop to 8 on 16 GB.
BS="${BS:-16}"
WORKERS="${WORKERS:-4}"
LSH="${LSH:-0.1}"
EXTRA=(${EXTRA:-})                       # caller-supplied extra flags

TRAIN_LOG_DIR="$PROJECT_DIR/logs/renew_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/renew_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

echo "============================================================"
echo "renew_single — $EXP"
echo "  config=$CONFIG  GPU=$GPU  epochs=$EPOCHS  lr=$LR  bs=$BS  lsh=$LSH"
echo "  extra='${EXTRA[*]:-}'"
echo "  $(date)"
echo "============================================================"

# --- Train -------------------------------------------------------------
echo "[$(date +%H:%M:%S)] TRAIN $EXP"
TRAIN_EC=0
CUDA_VISIBLE_DEVICES="$GPU" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
"$PYTHON" -u train.py \
    --config "$CONFIG" \
    --experiment-name "$EXP" \
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
    exit "$TRAIN_EC"
fi
DONE_LINE=$(grep "^Done\." "$TRAIN_LOG_DIR/${EXP}.log" | tail -1)
echo "  TRAIN_DONE — $DONE_LINE"

# --- Find checkpoint and test -----------------------------------------
CKPT_PATH=$(find checkpoints/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ]; then
    echo "  TEST_SKIP — no checkpoint found for $EXP"
    exit 0
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
    --checkpoint-path "$CKPT_PATH" \
    --eval-on test \
    --batch-size 1 \
    --vis-per-scene 100 \
    --rotate-canonical \
    "${EXTRA[@]}" \
    > "$TEST_LOG_DIR/${EXP}_test.log" 2>&1 || TEST_EC=$?

if [ "$TEST_EC" -ne 0 ]; then
    echo "  TEST_FAILED (exit=$TEST_EC). Tail:"
    tail -5 "$TEST_LOG_DIR/${EXP}_test.log"
    exit "$TEST_EC"
fi

ABS=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
RMSE=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
D1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
FOA=$(grep -oP 'FOA_L1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
echo "  TEST_OK  ABS=$ABS  RMSE=$RMSE  D1=$D1  FOA_L1=$FOA"
echo "============================================================"
echo "renew_single finished — $(date)"
echo "============================================================"

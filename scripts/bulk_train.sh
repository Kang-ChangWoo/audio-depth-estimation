#!/bin/bash
# ============================================================
# Bulk training: run all 4 models in parallel across GPU pairs
# GPUs 0,1: baseline | GPUs 2,3: echodiffusion
# GPUs 4,5: foa      | GPUs 6,7: vit
# ============================================================
set -euo pipefail

EXPERIMENT=${1:-"full_run"}
EPOCHS=${2:-40}
NUM_WORKERS=${3:-4}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export WANDB_PROJECT=neurips_audio_depth
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Bulk Training — experiment=$EXPERIMENT epochs=$EPOCHS"
echo "Project: $PROJECT_DIR"
echo "============================================================"

# ── GPUs 0,1: baseline ──
echo "[GPUs 0,1] Starting baseline..."
CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
    --config baseline \
    --experiment-name "${EXPERIMENT}_baseline" \
    --epochs "$EPOCHS" \
    --num-workers "$NUM_WORKERS" \
    > "$LOG_DIR/train_baseline_${EXPERIMENT}.log" 2>&1 &
PID_BASELINE=$!

# ── GPUs 2,3: echodiffusion ──
echo "[GPUs 2,3] Starting echodiffusion..."
CUDA_VISIBLE_DEVICES=2,3 python3 train.py \
    --config echodiffusion \
    --experiment-name "${EXPERIMENT}_echodiffusion" \
    --epochs "$EPOCHS" \
    --num-workers "$NUM_WORKERS" \
    > "$LOG_DIR/train_echodiffusion_${EXPERIMENT}.log" 2>&1 &
PID_ECHO=$!

# ── GPUs 4,5: foa ──
echo "[GPUs 4,5] Starting FOA..."
CUDA_VISIBLE_DEVICES=4,5 python3 train.py \
    --config foa \
    --experiment-name "${EXPERIMENT}_foa" \
    --epochs "$EPOCHS" \
    --num-workers "$NUM_WORKERS" \
    > "$LOG_DIR/train_foa_${EXPERIMENT}.log" 2>&1 &
PID_FOA=$!

# ── GPUs 6,7: vit ──
echo "[GPUs 6,7] Starting ViT..."
CUDA_VISIBLE_DEVICES=6,7 python3 train.py \
    --config vit \
    --experiment-name "${EXPERIMENT}_vit" \
    --epochs "$EPOCHS" \
    --num-workers "$NUM_WORKERS" \
    > "$LOG_DIR/train_vit_${EXPERIMENT}.log" 2>&1 &
PID_VIT=$!

# ── Wait for all ──
echo ""
echo "PIDs: baseline=$PID_BASELINE echo=$PID_ECHO foa=$PID_FOA vit=$PID_VIT"
echo "Waiting for all training jobs..."

FAIL=0
for PID_NAME in "baseline:$PID_BASELINE" "echodiffusion:$PID_ECHO" "foa:$PID_FOA" "vit:$PID_VIT"; do
    NAME="${PID_NAME%%:*}"
    PID="${PID_NAME##*:}"
    if wait "$PID"; then
        echo "  $NAME done (success)"
    else
        echo "  $NAME FAILED (exit=$?)"
        FAIL=1
    fi
done

echo ""
echo "============================================================"
if [ "$FAIL" -eq 0 ]; then
    echo "All training complete. Logs in: $LOG_DIR"
else
    echo "Some training jobs FAILED. Check logs in: $LOG_DIR"
fi
echo "============================================================"
exit $FAIL

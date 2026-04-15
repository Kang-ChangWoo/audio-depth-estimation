#!/bin/bash
# Test FOA_v2_js model (independent from bulk testing)
set -euo pipefail

# ── Parameters ──
GPU="${1:-0}"
EXPERIMENT="${2:-full_run}"
EVAL_ON="${3:-test}"
CKPT="${4:-best}"
VIS_PER_SCENE="${5:-100}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/test_foa_${EXPERIMENT}.log"
mkdir -p "$LOG_DIR"

EXP_NAME="${EXPERIMENT}_foa_v2_js"
CKPT_DIR="unet_256_soundspaces_BS32_Lr0.001_AdamW_${EXP_NAME}"
CKPT_PATH="$PROJECT_DIR/checkpoints/${CKPT_DIR}/best_model.pth"

echo "============================================================"
echo "  GPU:         $GPU"
echo "  Experiment:  $EXP_NAME"
echo "  Eval on:     $EVAL_ON"
echo "  Checkpoint:  $CKPT_PATH"
echo "  Vis/scene:   $VIS_PER_SCENE"
echo "  Log:         $LOG_FILE"
echo "============================================================"

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    echo "  (did you run train_JS.sh first?)"
    exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU" python3 test.py \
    --config foa_v2_js \
    --experiment-name "$EXP_NAME" \
    --checkpoint-path "$CKPT_PATH" \
    --eval-on "$EVAL_ON" \
    --checkpoints "$CKPT" \
    --vis-per-scene "$VIS_PER_SCENE" \
    2>&1 | tee "$LOG_FILE"

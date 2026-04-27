#!/bin/bash
# Train FOA_v2_js_rgb model (RGB teacher-guided with feature alignment)
set -euo pipefail

# ── Parameters ──
GPUS="${1:-0,1}"
EXPERIMENT="${2:-full_run}"
EPOCHS="${3:-40}"
NUM_WORKERS="${4:-8}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/train_foa_${EXPERIMENT}.log"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  GPUs:        $GPUS"
echo "  Experiment:  $EXPERIMENT"
echo "  Epochs:      $EPOCHS"
echo "  Workers:     $NUM_WORKERS"
echo "  Config:      foa_v2_js_rgb"
echo "  Log:         $LOG_FILE"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPUS" python3 train.py \
    --config foa_v2_js_rgb \
    --experiment-name "${EXPERIMENT}_foa_v2_js_rgb" \
    --epochs "$EPOCHS" \
    --batch-size 12 \
    --num-workers "$NUM_WORKERS" \
    2>&1 | tee "$LOG_FILE"

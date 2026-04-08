#!/bin/bash
# ============================================================
# Bulk testing: evaluate all 4 models in parallel across GPU pairs
# GPUs 0,1: baseline | GPUs 2,3: echodiffusion
# GPUs 4,5: foa      | GPUs 6,7: vit
# ============================================================
set -euo pipefail

EXPERIMENT=${1:-"full_run"}
EVAL_ON=${2:-"test"}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Bulk Testing — experiment=$EXPERIMENT eval_on=$EVAL_ON"
echo "============================================================"

# ── Run all tests in parallel ──
echo "[GPUs 0,1] Testing baseline..."
CUDA_VISIBLE_DEVICES=0,1 python3 test.py \
    --config baseline \
    --experiment-name "${EXPERIMENT}_baseline" \
    --eval-on "$EVAL_ON" \
    > "$LOG_DIR/test_baseline_${EXPERIMENT}.log" 2>&1 &
PID_B=$!

echo "[GPUs 2,3] Testing echodiffusion..."
CUDA_VISIBLE_DEVICES=2,3 python3 test.py \
    --config echodiffusion \
    --experiment-name "${EXPERIMENT}_echodiffusion" \
    --eval-on "$EVAL_ON" \
    > "$LOG_DIR/test_echodiffusion_${EXPERIMENT}.log" 2>&1 &
PID_E=$!

echo "[GPUs 4,5] Testing FOA..."
CUDA_VISIBLE_DEVICES=4,5 python3 test.py \
    --config foa \
    --experiment-name "${EXPERIMENT}_foa" \
    --eval-on "$EVAL_ON" \
    > "$LOG_DIR/test_foa_${EXPERIMENT}.log" 2>&1 &
PID_F=$!

echo "[GPUs 6,7] Testing ViT..."
CUDA_VISIBLE_DEVICES=6,7 python3 test.py \
    --config vit \
    --experiment-name "${EXPERIMENT}_vit" \
    --eval-on "$EVAL_ON" \
    > "$LOG_DIR/test_vit_${EXPERIMENT}.log" 2>&1 &
PID_V=$!

# ── Wait for all ──
echo ""
echo "Waiting for all test jobs..."

FAIL=0
for PID_NAME in "baseline:$PID_B" "echodiffusion:$PID_E" "foa:$PID_F" "vit:$PID_V"; do
    NAME="${PID_NAME%%:*}"
    PID="${PID_NAME##*:}"
    if wait "$PID"; then
        echo "  $NAME done (success)"
    else
        echo "  $NAME FAILED (exit=$?)"
        FAIL=1
    fi
done

# ── Print results summary ──
echo ""
echo "============================================================"
echo "Results Summary"
echo "============================================================"
for model in baseline echodiffusion foa vit; do
    LOG="$LOG_DIR/test_${model}_${EXPERIMENT}.log"
    if [ -f "$LOG" ]; then
        echo ""
        echo "--- $model ---"
        grep -E "ABS_REL|RMSE|Delta1|Log10|MAE|FOA" "$LOG" 2>/dev/null || echo "(no results)"
    fi
done
echo ""
echo "============================================================"
echo "Logs in: $LOG_DIR"
echo "============================================================"
exit $FAIL

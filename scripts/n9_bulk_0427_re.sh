#!/bin/bash
# ============================================================
# n9_bulk_0427_re.sh — re-run exp910 only on GPU 1.
#
# exp910 (median-output variant of exp907) failed in n9_bulk_0427.sh
# due to a broadcast bug in range_head.py:123 (median branch never
# expanded the batch dimension on `bins`). The fix landed earlier
# this session, so the run can be reattempted with the same hp.
#
# The other cancelled experiments are already covered:
#   exp909  on GPU 0  (running in the original sweep)
#   exp911  on GPU 0  (queued behind exp909)
#   exp912  on GPU 1  (in-flight in the original sweep)
#
# Run this AFTER the in-flight GPU-1 job finishes — it pins to GPU 1.
#
# Usage:
#     bash scripts/n9_bulk_0427_re.sh                  # default GPU 1
#     GPU=1 bash scripts/n9_bulk_0427_re.sh            # explicit
#     EPOCHS=20 bash scripts/n9_bulk_0427_re.sh        # override
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303_renew}"
EPOCHS="${EPOCHS:-20}"
WORKERS="${WORKERS:-4}"
VIS_PER_SCENE=0
GPU="${GPU:-1}"

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n9_0427_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n9_0427_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

CONFIG="echorange"
EXP="exp910_echorange_32log_d10_median_full_lr1e4_bs32"
LR="0.0001"
BS="32"
EXTRA=(
    --depth-head-type range
    --range-num-bins 32
    --range-bin-spacing log
    --range-min-depth 0.1
    --range-max-depth 10.0
    --range-soft-label-sigma 0.14
    --range-output-mode median
    --lambda-range-nll 1.0
    --lambda-berhu 1.0
    --lambda-silog 1.0
    --erp-cos-lat-weight
    --erp-far-mask
)

echo "============================================================"
echo "n9_bulk_0427_re — re-run $EXP on GPU $GPU"
echo "  EPOCHS=$EPOCHS  bs=$BS  lr=$LR"
echo "  $(date)"
echo "============================================================"

# ── TRAIN ────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] [GPU=$GPU] TRAIN $EXP"
JOB_TIMEOUT="${JOB_TIMEOUT:-54000}"
ec=0
CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    timeout --signal=TERM --kill-after=60 "$JOB_TIMEOUT" \
    "$PYTHON" train.py \
    --config "$CONFIG" \
    --experiment-name "$EXP" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BS" \
    --num-workers "$WORKERS" \
    --depth-dir "$DEPTH_DIR" \
    "${DATASET_ARGS[@]}" \
    "${EXTRA[@]}" \
    > "$TRAIN_LOG_DIR/${EXP}.log" 2>&1 || ec=$?

if [ "$ec" -eq 124 ] || [ "$ec" -eq 137 ]; then
    echo "  [GPU=$GPU] $EXP TRAIN_TIMEOUT (exit=$ec, budget=${JOB_TIMEOUT}s)"
    tail -3 "$TRAIN_LOG_DIR/${EXP}.log" | sed "s/^/      /"
    exit 1
fi
if [ "$ec" -ne 0 ] && [ "$ec" -ne 141 ]; then
    echo "  [GPU=$GPU] $EXP TRAIN_FAILED (exit=$ec)"
    tail -3 "$TRAIN_LOG_DIR/${EXP}.log" | sed "s/^/      /"
    exit 1
fi
echo "  [GPU=$GPU] $EXP TRAIN_DONE"

# ── TEST ─────────────────────────────────────────────────────
echo ""
CKPT=$(find checkpoints/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "  [test] $EXP NO_CHECKPOINT — skipping"
    exit 1
fi

echo "[$(date +%H:%M:%S)] [GPU=$GPU] TEST $EXP"
ec=0
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PYTHON" -u test.py \
    --config "$CONFIG" \
    --experiment-name "$EXP" \
    --checkpoint-path "$CKPT" \
    --eval-on test \
    --batch-size 4 \
    --num-workers "$WORKERS" \
    --vis-per-scene "$VIS_PER_SCENE" \
    --depth-dir "$DEPTH_DIR" \
    "${DATASET_ARGS[@]}" \
    "${EXTRA[@]}" \
    > "$TEST_LOG_DIR/${EXP}_test.log" 2>&1 || ec=$?

if [ "$ec" -ne 0 ]; then
    echo "  [test] $EXP TEST_FAILED (exit=$ec)"
    tail -3 "$TEST_LOG_DIR/${EXP}_test.log" | sed "s/^/      /"
    exit 1
fi
abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
echo "  [test] $EXP TEST_OK  ABS=$abs RMSE=$rmse D1=$d1"

echo ""
echo "============================================================"
echo "n9_bulk_0427_re finished — $(date)"
echo "============================================================"

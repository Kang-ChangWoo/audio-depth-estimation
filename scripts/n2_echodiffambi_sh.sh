#!/bin/bash
# ============================================================
# n2_echodiffambi_sh.sh — node2 variant of n9_echodiffambi_sh.sh.
#
# Same architecture and experiment list as n9_echodiffambi_sh.sh
# (EchoDiffusion + audio→SH coefficient prediction + real-SH coarse
# layout fusion at the depth-decoder output, max-capacity 167.92M
# parameters, see models/echodiffusion/echodiffusion_ambi_sh.py).
#
# Differences from n9 → n2:
#   1. DATASET_DIR — local SSD on node2 (/root/local1/...). NFS on node9
#      (/root/storage) is 100% full and concurrent-read-bound; that was
#      the root cause of the DataLoader timeout=120s failures observed
#      on the previous launch.
#   2. GPU_PAIRS — default to "0,1,2,3 4,5,6,7" (2 workers × 4 GPUs each
#      via DataParallel), matching this script's bs=128 / per-GPU=32
#      sizing comment. The n9 default was a single 3-GPU group.
#   3. Log directories prefixed with logs/n2_echodiff_ambi_sh_* so the
#      two servers don't fight over the same log files when sharing the
#      project tree via NFS.
#
# Active cells (potential picks based on prior sweeps):
#   exp720  lr=5e-4   bs=128   (echodiffusion's best historical LR)
#   exp721  lr=1e-4   bs=128   (n4_0425's best historical LR)
#   exp722  lr=5e-5   bs=128   (likely too slow — held)
#   exp723  lr=1e-3   bs=128   (likely unstable — held)
#
# Usage:
#     bash scripts/n2_echodiffambi_sh.sh
#     EPOCHS=20 bash scripts/n2_echodiffambi_sh.sh    # quick sweep
#     GPU_PAIRS="0,1 2,3 4,5 6,7" bash scripts/n2_echodiffambi_sh.sh
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"

# Local-SSD copy of matterport3d on node2. Verified existing path
# (note: NO underscore between '0303' and 'renew' — the earlier
# n9 default had a typo'd '0303_renew' that did not exist).
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303renew}"
EPOCHS="${EPOCHS:-40}"

# WORKERS=4 (per-job dataloader workers) keeps total dataloader processes
# at 2 × 4 = 8 across the two parallel jobs. The n9 default of 8 ×
# 2 jobs = 16 was on the edge of CPU/IO saturation even on local SSD.
WORKERS="${WORKERS:-4}"
VIS_PER_SCENE=0

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: DATASET_DIR not found: $DATASET_DIR" >&2
    echo "       Set DATASET_DIR=<path> on launch, or fix the default above." >&2
    exit 3
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n2_echodiff_ambi_sh_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n2_echodiff_ambi_sh_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

# 2 parallel workers × 4 GPUs each (DataParallel). Worker 0 runs on GPUs
# 0,1,2,3; worker 1 on GPUs 4,5,6,7. Override via GPU_PAIRS env var if
# only a subset is available (e.g. GPU_PAIRS="0,1,2,3" for one job).
GPU_PAIRS_STR="${GPU_PAIRS:-0,1,2,3 4,5,6,7}"
read -r -a GPU_PAIRS <<< "$GPU_PAIRS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

train_and_test() {
    local GPU="$1" CONFIG="$2" EXP="$3" LR="$4" BS="$5"
    shift 5
    local EXTRA=("$@")

    if [ -f "$TEST_LOG_DIR/${EXP}_test.log" ] && \
       grep -q 'ABS_REL:' "$TEST_LOG_DIR/${EXP}_test.log"; then
        echo "[$(date +%H:%M:%S)] [GPU=$GPU] SKIP $EXP (already completed)"
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TRAIN $EXP (cfg=$CONFIG lr=$LR bs=$BS extra='${EXTRA[*]}')"
    local JOB_TIMEOUT="${JOB_TIMEOUT:-54000}"
    local ec=0
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
        return
    fi
    if [ "$ec" -ne 0 ] && [ "$ec" -ne 141 ]; then
        echo "  [GPU=$GPU] $EXP TRAIN_FAILED (exit=$ec)"
        tail -3 "$TRAIN_LOG_DIR/${EXP}.log" | sed "s/^/      /"
        return
    fi
    echo "  [GPU=$GPU] $EXP TRAIN_DONE"

    local CKPT
    CKPT=$(find checkpoints/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "  [GPU=$GPU] $EXP TEST_SKIP (no checkpoint)"
        return
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
        echo "  [GPU=$GPU] $EXP TEST_FAILED (exit=$ec)"
    else
        local abs rmse d1
        abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        echo "  [GPU=$GPU] $EXP TEST_OK  ABS=$abs RMSE=$rmse D1=$d1"
    fi
}

# ============================================================
# Experiment list — 2 high-priority LR cells first, 2 deferred cells
# uncommented for completeness. Same as n9_echodiffambi_sh.sh.
# ============================================================
ALL_EXPS=(
    "echodiffusion_ambi_sh  exp720_eda_sh5_lr5e4_bs128      0.0005  128"
    "echodiffusion_ambi_sh  exp721_eda_sh5_lr1e4_bs128      0.0001  128"
    "echodiffusion_ambi_sh  exp722_eda_sh5_lr5e5_bs128      0.00005 128"
    "echodiffusion_ambi_sh  exp723_eda_sh5_lr1e3_bs128      0.001   128"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache (sequentially; avoids the worker-storm we hit
# at first-batch on NFS).
echo "Pre-warming dataset cache for radial depth on $DATASET_DIR..."
"$PYTHON" -c "
from data.dataset import SoundSpacesDataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/echodiffusion_ambi_sh.yaml')
cfg.dataset.depth_dir = '$DEPTH_DIR'
if '$DATASET_DIR':
    cfg.dataset.dataset_dir = '$DATASET_DIR'
for split in ['train', 'val', 'test']:
    SoundSpacesDataset(cfg, split=split)
print('Cache warm.')
" 2>&1 | tail -5
echo ""

echo "============================================================"
echo "n2_echodiffambi_sh — $TOTAL experiments (echodiff + audio→SH coarse-layout)"
echo "  EPOCHS=$EPOCHS  depth_dir=$DEPTH_DIR  bs=128  K=8  sh_order=5 (36 coeffs)  +CIDE  unet_ch=64  dec_ch=[256,128,64]"
echo "  dataset_dir=${DATASET_DIR}"
echo "  workers/job=$WORKERS  $NUM_WORKERS parallel jobs × GPUs: ${GPU_PAIRS[*]}"
echo "  $(date)"
echo "============================================================"

run_worker() {
    local WORKER_ID=$1
    local GPU="${GPU_PAIRS[$WORKER_ID]}"

    for (( i=WORKER_ID; i<TOTAL; i+=NUM_WORKERS )); do
        local SPEC="${ALL_EXPS[$i]}"
        read -r -a FIELDS <<< "$SPEC"
        local CONFIG="${FIELDS[0]}"
        local EXP="${FIELDS[1]}"
        local LR="${FIELDS[2]}"
        local BS="${FIELDS[3]}"
        local EXTRA=("${FIELDS[@]:4}")

        train_and_test "$GPU" "$CONFIG" "$EXP" "$LR" "$BS" "${EXTRA[@]}"
    done
}

PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
    run_worker "$w" &
    PIDS+=($!)
    echo "Worker $w launched (GPU ${GPU_PAIRS[$w]}, PID ${PIDS[-1]})"
done

echo "All $NUM_WORKERS workers running. Waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

echo ""
echo "============================================================"
echo "n2_echodiffambi_sh finished — $(date)"
echo "Workers failed: $FAIL / $NUM_WORKERS"
echo "============================================================"
echo ""
printf "%-50s %8s %8s %8s\n" "Experiment" "ABS_REL" "RMSE" "Delta1"
echo "---------------------------------------------------------------------"

SUCCESS=0
for SPEC in "${ALL_EXPS[@]}"; do
    EXP=$(echo "$SPEC" | awk '{print $2}')
    LOG="$TEST_LOG_DIR/${EXP}_test.log"
    if [ -f "$LOG" ]; then
        abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$LOG" | head -1)
        rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$LOG" | head -1)
        d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$LOG" | head -1)
        if [ -n "$abs" ]; then
            printf "%-50s %8s %8s %8s\n" "$EXP" "$abs" "$rmse" "$d1"
            SUCCESS=$((SUCCESS + 1))
        else
            printf "%-50s %8s\n" "$EXP" "FAILED"
        fi
    else
        printf "%-50s %8s\n" "$EXP" "NO_RESULT"
    fi
done

echo "---------------------------------------------------------------------"
echo "Success: $SUCCESS / $TOTAL"
echo "============================================================"

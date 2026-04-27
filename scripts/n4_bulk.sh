#!/bin/bash
# ============================================================
# n4_bulk.sh — n4_0425 (binaural UNet + oracle bin-gated FOA) sweep.
#
# Implements the bin-selection plan in
# docs/experiment_plan_bin_selection.md. The model is described in
# models/n4_0425/n4_0425.py: a learnable gate g ∈ R^{K=8} multiplies
# the dataset's oracle per-bin FOA representatives (rep_gt), the
# gated bins are projected via MLP and added at the UNet bottleneck.
#
# This sweep is sized at 20 runs, exp400-419:
#
#   λ_sparsity sweep (gate learnable, lr=1e-4):
#     exp400  λ=0.0     no sparsity pressure (oracle binaural+all-bins ceiling)
#     exp401  λ=0.01
#     exp402  λ=0.05
#     exp403  λ=0.1     (config default)
#     exp404  λ=0.5     strong sparsity
#
#   LR sweep (λ=0.05, gate learnable):
#     exp405-409  lr ∈ {5e-4, 3e-4, 1e-3, 5e-5, 1e-5}
#
#   Drop-one-bin retraining (gate frozen, λ=0, lr=1e-4):
#     exp410-417  drop bin k, k=0..7
#
#   Special cases:
#     exp418  all-zero gate (binaural-only baseline; bin path inert)
#     exp419  random K=4 gate (alternating: bins 0,2,4,6)
#
# Runs on node4 with 4 workers × 2 GPUs each (GPUs 0-7).
# Reads/writes from /root/local1/changwoo/matterport3d_0303renew (local SSD).
#
# Usage:
#     bash scripts/n4_bulk.sh
#     GPU_PAIRS="0,1 2,3 4,5 6,7" bash scripts/n4_bulk.sh
#     EPOCHS=20 bash scripts/n4_bulk.sh    # quick sweep
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
# Local-SSD copy of matterport3d_0303renew on node4 (see /root/local1).
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303renew}"
EPOCHS="${EPOCHS:-40}"
WORKERS="${WORKERS:-2}"
VIS_PER_SCENE=0

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n4_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n4_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

GPU_PAIRS_STR="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
read -r -a GPU_PAIRS <<< "$GPU_PAIRS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

train_and_test() {
    local GPU="$1" CONFIG="$2" EXP="$3" LR="$4" BS="$5"
    shift 5
    local EXTRA=("$@")

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TRAIN $EXP (cfg=$CONFIG lr=$LR bs=$BS extra='${EXTRA[*]}')"

    local JOB_TIMEOUT="${JOB_TIMEOUT:-54000}"   # 15 h default
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
# Experiment definitions: "CONFIG EXP LR BS [EXTRA…]"
# Common batch size bs=64 (54M UNet on 256x512, ~5-6 GiB/GPU on RTX 3090
# 24 GiB with 2-GPU DataParallel = 32 per GPU). ~25% utilization — leaves
# plenty of headroom in case stale processes squat on the card.
# ============================================================
ALL_EXPS=(
    # --- Front-loaded representatives (one per group, fans out across 4 workers) ---
    # 1. ceiling — no sparsity pressure, all bins kept (sparsity group rep)
    "n4_0425  exp400_n4_lam0.0_lr1e4_bs64     0.0001  64  --lambda-sparsity 0.0"
    # 2. default sparsity λ=0.1 (main config)
    "n4_0425  exp403_n4_lam0.1_lr1e4_bs64     0.0001  64  --lambda-sparsity 0.1"
    # 3. LR-sweep rep — highest LR (lr=1e-3)
    "n4_0425  exp407_n4_lam0.05_lr1e3_bs64    0.001   64  --lambda-sparsity 0.05"
    # 4. drop-bin rep — drop near bin 0 (early reflection ablation)
    "n4_0425  exp410_n4_drop0_lr1e4_bs64      0.0001  64  --gate-mask 0,1,1,1,1,1,1,1  --lambda-sparsity 0"
    # 5. floor — binaural-only baseline (all-zero gate, bin path effectively inert)
    "n4_0425  exp418_n4_binaural_lr1e4_bs64   0.0001  64  --gate-mask 0,0,0,0,0,0,0,0  --lambda-sparsity 0"

    # --- Remaining λ_sparsity sweep (gate learnable, lr=1e-4) ---
    "n4_0425  exp401_n4_lam0.01_lr1e4_bs64    0.0001  64  --lambda-sparsity 0.01"
    "n4_0425  exp402_n4_lam0.05_lr1e4_bs64    0.0001  64  --lambda-sparsity 0.05"
    "n4_0425  exp404_n4_lam0.5_lr1e4_bs64     0.0001  64  --lambda-sparsity 0.5"

    # --- Remaining LR sweep (λ=0.05, gate learnable) ---
    "n4_0425  exp405_n4_lam0.05_lr5e4_bs64    0.0005  64  --lambda-sparsity 0.05"
    "n4_0425  exp406_n4_lam0.05_lr3e4_bs64    0.0003  64  --lambda-sparsity 0.05"
    "n4_0425  exp408_n4_lam0.05_lr5e5_bs64    0.00005 64  --lambda-sparsity 0.05"
    "n4_0425  exp409_n4_lam0.05_lr1e5_bs64    0.00001 64  --lambda-sparsity 0.05"

    # --- Remaining drop-one-bin retraining (gate frozen, λ=0, lr=1e-4) ---
    "n4_0425  exp411_n4_drop1_lr1e4_bs64      0.0001  64  --gate-mask 1,0,1,1,1,1,1,1  --lambda-sparsity 0"
    "n4_0425  exp412_n4_drop2_lr1e4_bs64      0.0001  64  --gate-mask 1,1,0,1,1,1,1,1  --lambda-sparsity 0"
    "n4_0425  exp413_n4_drop3_lr1e4_bs64      0.0001  64  --gate-mask 1,1,1,0,1,1,1,1  --lambda-sparsity 0"
    "n4_0425  exp414_n4_drop4_lr1e4_bs64      0.0001  64  --gate-mask 1,1,1,1,0,1,1,1  --lambda-sparsity 0"
    "n4_0425  exp415_n4_drop5_lr1e4_bs64      0.0001  64  --gate-mask 1,1,1,1,1,0,1,1  --lambda-sparsity 0"
    "n4_0425  exp416_n4_drop6_lr1e4_bs64      0.0001  64  --gate-mask 1,1,1,1,1,1,0,1  --lambda-sparsity 0"
    "n4_0425  exp417_n4_drop7_lr1e4_bs64      0.0001  64  --gate-mask 1,1,1,1,1,1,1,0  --lambda-sparsity 0"

    # --- Random K=4 alternating mask (every other bin) ---
    "n4_0425  exp419_n4_alt_K4_lr1e4_bs64     0.0001  64  --gate-mask 1,0,1,0,1,0,1,0  --lambda-sparsity 0"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache (avoids workers scanning in parallel).
echo "Pre-warming dataset cache for radial depth on $DATASET_DIR..."
"$PYTHON" -c "
from data.dataset import SoundSpacesDataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/n4_0425.yaml')
cfg.dataset.depth_dir = '$DEPTH_DIR'
if '$DATASET_DIR':
    cfg.dataset.dataset_dir = '$DATASET_DIR'
for split in ['train', 'val', 'test']:
    SoundSpacesDataset(cfg, split=split)
print('Cache warm.')
" 2>&1 | tail -5
echo ""

echo "============================================================"
echo "n4_bulk — $TOTAL experiments (n4_0425 bin-gated UNet)"
echo "  EPOCHS=$EPOCHS  depth_dir=$DEPTH_DIR  bs=64  lambda_sparsity (per-row)"
echo "  dataset_dir=${DATASET_DIR:-<yaml default>}"
echo "  $NUM_WORKERS workers × 2 GPUs: ${GPU_PAIRS[*]}"
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
echo "n4_bulk finished — $(date)"
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

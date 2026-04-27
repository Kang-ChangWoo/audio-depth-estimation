#!/bin/bash
# ============================================================
# n9_bulk.sh — n9_0425 (binaural UNet + bin-gated FOA from a
# PRE-TRAINED n3_0425) sweep.
#
# Architecture: identical to n4_0425 except that the per-bin FOA
# representatives feeding the gate / EM-encoder come from a frozen-or-
# finetuned n3_0425 model instead of the dataset's oracle rep_gt.
# See models/n9_0425/n9_0425.py.
#
# Backbones: ONE best eigen n3 per K — RMS and sensitivity-check
# variants are skipped per design preference. The K=8 vs K=1 split is
# kept because the bin granularity may genuinely change behaviour:
#   K=8  (n3 = exp503, eigen-8, geometric distance edges)
#   K=1  (n3 = exp502, eigen-1, single 0→10 m round-trip window)
#
# 12 experiments, exp600–611:
#
#   A) K=8 vs K=1, frozen vs finetune (4 runs — the head-to-head test):
#     exp600 K=8 freeze    n3=exp503
#     exp601 K=8 finetune  n3=exp503
#     exp602 K=1 freeze    n3=exp502
#     exp603 K=1 finetune  n3=exp502
#
#   B) λ_sparsity sweep on best frozen K=8 (3 runs, completes a 4-point
#      sweep with exp600's λ=0.1):
#     exp604 freeze λ_sp=0      (no sparsity pressure)
#     exp605 freeze λ_sp=0.05
#     exp606 freeze λ_sp=0.5
#
#   C) λ_sh fine-tune sweep on best finetune K=8 (3 runs, completes a
#      4-point sweep with exp601's λ_sh=0.05):
#     exp607 finetune λ_sh=0      (no n3 supervision; depth loss only)
#     exp608 finetune λ_sh=0.01
#     exp609 finetune λ_sh=0.1
#
#   D) LR sweep on best finetune K=8 (2 runs):
#     exp610 finetune lr=5e-5
#     exp611 finetune lr=3e-4
#
# Common: bs=64, eigen distance edges. 4 workers × 2 GPUs each (GPUs 0-7).
#
# Usage:
#     bash scripts/n9_bulk.sh
#     GPU_PAIRS="0,1 2,3 4,5 6,7" bash scripts/n9_bulk.sh
#     EPOCHS=20 bash scripts/n9_bulk.sh                        # quick sweep
#     N3_CKPT_503=/path/to/best_model.pth bash scripts/n9_bulk.sh
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
# Local1 dataset path on this server (note _renew with underscore).
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303_renew}"
EPOCHS="${EPOCHS:-40}"
# 1 sequential job × 4 GPUs (DataParallel), so 1 main + 8 dataloader = 9 procs.
WORKERS="${WORKERS:-8}"
VIS_PER_SCENE=0

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n9_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n9_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

# ============================================================
# n3_0425 checkpoint paths (override via env if your dirs differ).
# Pattern: checkpoints/n3_0425_soundspaces_BS128_Lr<LR>_AdamW_<EXP>/best_model.pth
#
# Only the BEST eigen-K=8 (exp503) and best eigen-K=1 (exp502) backbones
# are used in this sweep.
# ============================================================
N3_CKPT_503="${N3_CKPT_503:-$PROJECT_DIR/checkpoints/n3_0425_soundspaces_BS128_Lr0.0001_AdamW_exp503_n3_eigen8_lr1e4_bs128/best_model.pth}"
N3_CKPT_502="${N3_CKPT_502:-$PROJECT_DIR/checkpoints/n3_0425_soundspaces_BS128_Lr0.0001_AdamW_exp502_n3_eigen1_lr1e4_bs128/best_model.pth}"

echo "n3 checkpoints:"
for v in N3_CKPT_503 N3_CKPT_502; do
    p="${!v}"
    if [ -f "$p" ]; then
        printf "  %-13s OK    %s\n" "$v" "$p"
    else
        printf "  %-13s MISS  %s\n" "$v" "$p"
    fi
done
echo ""

# This server has 4 GPUs (0,1,2,3). 1 sequential worker × 4 GPUs each — all
# 4 GPUs share one DataParallel job at a time. Matches renew_single_latter.sh.
GPU_PAIRS_STR="${GPU_PAIRS:-0,1,3}"
read -r -a GPU_PAIRS <<< "$GPU_PAIRS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

TEST_ONLY_EXPS=("exp601_n9_K8_finetune_lr1e4_bs256")

train_and_test() {
    local GPU="$1" CONFIG="$2" EXP="$3" LR="$4" BS="$5"
    shift 5
    local EXTRA=("$@")

    if [ -f "$TEST_LOG_DIR/${EXP}_test.log" ] && \
       grep -q 'ABS_REL:' "$TEST_LOG_DIR/${EXP}_test.log"; then
        echo "[$(date +%H:%M:%S)] [GPU=$GPU] SKIP $EXP (already completed)"
        return
    fi

    local SKIP_TRAIN=0
    for to in "${TEST_ONLY_EXPS[@]}"; do
        [ "$EXP" = "$to" ] && SKIP_TRAIN=1
    done

    if [ "$SKIP_TRAIN" -eq 1 ]; then
        echo "[$(date +%H:%M:%S)] [GPU=$GPU] SKIP_TRAIN $EXP (test-only)"
    else
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
    fi

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
# Experiment definitions: "CONFIG  EXP  LR  BS  [EXTRA…]"
#
# All rows use the eigen K=8 distance-bin layout (default in
# config/n9_0425.yaml), bs=64. The variable knobs are:
#   --n3-checkpoint        which n3 backbone to start from
#   --freeze-n3 / --no-freeze-n3
#   --lambda-sparsity      gate sparsity weight (group B)
#   --lambda-sh            n3 fine-tune signal toward rep_gt (group C)
# ============================================================
ALL_EXPS=(
    # --- A. K=8 vs K=1, Frozen vs Finetune (the head-to-head test) ---
    # bs=256 = 64/GPU on 4 GPUs DataParallel (4× the original n3-server bs=64).
    # exp600 deactivated (train + test skipped); exp601 train deactivated, test runs from existing checkpoint.
    # "n9_0425  exp600_n9_K8_freeze_lr1e4_bs256        0.0001  256  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --freeze-n3     --lambda-sparsity 0.1  --lambda-sh 0.0"
    "n9_0425  exp601_n9_K8_finetune_lr1e4_bs256      0.0001  256  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --no-freeze-n3  --lambda-sparsity 0.1  --lambda-sh 0.05"
    "n9_0425  exp602_n9_K1_freeze_lr1e4_bs256        0.0001  256  --rep-K 1  --n3-checkpoint $N3_CKPT_502  --freeze-n3     --lambda-sparsity 0.1  --lambda-sh 0.0"
    "n9_0425  exp603_n9_K1_finetune_lr1e4_bs256      0.0001  256  --rep-K 1  --n3-checkpoint $N3_CKPT_502  --no-freeze-n3  --lambda-sparsity 0.1  --lambda-sh 0.05"

    # --- B. λ_sparsity sweep on best frozen K=8 (n3=exp503) ---
    "n9_0425  exp604_n9_K8_freeze_lam0_lr1e4_bs256     0.0001  256  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --freeze-n3     --lambda-sparsity 0.0   --lambda-sh 0.0"
    "n9_0425  exp605_n9_K8_freeze_lam0.05_lr1e4_bs256  0.0001  256  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --freeze-n3     --lambda-sparsity 0.05  --lambda-sh 0.0"
    "n9_0425  exp606_n9_K8_freeze_lam0.5_lr1e4_bs256   0.0001  256  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --freeze-n3     --lambda-sparsity 0.5   --lambda-sh 0.0"

    # --- C. λ_sh fine-tune sweep on best finetune K=8 (n3=exp503) ---
    "n9_0425  exp607_n9_K8_finetune_lsh0_lr1e4_bs256     0.0001  256  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --no-freeze-n3  --lambda-sparsity 0.1   --lambda-sh 0.0"
    "n9_0425  exp608_n9_K8_finetune_lsh0.01_lr1e4_bs256  0.0001  256  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --no-freeze-n3  --lambda-sparsity 0.1   --lambda-sh 0.01"
    "n9_0425  exp609_n9_K8_finetune_lsh0.1_lr1e4_bs256   0.0001  256  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --no-freeze-n3  --lambda-sparsity 0.1   --lambda-sh 0.1"

    # --- D. LR sweep on best finetune K=8 (n3=exp503) ---
    "n9_0425  exp610_n9_K8_finetune_lr5e5_bs256       0.00005 256  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --no-freeze-n3  --lambda-sparsity 0.1   --lambda-sh 0.05"
    "n9_0425  exp611_n9_K8_finetune_lr3e4_bs256       0.0003  256  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --no-freeze-n3  --lambda-sparsity 0.1   --lambda-sh 0.05"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache (eigen-8 default — same as n4_bulk).
echo "Pre-warming dataset cache for radial depth on $DATASET_DIR..."
"$PYTHON" -c "
from data.dataset import SoundSpacesDataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/n9_0425.yaml')
cfg.dataset.depth_dir = '$DEPTH_DIR'
if '$DATASET_DIR':
    cfg.dataset.dataset_dir = '$DATASET_DIR'
for split in ['train', 'val', 'test']:
    SoundSpacesDataset(cfg, split=split)
print('Cache warm.')
" 2>&1 | tail -5
echo ""

echo "============================================================"
echo "n9_bulk — $TOTAL experiments (n9_0425 cascade: pre-trained n3 + bin-gated UNet)"
echo "  EPOCHS=$EPOCHS  depth_dir=$DEPTH_DIR  bs=256  K=8 (eigen-geometric)"
echo "  dataset_dir=${DATASET_DIR:-<yaml default>}"
echo "  $NUM_WORKERS workers × 4 GPUs: ${GPU_PAIRS[*]}"
echo "  $(date)"
echo "============================================================"

# ---- One-off: pre-train test for exp600 (DISABLED per user request) ----
# Was added to capture existing-checkpoint metrics before retraining; commented
# out because the test path was wedged on the broken GPU state (GPU2 NVML).
# EXP600_NAME="exp600_n9_K8_freeze_lr1e4_bs256"
# EXP600_PRE_CKPT=$(find checkpoints/ -name "best_model.pth" -path "*${EXP600_NAME}*" 2>/dev/null | head -1)
# if [ -n "$EXP600_PRE_CKPT" ]; then
#     echo "[$(date +%H:%M:%S)] PRE-TRAIN TEST exp600 (ckpt=$EXP600_PRE_CKPT)"
#     pre_ec=0
#     CUDA_VISIBLE_DEVICES="${GPU_PAIRS[0]}" PYTHONUNBUFFERED=1 \
#         OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
#         "$PYTHON" -u test.py \
#         --config n9_0425 \
#         --experiment-name "$EXP600_NAME" \
#         --checkpoint-path "$EXP600_PRE_CKPT" \
#         --eval-on test \
#         --batch-size 4 \
#         --num-workers "$WORKERS" \
#         --vis-per-scene 0 \
#         --depth-dir "$DEPTH_DIR" \
#         "${DATASET_ARGS[@]}" \
#         --rep-K 8 --n3-checkpoint "$N3_CKPT_503" --freeze-n3 \
#         --lambda-sparsity 0.1 --lambda-sh 0.0 \
#         > "$TEST_LOG_DIR/${EXP600_NAME}_pretrain_test.log" 2>&1 || pre_ec=$?
#     if [ "$pre_ec" -eq 0 ]; then
#         pre_abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP600_NAME}_pretrain_test.log" | head -1)
#         pre_rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP600_NAME}_pretrain_test.log" | head -1)
#         pre_d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP600_NAME}_pretrain_test.log" | head -1)
#         echo "  exp600 PRE-TRAIN  ABS=$pre_abs RMSE=$pre_rmse D1=$pre_d1"
#     else
#         echo "  exp600 PRE-TRAIN TEST FAILED (exit=$pre_ec)"
#         tail -3 "$TEST_LOG_DIR/${EXP600_NAME}_pretrain_test.log" | sed "s/^/      /"
#     fi
# else
#     echo "[$(date +%H:%M:%S)] PRE-TRAIN TEST exp600 — no existing checkpoint, skipping."
# fi
# echo ""

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
echo "n9_bulk finished — $(date)"
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

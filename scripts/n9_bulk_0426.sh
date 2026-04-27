#!/bin/bash
# ============================================================
# n9_bulk_0426.sh — n9_0426 (pretrained ViT/ResNet outer + n3_0425
# inner FOA cascade) sweep.
#
# Architecture: same I/O as n9_0425; the outer 8-level UNet is replaced
# by a torchvision ViT-B/16 or ResNet-50 (ImageNet pretrained). Outer
# input is concat(binaural, gated_em) — 3 channels. Inner FOA predictor
# is the same n3_0425 used by n9_0425 (loaded from N3_CKPT_503,
# eigen-K=8, frozen by default).
#
# 4 experiments, exp600–603 — {ViT, ResNet} × {freeze, finetune} on
# the outer pretrained backbone:
#     exp600  backbone=vit     freeze_backbone=True
#     exp601  backbone=vit     freeze_backbone=False
#     exp602  backbone=resnet  freeze_backbone=True
#     exp603  backbone=resnet  freeze_backbone=False
#
# Common: K=8 (eigen, geometric edges), bs=64, lr=1e-4, n3 frozen
# (n3=exp503, the same backbone n9_bulk.sh uses for its K=8 cells).
#
# Logs land in logs/n9_0426_{train,test}/ to avoid clashing with
# n9_bulk.sh's logs/n9_{train,test}/ (which uses the same exp600–603
# numbering but a different architecture).
#
# Usage:
#     bash scripts/n9_bulk_0426.sh
#     GPU_PAIRS="0,1 2,3 4,5 6,7" bash scripts/n9_bulk_0426.sh
#     EPOCHS=20 bash scripts/n9_bulk_0426.sh                        # quick sweep
#     N3_CKPT_503=/path/to/best_model.pth bash scripts/n9_bulk_0426.sh
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
# Local1 dataset path on this server (note _renew with underscore).
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303_renew}"
EPOCHS="${EPOCHS:-40}"
WORKERS="${WORKERS:-8}"
VIS_PER_SCENE=0

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n9_0426_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n9_0426_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

# ============================================================
# n3_0425 checkpoint path (override via env if your dir differs).
# Pattern: checkpoints/n3_0425_soundspaces_BS128_Lr<LR>_AdamW_<EXP>/best_model.pth
# Only the eigen-K=8 (exp503) backbone is used — same convention as
# n9_bulk.sh's K=8 cells.
# ============================================================
N3_CKPT_503="${N3_CKPT_503:-$PROJECT_DIR/checkpoints/n3_0425_soundspaces_BS128_Lr0.0001_AdamW_exp503_n3_eigen8_lr1e4_bs128/best_model.pth}"

echo "n3 checkpoint:"
if [ -f "$N3_CKPT_503" ]; then
    printf "  %-13s OK    %s\n" "N3_CKPT_503" "$N3_CKPT_503"
else
    printf "  %-13s MISS  %s\n" "N3_CKPT_503" "$N3_CKPT_503"
fi
echo ""

GPU_PAIRS_STR="${GPU_PAIRS:-0,1,3}"
read -r -a GPU_PAIRS <<< "$GPU_PAIRS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

TEST_ONLY_EXPS=()

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
# All rows: K=8 eigen, bs=64, lr=1e-4, inner n3 frozen (= exp503 ckpt).
# Outer-backbone knobs vary across rows:
#   --backbone {vit,resnet}
#   --freeze-backbone / --no-freeze-backbone
# ============================================================
ALL_EXPS=(
    # --- {ViT, ResNet} × {freeze, finetune} on the outer hourglass ---
    "n9_0426  exp600_n9_0426_vit_freeze_lr1e4_bs64        0.0001  64  --backbone vit     --freeze-backbone     --rep-K 8  --n3-checkpoint $N3_CKPT_503  --freeze-n3  --lambda-sparsity 0.1  --lambda-sh 0.0"
    "n9_0426  exp601_n9_0426_vit_finetune_lr1e4_bs64      0.0001  64  --backbone vit     --no-freeze-backbone  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --freeze-n3  --lambda-sparsity 0.1  --lambda-sh 0.0"
    "n9_0426  exp602_n9_0426_resnet_freeze_lr1e4_bs64     0.0001  64  --backbone resnet  --freeze-backbone     --rep-K 8  --n3-checkpoint $N3_CKPT_503  --freeze-n3  --lambda-sparsity 0.1  --lambda-sh 0.0"
    "n9_0426  exp603_n9_0426_resnet_finetune_lr1e4_bs64   0.0001  64  --backbone resnet  --no-freeze-backbone  --rep-K 8  --n3-checkpoint $N3_CKPT_503  --freeze-n3  --lambda-sparsity 0.1  --lambda-sh 0.0"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache (eigen-8 default — same as n9_bulk).
echo "Pre-warming dataset cache for radial depth on $DATASET_DIR..."
"$PYTHON" -c "
from data.dataset import SoundSpacesDataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/n9_0426.yaml')
cfg.dataset.depth_dir = '$DEPTH_DIR'
if '$DATASET_DIR':
    cfg.dataset.dataset_dir = '$DATASET_DIR'
for split in ['train', 'val', 'test']:
    SoundSpacesDataset(cfg, split=split)
print('Cache warm.')
" 2>&1 | tail -5
echo ""

echo "============================================================"
echo "n9_bulk_0426 — $TOTAL experiments (n9_0426: pretrained outer + n3_0425 inner)"
echo "  EPOCHS=$EPOCHS  depth_dir=$DEPTH_DIR  bs=64  K=8 (eigen-geometric)"
echo "  dataset_dir=${DATASET_DIR:-<yaml default>}"
echo "  $NUM_WORKERS workers × GPUs: ${GPU_PAIRS[*]}"
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
echo "n9_bulk_0426 finished — $(date)"
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

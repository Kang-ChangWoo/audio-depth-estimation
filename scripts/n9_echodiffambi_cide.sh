#!/bin/bash
# ============================================================
# n9_echodiffambi_cide.sh — EchoDiffusionAmbi + CIDE/Wav2Vec2 conditioning,
# 4-cell sweep on node2 (8× A6000), 2 workers × 4 GPUs DataParallel each.
#
# Architecture: EchoDiffusionAmbi with `use_cide=True`. Cross-attention
# context combines:
#   - 1 CIDE token from binaural waveform via Wav2Vec2 → CIDE module
#   - K=8 bin tokens (foa_mode='condition') OR none (foa_mode='input')
#
# 4 experiments (foa_mode × LR), all with CIDE on:
#     exp710  input      lr=5e-4   bs=32
#     exp711  input      lr=1e-4   bs=32
#     exp712  condition  lr=5e-4   bs=32
#     exp713  condition  lr=1e-4   bs=32
#
# These complement the no-CIDE sweep (exp700-707) at matched LR cells, so
# the table comparison reads:
#   exp700 (input, lr=5e-4, no-CIDE)   vs   exp710 (input, lr=5e-4, +CIDE)
#   exp701 (input, lr=1e-4, no-CIDE)   vs   exp711 (input, lr=1e-4, +CIDE)
#   exp704 (cond,  lr=5e-4, no-CIDE)   vs   exp712 (cond,  lr=5e-4, +CIDE)
#   exp705 (cond,  lr=1e-4, no-CIDE)   vs   exp713 (cond,  lr=1e-4, +CIDE)
#
# Compute setup (node2, A6000 ×8, refer n2_bulk):
#   2 parallel workers × 4 GPUs DataParallel each.
#   Worker 0: CUDA_VISIBLE_DEVICES=0,1,2,3
#   Worker 1: CUDA_VISIBLE_DEVICES=4,5,6,7
#   bs=32 → per-GPU batch = 16. CIDE+Wav2Vec2 adds ~94M frozen params
#   to the model; activations dominate. A6000 (48 GiB) holds bs=16/GPU
#   at ~25 GiB/GPU comfortably. Bump to bs=128 if first cell shows headroom.
#
# Logs: logs/echodiff_ambi_cide_{train,test}/.
#
# Usage:
#     bash scripts/n9_echodiffambi_cide.sh
#     EPOCHS=20 bash scripts/n9_echodiffambi_cide.sh    # quick sweep
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303_renew}"
EPOCHS="${EPOCHS:-40}"
WORKERS="${WORKERS:-8}"
VIS_PER_SCENE=0

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/echodiff_ambi_cide_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/echodiff_ambi_cide_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

GPU_PAIRS_STR="${GPU_PAIRS:-0 1 3}"
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
    # SINGLE-GPU TRAIN — multi-GPU NCCL is hopelessly poisoned on this node
    # (Error 2 even with P2P/SHM disabled). Drop to GPU 0 only for both
    # train and test; eliminates DataParallel.replicate() / NCCL entirely.
    # Tradeoff: ~3× slower per cell vs the (broken) 3-GPU DataParallel mode.
    local TRAIN_GPU
    TRAIN_GPU=$(echo "$GPU" | cut -d',' -f1)
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
    # Test runs at bs=4 — single-GPU is plenty fast and sidesteps the
    # multi-GPU NCCL Error 2 we hit earlier. Force GPU 0 only for test.
    local TEST_GPU
    TEST_GPU=$(echo "$GPU" | cut -d',' -f1)
    CUDA_VISIBLE_DEVICES="$TEST_GPU" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
# 4-cell experiment list: "CONFIG  EXP  LR  BS  [EXTRA…]"
# ============================================================
ALL_EXPS=(
    # exp710 DONE 2026-04-27 — ABS_REL=0.4382  RMSE=1.2160  D1=0.4947
    # (single-GPU test fallback after multi-GPU NCCL Error 2; checkpoint at
    #  checkpoints/echodiffusion_ambi_soundspaces_BS32_Lr0.0005_AdamW_exp710_eda_cide_input_lr5e4_bs32/best_model.pth)
    # "echodiffusion_ambi_cide  exp710_eda_cide_input_lr5e4_bs32       0.0005  32  --foa-mode input      --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"
    "echodiffusion_ambi_cide  exp711_eda_cide_input_lr1e4_bs32       0.0001  12  --foa-mode input      --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"
    "echodiffusion_ambi_cide  exp712_eda_cide_cond_lr5e4_bs32        0.0005  12  --foa-mode condition  --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"
    "echodiffusion_ambi_cide  exp713_eda_cide_cond_lr1e4_bs32        0.0001  12  --foa-mode condition  --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache.
echo "Pre-warming dataset cache for radial depth on $DATASET_DIR..."
"$PYTHON" -c "
from data.dataset import SoundSpacesDataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/echodiffusion_ambi_cide.yaml')
cfg.dataset.depth_dir = '$DEPTH_DIR'
if '$DATASET_DIR':
    cfg.dataset.dataset_dir = '$DATASET_DIR'
for split in ['train', 'val', 'test']:
    SoundSpacesDataset(cfg, split=split)
print('Cache warm.')
" 2>&1 | tail -5
echo ""

echo "============================================================"
echo "n9_echodiffambi_cide — $TOTAL experiments (echodiffusion_ambi + CIDE)"
echo "  EPOCHS=$EPOCHS  depth_dir=$DEPTH_DIR  bs=32  K=8 (eigen-geometric)"
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
echo "n9_echodiffambi_cide finished — $(date)"
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

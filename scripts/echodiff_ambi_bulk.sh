#!/bin/bash
# ============================================================
# echodiff_ambi_bulk.sh — EchoDiffusion + bin-gated FOA conditioning sweep.
#
# Architecture: EchoDiffusion's diffusion-UNet feature extractor (ASPP+ASFF
# spec encoder + DiffusionUNet at fixed t=1) but with the CIDE/Wav2Vec2
# branch REPLACED by oracle bin-gated FOA conditioning (n4_0425-style).
# See models/echodiffusion/echodiffusion_ambi.py for the full pipeline.
#
# Two foa_mode options compared head-to-head:
#   'input'      gated_em (1ch) concat with binaural spec → 3ch ASPP input.
#                Cross-attention context is a learnable null token.
#   'condition'  binaural spec stays 2ch; gated rep_gt → K=8 cross-attention
#                tokens of width 768 fed to the diffusion-UNet transformer
#                blocks.
#
# Sweep design — exp700-707 (8 cells):
#   foa_mode × 4 LR cells matched to echodiffusion (exp360-364) baselines:
#     exp700 input      lr=5e-4 bs=32  ← echodiff exp363's best HP cell
#     exp701 input      lr=1e-4 bs=32
#     exp702 input      lr=5e-5 bs=32
#     exp703 input      lr=1e-3 bs=32
#     exp704 condition  lr=5e-4 bs=32
#     exp705 condition  lr=1e-4 bs=32
#     exp706 condition  lr=5e-5 bs=32
#     exp707 condition  lr=1e-3 bs=32
#
# Important caveat (from docs/table.md, 2026-04-26):
#   - The original echodiffusion family has HP-induced std=0.0336 on RMSE.
#   - The n4-vs-baseline gap (oracle FOA helping) was 0.0104 RMSE.
#   - Therefore, any "ambisonic helps EchoDiffusion" gain ≤0.05 RMSE is
#     statistically unidentifiable WITHOUT a seed-variance baseline run on
#     the original echodiff exp363. Treat this sweep as exploratory.
#
# Logs: logs/echodiff_ambi_{train,test}/  (separate from echodiffusion logs).
#
# Usage:
#     bash scripts/echodiff_ambi_bulk.sh
#     GPU_PAIRS="0,1,3" bash scripts/echodiff_ambi_bulk.sh
#     EPOCHS=20 bash scripts/echodiff_ambi_bulk.sh    # quick sweep
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

TRAIN_LOG_DIR="$PROJECT_DIR/logs/echodiff_ambi_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/echodiff_ambi_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

GPU_PAIRS_STR="${GPU_PAIRS:-0,1,3}"
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
# Experiment definitions: "CONFIG  EXP  LR  BS  [EXTRA…]"
# ============================================================
ALL_EXPS=(
    # ----- foa_mode='input' (gated_em concat at spec input) -----
    "echodiffusion_ambi  exp700_eda_input_lr5e4_bs32      0.0005  32  --foa-mode input      --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"
    "echodiffusion_ambi  exp701_eda_input_lr1e4_bs32      0.0001  32  --foa-mode input      --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"
    "echodiffusion_ambi  exp702_eda_input_lr5e5_bs32      0.00005 32  --foa-mode input      --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"
    "echodiffusion_ambi  exp703_eda_input_lr1e3_bs32      0.001   32  --foa-mode input      --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"

    # ----- foa_mode='condition' (cross-attention bin tokens) -----
    "echodiffusion_ambi  exp704_eda_cond_lr5e4_bs32       0.0005  32  --foa-mode condition  --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"
    "echodiffusion_ambi  exp705_eda_cond_lr1e4_bs32       0.0001  32  --foa-mode condition  --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"
    "echodiffusion_ambi  exp706_eda_cond_lr5e5_bs32       0.00005 32  --foa-mode condition  --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"
    "echodiffusion_ambi  exp707_eda_cond_lr1e3_bs32       0.001   32  --foa-mode condition  --rep-K 8  --lambda-sparsity 0.05  --lambda-sh 0.0"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache.
echo "Pre-warming dataset cache for radial depth on $DATASET_DIR..."
"$PYTHON" -c "
from data.dataset import SoundSpacesDataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/echodiffusion_ambi.yaml')
cfg.dataset.depth_dir = '$DEPTH_DIR'
if '$DATASET_DIR':
    cfg.dataset.dataset_dir = '$DATASET_DIR'
for split in ['train', 'val', 'test']:
    SoundSpacesDataset(cfg, split=split)
print('Cache warm.')
" 2>&1 | tail -5
echo ""

echo "============================================================"
echo "echodiff_ambi_bulk — $TOTAL experiments (echodiffusion + bin-gated FOA)"
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
echo "echodiff_ambi_bulk finished — $(date)"
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

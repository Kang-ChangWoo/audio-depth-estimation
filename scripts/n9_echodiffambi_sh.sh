#!/bin/bash
# ============================================================
# n9_echodiffambi_sh.sh — EchoDiffusion + audio→SH coefficient prediction
# + real-SH coarse layout fusion at the depth-decoder output.
#
# Architecture (models/echodiffusion/echodiffusion_ambi_sh.py):
#   spec → ASPP+ASFF → DiffusionUNet (model_channels=64) → aggregator
#                                            │
#                                            ▼
#                                  TaperedDecoder (256→128→64)
#                                            │
#                                            ▼ gated residual fusion
#   rep_gt → MLP → SH coeffs → real SH basis → coarse layout (B, 1, H, W)
#                     │
#                     └─→ + CIDE token (Wav2Vec2) → cross-attn (B, 2, 768)
#
# === Max-capacity variant (cfg.model.use_cide=true, unet_ch=64,
#     decoder_ch=[256,128,64]). Total 167.92M (73.55M trainable +
#     94.4M frozen Wav2Vec2). ===
#
# 2-cell potential-first sweep (LR × fixed sh_order=5, bs=128):
#     exp720  lr=5e-4   bs=128   (echodiffusion's best historical LR)
#     exp721  lr=1e-4   bs=128   (n4_0425's best historical LR)
#
# Deferred (uncomment in ALL_EXPS if either potential cell shows signal):
#     exp722  lr=5e-5   bs=128   (likely too slow)
#     exp723  lr=1e-3   bs=128   (likely unstable)
#
# Compute setup (node2, A6000 ×8, refer n2_bulk):
#   2 parallel workers × 4 GPUs DataParallel each.
#   Worker 0 (GPUs 0,1,2,3) runs exp720; worker 1 (GPUs 4,5,6,7) runs exp721.
#   bs=128 → per-GPU batch = 32. The 73.55M trainable + Wav2Vec2 forward at
#   bs=32/GPU should land ~25-35 GiB on A6000 (48 GiB) — comfortable.
#   If first-cell shows headroom, bump to bs=256. If OOM, drop to bs=64.
#
# Caveat (docs/table.md): echodiffusion family has HP-induced std=0.0336
# on RMSE. SH-prior gain ≤0.05 RMSE is statistically unidentifiable
# without seed-variance baseline. Treat this sweep as exploratory.
#
# Logs: logs/echodiff_ambi_sh_{train,test}/.
#
# Usage:
#     bash scripts/n9_echodiffambi_sh.sh
#     EPOCHS=20 bash scripts/n9_echodiffambi_sh.sh    # quick sweep
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303renew}"
EPOCHS="${EPOCHS:-40}"
WORKERS="${WORKERS:-8}"
VIS_PER_SCENE=0

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/echodiff_ambi_sh_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/echodiff_ambi_sh_test"
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
# Experiment list — start with the 2 most likely-to-work LR cells.
#
# Active cells (potential picks based on prior sweeps):
#   exp720  lr=5e-4  — best LR for echodiffusion family (exp363 RMSE 1.2198)
#   exp721  lr=1e-4  — best LR for n4_0425 family       (exp401-403 RMSE ~1.20)
#
# Deferred cells (commented; uncomment if exp720/721 show signal worth more
# HP exploration):
#   exp722  lr=5e-5  — likely too slow to converge in 40 ep (n4 exp408 was best=1.23)
#   exp723  lr=1e-3  — unstable in n4 sweep (exp407 RMSE=1.27, ABS=0.52)
#
# Keep the LR cell tuple in sync with the bs (=128) → effective per-step
# learning amount = LR × bs.
# ============================================================
ALL_EXPS=(
    "echodiffusion_ambi_sh  exp720_eda_sh5_lr5e4_bs128      0.0005  128"
    "echodiffusion_ambi_sh  exp721_eda_sh5_lr1e4_bs128      0.0001  128"
    "echodiffusion_ambi_sh  exp722_eda_sh5_lr5e5_bs128      0.00005 128"
    "echodiffusion_ambi_sh  exp723_eda_sh5_lr1e3_bs128      0.001   128"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache.
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
echo "n9_echodiffambi_sh — $TOTAL experiments (echodiff + audio→SH coarse-layout)"
echo "  EPOCHS=$EPOCHS  depth_dir=$DEPTH_DIR  bs=128  K=8  sh_order=5 (36 coeffs)  +CIDE  unet_ch=64  dec_ch=[256,128,64]"
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
echo "n9_echodiffambi_sh finished — $(date)"
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

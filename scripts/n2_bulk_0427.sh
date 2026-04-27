#!/bin/bash
# ============================================================
# n2_bulk_0427.sh — EchoDiffusion + SH side-prior (n2_0427) sweep.
# Modeled on n2_echodiffambi_sh.sh (node2, A6000 ×8 layout).
#
# Two model variants live in models/n2_0427/:
#   echodiff_sh_side_plus  ← Plus variant (UNet=64, tapered decoder,
#                            split SH head, SH cross-attn token)
#   echodiff_sh_side       ← concise variant (deferred — see commented
#                            block at the bottom of ALL_EXPS)
#
# === Active sweep: Plus variant only ===
#
# Two cells per the experimental design:
#   exp730  Plus baseline           — side_fusion=False
#                                     pure binaural EchoDiffusion path,
#                                     SH heads compute reps but never
#                                     feed into the depth path.
#   exp731  Plus real-oracle UB     — side_fusion=True
#                                     oracle_mode=True (rep_gt instead of rep_pred)
#                                     oracle_gate_mode='ones' (gate forced to 1s)
#                                     ablation: tells you the absolute upper
#                                     bound when SH side info is perfect.
#
# Compute setup (node2, A6000 ×8, refer n2_bulk):
#   2 parallel workers × 4 GPUs DataParallel each.
#   Worker 0 (GPUs 0,1,2,3) runs exp730 (baseline).
#   Worker 1 (GPUs 4,5,6,7) runs exp731 (oracle UB).
#   bs=128 → per-GPU batch = 32. Plus model is 75.32M trainable
#   (no Wav2Vec2). Should land ~15-20 GiB/GPU on A6000 (48 GiB).
#
# Caveat (docs/table.md): echodiffusion family HP-induced std on RMSE
# is 0.0336. Any "side prior helps" gain ≤0.05 RMSE is not statistically
# distinguishable without seed-variance baseline. Treat as exploratory.
#
# Logs: logs/n2_0427_{train,test}/.
#
# Usage:
#     bash scripts/n2_bulk_0427.sh
#     EPOCHS=20 bash scripts/n2_bulk_0427.sh    # quick sweep
#     GPU_PAIRS="0,1,2,3 4,5,6,7" bash scripts/n2_bulk_0427.sh
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303renew}"
EPOCHS="${EPOCHS:-40}"
# 2 parallel jobs × 4 workers each = 8 dataloader procs total. The previous
# default of 8/job × 2 jobs = 16 saturated the disk pipeline at first-batch
# even on local SSD; halving fixes that without measurable epoch slowdown.
WORKERS="${WORKERS:-4}"
VIS_PER_SCENE=0

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: DATASET_DIR not found: $DATASET_DIR" >&2
    echo "       Set DATASET_DIR=<path> on launch or fix the default above." >&2
    exit 3
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n2_0427_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n2_0427_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

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
# Plus-variant cells (active)
# ============================================================
ALL_EXPS=(
    # Baseline: side_fusion off → pure binaural EchoDiffusion path.
    "echodiff_sh_side_plus  exp730_sideplus_baseline_lr5e4_bs128         0.0005  128  --no-side-fusion --no-oracle-mode  --rep-K 8  --lambda-sh 0.0  --lambda-sparsity 0.0  --lambda-energy 0.0"
    # Real oracle upper bound: side_fusion on, rep_gt fed in, gate forced to ones.
    "echodiff_sh_side_plus  exp731_sideplus_oracleUB_gate-ones_lr5e4_bs128  0.0005  128  --side-fusion --oracle-mode --oracle-gate-mode ones  --rep-K 8  --lambda-sh 0.0  --lambda-sparsity 0.0  --lambda-energy 0.0"

    # ----- DEFERRED: concise echodiff_sh_side cells (uncomment when ready) -----
    # "echodiff_sh_side  exp732_side_baseline_lr5e4_bs128            0.0005  128  --no-side-fusion --no-oracle-mode  --rep-K 8  --lambda-sh 0.0  --lambda-sparsity 0.0  --lambda-energy 0.0"
    # "echodiff_sh_side  exp733_side_oracleUB_lr5e4_bs128             0.0005  128  --side-fusion --oracle-mode  --rep-K 8  --lambda-sh 0.0  --lambda-sparsity 0.0  --lambda-energy 0.0"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache.
echo "Pre-warming dataset cache for radial depth on $DATASET_DIR..."
"$PYTHON" -c "
from data.dataset import SoundSpacesDataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/echodiff_sh_side_plus.yaml')
cfg.dataset.depth_dir = '$DEPTH_DIR'
if '$DATASET_DIR':
    cfg.dataset.dataset_dir = '$DATASET_DIR'
for split in ['train', 'val', 'test']:
    SoundSpacesDataset(cfg, split=split)
print('Cache warm.')
" 2>&1 | tail -5
echo ""

echo "============================================================"
echo "n2_bulk_0427 — $TOTAL experiments (echodiff_sh_side_plus, baseline + oracle UB)"
echo "  EPOCHS=$EPOCHS  depth_dir=$DEPTH_DIR  bs=128  K=8 (eigen-geometric)"
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
echo "n2_bulk_0427 finished — $(date)"
echo "Workers failed: $FAIL / $NUM_WORKERS"
echo "============================================================"
echo ""
printf "%-65s %8s %8s %8s\n" "Experiment" "ABS_REL" "RMSE" "Delta1"
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
            printf "%-65s %8s %8s %8s\n" "$EXP" "$abs" "$rmse" "$d1"
            SUCCESS=$((SUCCESS + 1))
        else
            printf "%-65s %8s\n" "$EXP" "FAILED"
        fi
    else
        printf "%-65s %8s\n" "$EXP" "NO_RESULT"
    fi
done

echo "---------------------------------------------------------------------"
echo "Success: $SUCCESS / $TOTAL"
echo "============================================================"

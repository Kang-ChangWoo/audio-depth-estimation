#!/bin/bash
# ============================================================
# n3_bulk_0425.sh — n3_0425 (binaural → FOA representation) sweep.
#
# Implements `docs/experiment_plan_predict_FOA.md` restricted to
# K ∈ {1, 8}. K=8 uses the geometric distance edges from n9_0424;
# K=1 is a single window over [0, ~2570 samples] = the same total
# round-trip 0→10 m range, just unbinned. 6 experiments — the 2×2
# {RMS, Eigen} × {K=1, K=8} matrix plus 2 Eigen-8 sensitivity checks.
#
# Coverage:
#   exp500  RMS-1      baseline    (lr=1e-4)
#   exp501  RMS-8      baseline    (lr=1e-4)
#   exp502  Eigen-1    baseline    (lr=1e-4, lambda_dir=0.1)
#   exp503  Eigen-8    baseline    (lr=1e-4, lambda_dir=0.1)
#   exp504  Eigen-8    lambda_dir=0  (L1-only sensitivity check)
#   exp505  Eigen-8    lr=5e-4       (LR sensitivity check)
#
# Loss (per _train_step_n3_0425 in train.py):
#     L = L1(pred_rep, rep_gt) + lambda_dir * (1 - cos(pred_rep, rep_gt))
# Cosine direction term is applied for eigen variants only.
#
# Runs on n3 server with 4 workers × 2 GPUs each. Reads from the NFS
# storage path `/root/storage/matterport3d_0303renew`.
#
# Usage:
#     bash scripts/n3_bulk_0425.sh
#     GPU_PAIRS="0,1 2,3 4,5 6,7"   bash scripts/n3_bulk_0425.sh
#     EPOCHS=20                     bash scripts/n3_bulk_0425.sh   # quick sweep
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"      # consumed by data/dataset only for shape;
                                  # n3_0425 does not actually use the depth GT.
# Local1 dataset path on this server (matches renew_single_latter.sh).
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303_renew}"
EPOCHS="${EPOCHS:-20}"
# WORKERS=8 dataloader processes per training job. Single sequential job
# below (all 4 GPUs in DataParallel), so total CPU footprint = 1 main + 8
# dataloader = 9 procs.
WORKERS="${WORKERS:-8}"
VIS_PER_SCENE=0

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n3_0425_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n3_0425_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

# This server has 4 GPUs (0,1,2,3). 1 sequential worker × 4 GPUs each —
# all 4 GPUs share one DataParallel job at a time. Matches the
# renew_single_latter.sh style; lets each experiment use a 4× larger
# effective batch with negligible CPU overhead.
GPU_PAIRS_STR="${GPU_PAIRS:-0,1,2,3}"
read -r -a GPU_PAIRS <<< "$GPU_PAIRS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

train_and_test() {
    local GPU="$1" CONFIG="$2" EXP="$3" LR="$4" BS="$5"
    shift 5
    local EXTRA=("$@")

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
        # n3_0425 reuses depth labels for FOA-pred metrics:
        #   ABS_REL = L1, RMSE = RMSE, Delta1 = mean cosine (NaN for rms).
        abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        d1=$(grep -oP 'Delta1:\s*\K[\-0-9.nan]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        echo "  [GPU=$GPU] $EXP TEST_OK  L1=$abs RMSE=$rmse COS=$d1"
    fi
}

# ============================================================
# Experiment definitions: "CONFIG EXP LR BS [EXTRA…]"
# All rows: K=8 (geometric distance edges from n9_0424), bs=128.
#
# Groups:
#   A — RMS-8 LR sweep              (rep_kind=rms,   no cosine)
#   B — Eigen-8 LR sweep            (rep_kind=eigen, lambda_dir=0.1)
#   C — Eigen-8 lambda_dir sweep    (lr=1e-4)
#   D — Eigen-8 ngf sweep           (lr=1e-4, lambda_dir=0.1)
# ============================================================
ALL_EXPS=(
    # --- Front-loaded baselines (the 2×2 matrix) — bs=512 (4× the per-GPU
    # default; 4 GPUs × bs128/GPU via DataParallel) ---
    # 1. RMS-1   baseline
    "n3_0425  exp500_n3_rms1_lr1e4_bs512         0.0001  512  --rep-kind rms    --rep-K 1"
    # 2. RMS-8   baseline
    "n3_0425  exp501_n3_rms8_lr1e4_bs512         0.0001  512  --rep-kind rms    --rep-K 8"
    # 3. Eigen-1 baseline (lambda_dir=0.1)
    "n3_0425  exp502_n3_eigen1_lr1e4_bs512       0.0001  512  --rep-kind eigen  --rep-K 1  --lambda-dir 0.1"
    # 4. Eigen-8 baseline (lambda_dir=0.1)
    "n3_0425  exp503_n3_eigen8_lr1e4_bs512       0.0001  512  --rep-kind eigen  --rep-K 8  --lambda-dir 0.1"

    # --- Eigen-8 sensitivity checks ---
    "n3_0425  exp504_n3_eigen8_ldir0_lr1e4_bs512 0.0001  512  --rep-kind eigen  --rep-K 8  --lambda-dir 0.0"
    "n3_0425  exp505_n3_eigen8_lr5e4_bs512       0.0005  512  --rep-kind eigen  --rep-K 8  --lambda-dir 0.1"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache (rep_kind/rep_K are cheap per-sample; no cache key
# dependence — calling once with the default config is enough).
echo "Pre-warming dataset cache for radial depth on $DATASET_DIR..."
"$PYTHON" -c "
from data.dataset import SoundSpacesDataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/n3_0425.yaml')
cfg.dataset.depth_dir = '$DEPTH_DIR'
if '$DATASET_DIR':
    cfg.dataset.dataset_dir = '$DATASET_DIR'
for split in ['train', 'val', 'test']:
    SoundSpacesDataset(cfg, split=split)
print('Cache warm.')
" 2>&1 | tail -5
echo ""

echo "============================================================"
echo "n3_bulk_0425 — $TOTAL experiments (n3_0425 binaural→FOA prediction)"
echo "  EPOCHS=$EPOCHS  bs=128  K=8 (geometric edges, n9 layout)"
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
echo "n3_bulk_0425 finished — $(date)"
echo "Workers failed: $FAIL / $NUM_WORKERS"
echo "============================================================"
echo ""
printf "%-50s %10s %10s %10s\n" "Experiment" "L1" "RMSE" "COS"
echo "---------------------------------------------------------------------------"

SUCCESS=0
for SPEC in "${ALL_EXPS[@]}"; do
    EXP=$(echo "$SPEC" | awk '{print $2}')
    LOG="$TEST_LOG_DIR/${EXP}_test.log"
    if [ -f "$LOG" ]; then
        abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$LOG" | head -1)
        rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$LOG" | head -1)
        d1=$(grep -oP 'Delta1:\s*\K[\-0-9.nan]+' "$LOG" | head -1)
        if [ -n "$abs" ]; then
            printf "%-50s %10s %10s %10s\n" "$EXP" "$abs" "$rmse" "$d1"
            SUCCESS=$((SUCCESS + 1))
        else
            printf "%-50s %10s\n" "$EXP" "FAILED"
        fi
    else
        printf "%-50s %10s\n" "$EXP" "NO_RESULT"
    fi
done

echo "---------------------------------------------------------------------------"
echo "Success: $SUCCESS / $TOTAL"
echo "============================================================"

#!/bin/bash
# ============================================================
# n9_bulk_0427.sh — EchoRange follow-up sweep (bs=32, exp907–912)
#
# Round 1 (exp901–906) results are summarised in
# logs/n9_0427_train/E20260428-exp900-906-EchoRange_bin_sigma_ERP_sweep.md
# Best (exp906): 32-bin log + sigma=0.14 + cos-lat + far-mask, ABS_REL 0.4814.
# Open issues from round 1:
#   • our best (bs=16) sits 12 % above original echodiffusion best (bs=32, ABS=0.4300)
#   • our echorange-scalar (exp900) trails echodiffusion-scalar by 0.09 ABS_REL
#   • combined ERP fix helped, but cos-lat / far-mask never split apart
#   • output mode (median vs expectation) was never compared
#
# Round 2 (this sweep, 6 exps @ bs=32) — addresses the four open issues:
#   exp907  bin=32 + cos-lat + far-mask, **bs=32**          ← Q-A: does bs=32 close the echodiff gap?
#   exp908  bin=32 + cos-lat ONLY                           ← Q-B1: cos-lat alone
#   exp909  bin=32 + far-mask ONLY                          ← Q-B2: far-mask alone
#   exp910  bin=32 + cos-lat + far-mask, **median output**  ← Q-C: output mode
#   exp911  echorange-scalar at bs=32                       ← Q-D1: scalar gap, same hp as round 1 best
#   exp912  echodiffusion (original config) at bs=32        ← Q-D2: scalar gap, same env as exp911
#
# All exps fixed at: 20 epochs, AdamW lr=1e-4, range_min_depth=0.1,
# range_max_depth=10.0, log spacing, sigma=0.14 (where range head used),
# lambda_NLL=lambda_BerHu=lambda_SILog=1.0.
#
# Validation: every 2 epochs (cfg.train.validation_iter=2).
# Train logging: standard heartbeat (~5 prints/epoch) + per-epoch
# loss summary; no per-minibatch spam.
#
# Hardware: n9 server, single-GPU per process across 2 GPUs in parallel.
# DataParallel was killed by GPU 2's driver-level fault ("Unable to
# determine the device handle for GPU2"), which broke NVML and made
# NCCL unusable system-wide — even DP[0,1] failed with NCCL Error 2.
# GPU 3 also became unreachable solo (its enumeration depends on GPU 2).
# Only GPUs 0 and 1 work, each as an independent single-GPU process
# (no NCCL involvement). Two workers run in parallel, each pinned to
# its own GPU; experiments are distributed round-robin across them.
#
# Usage:
#     bash scripts/n9_bulk_0427.sh                 # GPUs 0+1 parallel
#     EPOCHS=10 bash scripts/n9_bulk_0427.sh       # quick sweep
#     GPUS="0" bash scripts/n9_bulk_0427.sh        # single GPU 0 only
#     GPUS="1" bash scripts/n9_bulk_0427.sh        # single GPU 1 only
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

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n9_0427_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n9_0427_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

# 2-GPU parallel mode — each worker uses one physical GPU (no NCCL,
# no DataParallel). $GPUS is a space-separated list of GPU indices.
# Defaults to "0 1" (the only working pair on this server).
GPU_PAIRS_STR="${GPUS:-0 1}"
read -r -a GPU_PAIRS <<< "$GPU_PAIRS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

train_one() {
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
}

test_only() {
    local GPUS="$1" CONFIG="$2" EXP="$3"
    shift 3
    local EXTRA=("$@")

    local CKPT
    CKPT=$(find checkpoints/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "  [test-only] $EXP NO_CHECKPOINT — skipping"
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPUS] TEST(only) $EXP"
    local ec=0
    CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 \
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
        echo "  [test-only] $EXP TEST_FAILED (exit=$ec)"
        tail -3 "$TEST_LOG_DIR/${EXP}_test.log" | sed "s/^/      /"
    else
        local abs rmse d1
        abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        echo "  [test-only] $EXP TEST_OK  ABS=$abs RMSE=$rmse D1=$d1"
    fi
}

# ============================================================
# Experiment definitions: "CONFIG  EXP  LR  BS  [EXTRA…]"
# Round 2 sweep — bs=32 single-GPU, six experiments addressing the
# round-1 open issues (see header). Round-1 archived under
# logs/n9_0427_train/E20260428-exp900-906-*.{md,pth}; the round-1
# entries are kept here as commented references for re-test only.
# ============================================================
ALL_EXPS=(
    # ── Round 1 archived (do NOT re-train; use checkpoints under logs/) ──
    # "echorange  exp906_echorange_32log_d10_coslat_farmask_lr1e4_bs16  0.0001  16  --depth-head-type range  --range-num-bins 32  --range-bin-spacing log  --range-min-depth 0.1  --range-max-depth 10.0  --range-soft-label-sigma 0.14  --range-output-mode expectation  --lambda-range-nll 1.0  --lambda-berhu 1.0  --lambda-silog 1.0  --erp-cos-lat-weight  --erp-far-mask"

    # ── Round 2: 6 experiments at bs=32 (sequential 907 → 912). ──

    # exp907  Q-A: bs=32 replicate of round-1 best (exp906) — does the
    # larger batch close the 12 % ABS_REL gap to echodiffusion (0.4300)?
    "echorange      exp907_echorange_32log_d10_coslat_farmask_lr1e4_bs32  0.0001  32  --depth-head-type range  --range-num-bins 32  --range-bin-spacing log  --range-min-depth 0.1  --range-max-depth 10.0  --range-soft-label-sigma 0.14  --range-output-mode expectation  --lambda-range-nll 1.0  --lambda-berhu 1.0  --lambda-silog 1.0  --erp-cos-lat-weight  --erp-far-mask"

    # exp908  Q-B1: bin=32 + cos(lat) ONLY at bs=32. Attributes ERP gain.
    "echorange      exp908_echorange_32log_d10_coslat_lr1e4_bs32          0.0001  32  --depth-head-type range  --range-num-bins 32  --range-bin-spacing log  --range-min-depth 0.1  --range-max-depth 10.0  --range-soft-label-sigma 0.14  --range-output-mode expectation  --lambda-range-nll 1.0  --lambda-berhu 1.0  --lambda-silog 1.0  --erp-cos-lat-weight"

    # exp909  Q-B2: bin=32 + far-mask ONLY at bs=32. Attributes ERP gain.
    "echorange      exp909_echorange_32log_d10_farmask_lr1e4_bs32         0.0001  32  --depth-head-type range  --range-num-bins 32  --range-bin-spacing log  --range-min-depth 0.1  --range-max-depth 10.0  --range-soft-label-sigma 0.14  --range-output-mode expectation  --lambda-range-nll 1.0  --lambda-berhu 1.0  --lambda-silog 1.0  --erp-far-mask"

    # exp910  Q-C: same as exp907 but median output (vs expectation).
    "echorange      exp910_echorange_32log_d10_median_full_lr1e4_bs32     0.0001  32  --depth-head-type range  --range-num-bins 32  --range-bin-spacing log  --range-min-depth 0.1  --range-max-depth 10.0  --range-soft-label-sigma 0.14  --range-output-mode median       --lambda-range-nll 1.0  --lambda-berhu 1.0  --lambda-silog 1.0  --erp-cos-lat-weight  --erp-far-mask"

    # exp911  Q-D1: echorange-scalar at bs=32 — does the scalar gap to
    # echodiffusion (0.43 vs our 0.52) shrink at the same batch size?
    "echorange      exp911_echorange_scalar_lr1e4_bs32                    0.0001  32  --depth-head-type scalar"

    # exp912  Q-D2: echodiffusion (original config) at bs=32 single-GPU.
    # Same env as exp911 → fair scalar comparison. CONFIG is different.
    "echodiffusion  exp912_echodiffusion_lr1e4_bs32                       0.0001  32"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache (single train.py boot is enough for the cache).
echo "Pre-warming dataset cache for radial depth on $DATASET_DIR..."
"$PYTHON" -c "
from data.dataset import SoundSpacesDataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/echorange.yaml')
cfg.dataset.depth_dir = '$DEPTH_DIR'
if '$DATASET_DIR':
    cfg.dataset.dataset_dir = '$DATASET_DIR'
for split in ['train', 'val', 'test']:
    SoundSpacesDataset(cfg, split=split)
print('Cache warm.')
" 2>&1 | tail -5
echo ""

echo "============================================================"
echo "n9_bulk_0427 — $TOTAL experiments (EchoRange bin sweep, 0–10m)"
echo "  EPOCHS=$EPOCHS  bs=64 (effective; ~21/GPU × 3 GPUs DP)"
echo "  dataset_dir=${DATASET_DIR:-<yaml default>}"
echo "  $NUM_WORKERS sequential worker × 3 GPUs: ${GPU_PAIRS[*]}"
echo "  $(date)"
echo "============================================================"

# exp900 was trained earlier and tested separately; commented out so reruns
# of this script don't repeat the test pass.
# test_only "$GPU_PAIRS_STR" "echorange" "exp900_echorange_scalar_lr1e4_bs64" --depth-head-type scalar
# echo ""

run_train_worker() {
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
        train_one "$GPU" "$CONFIG" "$EXP" "$LR" "$BS" "${EXTRA[@]}"
    done
}

run_test_worker() {
    local WORKER_ID=$1
    local GPU="${GPU_PAIRS[$WORKER_ID]}"
    for (( i=WORKER_ID; i<TOTAL; i+=NUM_WORKERS )); do
        local SPEC="${ALL_EXPS[$i]}"
        read -r -a FIELDS <<< "$SPEC"
        local CONFIG="${FIELDS[0]}"
        local EXP="${FIELDS[1]}"
        local EXTRA=("${FIELDS[@]:4}")
        test_only "$GPU" "$CONFIG" "$EXP" "${EXTRA[@]}"
    done
}

# ── Phase 1: TRAIN ALL ───────────────────────────────────────
echo "[Phase 1/2] TRAIN — launching $NUM_WORKERS train worker(s)…"
PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
    run_train_worker "$w" &
    PIDS+=($!)
    echo "  Train worker $w launched (GPU ${GPU_PAIRS[$w]}, PID ${PIDS[-1]})"
done

TRAIN_FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || TRAIN_FAIL=$((TRAIN_FAIL + 1))
done
echo "[Phase 1/2] TRAIN done — failed workers: $TRAIN_FAIL / $NUM_WORKERS"
echo ""

# ── Phase 2: TEST ALL ────────────────────────────────────────
echo "[Phase 2/2] TEST — launching $NUM_WORKERS test worker(s)…"
PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
    run_test_worker "$w" &
    PIDS+=($!)
    echo "  Test worker $w launched (GPU ${GPU_PAIRS[$w]}, PID ${PIDS[-1]})"
done

TEST_FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || TEST_FAIL=$((TEST_FAIL + 1))
done
echo "[Phase 2/2] TEST done — failed workers: $TEST_FAIL / $NUM_WORKERS"

FAIL=$((TRAIN_FAIL + TEST_FAIL))

echo ""
echo "============================================================"
echo "n9_bulk_0427 finished — $(date)"
echo "Train workers failed: $TRAIN_FAIL / $NUM_WORKERS"
echo "Test workers failed:  $TEST_FAIL / $NUM_WORKERS"
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

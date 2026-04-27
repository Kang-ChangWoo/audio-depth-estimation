#!/bin/bash
# ============================================================
# n1_bulk.sh — N1 0420 experiment pipeline (exp241-246)
#
# Cross-group hybrids: N2-style temporal FOA features (early/mid/late
# energy maps + per-bin RMS) fed into the pretrained ViT-B/16 backbone.
# Fills the ViT x temporal cell that was empty in n2_bulk.sh (UNet +
# temporal) and n3_bulk.sh (ViT + N3 hybrids, no temporal).
#
# Design rationale (see report_e §12 discussion and chat 2026-04-20):
#   - ViT pretraining buys ~0.04 ABS_REL headroom over UNet (pvitfoa_v3
#     val=0.3817 vs UNet exp187=0.4728).
#   - N2 temporal maps encode per-bin directional energy -> near-field
#     cues -> ABS_REL-sensitive pixels.
#   - Two orthogonal strengths; the combination is untested.
# Risks: adapter shift (ViT expects 3ch RGB; nc=5 is further from
# pretrain manifold), and UNet-side N2 runs haven't conclusively beaten
# UNet-side baselines (exp192 worse than exp190). Do not burn compute
# on the stacked bet before exp201/202 and exp219 report.
#
# Models (models/n1_4020/pvit_temap.py):
#   pvit_n1_temap_input    — 5ch concat, aux SH head.           (exp241-243)
#   pvit_n1_temap_rms_film — 2ch binaural + 12-dim RMS -> FiLM. (exp244)
#   pvit_n1_temap_eattn    — 5ch concat + 3 energy heads, each
#                            supervised against GT energy map.  (exp245)
#   pvit_n1_temap_mssh     — 5ch concat + multi-scale SH (4 taps). (exp246)
#
# All variants route through the N2 train/val/test path (dataset_n2
# 7-tuple, _train_step_n2). Model-wise they are ViT; data-wise they
# are N2 — registered in _N2_CLASSES / _N1_PVIT_NAMES. sh_dim=12 to
# match the temporal_rms target.
#
# SPEC format: "CONFIG   EXP   LR   LSH   [extra train.py flags]"
#
# Runtime: 6 GPUs (0,1,3,4,5,6 — GPU 2 reserved), 3 workers (2 GPUs
# each). Each worker: train -> test -> next experiment, sequentially.
# ============================================================
set -euo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

# Pin the interpreter explicitly. Honor a caller-supplied $PYTHON, but
# verify up front that it can import everything the training pipeline
# touches — catches silent env-shadowing (tmux venvs, PYTHONPATH leaks,
# wrong conda activation) before burning GPU time.
PYTHON="${PYTHON:-/opt/conda/bin/python3}"
_REQUIRED_MODULES="torch torchvision torchaudio timm scipy numpy yaml einops"
_MISSING=$("$PYTHON" - <<EOF 2>&1
import importlib, sys
missing = []
for m in "$_REQUIRED_MODULES".split():
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}: {e.__class__.__name__}")
print(" ".join(missing) if missing else "OK")
EOF
)
if [ "$_MISSING" != "OK" ]; then
    echo "ERROR: $PYTHON is missing required modules: $_MISSING" >&2
    echo "       (expected: $_REQUIRED_MODULES)" >&2
    echo "       Set PYTHON=/path/to/python that has these, or install them." >&2
    echo "       Known-good interpreter on this box: /opt/conda/bin/python3" >&2
    exit 1
fi
echo "Using PYTHON=$PYTHON"

WORKERS=2              # dataloader workers per train.py
OMP_THREADS=4
EPOCHS=40
BATCH_SIZE=16          # ViT 90M+ params -> BS=16 on 2 GPUs is the v3 baseline
VIS_PER_SCENE=100

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n1_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n1_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

# ---- Train one experiment, then test it immediately ----
train_and_test() {
    local GPU="$1"
    local CONFIG="$2"
    local EXP="$3"
    local LR="$4"
    local LSH="$5"
    shift 5
    local EXTRA=("$@")

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TRAIN $EXP (config=$CONFIG lr=$LR lsh=$LSH extra='${EXTRA[*]}')"

    local TRAIN_EC=0
    CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$OMP_THREADS" \
    OPENBLAS_NUM_THREADS="$OMP_THREADS" NUMEXPR_NUM_THREADS="$OMP_THREADS" \
    "$PYTHON" train.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$WORKERS" \
        --lambda-sh "$LSH" \
        --rotate-canonical \
        "${EXTRA[@]}" \
        > "$TRAIN_LOG_DIR/${EXP}.log" 2>&1 || TRAIN_EC=$?

    if [ "$TRAIN_EC" -ne 0 ] && [ "$TRAIN_EC" -ne 141 ]; then
        local TAIL_ERR=$(tail -3 "$TRAIN_LOG_DIR/${EXP}.log" | tr '\n' ' ')
        echo "  [GPU=$GPU] $EXP TRAIN_FAILED (exit=$TRAIN_EC) | ${TAIL_ERR}"
        return
    fi

    local DONE_LINE=$(grep "^Done\." "$TRAIN_LOG_DIR/${EXP}.log" | tail -1)
    echo "  [GPU=$GPU] $EXP TRAIN_DONE — $DONE_LINE"

    local CKPT_PATH=$(find checkpoints/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
    if [ -z "$CKPT_PATH" ]; then
        echo "  [GPU=$GPU] $EXP TEST_SKIP (no checkpoint)"
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TEST  $EXP"
    local TEST_EC=0
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
        "$PYTHON" -u test.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --checkpoint-path "$CKPT_PATH" \
        --eval-on test \
        --batch-size 1 \
        --vis-per-scene "$VIS_PER_SCENE" \
        --rotate-canonical \
        "${EXTRA[@]}" \
        > "$TEST_LOG_DIR/${EXP}_test.log" 2>&1 || TEST_EC=$?

    if [ "$TEST_EC" -ne 0 ]; then
        echo "  [GPU=$GPU] $EXP TEST_FAILED (exit=$TEST_EC)"
    else
        local ABS=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        local RMSE=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        local D1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        echo "  [GPU=$GPU] $EXP TEST_OK  ABS=$ABS RMSE=$RMSE D1=$D1"
    fi
}

# ---- Test-only: rerun test for an already-trained experiment ----
test_only() {
    local GPU="$1"
    local CONFIG="$2"
    local EXP="$3"

    local CKPT_PATH=$(find checkpoints/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
    if [ -z "$CKPT_PATH" ]; then
        echo "  [GPU=$GPU] $EXP TEST_SKIP (no checkpoint)"
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TEST $EXP (retest)"
    local TEST_EC=0
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
        "$PYTHON" -u test.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --checkpoint-path "$CKPT_PATH" \
        --eval-on test \
        --batch-size 1 \
        --vis-per-scene "$VIS_PER_SCENE" \
        --rotate-canonical \
        > "$TEST_LOG_DIR/${EXP}_test.log" 2>&1 || TEST_EC=$?

    if [ "$TEST_EC" -ne 0 ]; then
        echo "  [GPU=$GPU] $EXP TEST_FAILED (exit=$TEST_EC)"
        tail -3 "$TEST_LOG_DIR/${EXP}_test.log" 2>/dev/null
    else
        local ABS=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        local RMSE=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        local D1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        echo "  [GPU=$GPU] $EXP TEST_OK  ABS=$ABS RMSE=$RMSE D1=$D1"
    fi
}

# 3 GPU-pair workers. GPU 2 is intentionally excluded.
GPU_PAIRS=("0,1" "3,4" "5,6")
NUM_WORKERS=3

# ============================================================
# Phase 1: retest experiments that trained OK but test failed
# (root cause: test_utils.py did not route pvit_n1_* to N2 input
#  assembly — fixed 2026-04-21)
# ============================================================
RETEST_EXPS=(
    "pvit_n1_temap_input  exp241_n1_pvit_temap_lr1e4_lsh0.1"
    "pvit_n1_temap_input  exp242_n1_pvit_temap_lr5e5_lsh0.1"
    "pvit_n1_temap_input  exp243_n1_pvit_temap_lr1e4_lsh0.3"
)

if [ ${#RETEST_EXPS[@]} -gt 0 ]; then
    echo "============================================================"
    echo "Phase 1: retest ${#RETEST_EXPS[@]} experiments (train done, test failed)"
    echo "$(date)"
    echo "============================================================"
    RETEST_PIDS=()
    for i in "${!RETEST_EXPS[@]}"; do
        SPEC="${RETEST_EXPS[$i]}"
        CONFIG=$(echo "$SPEC" | awk '{print $1}')
        EXP=$(echo    "$SPEC" | awk '{print $2}')
        GPU="${GPU_PAIRS[$((i % NUM_WORKERS))]}"
        test_only "$GPU" "$CONFIG" "$EXP" &
        RETEST_PIDS+=($!)
    done
    for pid in "${RETEST_PIDS[@]}"; do wait "$pid" || true; done
    echo "Phase 1 retests finished — $(date)"
    echo ""
fi

# ============================================================
# Phase 2: train + test remaining experiments
# ============================================================
# Experiment definitions — exp241-246.
# Baselines to compare against:
#   ViT exp160-165 (pvitfoav1-v3)    val ABS_REL = 0.3817 (best)
#   UNet exp187 (n2_6ch_input)        ABS=0.4728  RMSE=1.0713  D1=0.5111
#   UNet exp201 (n2_temap_input)      pending — crashed, retraining
#   UNet exp192 (n2_temporal_energy)  ABS=0.4734  RMSE=1.0805  D1=0.4912
# ============================================================
ALL_EXPS=(
    # --- A: ViT + 5ch concat input (DONE train 40/40, retested above) ---
    # "pvit_n1_temap_input      exp241_n1_pvit_temap_lr1e4_lsh0.1     0.0001 0.1"   # DONE 40/40
    # "pvit_n1_temap_input      exp242_n1_pvit_temap_lr5e5_lsh0.1     0.00005 0.1"   # DONE 40/40
    # "pvit_n1_temap_input      exp243_n1_pvit_temap_lr1e4_lsh0.3     0.0001 0.3"    # DONE 40/40
    # --- B: ViT + RMS-FiLM (binaural input, RMS modulates ViT blocks) ---
    "pvit_n1_temap_rms_film   exp244_n1_pvit_trms_film_lr1e4        0.0001 0.1"
    # --- C: ViT + 5ch concat + energy-map prediction + supervision ---
    "pvit_n1_temap_eattn      exp245_n1_pvit_eattn_lr1e4            0.0001 0.1  --lambda-energy 0.1"
    # --- D: ViT + 5ch concat + multi-scale SH ---
    "pvit_n1_temap_mssh       exp246_n1_pvit_mssh_lr1e4             0.0001 0.1"
)

TOTAL=${#ALL_EXPS[@]}

echo "============================================================"
echo "n1_bulk.sh — $TOTAL experiments, 3 workers (GPUs 0,1,3,4,5,6), BS=$BATCH_SIZE"
echo "$(date)"
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
        local LSH="${FIELDS[3]}"
        local EXTRA=("${FIELDS[@]:4}")

        train_and_test "$GPU" "$CONFIG" "$EXP" "$LR" "$LSH" "${EXTRA[@]}"
    done
}

PIDS=()
for w in 0 1 2; do
    run_worker $w &
    PIDS+=($!)
    echo "Worker $w launched (GPU ${GPU_PAIRS[$w]}, PID ${PIDS[-1]})"
done

echo "All 3 workers running. Waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "n1_bulk.sh finished — $(date)"
echo "Workers failed: $FAIL / $NUM_WORKERS"
echo "============================================================"
echo ""
printf "%-50s %8s %8s %8s\n" "Experiment" "ABS_REL" "RMSE" "Delta1"
echo "---------------------------------------------------------------------"

SUCCESS=0
FAILURES=0
MISSING=0

for SPEC in "${ALL_EXPS[@]}"; do
    EXP=$(echo "$SPEC" | awk '{print $2}')
    LOG="$TEST_LOG_DIR/${EXP}_test.log"
    if [ -f "$LOG" ]; then
        ABS=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$LOG" | head -1)
        RMSE=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$LOG" | head -1)
        D1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$LOG" | head -1)
        if [ -n "$ABS" ]; then
            printf "%-50s %8s %8s %8s\n" "$EXP" "$ABS" "$RMSE" "$D1"
            SUCCESS=$((SUCCESS + 1))
        else
            printf "%-50s %8s\n" "$EXP" "FAILED"
            FAILURES=$((FAILURES + 1))
        fi
    else
        printf "%-50s %8s\n" "$EXP" "NO_RESULT"
        MISSING=$((MISSING + 1))
    fi
done

echo "---------------------------------------------------------------------"
echo "Success: $SUCCESS  Failed: $FAILURES  No result: $MISSING  Total: $TOTAL"
echo "============================================================"

#!/bin/bash
# ============================================================
# summary_test_tbd.sh — Experiments that need testing
#
# Two categories:
#   1. Trained (best_model.pth exists) but NEVER tested
#   2. Previously tested but FAILED (checkpoint load issues)
#
# Usage: GPUS="0,1" bash summary_test_tbd.sh [GROUP]
#   GROUP: all (default), foa0415, foa0415_cw_fire, failed_variants
# ============================================================
set -euo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

GPUS="${GPUS:-0,1}"
GROUP="${1:-all}"
VIS_PER_SCENE="${VIS_PER_SCENE:-100}"

LOG_DIR="$PROJECT_DIR/logs/summary_test"
mkdir -p "$LOG_DIR"

run_test() {
    local CONFIG="$1"
    local EXP="$2"
    local CKPT_PATH="$3"

    if [ ! -f "$CKPT_PATH" ]; then
        echo "  SKIP: $EXP — checkpoint not found ($CKPT_PATH)"
        return
    fi

    echo "[$(date +%H:%M:%S)] Testing $EXP [$CONFIG]"

    CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        python3 -u test.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --checkpoint-path "$CKPT_PATH" \
        --eval-on test \
        --batch-size 1 \
        --vis-per-scene "$VIS_PER_SCENE" \
        > "$LOG_DIR/${EXP}_test.log" 2>&1

    local EC=$?
    if [ "$EC" -ne 0 ]; then
        echo "  $EXP FAILED (exit=$EC)"
    else
        local ABS=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$LOG_DIR/${EXP}_test.log" | head -1)
        local RMSE=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$LOG_DIR/${EXP}_test.log" | head -1)
        echo "  $EXP OK (ABS=$ABS RMSE=$RMSE)"
    fi
}

# Helper to find checkpoint path by experiment name pattern
find_ckpt() {
    local PATTERN="$1"
    local FOUND=$(find checkpoints/ -name "best_model.pth" -path "*${PATTERN}*" | head -1)
    echo "$FOUND"
}

# ============================================================
# GROUP: foa0415 — 20 FOA 0415 experiments never tested
#   (exp130-154, minus the 5 tested in CW_fire: 130,135,140,145,150)
# ============================================================
test_foa0415_tbd() {
    echo "=== FOA 0415 untested (20 experiments) ==="
    local VARIANTS=("foa_0415_v1:foa0415v1" "foa_0415_v2:foa0415v2" "foa_0415_v3:foa0415v3" "foa_0415_v4:foa0415v4" "foa_0415_v5:foa0415v5")
    local SETTINGS=("1e3 0.1" "1e3 0.3" "5e4 0.1" "5e4 0.5" "1e4 0.1")
    local BASE=130
    # Already tested: exp130,135,140,145,150 (lr1e3_lsh0.1 for each variant)
    local SKIP_EXPS="130 135 140 145 150"

    for v in "${!VARIANTS[@]}"; do
        IFS=: read -r CONFIG SHORT <<< "${VARIANTS[$v]}"
        for s in "${!SETTINGS[@]}"; do
            read -r LR_TAG LSH <<< "${SETTINGS[$s]}"
            IDX=$((BASE + v * 5 + s))
            # Skip already tested
            echo "$SKIP_EXPS" | grep -qw "$IDX" && continue
            EXP="exp${IDX}_${SHORT}_lr${LR_TAG}_lsh${LSH}"
            CKPT=$(find_ckpt "exp${IDX}_${SHORT}")
            [ -n "$CKPT" ] && run_test "$CONFIG" "$EXP" "$CKPT"
        done
    done
}

# ============================================================
# GROUP: failed_variants — 25 experiments that FAILED test (no KL)
#   These need debugging — the checkpoint loads fail due to
#   DataParallel/naming issues. Try with strict=False.
# ============================================================
test_failed_variants_tbd() {
    echo "=== Previously failed variants (25 experiments, no KL) ==="
    echo "NOTE: These failed in bulk0410 due to checkpoint loading issues."
    echo "Try re-testing — the test.py may have been fixed since then."

    # CrossAttn no KL (5)
    for i in 16 17 18 19 20; do
        CKPT=$(find_ckpt "exp${i}_crossattn")
        [ -n "$CKPT" ] && run_test foa_crossattn "exp${i}_crossattn_retest" "$CKPT"
    done
    # FeatBank no KL (5)
    for i in 21 22 23 24 25; do
        CKPT=$(find_ckpt "exp${i}_featbank")
        [ -n "$CKPT" ] && run_test foa_featbank "exp${i}_featbank_retest" "$CKPT"
    done
    # MSAttn no KL (5)
    for i in 26 27 28 29 30; do
        CKPT=$(find_ckpt "exp${i}_msattn")
        [ -n "$CKPT" ] && run_test foa_msattn "exp${i}_msattn_retest" "$CKPT"
    done
    # ChannelAttn no KL (5)
    for i in 31 32 33 34 35; do
        CKPT=$(find_ckpt "exp${i}_channelattn")
        [ -n "$CKPT" ] && run_test foa_channelattn "exp${i}_channelattn_retest" "$CKPT"
    done
    # FOA v2 (5)
    for i in 56 57 58 59 60; do
        CKPT=$(find_ckpt "exp${i}_foav2")
        [ -n "$CKPT" ] && run_test foa_v2 "exp${i}_foav2_retest" "$CKPT"
    done
}

# ============================================================
# DISPATCHER
# ============================================================
case "$GROUP" in
    all)           test_foa0415_tbd; test_failed_variants_tbd ;;
    foa0415)       test_foa0415_tbd ;;
    failed_variants) test_failed_variants_tbd ;;
    *)             echo "Unknown group: $GROUP"; exit 1 ;;
esac

echo "============================================================"
echo "summary_test_tbd.sh ($GROUP) finished — $(date)"
echo "============================================================"

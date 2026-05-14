#!/bin/bash
# ============================================================
# summary_train_tbd.sh — Unfinished training experiments to complete
#
# These experiments either:
#   - Stalled mid-training (have checkpoint dir but incomplete)
#   - Never started (killed before launch in bulk0408)
#   - Have no best_model.pth
#
# Usage: GPUS="0,1" bash summary_train_tbd.sh [GROUP]
#   GROUP: all (default), echonet, crossattn, foa_missing, pvitvoa
# ============================================================
set -euo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

GPUS="${GPUS:-0,1}"
GROUP="${1:-all}"
WORKERS="${WORKERS:-8}"

run_train() {
    local CONFIG="$1"
    local EXP="$2"
    local EPOCHS="$3"
    local LR="$4"
    local BS="$5"
    shift 5
    local EXTRA_ARGS="$@"

    echo "[$(date +%H:%M:%S)] Training $EXP (config=$CONFIG, lr=$LR, bs=$BS, epochs=$EPOCHS)"

    CUDA_VISIBLE_DEVICES="$GPUS" python3 train.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --epochs "$EPOCHS" \
        --learning-rate "$LR" \
        --batch-size "$BS" \
        --num-workers "$WORKERS" \
        $EXTRA_ARGS \
        2>&1 | tee "logs/summary_train/${EXP}.log"
}

mkdir -p logs/summary_train

# ============================================================
# GROUP: echonet — 4 stalled experiments (17-20/40 epochs)
# ============================================================
train_echonet_tbd() {
    echo "=== EchoNet: 4 experiments stalled at 17-20 epochs ==="
    run_train echonet exp66_echonet_lr1e3_bs8  40 0.001  8
    run_train echonet exp67_echonet_lr5e4_bs16 40 0.0005 16
    run_train echonet exp68_echonet_lr1e4_bs16 40 0.0001 16
    run_train echonet exp69_echonet_lr1e3_bs16 40 0.001  16
}

# ============================================================
# GROUP: crossattn — 1 stalled experiment (4/40 epochs)
# ============================================================
train_crossattn_tbd() {
    echo "=== CrossAttn: exp78 stalled at 4/40 epochs ==="
    run_train foa_crossattn exp78_crossattn_lr1e3_fw0.05_kl0.01 40 0.001 32
}

# ============================================================
# GROUP: foa_missing — 10 never-started FOA experiments (bulk0408 killed)
# ============================================================
train_foa_missing() {
    echo "=== FOA missing: 10 experiments never started ==="
    run_train foa             exp98_foa_lr7e4_dw1.0_fw0.1_hw0.1    40 0.0007 32
    run_train foa             exp102_foa_lr5e4_dw1.5_fw0.1_hw0.1   40 0.0005 32
    run_train foa             exp106_foa_lr1e3_fw0.1_hw0.3          40 0.001  32
    run_train foa             exp110_foa_lr5e4_fw0.2_hw0.1_freeze5  40 0.0005 32
    run_train foa             exp114_foa_lr5e4_dw0.5_fw0.1_hw0.2   40 0.0005 32
    run_train foa             exp118_foa_lr7e4_fw0.15_hw0.15        40 0.0007 32
    run_train foa_featbank    exp82_featbank_lr5e4_fw0.2_kl0.005    40 0.0005 32
    run_train foa_msattn      exp86_msattn_lr1e3_fw0.1_kl0.02      40 0.001  32
    run_train foa_msattn      exp90_msattn_lr1e3_fw0.3_kl0.01      40 0.001  32
    run_train foa_channelattn exp94_channelattn_lr5e4_fw0.1_kl0.01  40 0.0005 32
}

# ============================================================
# GROUP: pvitvoa — deprecated in Phase H2 (v4/v5 moved to models/deprecated/)
# ============================================================
train_pvitvoa_tbd() {
    echo "=== PreViT+FOA: v4/v5 deprecated in Phase H2 (no longer scheduled) ==="
}

# ============================================================
# DISPATCHER
# ============================================================
case "$GROUP" in
    all)
        train_echonet_tbd; train_crossattn_tbd; train_foa_missing; train_pvitvoa_tbd
        ;;
    echonet)    train_echonet_tbd ;;
    crossattn)  train_crossattn_tbd ;;
    foa_missing) train_foa_missing ;;
    pvitvoa)    train_pvitvoa_tbd ;;
    *)          echo "Unknown group: $GROUP"; exit 1 ;;
esac

echo "============================================================"
echo "summary_train_tbd.sh ($GROUP) finished — $(date)"
echo "Total TBD experiments: 19"
echo "============================================================"

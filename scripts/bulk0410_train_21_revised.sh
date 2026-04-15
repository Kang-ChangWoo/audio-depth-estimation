#!/bin/bash
# ============================================================
# bulk0410_sum_21_revised — reruns for failed / partial / never-ran
#
# Archived companion to bulk0410_sum_41.sh. Contains:
#
#   A. Partial training runs that did NOT reach 40/40 epochs
#      (killed by the scheduler mid-run):
#        exp66–69  echonet            (17–20/40 epochs)
#        exp78     foa_crossattn      (4/40 epochs)
#
#   B. Experiments from bulk0408_65exps.sh that never started
#      (training process was killed before reaching them):
#        exp82     foa_featbank
#        exp86,90  foa_msattn
#        exp94     foa_channelattn
#        exp98,102,106,110,114,118  foa (wider search)
#
#   C. Reruns moved verbatim from the now-removed
#      bulk0410_failedexps.sh (originally failed on their first
#      attempt in bulk0407_revised / bulk0408_5):
#        exp57,59,60 foa_v2
#        exp121,123,125 echodiffusion (wav2vec finetune)
#
# Purpose: when this experimental grid is applied to a different
# dataset, this script is the "retry bucket" — anything that
# failed on the original SoundSpaces pass and therefore has a
# higher a-priori risk of failing again.
#
# Original experiment indices are preserved exactly.
# 4 persistent workers × 2 GPUs each = 8 GPUs (0-7)
# ============================================================

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/train.py" ]; then
    echo "ERROR: train.py not found in $PROJECT_DIR"
    return 1 2>/dev/null || exit 1
fi

EPOCHS=40
NUM_WORKERS=4
LOG_DIR="$PROJECT_DIR/logs/bulk0410_sum_revised"
mkdir -p "$LOG_DIR"
TOTAL=21

echo "============================================================"
echo "bulk0410_sum_21_revised — 21 rerun experiments — $(date)"
echo "4 workers × 2 GPUs each (GPUs 0-7)"
echo "Epochs: $EPOCHS | Workers: $NUM_WORKERS"
echo "Logs: $LOG_DIR"
echo "============================================================"
echo ""

run_exp() {
    local GPUS="$1"; shift
    local EXP_IDX="$1"; shift
    local CONFIG="$1"; shift
    local EXP_NAME="$1"; shift

    echo "[$(date +%H:%M:%S)] [GPU=$GPUS] Exp $EXP_IDX/$TOTAL: $EXP_NAME"

    cd "$PROJECT_DIR"
    CUDA_VISIBLE_DEVICES="$GPUS" python3 "$PROJECT_DIR/train.py" \
        --config "$CONFIG" \
        --experiment-name "$EXP_NAME" \
        --epochs "$EPOCHS" \
        --num-workers "$NUM_WORKERS" \
        "$@" \
        > "$LOG_DIR/${EXP_NAME}.log" 2>&1

    local EC=$?
    if [ "$EC" -ne 0 ]; then
        echo "  [GPU=$GPUS] $EXP_NAME FAILED (exit=$EC)"
    else
        echo "  [GPU=$GPUS] $EXP_NAME OK"
    fi
}

# ════════════════════════════════════════════════════════════
# Worker 0: GPUs 0,1 — 6 experiments
# ════════════════════════════════════════════════════════════
worker_0() {
    local G="0,1"
    # Partial echonet
    run_exp $G 66 echonet exp66_echonet_lr1e3_bs8 --lr 0.001 --batch-size 8
    # Partial crossattn KL
    run_exp $G 78 foa_crossattn exp78_crossattn_lr1e3_fw0.05_kl0.01 --lr 0.001 --foa-weight 0.05 --kl-weight 0.01
    # Never-ran msattn KL
    run_exp $G 90 foa_msattn exp90_msattn_lr1e3_fw0.3_kl0.01 --lr 0.001 --foa-weight 0.3 --kl-weight 0.01
    # Never-ran foa (wider search)
    run_exp $G 106 foa exp106_foa_lr1e3_fw0.1_hw0.3 --lr 0.001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.3
    # Moved from bulk0410_failedexps.sh
    run_exp $G 57 foa_v2 exp57_foav2_lr5e4_dw1.0_fw0.1_hw0.1 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 123 echodiffusion exp123_echodiff_wav2vec_lr1e4_bs32 --lr 0.0001 --batch-size 32
}

# ════════════════════════════════════════════════════════════
# Worker 1: GPUs 2,3 — 5 experiments
# ════════════════════════════════════════════════════════════
worker_1() {
    local G="2,3"
    # Partial echonet
    run_exp $G 67 echonet exp67_echonet_lr5e4_bs16 --lr 0.0005 --batch-size 16
    # Never-ran featbank KL
    run_exp $G 82 foa_featbank exp82_featbank_lr5e4_fw0.2_kl0.005 --lr 0.0005 --foa-weight 0.2 --kl-weight 0.005
    # Never-ran channelattn KL
    run_exp $G 94 foa_channelattn exp94_channelattn_lr5e4_fw0.1_hw0.2_kl0.01 --lr 0.0005 --foa-weight 0.1 --hist-weight 0.2 --kl-weight 0.01
    # Never-ran foa (wider search)
    run_exp $G 110 foa exp110_foa_lr5e4_fw0.2_hw0.1_freeze5 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.1 --foa-freeze-epochs 5
    # Moved from bulk0410_failedexps.sh
    run_exp $G 59 foa_v2 exp59_foav2_lr1e3_dw1.0_fw0.2_hw0.1 --lr 0.001 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.1
}

# ════════════════════════════════════════════════════════════
# Worker 2: GPUs 4,5 — 5 experiments
# ════════════════════════════════════════════════════════════
worker_2() {
    local G="4,5"
    # Partial echonet
    run_exp $G 68 echonet exp68_echonet_lr1e4_bs16 --lr 0.0001 --batch-size 16
    # Never-ran msattn KL
    run_exp $G 86 foa_msattn exp86_msattn_lr1e3_fw0.1_kl0.02 --lr 0.001 --foa-weight 0.1 --kl-weight 0.02
    # Never-ran foa (wider search)
    run_exp $G 98 foa exp98_foa_lr7e4_dw1.0_fw0.1_hw0.1 --lr 0.0007 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 114 foa exp114_foa_lr5e4_dw0.5_fw0.1_hw0.2 --lr 0.0005 --depth-weight 0.5 --foa-weight 0.1 --hist-weight 0.2
    # Moved from bulk0410_failedexps.sh
    run_exp $G 60 foa_v2 exp60_foav2_lr5e4_dw1.0_fw0.2_hw0.2 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.2
}

# ════════════════════════════════════════════════════════════
# Worker 3: GPUs 6,7 — 5 experiments
# ════════════════════════════════════════════════════════════
worker_3() {
    local G="6,7"
    # Partial echonet
    run_exp $G 69 echonet exp69_echonet_lr1e3_bs16 --lr 0.001 --batch-size 16
    # Never-ran foa (wider search)
    run_exp $G 102 foa exp102_foa_lr5e4_dw1.5_fw0.1_hw0.1 --lr 0.0005 --depth-weight 1.5 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 118 foa exp118_foa_lr7e4_fw0.15_hw0.15 --lr 0.0007 --depth-weight 1.0 --foa-weight 0.15 --hist-weight 0.15
    # Moved from bulk0410_failedexps.sh
    run_exp $G 121 echodiffusion exp121_echodiff_wav2vec_lr1e4_bs16 --lr 0.0001 --batch-size 16
    run_exp $G 125 echodiffusion exp125_echodiff_wav2vec_lr1e4_bs8 --lr 0.0001 --batch-size 8
}

# ── Launch all 4 workers in background ─────────────────────
worker_0 &
W0=$!
worker_1 &
W1=$!
worker_2 &
W2=$!
worker_3 &
W3=$!

echo "Workers launched: W0=$W0(GPU0,1) W1=$W1(GPU2,3) W2=$W2(GPU4,5) W3=$W3(GPU6,7)"
echo ""

wait $W0 $W1 $W2 $W3 2>/dev/null

# ── Results summary ─────────────────────────────────────────
echo ""
echo "============================================================"
echo "bulk0410_sum_21_revised finished — $(date)"
echo "============================================================"
echo ""
printf "%-4s %-55s %8s %8s %8s\n" "Exp" "Name" "RMSE" "ABS_REL" "Score"
echo "-------------------------------------------------------------------"

SUCCESS=0
FAILURES=0
for name in \
    exp66_echonet_lr1e3_bs8 exp67_echonet_lr5e4_bs16 \
    exp68_echonet_lr1e4_bs16 exp69_echonet_lr1e3_bs16 \
    exp78_crossattn_lr1e3_fw0.05_kl0.01 \
    exp82_featbank_lr5e4_fw0.2_kl0.005 \
    exp86_msattn_lr1e3_fw0.1_kl0.02 exp90_msattn_lr1e3_fw0.3_kl0.01 \
    exp94_channelattn_lr5e4_fw0.1_hw0.2_kl0.01 \
    exp98_foa_lr7e4_dw1.0_fw0.1_hw0.1 exp102_foa_lr5e4_dw1.5_fw0.1_hw0.1 \
    exp106_foa_lr1e3_fw0.1_hw0.3 exp110_foa_lr5e4_fw0.2_hw0.1_freeze5 \
    exp114_foa_lr5e4_dw0.5_fw0.1_hw0.2 exp118_foa_lr7e4_fw0.15_hw0.15 \
    exp57_foav2_lr5e4_dw1.0_fw0.1_hw0.1 exp59_foav2_lr1e3_dw1.0_fw0.2_hw0.1 \
    exp60_foav2_lr5e4_dw1.0_fw0.2_hw0.2 \
    exp121_echodiff_wav2vec_lr1e4_bs16 exp123_echodiff_wav2vec_lr1e4_bs32 \
    exp125_echodiff_wav2vec_lr1e4_bs8; do

    LOG="$LOG_DIR/${name}.log"
    IDX=$(echo "$name" | grep -oP '^exp\K\d+')
    if [ ! -f "$LOG" ]; then
        printf "%-4s %-55s %8s\n" "$IDX" "$name" "MISSING"
        FAILURES=$((FAILURES + 1))
        continue
    fi
    BEST=$(grep ">> Best" "$LOG" 2>/dev/null | tail -1)
    if [ -n "$BEST" ]; then
        SCORE=$(echo "$BEST" | grep -oP 'score:\K[0-9.]+' || echo "N/A")
        RMSE=$(echo "$BEST" | grep -oP 'RMSE:\K[0-9.]+' || echo "N/A")
        ABS=$(echo "$BEST" | grep -oP 'ABS:\K[0-9.]+' || echo "N/A")
        printf "%-4s %-55s %8s %8s %8s\n" "$IDX" "$name" "$RMSE" "$ABS" "$SCORE"
        SUCCESS=$((SUCCESS + 1))
    else
        ERR=0
        grep -q "Traceback\|Error" "$LOG" 2>/dev/null && ERR=1
        if [ "$ERR" -eq 1 ]; then
            printf "%-4s %-55s %8s\n" "$IDX" "$name" "FAILED"
        else
            printf "%-4s %-55s %8s\n" "$IDX" "$name" "NO_RESULT"
        fi
        FAILURES=$((FAILURES + 1))
    fi
done

echo "-------------------------------------------------------------------"
echo "Success: $SUCCESS  Failed: $FAILURES  Total: $TOTAL"
echo "============================================================"

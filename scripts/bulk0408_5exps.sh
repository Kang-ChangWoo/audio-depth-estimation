#!/bin/bash
# ============================================================
# 5 EchoDiffusion finetuning experiments (with real Wav2Vec2)
# Indices 121-125, 3 workers × 2 GPUs (GPUs 4,5 / 6,7 / 8,9)
# ============================================================

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/train.py" ]; then
    echo "ERROR: train.py not found in $PROJECT_DIR"
    return 1 2>/dev/null || exit 1
fi

EPOCHS=40
NUM_WORKERS=4
LOG_DIR="$PROJECT_DIR/logs/bulk0408"
mkdir -p "$LOG_DIR"
TOTAL=5

echo "============================================================"
echo "5 EchoDiffusion finetuning (Wav2Vec2) — $(date)"
echo "Indices 121-125 | GPUs 4,5 + 6,7 + 8,9 | Epochs: $EPOCHS"
echo "Logs: $LOG_DIR"
echo "============================================================"
echo ""

run_exp() {
    local GPUS="$1"; shift
    local EXP_IDX="$1"; shift
    local CONFIG="$1"; shift
    local EXP_NAME="$1"; shift

    echo "[$(date +%H:%M:%S)] [GPU=$GPUS] Exp $EXP_IDX: $EXP_NAME"

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

# ── Worker 0: GPUs 4,5 — 2 experiments ──
worker_0() {
    local G="4,5"
    run_exp $G 121 echodiffusion exp121_echodiff_wav2vec_lr1e4_bs16 --lr 0.0001 --batch-size 16
    run_exp $G 122 echodiffusion exp122_echodiff_wav2vec_lr5e4_bs16 --lr 0.0005 --batch-size 16
}

# ── Worker 1: GPUs 6,7 — 2 experiments ──
worker_1() {
    local G="6,7"
    run_exp $G 123 echodiffusion exp123_echodiff_wav2vec_lr1e4_bs32 --lr 0.0001 --batch-size 32
    run_exp $G 124 echodiffusion exp124_echodiff_wav2vec_lr5e5_bs16 --lr 0.00005 --batch-size 16
}

# ── Worker 2: GPUs 8,9 — 1 experiment ──
worker_2() {
    local G="8,9"
    run_exp $G 125 echodiffusion exp125_echodiff_wav2vec_lr1e4_bs8 --lr 0.0001 --batch-size 8
}

# ── Launch workers ──────────────────────────────────────────
worker_0 &
W0=$!
worker_1 &
W1=$!
worker_2 &
W2=$!

echo "Workers: W0=$W0(GPU4,5) W1=$W1(GPU6,7) W2=$W2(GPU8,9)"
echo ""

wait $W0 $W1 $W2 2>/dev/null

# ── Results ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "EchoDiffusion finetuning done — $(date)"
echo "============================================================"
echo ""
printf "%-4s %-45s %8s %8s %8s\n" "Exp" "Name" "RMSE" "ABS_REL" "Score"
echo "--------------------------------------------------------------"

for i in 121 122 123 124 125; do
    LOG=$(ls "$LOG_DIR"/exp${i}_*.log 2>/dev/null | head -1)
    if [ -z "$LOG" ]; then
        printf "%-4s %-45s %8s\n" "$i" "(no log)" "MISSING"
        continue
    fi
    NAME=$(basename "$LOG" .log)
    BEST=$(grep ">> Best" "$LOG" 2>/dev/null | tail -1)
    if [ -n "$BEST" ]; then
        SCORE=$(echo "$BEST" | grep -oP 'score:\K[0-9.]+' || echo "N/A")
        RMSE=$(echo "$BEST" | grep -oP 'RMSE:\K[0-9.]+' || echo "N/A")
        ABS=$(echo "$BEST" | grep -oP 'ABS:\K[0-9.]+' || echo "N/A")
        printf "%-4s %-45s %8s %8s %8s\n" "$i" "$NAME" "$RMSE" "$ABS" "$SCORE"
    else
        printf "%-4s %-45s %8s\n" "$i" "$NAME" "FAILED"
    fi
done
echo "--------------------------------------------------------------"

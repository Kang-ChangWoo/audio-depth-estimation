#!/bin/bash
# ============================================================
# REVISED: Rerun only failed experiments + add foa_v2 trials
# 27 failed reruns (same indices) + 5 new foa_v2 (indices 56-60)
# = 32 experiments total
# 4 workers × 2 GPUs each = GPUs 0-7
# ============================================================

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/train.py" ]; then
    echo "ERROR: train.py not found in $PROJECT_DIR"
    return 1 2>/dev/null || exit 1
fi

EPOCHS=40
NUM_WORKERS=4
LOG_DIR="$PROJECT_DIR/logs/bulk0407"
mkdir -p "$LOG_DIR"
TOTAL=32

echo "============================================================"
echo "REVISED 55exps — Rerun 27 failed + 5 new foa_v2 — $(date)"
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

# ── Worker 0: GPUs 0,1 — 8 experiments ──
worker_0() {
    local G="0,1"
    # FOA variant reruns
    run_exp $G 16 foa_crossattn exp16_crossattn_lr1e3_fw0.1 --lr 0.001 --foa-weight 0.1
    run_exp $G 21 foa_featbank exp21_featbank_lr1e3_fw0.1 --lr 0.001 --foa-weight 0.1
    run_exp $G 26 foa_msattn exp26_msattn_lr1e3_fw0.1 --lr 0.001 --foa-weight 0.1
    run_exp $G 31 foa_channelattn exp31_channelattn_lr1e3_fw0.1 --lr 0.001 --foa-weight 0.1
    # FOA original reruns (import-error failures)
    run_exp $G 43 foa exp43_foa_lr1e3_dw1.0_fw0.1_hw0.2 --lr 0.001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.2
    run_exp $G 51 foa exp51_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze10 --lr 0.001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1 --foa-freeze-epochs 10
    # New foa_v2
    run_exp $G 56 foa_v2 exp56_foav2_lr1e3_dw1.0_fw0.1_hw0.1 --lr 0.001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 57 foa_v2 exp57_foav2_lr5e4_dw1.0_fw0.1_hw0.1 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
}

# ── Worker 1: GPUs 2,3 — 8 experiments ──
worker_1() {
    local G="2,3"
    run_exp $G 17 foa_crossattn exp17_crossattn_lr5e4_fw0.1 --lr 0.0005 --foa-weight 0.1
    run_exp $G 22 foa_featbank exp22_featbank_lr5e4_fw0.1 --lr 0.0005 --foa-weight 0.1
    run_exp $G 27 foa_msattn exp27_msattn_lr5e4_fw0.1 --lr 0.0005 --foa-weight 0.1
    run_exp $G 32 foa_channelattn exp32_channelattn_lr5e4_fw0.1 --lr 0.0005 --foa-weight 0.1
    run_exp $G 48 foa exp48_foa_lr1e3_dw1.0_fw0.05_hw0.1 --lr 0.001 --depth-weight 1.0 --foa-weight 0.05 --hist-weight 0.1
    run_exp $G 52 foa exp52_foa_lr5e4_dw1.0_fw0.2_hw0.2 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.2
    run_exp $G 58 foa_v2 exp58_foav2_lr1e4_dw1.0_fw0.1_hw0.1 --lr 0.0001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 59 foa_v2 exp59_foav2_lr1e3_dw1.0_fw0.2_hw0.1 --lr 0.001 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.1
}

# ── Worker 2: GPUs 4,5 — 8 experiments ──
worker_2() {
    local G="4,5"
    run_exp $G 18 foa_crossattn exp18_crossattn_lr1e4_fw0.1 --lr 0.0001 --foa-weight 0.1
    run_exp $G 19 foa_crossattn exp19_crossattn_lr1e3_fw0.2 --lr 0.001 --foa-weight 0.2
    run_exp $G 23 foa_featbank exp23_featbank_lr1e4_fw0.1 --lr 0.0001 --foa-weight 0.1
    run_exp $G 24 foa_featbank exp24_featbank_lr1e3_fw0.2 --lr 0.001 --foa-weight 0.2
    run_exp $G 28 foa_msattn exp28_msattn_lr1e4_fw0.1 --lr 0.0001 --foa-weight 0.1
    run_exp $G 29 foa_msattn exp29_msattn_lr1e3_fw0.2 --lr 0.001 --foa-weight 0.2
    run_exp $G 53 foa exp53_foa_lr5e4_dw0.5_fw0.2_hw0.1 --lr 0.0005 --depth-weight 0.5 --foa-weight 0.2 --hist-weight 0.1
    run_exp $G 60 foa_v2 exp60_foav2_lr5e4_dw1.0_fw0.2_hw0.2 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.2
}

# ── Worker 3: GPUs 6,7 — 8 experiments ──
worker_3() {
    local G="6,7"
    run_exp $G 20 foa_crossattn exp20_crossattn_lr5e4_fw0.2 --lr 0.0005 --foa-weight 0.2
    run_exp $G 25 foa_featbank exp25_featbank_lr5e4_fw0.2 --lr 0.0005 --foa-weight 0.2
    run_exp $G 30 foa_msattn exp30_msattn_lr5e4_fw0.2 --lr 0.0005 --foa-weight 0.2
    run_exp $G 33 foa_channelattn exp33_channelattn_lr1e4_fw0.1 --lr 0.0001 --foa-weight 0.1
    run_exp $G 34 foa_channelattn exp34_channelattn_lr1e3_fw0.2 --lr 0.001 --foa-weight 0.2
    run_exp $G 35 foa_channelattn exp35_channelattn_lr5e4_fw0.2 --lr 0.0005 --foa-weight 0.2
    run_exp $G 54 foa exp54_foa_lr1e4_dw1.0_fw0.2_hw0.1 --lr 0.0001 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.1
    run_exp $G 55 foa exp55_foa_lr1e4_bs16_dw1.0_fw0.1_hw0.1 --lr 0.0001 --batch-size 16 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
}

# ── Launch all 4 workers ────────────────────────────────────
worker_0 &
W0=$!
worker_1 &
W1=$!
worker_2 &
W2=$!
worker_3 &
W3=$!

echo "Workers: W0=$W0(GPU0,1) W1=$W1(GPU2,3) W2=$W2(GPU4,5) W3=$W3(GPU6,7)"
echo ""

wait $W0 $W1 $W2 $W3 2>/dev/null

# ── Results summary ─────────────────────────────────────────
echo ""
echo "============================================================"
echo "Revised experiments finished — $(date)"
echo "============================================================"
echo ""
printf "%-4s %-45s %8s %8s %8s\n" "Exp" "Name" "RMSE" "ABS_REL" "Score"
echo "--------------------------------------------------------------"

SUCCESS=0
FAILURES=0
for i in 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 43 48 51 52 53 54 55 56 57 58 59 60; do
    LOG=$(ls "$LOG_DIR"/exp$(printf '%02d' $i)_*.log 2>/dev/null | head -1)
    if [ -z "$LOG" ]; then
        printf "%-4s %-45s %8s\n" "$i" "(no log)" "MISSING"
        FAILURES=$((FAILURES + 1))
        continue
    fi
    NAME=$(basename "$LOG" .log)
    BEST=$(grep ">> Best" "$LOG" 2>/dev/null | tail -1)
    if [ -n "$BEST" ]; then
        SCORE=$(echo "$BEST" | grep -oP 'score:\K[0-9.]+' || echo "N/A")
        RMSE=$(echo "$BEST" | grep -oP 'RMSE:\K[0-9.]+' || echo "N/A")
        ABS=$(echo "$BEST" | grep -oP 'ABS:\K[0-9.]+' || echo "N/A")
        printf "%-4s %-45s %8s %8s %8s\n" "$i" "$NAME" "$RMSE" "$ABS" "$SCORE"
        SUCCESS=$((SUCCESS + 1))
    else
        ERR=0
        grep -q "Traceback\|Error" "$LOG" 2>/dev/null && ERR=1
        if [ "$ERR" -eq 1 ]; then
            printf "%-4s %-45s %8s\n" "$i" "$NAME" "FAILED"
        else
            printf "%-4s %-45s %8s\n" "$i" "$NAME" "NO_RESULT"
        fi
        FAILURES=$((FAILURES + 1))
    fi
done

echo "--------------------------------------------------------------"
echo "Success: $SUCCESS  Failed: $FAILURES  Total: $TOTAL"
echo "============================================================"

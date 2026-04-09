#!/bin/bash
# ============================================================
# 55 Experiments — 2026-04-07
# 5 persistent workers × 2 GPUs each = 10 GPUs (0-9)
# Jobs are pre-assigned round-robin to workers.
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
TOTAL=55

echo "============================================================"
echo "55 Experiments — $(date)"
echo "5 workers × 2 GPUs each (GPUs 0-9)"
echo "Epochs: $EPOCHS | Workers: $NUM_WORKERS"
echo "Logs: $LOG_DIR"
echo "============================================================"
echo ""

# ── run_exp: run a single experiment ────────────────────────
run_exp() {
    local GPUS="$1"; shift
    local EXP_IDX="$1"; shift
    local CONFIG="$1"; shift
    local EXP_NAME="$1"; shift
    # Remaining args are EXTRA_ARGS

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

# ── Worker 0: GPUs 0,1 — Exps 01,06,11,16,21,26,31,36,41,46,51 ──
worker_0() {
    local G="0,1"
    run_exp $G 01 baseline exp01_baseline_lr1e3_bs32 --lr 0.001 --batch-size 32
    run_exp $G 06 vit exp06_vit_lr1e4_bs32 --lr 0.0001 --batch-size 32
    run_exp $G 11 echodiffusion exp11_echodiff_lr1e4_bs32 --lr 0.0001 --batch-size 32
    run_exp $G 16 foa_crossattn exp16_crossattn_lr1e3_fw0.1 --lr 0.001 --foa-weight 0.1
    run_exp $G 21 foa_featbank exp21_featbank_lr1e3_fw0.1 --lr 0.001 --foa-weight 0.1
    run_exp $G 26 foa_msattn exp26_msattn_lr1e3_fw0.1 --lr 0.001 --foa-weight 0.1
    run_exp $G 31 foa_channelattn exp31_channelattn_lr1e3_fw0.1 --lr 0.001 --foa-weight 0.1
    run_exp $G 36 foa exp36_foa_lr1e3_dw1.0_fw0.1_hw0.1 --lr 0.001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 41 foa exp41_foa_lr1e3_dw1.0_fw0.2_hw0.1 --lr 0.001 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.1
    run_exp $G 46 foa exp46_foa_lr1e3_dw0.5_fw0.1_hw0.1 --lr 0.001 --depth-weight 0.5 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 51 foa exp51_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze10 --lr 0.001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1 --foa-freeze-epochs 10
}

# ── Worker 1: GPUs 2,3 — Exps 02,07,12,17,22,27,32,37,42,47,52 ──
worker_1() {
    local G="2,3"
    run_exp $G 02 baseline exp02_baseline_lr5e4_bs32 --lr 0.0005 --batch-size 32
    run_exp $G 07 vit exp07_vit_lr5e5_bs32 --lr 0.00005 --batch-size 32
    run_exp $G 12 echodiffusion exp12_echodiff_lr5e5_bs32 --lr 0.00005 --batch-size 32
    run_exp $G 17 foa_crossattn exp17_crossattn_lr5e4_fw0.1 --lr 0.0005 --foa-weight 0.1
    run_exp $G 22 foa_featbank exp22_featbank_lr5e4_fw0.1 --lr 0.0005 --foa-weight 0.1
    run_exp $G 27 foa_msattn exp27_msattn_lr5e4_fw0.1 --lr 0.0005 --foa-weight 0.1
    run_exp $G 32 foa_channelattn exp32_channelattn_lr5e4_fw0.1 --lr 0.0005 --foa-weight 0.1
    run_exp $G 37 foa exp37_foa_lr5e4_dw1.0_fw0.1_hw0.1 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 42 foa exp42_foa_lr5e4_dw1.0_fw0.2_hw0.1 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.1
    run_exp $G 47 foa exp47_foa_lr1e3_dw2.0_fw0.1_hw0.1 --lr 0.001 --depth-weight 2.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 52 foa exp52_foa_lr5e4_dw1.0_fw0.2_hw0.2 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.2
}

# ── Worker 2: GPUs 4,5 — Exps 03,08,13,18,23,28,33,38,43,48,53 ──
worker_2() {
    local G="4,5"
    run_exp $G 03 baseline exp03_baseline_lr1e4_bs32 --lr 0.0001 --batch-size 32
    run_exp $G 08 vit exp08_vit_lr1e4_bs16 --lr 0.0001 --batch-size 16
    run_exp $G 13 echodiffusion exp13_echodiff_lr1e4_bs16 --lr 0.0001 --batch-size 16
    run_exp $G 18 foa_crossattn exp18_crossattn_lr1e4_fw0.1 --lr 0.0001 --foa-weight 0.1
    run_exp $G 23 foa_featbank exp23_featbank_lr1e4_fw0.1 --lr 0.0001 --foa-weight 0.1
    run_exp $G 28 foa_msattn exp28_msattn_lr1e4_fw0.1 --lr 0.0001 --foa-weight 0.1
    run_exp $G 33 foa_channelattn exp33_channelattn_lr1e4_fw0.1 --lr 0.0001 --foa-weight 0.1
    run_exp $G 38 foa exp38_foa_lr1e4_dw1.0_fw0.1_hw0.1 --lr 0.0001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 43 foa exp43_foa_lr1e3_dw1.0_fw0.1_hw0.2 --lr 0.001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.2
    run_exp $G 48 foa exp48_foa_lr1e3_dw1.0_fw0.05_hw0.1 --lr 0.001 --depth-weight 1.0 --foa-weight 0.05 --hist-weight 0.1
    run_exp $G 53 foa exp53_foa_lr5e4_dw0.5_fw0.2_hw0.1 --lr 0.0005 --depth-weight 0.5 --foa-weight 0.2 --hist-weight 0.1
}

# ── Worker 3: GPUs 6,7 — Exps 04,09,14,19,24,29,34,39,44,49,54 ──
worker_3() {
    local G="6,7"
    run_exp $G 04 baseline exp04_baseline_lr1e3_bs16 --lr 0.001 --batch-size 16
    run_exp $G 09 vit exp09_vit_lr5e4_bs32 --lr 0.0005 --batch-size 32
    run_exp $G 14 echodiffusion exp14_echodiff_lr5e4_bs32 --lr 0.0005 --batch-size 32
    run_exp $G 19 foa_crossattn exp19_crossattn_lr1e3_fw0.2 --lr 0.001 --foa-weight 0.2
    run_exp $G 24 foa_featbank exp24_featbank_lr1e3_fw0.2 --lr 0.001 --foa-weight 0.2
    run_exp $G 29 foa_msattn exp29_msattn_lr1e3_fw0.2 --lr 0.001 --foa-weight 0.2
    run_exp $G 34 foa_channelattn exp34_channelattn_lr1e3_fw0.2 --lr 0.001 --foa-weight 0.2
    run_exp $G 39 foa exp39_foa_lr1e3_bs16_dw1.0_fw0.1_hw0.1 --lr 0.001 --batch-size 16 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 44 foa exp44_foa_lr5e4_dw1.0_fw0.1_hw0.2 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.2
    run_exp $G 49 foa exp49_foa_lr1e3_dw1.0_fw0.1_hw0.05 --lr 0.001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.05
    run_exp $G 54 foa exp54_foa_lr1e4_dw1.0_fw0.2_hw0.1 --lr 0.0001 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.1
}

# ── Worker 4: GPUs 8,9 — Exps 05,10,15,20,25,30,35,40,45,50,55 ──
worker_4() {
    local G="8,9"
    run_exp $G 05 baseline exp05_baseline_lr5e4_bs16 --lr 0.0005 --batch-size 16
    run_exp $G 10 vit exp10_vit_lr1e5_bs32 --lr 0.00001 --batch-size 32
    run_exp $G 15 echodiffusion exp15_echodiff_lr1e5_bs32 --lr 0.00001 --batch-size 32
    run_exp $G 20 foa_crossattn exp20_crossattn_lr5e4_fw0.2 --lr 0.0005 --foa-weight 0.2
    run_exp $G 25 foa_featbank exp25_featbank_lr5e4_fw0.2 --lr 0.0005 --foa-weight 0.2
    run_exp $G 30 foa_msattn exp30_msattn_lr5e4_fw0.2 --lr 0.0005 --foa-weight 0.2
    run_exp $G 35 foa_channelattn exp35_channelattn_lr5e4_fw0.2 --lr 0.0005 --foa-weight 0.2
    run_exp $G 40 foa exp40_foa_lr5e4_bs16_dw1.0_fw0.1_hw0.1 --lr 0.0005 --batch-size 16 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 45 foa exp45_foa_lr1e3_dw1.0_fw0.2_hw0.2 --lr 0.001 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.2
    run_exp $G 50 foa exp50_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze5 --lr 0.001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1 --foa-freeze-epochs 5
    run_exp $G 55 foa exp55_foa_lr1e4_bs16_dw1.0_fw0.1_hw0.1 --lr 0.0001 --batch-size 16 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
}

# ── Launch all 5 workers in background ──────────────────────
worker_0 &
W0=$!
worker_1 &
W1=$!
worker_2 &
W2=$!
worker_3 &
W3=$!
worker_4 &
W4=$!

echo "Workers launched: W0=$W0 W1=$W1 W2=$W2 W3=$W3 W4=$W4"
echo ""

# ── Wait ────────────────────────────────────────────────────
wait $W0 $W1 $W2 $W3 $W4 2>/dev/null

# ── Results summary ─────────────────────────────────────────
echo ""
echo "============================================================"
echo "All 55 experiments finished — $(date)"
echo "============================================================"
echo ""
printf "%-4s %-45s %8s %8s %8s\n" "Exp" "Name" "RMSE" "ABS_REL" "Score"
echo "--------------------------------------------------------------"

SUCCESS=0
FAILURES=0
for i in $(seq -w 1 55); do
    LOG=$(ls "$LOG_DIR"/exp${i}_*.log 2>/dev/null | head -1)
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
        ERR=$(grep -c "Traceback\|Error" "$LOG" 2>/dev/null || true)
        ERR=${ERR:-0}
        if [ "$ERR" -gt 0 ] 2>/dev/null; then
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

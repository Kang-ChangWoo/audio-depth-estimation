#!/bin/bash
# ============================================================
# 65 Experiments — 2026-04-08
# 4 persistent workers × 2 GPUs each = 8 GPUs (0-7)
# Index 56–120 (continues from bulk0407_55exps.sh exps 01–55)
#
# Group A (exps 56–75):  New model baselines, 5 each
#   56-60  pretrain_resnet
#   61-65  pretrain_vit
#   66-70  echonet
#   71-75  batvision
#
# Group B (exps 76–95):  FOA variants, 5 each (new combos vs 0407)
#   76-80  foa_crossattn   (vary kl_weight, fw, hw — avoid 0407)
#   81-85  foa_featbank
#   86-90  foa_msattn
#   91-95  foa_channelattn
#
# Group C (exps 96–120): FOA main, 25 trials (new combos vs 0407)
#   wider lr/dw/fw/hw search + freeze epochs
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
TOTAL=65
START_IDX=56
END_IDX=120

echo "============================================================"
echo "65 Experiments (exp56–exp120) — $(date)"
echo "4 workers × 2 GPUs each (GPUs 0-7)"
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

    echo "[$(date +%H:%M:%S)] [GPU=$GPUS] Exp $EXP_IDX/$END_IDX: $EXP_NAME"

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
# Worker 0: GPUs 0,1
#   Round-robin: 56,60,64,68,72, 76,80,84,88,92, 96,100,104,108,112,116,120
# ════════════════════════════════════════════════════════════
worker_0() {
    local G="0,1"

    # ── Group A: new model baselines ──
    run_exp $G 56 pretrain_resnet  exp56_resnet_lr1e4_bs32   --lr 0.0001 --batch-size 32
    run_exp $G 60 pretrain_resnet  exp60_resnet_lr3e4_bs32   --lr 0.0003 --batch-size 32
    run_exp $G 64 pretrain_vit     exp64_vit_lr1e4_bs8       --lr 0.0001 --batch-size 8
    run_exp $G 68 echonet          exp68_echonet_lr1e4_bs16  --lr 0.0001 --batch-size 16
    run_exp $G 72 batvision        exp72_batvision_lr5e4_bs32 --lr 0.0005 --batch-size 32

    # ── Group B: FOA variants (new combos) ──
    run_exp $G 76 foa_crossattn    exp76_crossattn_lr1e3_fw0.1_kl0.02     --lr 0.001  --foa-weight 0.1 --kl-weight 0.02
    run_exp $G 80 foa_crossattn    exp80_crossattn_lr1e3_fw0.3_kl0.01     --lr 0.001  --foa-weight 0.3 --kl-weight 0.01
    run_exp $G 84 foa_featbank     exp84_featbank_lr5e4_fw0.1_hw0.2_kl0.01 --lr 0.0005 --foa-weight 0.1 --hist-weight 0.2 --kl-weight 0.01
    run_exp $G 88 foa_msattn       exp88_msattn_lr1e3_fw0.05_kl0.01       --lr 0.001  --foa-weight 0.05 --kl-weight 0.01
    run_exp $G 92 foa_channelattn  exp92_channelattn_lr5e4_fw0.2_kl0.005  --lr 0.0005 --foa-weight 0.2 --kl-weight 0.005

    # ── Group C: FOA main ──
    run_exp $G 96  foa exp96_foa_lr2e4_dw1.0_fw0.1_hw0.1           --lr 0.0002 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 100 foa exp100_foa_lr1e3_fw0.15_hw0.1                --lr 0.001  --depth-weight 1.0 --foa-weight 0.15 --hist-weight 0.1
    run_exp $G 104 foa exp104_foa_lr5e4_fw0.1_hw0.15               --lr 0.0005 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.15
    run_exp $G 108 foa exp108_foa_lr5e4_dw2.0_fw0.1_hw0.1          --lr 0.0005 --depth-weight 2.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 112 foa exp112_foa_lr5e4_dw1.0_fw0.1_hw0.1_freeze10 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1 --foa-freeze-epochs 10
    run_exp $G 116 foa exp116_foa_lr3e4_fw0.2_hw0.1                --lr 0.0003 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.1
    run_exp $G 120 foa exp120_foa_lr5e4_fw0.05_hw0.1_freeze5       --lr 0.0005 --depth-weight 1.0 --foa-weight 0.05 --hist-weight 0.1 --foa-freeze-epochs 5
}

# ════════════════════════════════════════════════════════════
# Worker 1: GPUs 2,3
#   Round-robin: 57,61,65,69,73, 77,81,85,89,93, 97,101,105,109,113,117
# ════════════════════════════════════════════════════════════
worker_1() {
    local G="2,3"

    # ── Group A ──
    run_exp $G 57 pretrain_resnet  exp57_resnet_lr5e5_bs32   --lr 0.00005 --batch-size 32
    run_exp $G 61 pretrain_vit     exp61_vit_lr1e4_bs16      --lr 0.0001  --batch-size 16
    run_exp $G 65 pretrain_vit     exp65_vit_lr3e5_bs16      --lr 0.00003 --batch-size 16
    run_exp $G 69 echonet          exp69_echonet_lr1e3_bs16  --lr 0.001   --batch-size 16
    run_exp $G 73 batvision        exp73_batvision_lr1e4_bs32 --lr 0.0001 --batch-size 32

    # ── Group B ──
    run_exp $G 77 foa_crossattn    exp77_crossattn_lr5e4_fw0.2_kl0.005    --lr 0.0005 --foa-weight 0.2 --kl-weight 0.005
    run_exp $G 81 foa_featbank     exp81_featbank_lr1e3_fw0.1_kl0.02      --lr 0.001  --foa-weight 0.1 --kl-weight 0.02
    run_exp $G 85 foa_featbank     exp85_featbank_lr1e3_fw0.3_kl0.01      --lr 0.001  --foa-weight 0.3 --kl-weight 0.01
    run_exp $G 89 foa_msattn       exp89_msattn_lr5e4_fw0.1_hw0.2_kl0.01 --lr 0.0005 --foa-weight 0.1 --hist-weight 0.2 --kl-weight 0.01
    run_exp $G 93 foa_channelattn  exp93_channelattn_lr1e3_fw0.05_kl0.01  --lr 0.001  --foa-weight 0.05 --kl-weight 0.01

    # ── Group C ──
    run_exp $G 97  foa exp97_foa_lr3e4_dw1.0_fw0.1_hw0.1           --lr 0.0003 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 101 foa exp101_foa_lr1e3_fw0.1_hw0.15               --lr 0.001  --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.15
    run_exp $G 105 foa exp105_foa_lr1e3_fw0.3_hw0.1                --lr 0.001  --depth-weight 1.0 --foa-weight 0.3 --hist-weight 0.1
    run_exp $G 109 foa exp109_foa_lr1e3_dw1.0_fw0.2_hw0.2_freeze5  --lr 0.001  --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.2 --foa-freeze-epochs 5
    run_exp $G 113 foa exp113_foa_lr1e3_dw1.0_fw0.05_hw0.05        --lr 0.001  --depth-weight 1.0 --foa-weight 0.05 --hist-weight 0.05
    run_exp $G 117 foa exp117_foa_lr2e4_dw1.0_fw0.2_hw0.2          --lr 0.0002 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.2
}

# ════════════════════════════════════════════════════════════
# Worker 2: GPUs 4,5
#   Round-robin: 58,62,66,70,74, 78,82,86,90,94, 98,102,106,110,114,118
# ════════════════════════════════════════════════════════════
worker_2() {
    local G="4,5"

    # ── Group A ──
    run_exp $G 58 pretrain_resnet  exp58_resnet_lr5e4_bs32   --lr 0.0005 --batch-size 32
    run_exp $G 62 pretrain_vit     exp62_vit_lr5e5_bs16      --lr 0.00005 --batch-size 16
    run_exp $G 66 echonet          exp66_echonet_lr1e3_bs8   --lr 0.001  --batch-size 8
    run_exp $G 70 echonet          exp70_echonet_lr2e3_bs16  --lr 0.002  --batch-size 16
    run_exp $G 74 batvision        exp74_batvision_lr1e3_bs16 --lr 0.001 --batch-size 16

    # ── Group B ──
    run_exp $G 78 foa_crossattn    exp78_crossattn_lr1e3_fw0.05_kl0.01    --lr 0.001  --foa-weight 0.05 --kl-weight 0.01
    run_exp $G 82 foa_featbank     exp82_featbank_lr5e4_fw0.2_kl0.005     --lr 0.0005 --foa-weight 0.2 --kl-weight 0.005
    run_exp $G 86 foa_msattn       exp86_msattn_lr1e3_fw0.1_kl0.02       --lr 0.001  --foa-weight 0.1 --kl-weight 0.02
    run_exp $G 90 foa_msattn       exp90_msattn_lr1e3_fw0.3_kl0.01       --lr 0.001  --foa-weight 0.3 --kl-weight 0.01
    run_exp $G 94 foa_channelattn  exp94_channelattn_lr5e4_fw0.1_hw0.2_kl0.01 --lr 0.0005 --foa-weight 0.1 --hist-weight 0.2 --kl-weight 0.01

    # ── Group C ──
    run_exp $G 98  foa exp98_foa_lr7e4_dw1.0_fw0.1_hw0.1            --lr 0.0007 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 102 foa exp102_foa_lr5e4_dw1.5_fw0.1_hw0.1           --lr 0.0005 --depth-weight 1.5 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 106 foa exp106_foa_lr1e3_fw0.1_hw0.3                 --lr 0.001  --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.3
    run_exp $G 110 foa exp110_foa_lr5e4_fw0.2_hw0.1_freeze5         --lr 0.0005 --depth-weight 1.0 --foa-weight 0.2 --hist-weight 0.1 --foa-freeze-epochs 5
    run_exp $G 114 foa exp114_foa_lr5e4_dw0.5_fw0.1_hw0.2           --lr 0.0005 --depth-weight 0.5 --foa-weight 0.1 --hist-weight 0.2
    run_exp $G 118 foa exp118_foa_lr7e4_fw0.15_hw0.15               --lr 0.0007 --depth-weight 1.0 --foa-weight 0.15 --hist-weight 0.15
}

# ════════════════════════════════════════════════════════════
# Worker 3: GPUs 6,7
#   Round-robin: 59,63,67,71,75, 79,83,87,91,95, 99,103,107,111,115,119
# ════════════════════════════════════════════════════════════
worker_3() {
    local G="6,7"

    # ── Group A ──
    run_exp $G 59 pretrain_resnet  exp59_resnet_lr1e4_bs16   --lr 0.0001 --batch-size 16
    run_exp $G 63 pretrain_vit     exp63_vit_lr5e4_bs16      --lr 0.0005 --batch-size 16
    run_exp $G 67 echonet          exp67_echonet_lr5e4_bs16  --lr 0.0005 --batch-size 16
    run_exp $G 71 batvision        exp71_batvision_lr1e3_bs32 --lr 0.001 --batch-size 32
    run_exp $G 75 batvision        exp75_batvision_lr2e3_bs32 --lr 0.002 --batch-size 32

    # ── Group B ──
    run_exp $G 79 foa_crossattn    exp79_crossattn_lr5e4_fw0.1_hw0.2_kl0.01 --lr 0.0005 --foa-weight 0.1 --hist-weight 0.2 --kl-weight 0.01
    run_exp $G 83 foa_featbank     exp83_featbank_lr1e3_fw0.05_kl0.01     --lr 0.001  --foa-weight 0.05 --kl-weight 0.01
    run_exp $G 87 foa_msattn       exp87_msattn_lr5e4_fw0.2_kl0.005      --lr 0.0005 --foa-weight 0.2 --kl-weight 0.005
    run_exp $G 91 foa_channelattn  exp91_channelattn_lr1e3_fw0.1_kl0.02  --lr 0.001  --foa-weight 0.1 --kl-weight 0.02
    run_exp $G 95 foa_channelattn  exp95_channelattn_lr1e3_fw0.3_kl0.01  --lr 0.001  --foa-weight 0.3 --kl-weight 0.01

    # ── Group C ──
    run_exp $G 99  foa exp99_foa_lr1e3_dw1.5_fw0.1_hw0.1             --lr 0.001  --depth-weight 1.5 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 103 foa exp103_foa_lr5e4_fw0.15_hw0.1                 --lr 0.0005 --depth-weight 1.0 --foa-weight 0.15 --hist-weight 0.1
    run_exp $G 107 foa exp107_foa_lr5e4_fw0.3_hw0.1                  --lr 0.0005 --depth-weight 1.0 --foa-weight 0.3 --hist-weight 0.1
    run_exp $G 111 foa exp111_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze15  --lr 0.001  --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1 --foa-freeze-epochs 15
    run_exp $G 115 foa exp115_foa_lr1e3_dw2.0_fw0.2_hw0.1           --lr 0.001  --depth-weight 2.0 --foa-weight 0.2 --hist-weight 0.1
    run_exp $G 119 foa exp119_foa_lr1e3_fw0.1_hw0.2_freeze3         --lr 0.001  --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.2 --foa-freeze-epochs 3
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

echo "Workers launched: W0=$W0 W1=$W1 W2=$W2 W3=$W3"
echo ""

# ── Wait ───────────────────────────────────────────────────
wait $W0 $W1 $W2 $W3 2>/dev/null

# ── Results summary ────────────────────────────────────────
echo ""
echo "============================================================"
echo "All 65 experiments (exp56–exp120) finished — $(date)"
echo "============================================================"
echo ""
printf "%-4s %-50s %8s %8s %8s\n" "Exp" "Name" "RMSE" "ABS_REL" "Score"
echo "-------------------------------------------------------------------"

SUCCESS=0
FAILURES=0
for i in $(seq $START_IDX $END_IDX); do
    LOG=$(ls "$LOG_DIR"/exp${i}_*.log 2>/dev/null | head -1)
    if [ -z "$LOG" ]; then
        # skip indices not assigned (round-robin gaps don't exist here)
        continue
    fi
    NAME=$(basename "$LOG" .log)
    BEST=$(grep ">> Best" "$LOG" 2>/dev/null | tail -1)
    if [ -n "$BEST" ]; then
        SCORE=$(echo "$BEST" | grep -oP 'score:\K[0-9.]+' || echo "N/A")
        RMSE=$(echo "$BEST" | grep -oP 'RMSE:\K[0-9.]+' || echo "N/A")
        ABS=$(echo "$BEST" | grep -oP 'ABS:\K[0-9.]+' || echo "N/A")
        printf "%-4s %-50s %8s %8s %8s\n" "$i" "$NAME" "$RMSE" "$ABS" "$SCORE"
        SUCCESS=$((SUCCESS + 1))
    else
        ERR=$(grep -c "Traceback\|Error" "$LOG" 2>/dev/null || true)
        ERR=${ERR:-0}
        if [ "$ERR" -gt 0 ] 2>/dev/null; then
            printf "%-4s %-50s %8s\n" "$i" "$NAME" "FAILED"
        else
            printf "%-4s %-50s %8s\n" "$i" "$NAME" "NO_RESULT"
        fi
        FAILURES=$((FAILURES + 1))
    fi
done

echo "-------------------------------------------------------------------"
echo "Success: $SUCCESS  Failed: $FAILURES  Total: $TOTAL"
echo "============================================================"

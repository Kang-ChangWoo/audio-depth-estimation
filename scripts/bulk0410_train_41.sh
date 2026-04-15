#!/bin/bash
# ============================================================
# bulk0410_sum_41 — archive of 41 successful 40-epoch runs
#
# Consolidates the "known good" training runs from:
#   - bulk0407_55exps.sh           (exp01–exp55)
#   - bulk0407_55exps_revised.sh   (exp16–35, 43, 48, 51–60 reruns)
#   - bulk0408_5exps.sh            (exp121–125 echodiff+wav2vec)
#   - bulk0408_65exps.sh           (exp56–120)
#   - bulk0410_failedexps.sh       (exp57/59/60 foav2 + exp121/123/125 reruns)
#
# Filter rules:
#   * Only runs whose training log reached epoch 40/40.
#   * FOA family (foa, foa_v2, foa_crossattn, foa_featbank,
#     foa_msattn, foa_channelattn) is trimmed to 5 representative
#     combinations: exp18 crossattn, exp27 msattn, exp31 channelattn,
#     exp49 foa (mandatory — current best, test RMSE 1.0803), and
#     exp56 foav2. All other FOA variants are intentionally removed.
#   * Echo-Net trimmed to exp70 (only one that reached 40 epochs).
#   * Original experiment indices are preserved exactly.
#
# This script is archived as a reusable template for running the
# same experimental grid on a different dataset. Failed / never-ran
# experiments live in bulk0410_sum_21_revised.sh.
#
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
LOG_DIR="$PROJECT_DIR/logs/bulk0410_sum"
mkdir -p "$LOG_DIR"
TOTAL=41

echo "============================================================"
echo "bulk0410_sum_41 — 41 experiments — $(date)"
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
# Worker 0: GPUs 0,1 — 11 experiments
# ════════════════════════════════════════════════════════════
worker_0() {
    local G="0,1"
    # Baseline UNet
    run_exp $G 01 baseline exp01_baseline_lr1e3_bs32 --lr 0.001 --batch-size 32
    run_exp $G 05 baseline exp05_baseline_lr5e4_bs16 --lr 0.0005 --batch-size 16
    # AudioDepthViT
    run_exp $G 09 vit exp09_vit_lr5e4_bs32 --lr 0.0005 --batch-size 32
    # EchoDiffusion
    run_exp $G 13 echodiffusion exp13_echodiff_lr1e4_bs16 --lr 0.0001 --batch-size 16
    # FOA family (trimmed)
    run_exp $G 27 foa_msattn exp27_msattn_lr5e4_fw0.1 --lr 0.0005 --foa-weight 0.1
    # Pretrained ResNet-50
    run_exp $G 56 pretrain_resnet exp56_resnet_lr1e4_bs32 --lr 0.0001 --batch-size 32
    run_exp $G 60 pretrain_resnet exp60_resnet_lr3e4_bs32 --lr 0.0003 --batch-size 32
    # Pretrained ViT-B/16
    run_exp $G 64 pretrain_vit exp64_vit_lr1e4_bs8 --lr 0.0001 --batch-size 8
    # BatVision
    run_exp $G 73 batvision exp73_batvision_lr1e4_bs32 --lr 0.0001 --batch-size 32
    # EchoDiff + Wav2Vec
    run_exp $G 121 echodiffusion exp121_echodiff_wav2vec_lr1e4_bs16 --lr 0.0001 --batch-size 16
    run_exp $G 125 echodiffusion exp125_echodiff_wav2vec_lr1e4_bs8 --lr 0.0001 --batch-size 8
}

# ════════════════════════════════════════════════════════════
# Worker 1: GPUs 2,3 — 10 experiments
# ════════════════════════════════════════════════════════════
worker_1() {
    local G="2,3"
    run_exp $G 02 baseline exp02_baseline_lr5e4_bs32 --lr 0.0005 --batch-size 32
    run_exp $G 06 vit exp06_vit_lr1e4_bs32 --lr 0.0001 --batch-size 32
    run_exp $G 10 vit exp10_vit_lr1e5_bs32 --lr 0.00001 --batch-size 32
    run_exp $G 14 echodiffusion exp14_echodiff_lr5e4_bs32 --lr 0.0005 --batch-size 32
    # FOA family (trimmed): exp31 foa_channelattn
    run_exp $G 31 foa_channelattn exp31_channelattn_lr1e3_fw0.1 --lr 0.001 --foa-weight 0.1
    run_exp $G 57 pretrain_resnet exp57_resnet_lr5e5_bs32 --lr 0.00005 --batch-size 32
    run_exp $G 61 pretrain_vit exp61_vit_lr1e4_bs16 --lr 0.0001 --batch-size 16
    run_exp $G 65 pretrain_vit exp65_vit_lr3e5_bs16 --lr 0.00003 --batch-size 16
    run_exp $G 74 batvision exp74_batvision_lr1e3_bs16 --lr 0.001 --batch-size 16
    run_exp $G 122 echodiffusion exp122_echodiff_wav2vec_lr5e4_bs16 --lr 0.0005 --batch-size 16
}

# ════════════════════════════════════════════════════════════
# Worker 2: GPUs 4,5 — 10 experiments
# ════════════════════════════════════════════════════════════
worker_2() {
    local G="4,5"
    run_exp $G 03 baseline exp03_baseline_lr1e4_bs32 --lr 0.0001 --batch-size 32
    run_exp $G 07 vit exp07_vit_lr5e5_bs32 --lr 0.00005 --batch-size 32
    run_exp $G 11 echodiffusion exp11_echodiff_lr1e4_bs32 --lr 0.0001 --batch-size 32
    run_exp $G 15 echodiffusion exp15_echodiff_lr1e5_bs32 --lr 0.00001 --batch-size 32
    # FOA family (trimmed): exp49 foa — MANDATORY (current best test RMSE 1.0803)
    run_exp $G 49 foa exp49_foa_lr1e3_dw1.0_fw0.1_hw0.05 --lr 0.001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.05
    run_exp $G 58 pretrain_resnet exp58_resnet_lr5e4_bs32 --lr 0.0005 --batch-size 32
    run_exp $G 62 pretrain_vit exp62_vit_lr5e5_bs16 --lr 0.00005 --batch-size 16
    # Echo-Net (only the single 40-epoch run survives the filter)
    run_exp $G 70 echonet exp70_echonet_lr2e3_bs16 --lr 0.002 --batch-size 16
    run_exp $G 75 batvision exp75_batvision_lr2e3_bs32 --lr 0.002 --batch-size 32
    run_exp $G 123 echodiffusion exp123_echodiff_wav2vec_lr1e4_bs32 --lr 0.0001 --batch-size 32
}

# ════════════════════════════════════════════════════════════
# Worker 3: GPUs 6,7 — 10 experiments
# ════════════════════════════════════════════════════════════
worker_3() {
    local G="6,7"
    run_exp $G 04 baseline exp04_baseline_lr1e3_bs16 --lr 0.001 --batch-size 16
    run_exp $G 08 vit exp08_vit_lr1e4_bs16 --lr 0.0001 --batch-size 16
    run_exp $G 12 echodiffusion exp12_echodiff_lr5e5_bs32 --lr 0.00005 --batch-size 32
    # FOA family (trimmed): exp18 foa_crossattn
    run_exp $G 18 foa_crossattn exp18_crossattn_lr1e4_fw0.1 --lr 0.0001 --foa-weight 0.1
    # FOA family (trimmed): exp56 foa_v2
    run_exp $G 56 foa_v2 exp56_foav2_lr1e3_dw1.0_fw0.1_hw0.1 --lr 0.001 --depth-weight 1.0 --foa-weight 0.1 --hist-weight 0.1
    run_exp $G 59 pretrain_resnet exp59_resnet_lr1e4_bs16 --lr 0.0001 --batch-size 16
    run_exp $G 63 pretrain_vit exp63_vit_lr5e4_bs16 --lr 0.0005 --batch-size 16
    run_exp $G 71 batvision exp71_batvision_lr1e3_bs32 --lr 0.001 --batch-size 32
    run_exp $G 72 batvision exp72_batvision_lr5e4_bs32 --lr 0.0005 --batch-size 32
    run_exp $G 124 echodiffusion exp124_echodiff_wav2vec_lr5e5_bs16 --lr 0.00005 --batch-size 16
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
echo "bulk0410_sum_41 finished — $(date)"
echo "============================================================"
echo ""
printf "%-4s %-50s %8s %8s %8s\n" "Exp" "Name" "RMSE" "ABS_REL" "Score"
echo "-------------------------------------------------------------------"

SUCCESS=0
FAILURES=0
for name in \
    exp01_baseline_lr1e3_bs32 exp02_baseline_lr5e4_bs32 exp03_baseline_lr1e4_bs32 \
    exp04_baseline_lr1e3_bs16 exp05_baseline_lr5e4_bs16 \
    exp06_vit_lr1e4_bs32 exp07_vit_lr5e5_bs32 exp08_vit_lr1e4_bs16 \
    exp09_vit_lr5e4_bs32 exp10_vit_lr1e5_bs32 \
    exp11_echodiff_lr1e4_bs32 exp12_echodiff_lr5e5_bs32 exp13_echodiff_lr1e4_bs16 \
    exp14_echodiff_lr5e4_bs32 exp15_echodiff_lr1e5_bs32 \
    exp18_crossattn_lr1e4_fw0.1 exp27_msattn_lr5e4_fw0.1 \
    exp31_channelattn_lr1e3_fw0.1 exp49_foa_lr1e3_dw1.0_fw0.1_hw0.05 \
    exp56_foav2_lr1e3_dw1.0_fw0.1_hw0.1 \
    exp56_resnet_lr1e4_bs32 exp57_resnet_lr5e5_bs32 exp58_resnet_lr5e4_bs32 \
    exp59_resnet_lr1e4_bs16 exp60_resnet_lr3e4_bs32 \
    exp61_vit_lr1e4_bs16 exp62_vit_lr5e5_bs16 exp63_vit_lr5e4_bs16 \
    exp64_vit_lr1e4_bs8 exp65_vit_lr3e5_bs16 \
    exp70_echonet_lr2e3_bs16 \
    exp71_batvision_lr1e3_bs32 exp72_batvision_lr5e4_bs32 exp73_batvision_lr1e4_bs32 \
    exp74_batvision_lr1e3_bs16 exp75_batvision_lr2e3_bs32 \
    exp121_echodiff_wav2vec_lr1e4_bs16 exp122_echodiff_wav2vec_lr5e4_bs16 \
    exp123_echodiff_wav2vec_lr1e4_bs32 exp124_echodiff_wav2vec_lr5e5_bs16 \
    exp125_echodiff_wav2vec_lr1e4_bs8; do

    LOG="$LOG_DIR/${name}.log"
    IDX=$(echo "$name" | grep -oP '^exp\K\d+')
    if [ ! -f "$LOG" ]; then
        printf "%-4s %-50s %8s\n" "$IDX" "$name" "MISSING"
        FAILURES=$((FAILURES + 1))
        continue
    fi
    BEST=$(grep ">> Best" "$LOG" 2>/dev/null | tail -1)
    if [ -n "$BEST" ]; then
        SCORE=$(echo "$BEST" | grep -oP 'score:\K[0-9.]+' || echo "N/A")
        RMSE=$(echo "$BEST" | grep -oP 'RMSE:\K[0-9.]+' || echo "N/A")
        ABS=$(echo "$BEST" | grep -oP 'ABS:\K[0-9.]+' || echo "N/A")
        printf "%-4s %-50s %8s %8s %8s\n" "$IDX" "$name" "$RMSE" "$ABS" "$SCORE"
        SUCCESS=$((SUCCESS + 1))
    else
        ERR=0
        grep -q "Traceback\|Error" "$LOG" 2>/dev/null && ERR=1
        if [ "$ERR" -eq 1 ]; then
            printf "%-4s %-50s %8s\n" "$IDX" "$name" "FAILED"
        else
            printf "%-4s %-50s %8s\n" "$IDX" "$name" "NO_RESULT"
        fi
        FAILURES=$((FAILURES + 1))
    fi
done

echo "-------------------------------------------------------------------"
echo "Success: $SUCCESS  Failed: $FAILURES  Total: $TOTAL"
echo "============================================================"

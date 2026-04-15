#!/bin/bash
# ============================================================
# bulk0410_test_78exps — archive of 78 already-tested experiments
#
# These experiments already have completed test logs under
# logs/bulk0410/ with valid RMSE / ABS_REL / δ metrics on the
# SoundSpaces test split (9 scenes, 3192 samples). This script
# is archived so the same evaluation set can be re-applied to a
# different dataset without losing the list of "known tested"
# configurations.
#
# Auto-discovers checkpoints under $PROJECT_DIR/checkpoints and
# filters against the WHITELIST below. Config is derived from the
# checkpoint directory name.
#
# Runs 4 workers × 1 GPU each = GPUs 0-3.
# Logs: logs/bulk0410_test_78/
# ============================================================

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/test.py" ]; then
    echo "ERROR: test.py not found in $PROJECT_DIR"
    return 1 2>/dev/null || exit 1
fi

LOG_DIR="$PROJECT_DIR/logs/bulk0410_test_78"
mkdir -p "$LOG_DIR"
VIS_PER_SCENE=100

# ── WHITELIST: the 78 experiments this script is responsible for ──
WHITELIST=(
    exp01_baseline_lr1e3_bs32 exp02_baseline_lr5e4_bs32 exp03_baseline_lr1e4_bs32
    exp04_baseline_lr1e3_bs16 exp05_baseline_lr5e4_bs16
    exp06_vit_lr1e4_bs32 exp07_vit_lr5e5_bs32 exp08_vit_lr1e4_bs16
    exp09_vit_lr5e4_bs32 exp10_vit_lr1e5_bs32
    exp11_echodiff_lr1e4_bs32 exp12_echodiff_lr5e5_bs32 exp13_echodiff_lr1e4_bs16
    exp14_echodiff_lr5e4_bs32 exp15_echodiff_lr1e5_bs32
    exp16_crossattn_lr1e3_fw0.1 exp17_crossattn_lr5e4_fw0.1 exp18_crossattn_lr1e4_fw0.1
    exp19_crossattn_lr1e3_fw0.2 exp20_crossattn_lr5e4_fw0.2
    exp21_featbank_lr1e3_fw0.1 exp22_featbank_lr5e4_fw0.1 exp23_featbank_lr1e4_fw0.1
    exp24_featbank_lr1e3_fw0.2 exp25_featbank_lr5e4_fw0.2
    exp26_msattn_lr1e3_fw0.1 exp27_msattn_lr5e4_fw0.1 exp28_msattn_lr1e4_fw0.1
    exp29_msattn_lr1e3_fw0.2 exp30_msattn_lr5e4_fw0.2
    exp31_channelattn_lr1e3_fw0.1 exp32_channelattn_lr5e4_fw0.1 exp33_channelattn_lr1e4_fw0.1
    exp34_channelattn_lr1e3_fw0.2 exp35_channelattn_lr5e4_fw0.2
    exp36_foa_lr1e3_dw1.0_fw0.1_hw0.1 exp37_foa_lr5e4_dw1.0_fw0.1_hw0.1
    exp38_foa_lr1e4_dw1.0_fw0.1_hw0.1 exp39_foa_lr1e3_bs16_dw1.0_fw0.1_hw0.1
    exp40_foa_lr5e4_bs16_dw1.0_fw0.1_hw0.1 exp41_foa_lr1e3_dw1.0_fw0.2_hw0.1
    exp42_foa_lr5e4_dw1.0_fw0.2_hw0.1 exp43_foa_lr1e3_dw1.0_fw0.1_hw0.2
    exp44_foa_lr5e4_dw1.0_fw0.1_hw0.2 exp45_foa_lr1e3_dw1.0_fw0.2_hw0.2
    exp46_foa_lr1e3_dw0.5_fw0.1_hw0.1 exp47_foa_lr1e3_dw2.0_fw0.1_hw0.1
    exp48_foa_lr1e3_dw1.0_fw0.05_hw0.1 exp49_foa_lr1e3_dw1.0_fw0.1_hw0.05
    exp50_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze5 exp51_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze10
    exp52_foa_lr5e4_dw1.0_fw0.2_hw0.2 exp53_foa_lr5e4_dw0.5_fw0.2_hw0.1
    exp54_foa_lr1e4_dw1.0_fw0.2_hw0.1 exp55_foa_lr1e4_bs16_dw1.0_fw0.1_hw0.1
    exp56_foav2_lr1e3_dw1.0_fw0.1_hw0.1 exp57_foav2_lr5e4_dw1.0_fw0.1_hw0.1
    exp58_foav2_lr1e4_dw1.0_fw0.1_hw0.1 exp59_foav2_lr1e3_dw1.0_fw0.2_hw0.1
    exp60_foav2_lr5e4_dw1.0_fw0.2_hw0.2
    exp56_resnet_lr1e4_bs32 exp57_resnet_lr5e5_bs32 exp58_resnet_lr5e4_bs32
    exp59_resnet_lr1e4_bs16 exp60_resnet_lr3e4_bs32
    exp61_vit_lr1e4_bs16 exp62_vit_lr5e5_bs16 exp63_vit_lr5e4_bs16
    exp64_vit_lr1e4_bs8 exp65_vit_lr3e5_bs16
    exp66_echonet_lr1e3_bs8 exp67_echonet_lr5e4_bs16 exp68_echonet_lr1e4_bs16
    exp69_echonet_lr1e3_bs16
    exp121_echodiff_wav2vec_lr1e4_bs16 exp122_echodiff_wav2vec_lr5e4_bs16
    exp123_echodiff_wav2vec_lr1e4_bs32 exp124_echodiff_wav2vec_lr5e5_bs16
)

in_whitelist() {
    local needle="$1"
    for w in "${WHITELIST[@]}"; do
        [ "$w" = "$needle" ] && return 0
    done
    return 1
}

get_config() {
    local dir="$1"
    if [[ "$dir" == echodiffusion_* ]]; then echo "echodiffusion"; return; fi
    if [[ "$dir" == echonet_* ]]; then echo "echonet"; return; fi
    if [[ "$dir" == pretrained_resnet_* ]]; then echo "pretrain_resnet"; return; fi
    if [[ "$dir" == pretrained_vit_* ]]; then echo "pretrain_vit"; return; fi
    if [[ "$dir" == vit_* ]]; then echo "vit"; return; fi
    if [[ "$dir" == batvision_* ]]; then echo "batvision"; return; fi
    if [[ "$dir" == *_batvision_* ]]; then echo "batvision"; return; fi
    if [[ "$dir" == *_baseline_* ]]; then echo "baseline"; return; fi
    if [[ "$dir" == *_crossattn_* ]]; then echo "foa_crossattn"; return; fi
    if [[ "$dir" == *_featbank_* ]]; then echo "foa_featbank"; return; fi
    if [[ "$dir" == *_msattn_* ]]; then echo "foa_msattn"; return; fi
    if [[ "$dir" == *_channelattn_* ]]; then echo "foa_channelattn"; return; fi
    if [[ "$dir" == *_foav2_* ]]; then echo "foa_v2"; return; fi
    if [[ "$dir" == *_foa_* ]]; then echo "foa"; return; fi
    echo "UNKNOWN"
}

get_exp_name() {
    echo "$1" | grep -oP 'exp\d+_.*'
}

run_test() {
    local GPU="$1"
    local CKPT_DIR="$2"
    local CONFIG="$3"
    local EXP_NAME="$4"
    local IDX="$5"
    local TOTAL="$6"

    local CKPT_PATH="$PROJECT_DIR/checkpoints/$CKPT_DIR/best_model.pth"

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] ($IDX/$TOTAL) $EXP_NAME [$CONFIG]"

    CUDA_VISIBLE_DEVICES="$GPU" python3 "$PROJECT_DIR/test.py" \
        --config "$CONFIG" \
        --experiment-name "$EXP_NAME" \
        --checkpoint-path "$CKPT_PATH" \
        --eval-on test \
        --vis-per-scene "$VIS_PER_SCENE" \
        > "$LOG_DIR/${EXP_NAME}_test.log" 2>&1

    local EC=$?
    if [ "$EC" -ne 0 ]; then
        echo "  [GPU=$GPU] $EXP_NAME FAILED (exit=$EC)"
    else
        echo "  [GPU=$GPU] $EXP_NAME OK"
    fi
}

# ── Discover matching checkpoints ──────────────────────────
CKPT_DIRS=()
CONFIGS=()
EXP_NAMES=()

for d in "$PROJECT_DIR"/checkpoints/*/; do
    [ ! -f "$d/best_model.pth" ] && continue
    dn=$(basename "$d")
    en=$(get_exp_name "$dn")
    [ -z "$en" ] && continue
    in_whitelist "$en" || continue
    cfg=$(get_config "$dn")
    if [ "$cfg" = "UNKNOWN" ]; then
        echo "WARN: skipping $dn (unknown config)"
        continue
    fi
    CKPT_DIRS+=("$dn")
    CONFIGS+=("$cfg")
    EXP_NAMES+=("$en")
done

TOTAL=${#CKPT_DIRS[@]}

echo "============================================================"
echo "bulk0410_test_78exps — $TOTAL / 78 matched — $(date)"
echo "4 workers × 1 GPU each (GPUs 0-3)"
echo "Vis per scene: $VIS_PER_SCENE"
echo "Logs: $LOG_DIR"
echo "============================================================"
echo ""

worker() {
    local GPU="$1"
    local WID="$2"
    for idx in "${!CKPT_DIRS[@]}"; do
        if [ $((idx % 4)) -eq "$WID" ]; then
            run_test "$GPU" "${CKPT_DIRS[$idx]}" "${CONFIGS[$idx]}" "${EXP_NAMES[$idx]}" "$((idx+1))" "$TOTAL"
        fi
    done
}

worker 0 0 &
W0=$!
worker 1 1 &
W1=$!
worker 2 2 &
W2=$!
worker 3 3 &
W3=$!

echo "Workers: W0=$W0(GPU0) W1=$W1(GPU1) W2=$W2(GPU2) W3=$W3(GPU3)"
echo ""

wait $W0 $W1 $W2 $W3 2>/dev/null

# ── Results summary ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "bulk0410_test_78exps finished — $(date)"
echo "============================================================"
echo ""
printf "%-6s %-55s %8s %8s %8s %8s %8s %8s %8s\n" "Exp" "Name" "ABS_REL" "RMSE" "Delta1" "Delta2" "Delta3" "Log10" "MAE"
echo "-------------------------------------------------------------------------------------------------------------------------------"

SUCCESS=0
FAILURES=0
for idx in "${!EXP_NAMES[@]}"; do
    en="${EXP_NAMES[$idx]}"
    LOG="$LOG_DIR/${en}_test.log"
    if [ ! -f "$LOG" ]; then
        printf "%-6s %-55s %8s\n" "$((idx+1))" "$en" "MISSING"
        FAILURES=$((FAILURES + 1))
        continue
    fi
    ABS_REL=$(grep "ABS_REL:" "$LOG" 2>/dev/null | awk '{print $2}')
    RMSE=$(grep "RMSE:" "$LOG" 2>/dev/null | awk '{print $2}')
    D1=$(grep "Delta1:" "$LOG" 2>/dev/null | awk '{print $2}')
    D2=$(grep "Delta2:" "$LOG" 2>/dev/null | awk '{print $2}')
    D3=$(grep "Delta3:" "$LOG" 2>/dev/null | awk '{print $2}')
    L10=$(grep "Log10:" "$LOG" 2>/dev/null | awk '{print $2}')
    MAE=$(grep "MAE:" "$LOG" 2>/dev/null | awk '{print $2}')

    if [ -n "$RMSE" ]; then
        printf "%-6s %-55s %8s %8s %8s %8s %8s %8s %8s\n" \
            "$((idx+1))" "$en" "$ABS_REL" "$RMSE" "$D1" "$D2" "$D3" "$L10" "$MAE"
        SUCCESS=$((SUCCESS + 1))
    else
        printf "%-6s %-55s %8s\n" "$((idx+1))" "$en" "FAILED"
        FAILURES=$((FAILURES + 1))
    fi
done

echo "-------------------------------------------------------------------------------------------------------------------------------"
echo "Success: $SUCCESS  Failed: $FAILURES  Total: $TOTAL / 78"
echo "============================================================"

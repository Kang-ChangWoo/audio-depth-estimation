#!/bin/bash
# ============================================================
# bulk0410_test_41exps_revised — evaluate the 41 experiments
# that have trained checkpoints but no test log yet.
#
# Coverage:
#   - exp70                 echonet         (1)
#   - exp71–75              batvision       (5)
#   - exp76,77,79,80        foa_crossattn+KL (4; exp78 still partial)
#   - exp81,83,84,85        foa_featbank+KL  (4; exp82 never trained)
#   - exp87,88,89           foa_msattn+KL    (3; exp86,90 never trained)
#   - exp91,92,93,95        foa_channelattn+KL (4; exp94 never trained)
#   - exp96,97,99,100,101,103,104,105,107,108,109,111,112,113,
#     115,116,117,119,120  foa (wider search) (19; exp98,102,106,
#                          110,114,118 never trained)
#   - exp125                echodiffusion+wav2vec (1)
#
# Auto-discovers checkpoints under $PROJECT_DIR/checkpoints and
# filters against the WHITELIST below. Config is derived from
# the checkpoint directory name.
#
# Runs 4 workers × 1 GPU each = GPUs 4,5,6,7.
# Logs: logs/bulk0410_test_41/
# ============================================================

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

# Activate conda env
if [ -z "$CONDA_SHLVL" ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi
conda activate shared_audio || { echo "ERROR: failed to activate conda env 'shared_audio'"; exit 1; }

if [ ! -f "$PROJECT_DIR/test.py" ]; then
    echo "ERROR: test.py not found in $PROJECT_DIR"
    return 1 2>/dev/null || exit 1
fi

LOG_DIR="$PROJECT_DIR/logs/bulk0410_test_41"
mkdir -p "$LOG_DIR"
VIS_PER_SCENE=100

WHITELIST=(
    exp70_echonet_lr2e3_bs16
    exp71_batvision_lr1e3_bs32 exp72_batvision_lr5e4_bs32 exp73_batvision_lr1e4_bs32
    exp74_batvision_lr1e3_bs16 exp75_batvision_lr2e3_bs32
    exp76_crossattn_lr1e3_fw0.1_kl0.02 exp77_crossattn_lr5e4_fw0.2_kl0.005
    exp79_crossattn_lr5e4_fw0.1_hw0.2_kl0.01 exp80_crossattn_lr1e3_fw0.3_kl0.01
    exp81_featbank_lr1e3_fw0.1_kl0.02 exp83_featbank_lr1e3_fw0.05_kl0.01
    exp84_featbank_lr5e4_fw0.1_hw0.2_kl0.01 exp85_featbank_lr1e3_fw0.3_kl0.01
    exp87_msattn_lr5e4_fw0.2_kl0.005 exp88_msattn_lr1e3_fw0.05_kl0.01
    exp89_msattn_lr5e4_fw0.1_hw0.2_kl0.01
    exp91_channelattn_lr1e3_fw0.1_kl0.02 exp92_channelattn_lr5e4_fw0.2_kl0.005
    exp93_channelattn_lr1e3_fw0.05_kl0.01 exp95_channelattn_lr1e3_fw0.3_kl0.01
    exp96_foa_lr2e4_dw1.0_fw0.1_hw0.1 exp97_foa_lr3e4_dw1.0_fw0.1_hw0.1
    exp99_foa_lr1e3_dw1.5_fw0.1_hw0.1 exp100_foa_lr1e3_fw0.15_hw0.1
    exp101_foa_lr1e3_fw0.1_hw0.15 exp103_foa_lr5e4_fw0.15_hw0.1
    exp104_foa_lr5e4_fw0.1_hw0.15 exp105_foa_lr1e3_fw0.3_hw0.1
    exp107_foa_lr5e4_fw0.3_hw0.1 exp108_foa_lr5e4_dw2.0_fw0.1_hw0.1
    exp109_foa_lr1e3_dw1.0_fw0.2_hw0.2_freeze5
    exp111_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze15
    exp112_foa_lr5e4_dw1.0_fw0.1_hw0.1_freeze10
    exp113_foa_lr1e3_dw1.0_fw0.05_hw0.05
    exp115_foa_lr1e3_dw2.0_fw0.2_hw0.1 exp116_foa_lr3e4_fw0.2_hw0.1
    exp117_foa_lr2e4_dw1.0_fw0.2_hw0.2
    exp119_foa_lr1e3_fw0.1_hw0.2_freeze3
    exp120_foa_lr5e4_fw0.05_hw0.1_freeze5
    exp125_echodiff_wav2vec_lr1e4_bs8
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

GPU_LIST=(4 5 6 7)
NUM_WORKERS=${#GPU_LIST[@]}

echo "============================================================"
echo "bulk0410_test_41exps_revised — $TOTAL / 41 matched — $(date)"
echo "$NUM_WORKERS workers × 1 GPU each (GPUs ${GPU_LIST[*]})"
echo "Vis per scene: $VIS_PER_SCENE"
echo "Logs: $LOG_DIR"
echo "============================================================"
echo ""

worker() {
    local GPU="$1"
    local WID="$2"
    for idx in "${!CKPT_DIRS[@]}"; do
        if [ $((idx % NUM_WORKERS)) -eq "$WID" ]; then
            run_test "$GPU" "${CKPT_DIRS[$idx]}" "${CONFIGS[$idx]}" "${EXP_NAMES[$idx]}" "$((idx+1))" "$TOTAL"
        fi
    done
}

WPIDS=()
for wid in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$wid]}"
    worker "$gpu" "$wid" &
    WPIDS+=($!)
    echo "Worker W$wid pid=${WPIDS[$wid]} GPU=$gpu"
done
echo ""

wait "${WPIDS[@]}" 2>/dev/null

echo ""
echo "============================================================"
echo "bulk0410_test_41exps_revised finished — $(date)"
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
echo "Success: $SUCCESS  Failed: $FAILURES  Total: $TOTAL / 41"
echo "============================================================"

#!/bin/bash
# ============================================================
# summary_test.sh — Consolidated test script for ALL completed experiments
#
# Auto-discovers checkpoints and matches to configs.
# Only tests experiments with best_model.pth.
#
# Usage: GPUS="0,1" WORKERS=4 bash summary_test.sh
# ============================================================
set -euo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

GPUS="${GPUS:-0,1}"
NUM_WORKERS="${WORKERS:-4}"
VIS_PER_SCENE="${VIS_PER_SCENE:-100}"

LOG_DIR="$PROJECT_DIR/logs/summary_test"
mkdir -p "$LOG_DIR"

get_config() {
    local dir="$1"
    case "$dir" in
        *pretrained_vit_foa_v3*) echo "pretrain_vit_foa_v3" ;;
        *pretrained_vit_foa*)    echo "pretrain_vit_foa" ;;
        *pretrained_vit*)        echo "pretrain_vit" ;;
        *pretrained_resnet*)     echo "pretrain_resnet" ;;
        *echodiffusion*)         echo "echodiffusion" ;;
        *echonet*)               echo "echonet" ;;
        *_crossattn_*)           echo "foa_crossattn" ;;
        *_featbank_*)            echo "foa_featbank" ;;
        *_msattn_*)              echo "foa_msattn" ;;
        *_channelattn_*)         echo "foa_channelattn" ;;
        *_foav2_*|*_foa_v2*)     echo "foa_v2" ;;
        *CW_rot_foa_v2*)         echo "foa_v2" ;;
        *CW_rot_foa_crossattn*)  echo "foa_crossattn" ;;
        *CW_rot_foa_featbank*)   echo "foa_featbank" ;;
        *CW_rot_foa_msattn*)     echo "foa_msattn" ;;
        *CW_rot_foa_channelattn*) echo "foa_channelattn" ;;
        *CW_rot_foa*)            echo "foa" ;;
        *foa0415v1*|*foa_0415_v1*) echo "foa_0415_v1" ;;
        *foa0415v2*|*foa_0415_v2*) echo "foa_0415_v2" ;;
        *foa0415v3*|*foa_0415_v3*) echo "foa_0415_v3" ;;
        *foa0415v4*|*foa_0415_v4*) echo "foa_0415_v4" ;;
        *foa0415v5*|*foa_0415_v5*) echo "foa_0415_v5" ;;
        # N3 variants (exp166-186)
        *oracle_nc3*)            echo "foa_oracle_nc3" ;;
        *oracle_nc1*)            echo "foa_oracle_nc1" ;;
        *n3film*)                echo "n3_film" ;;
        *n3mssh*)                echo "n3_multiscale_sh" ;;
        *n3eattn*)               echo "n3_energy_attn" ;;
        *n3twin*)                echo "n3_temporal_window" ;;
        *v1base*)                echo "foa_0415_v1" ;;
        *_foa_*)                 echo "foa" ;;
        *batvision*)             echo "batvision" ;;
        *vit_*)                  echo "vit" ;;
        *baseline*)              echo "baseline" ;;
        *)                       echo "UNKNOWN" ;;
    esac
}

get_exp_name() {
    local dir="$1"
    # Extract experiment name from checkpoint dir (after last AdamW_)
    echo "$dir" | sed 's/.*AdamW_//'
}

# Models trained with --rotate-canonical must be tested with it too.
needs_rotate_canonical() {
    local dir="$1"
    case "$dir" in
        *foa0415*|*foa_0415_*|*pvitfoa*|*pretrained_vit_foa*|*CW_rot_*|\
        *n3film*|*n3mssh*|*n3eattn*|*n3twin*|*oracle_nc*|*v1base*)
            return 0 ;;
        *) return 1 ;;
    esac
}

echo "============================================================"
echo "summary_test.sh — $(date)"
echo "GPUs: $GPUS | Workers: $NUM_WORKERS | Vis: $VIS_PER_SCENE"
echo "============================================================"

TOTAL=0
SUCCESS=0
FAILURES=0

for CKPT_DIR in checkpoints/*/; do
    [ -f "$CKPT_DIR/best_model.pth" ] || continue

    DIR_NAME=$(basename "$CKPT_DIR")

    # Skip JS-related
    case "$DIR_NAME" in
        *foa_v2_js*|*js_smoke*|*hope*|*dryrun*) continue ;;
    esac

    CONFIG=$(get_config "$DIR_NAME")
    EXP=$(get_exp_name "$DIR_NAME")
    CKPT_PATH="$CKPT_DIR/best_model.pth"
    LOG_FILE="$LOG_DIR/${EXP}_test.log"

    if [ "$CONFIG" = "UNKNOWN" ]; then
        echo "  SKIP: $EXP (unknown config)"
        continue
    fi

    TOTAL=$((TOTAL + 1))
    EXTRA_FLAGS=""
    if needs_rotate_canonical "$DIR_NAME"; then
        EXTRA_FLAGS="--rotate-canonical"
    fi
    echo "[$(date +%H:%M:%S)] ($TOTAL) Testing $EXP [$CONFIG]${EXTRA_FLAGS:+ (rotate-canonical)}"

    CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        python3 -u test.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --checkpoint-path "$CKPT_PATH" \
        --eval-on test \
        --batch-size 1 \
        --vis-per-scene "$VIS_PER_SCENE" \
        $EXTRA_FLAGS \
        > "$LOG_FILE" 2>&1

    EC=$?
    if [ "$EC" -ne 0 ]; then
        echo "  $EXP FAILED (exit=$EC)"
        FAILURES=$((FAILURES + 1))
    else
        ABS=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$LOG_FILE" | head -1)
        RMSE=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$LOG_FILE" | head -1)
        D1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$LOG_FILE" | head -1)
        echo "  $EXP OK (ABS=$ABS RMSE=$RMSE D1=$D1)"
        SUCCESS=$((SUCCESS + 1))
    fi
done

echo ""
echo "============================================================"
echo "summary_test.sh finished — $(date)"
echo "Success: $SUCCESS  Failed: $FAILURES  Total: $TOTAL"
echo "============================================================"

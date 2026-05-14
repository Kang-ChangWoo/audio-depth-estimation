#!/bin/bash
# ============================================================
# test.sh — Single-experiment test runner
#
# Env vars:
#   CONFIG      Config name (e.g. foa, foa_oracle_nc3, n3_energy_attn)  [required]
#   EXP         Experiment name (e.g. exp172_n3eattn_lr1e3_lsh0.1)      [required]
#   CKPT        Path to best_model.pth [optional — auto-discovered if omitted]
#   GPUS        CUDA_VISIBLE_DEVICES                                    [default: 0]
#   BS          Batch size                                              [default: 1]
#   EVAL_ON     test|val                                                [default: test]
#   VIS         Visualizations per scene                                [default: 100]
#   ROTATE      "1" to pass --rotate-canonical                          [auto-detect]
#
# Examples:
#   CONFIG=foa_oracle_nc3 EXP=exp179_oracle_nc3_lr5e4_lsh0.1 bash scripts/test.sh
#   CONFIG=n3_energy_attn EXP=exp172_n3eattn_lr1e3_lsh0.1 GPUS=0,1 bash scripts/test.sh
#   CONFIG=foa EXP=exp111_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze15 bash scripts/test.sh
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CONFIG:?CONFIG env var required (e.g. foa_oracle_nc3)}"
: "${EXP:?EXP env var required (e.g. exp179_oracle_nc3_lr5e4_lsh0.1)}"
GPUS="${GPUS:-0}"
BS="${BS:-1}"
EVAL_ON="${EVAL_ON:-test}"
VIS="${VIS:-100}"

# Auto-discover checkpoint if CKPT not provided.
if [ -z "${CKPT:-}" ]; then
    CKPT=$(find runs/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: no checkpoint found for EXP=$EXP under runs/" >&2
        exit 1
    fi
fi

# Auto-detect rotate-canonical unless user explicitly sets ROTATE.
ROTATE="${ROTATE:-auto}"
if [ "$ROTATE" = "auto" ]; then
    case "$EXP$CKPT" in
        *foa0415*|*foa_0415_*|*pvitfoa*|*pretrained_vit_foa*|*CW_rot_*|\
        *n3film*|*n3mssh*|*n3eattn*|*n3twin*|*oracle_nc*|*v1base*|\
        *n2_6ch*|*n2_trms*|*n2_tenergy*|*n2_dual*|*n2_stft*|*n2_temap*|*n2_xattn*)
            ROTATE=1 ;;
        *) ROTATE=0 ;;
    esac
fi
EXTRA=""
[ "$ROTATE" = "1" ] && EXTRA="--rotate-canonical"

echo "[$(date +%H:%M:%S)] TEST $EXP [$CONFIG] on GPU=$GPUS ckpt=$CKPT${EXTRA:+ rotate-canonical}"

CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python3 -u test.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --checkpoint-path "$CKPT" \
        --eval-on "$EVAL_ON" \
        --batch-size "$BS" \
        --vis-per-scene "$VIS" \
        $EXTRA

#!/bin/bash
# Test the 6 FOA models trained by scripts/train_CW.sh.
#
# Checkpoint layout produced by train_CW.sh:
#   checkpoints/unet_256_soundspaces_BS32_Lr0.001_AdamW_${TAG}_${cfg}/best_model.pth
# with ${TAG} defaulting to "CW_rot" and ${cfg} in MODELS below.
#
# Layout: 1 GPU per model, all 6 models launched concurrently
# (GPUs 0..5 by default, edit GPUS= below to change the mapping).
#
# Usage:
#   bash scripts/test_CW.sh [EXPERIMENT_TAG] [EVAL_ON] [CKPT] [VIS_PER_SCENE]
#
# Examples:
#   bash scripts/test_CW.sh                         # CW_rot, test split, best
#   bash scripts/test_CW.sh CW_rot val              # eval on val split
#   bash scripts/test_CW.sh CW_rot test 40          # use checkpoint_40.pth
set -euo pipefail

EXPERIMENT="${1:-CW_rot}"
EVAL_ON="${2:-test}"
CKPT="${3:-best}"
VIS_PER_SCENE="${4:-100}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MODELS=(foa foa_v2 foa_channelattn foa_crossattn foa_featbank foa_msattn)
# One GPU per model, all 6 launched concurrently.
GPUS=(0 1 2 3 4 5)

# Must match what train.py builds from generator/dataset/BS/LR/optimizer/exp_name.
# All 6 FOA configs share BS=32, Lr=0.001, AdamW, unet_256, soundspaces (verified).
BS=32
LR=0.001
OPT="AdamW"
GENERATOR="unet_256"
DS_NAME="soundspaces"

ckpt_path_for() {
    local cfg="$1"
    local name="${EXPERIMENT}_${cfg}"
    local ckpt_dir="${GENERATOR}_${DS_NAME}_BS${BS}_Lr${LR}_${OPT}_${name}"
    if [[ "${CKPT}" == "best" ]]; then
        echo "${PROJECT_DIR}/checkpoints/${ckpt_dir}/best_model.pth"
    else
        echo "${PROJECT_DIR}/checkpoints/${ckpt_dir}/checkpoint_${CKPT}.pth"
    fi
}

echo "============================================================"
echo "  Experiment tag: ${EXPERIMENT}"
echo "  Eval on:        ${EVAL_ON}"
echo "  Checkpoint:     ${CKPT}"
echo "  Vis/scene:      ${VIS_PER_SCENE}"
echo "  FOA rotation:   ON (--rotate-canonical)"
echo "============================================================"
echo "  Resolved checkpoints:"
missing=0
for cfg in "${MODELS[@]}"; do
    p="$(ckpt_path_for "${cfg}")"
    if [[ -f "${p}" ]]; then
        echo "    [OK]   ${cfg}: ${p}"
    else
        echo "    [MISS] ${cfg}: ${p}"
        missing=$((missing+1))
    fi
done
echo "============================================================"
if (( missing == ${#MODELS[@]} )); then
    echo "ERROR: no checkpoints found for tag '${EXPERIMENT}'. Did train_CW.sh finish?"
    exit 1
fi
if (( missing > 0 )); then
    echo "WARN: ${missing} checkpoint(s) missing — those models will be skipped."
fi

run_test() {
    local gpu="$1"
    local cfg="$2"
    local name="${EXPERIMENT}_${cfg}"
    local ckpt_path
    ckpt_path="$(ckpt_path_for "${cfg}")"
    local log="${LOG_DIR}/test_${name}.log"

    if [[ ! -f "${ckpt_path}" ]]; then
        echo "[$(date +%T)] SKIP ${cfg}: checkpoint not found" | tee "${log}"
        return 0
    fi

    echo "[$(date +%T)] start: cfg=${cfg} gpu=${gpu} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 test.py \
        --config "${cfg}" \
        --experiment-name "${name}" \
        --checkpoint-path "${ckpt_path}" \
        --eval-on "${EVAL_ON}" \
        --checkpoints "${CKPT}" \
        --vis-per-scene "${VIS_PER_SCENE}" \
        --rotate-canonical \
        > "${log}" 2>&1
    echo "[$(date +%T)] done:  cfg=${cfg}"
}

pids=()
n=${#MODELS[@]}
for ((i=0; i<n; i++)); do
    run_test "${GPUS[i]}" "${MODELS[i]}" &
    pids+=($!)
    # Small stagger to avoid simultaneous CUDA init thundering herd.
    if (( i+1 < n )); then
        sleep 8
    fi
done
wait "${pids[@]}"

echo
echo "All 6 FOA models tested. Per-model logs: ${LOG_DIR}/test_${EXPERIMENT}_*.log"
echo "Summary:"
for cfg in "${MODELS[@]}"; do
    log="${LOG_DIR}/test_${EXPERIMENT}_${cfg}.log"
    if [[ -f "${log}" ]]; then
        rmse=$(grep -E "^\s*RMSE:" "${log}" | tail -1 | awk '{print $2}')
        absrel=$(grep -E "^\s*ABS_REL:" "${log}" | tail -1 | awk '{print $2}')
        printf "  %-20s  ABS_REL=%s  RMSE=%s\n" "${cfg}" "${absrel:-N/A}" "${rmse:-N/A}"
    fi
done

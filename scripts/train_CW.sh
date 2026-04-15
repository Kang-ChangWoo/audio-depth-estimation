#!/bin/bash
# Train the 6 FOA models (foa1, foa2 + 4 variants) with canonical-frame
# FOA rotation (data/dataset_rotated.py).
#
# Layout: 4 GPUs per model, 2 models in parallel.
#   slot A -> GPUs 0,1,2,3
#   slot B -> GPUs 4,5,6,7
# Six models -> three sequential rounds of 2.
#
# Usage: bash scripts/train_CW.sh [EXPERIMENT_TAG] [EPOCHS] [NUM_WORKERS]
set -euo pipefail

EXPERIMENT="${1:-CW_rot}"
EPOCHS="${2:-40}"
NUM_WORKERS="${3:-8}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MODELS=(foa foa_v2 foa_channelattn foa_crossattn foa_featbank foa_msattn)
GPUS_A="0,1,2,3"
GPUS_B="4,5,6,7"

echo "============================================================"
echo "  Experiment tag: ${EXPERIMENT}"
echo "  Epochs:         ${EPOCHS}"
echo "  Workers:        ${NUM_WORKERS}"
echo "  Models:         ${MODELS[*]}"
echo "  Parallelism:    2 models x 4 GPUs (${GPUS_A} | ${GPUS_B})"
echo "  FOA rotation:   ON (--rotate-canonical)"
echo "============================================================"

run_train() {
    local gpus="$1"
    local cfg="$2"
    local name="${EXPERIMENT}_${cfg}"
    local log="$LOG_DIR/train_${name}.log"
    echo "[$(date +%T)] start: cfg=${cfg} gpus=${gpus} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpus}" python3 train.py \
        --config "${cfg}" \
        --experiment-name "${name}" \
        --rotate-canonical \
        --epochs "${EPOCHS}" \
        --num-workers "${NUM_WORKERS}" \
        > "${log}" 2>&1
    echo "[$(date +%T)] done:  cfg=${cfg}"
}

# Prime the filesystem dentry cache once so both parallel jobs don't
# race through 100k+ stat syscalls on cold cache.
DATASET_DIR="$(python3 -c 'import yaml,sys;print(yaml.safe_load(open("config/foa.yaml"))["dataset"]["dataset_dir"])')"
echo "[$(date +%T)] priming FS cache: ${DATASET_DIR}"
find "${DATASET_DIR}" -maxdepth 3 -type f >/dev/null 2>&1 || true

n=${#MODELS[@]}
for ((i=0; i<n; i+=2)); do
    cfg_a="${MODELS[i]}"
    run_train "${GPUS_A}" "${cfg_a}" &
    pid_a=$!

    if (( i+1 < n )); then
        # Stagger so the two jobs don't collide on CUDA / NCCL init
        # and on the dataset fast-cache write the very first time.
        sleep 20
        cfg_b="${MODELS[i+1]}"
        run_train "${GPUS_B}" "${cfg_b}" &
        pid_b=$!
        wait "${pid_a}" "${pid_b}"
    else
        wait "${pid_a}"
    fi
done

echo "All 6 FOA models trained with canonical rotation."

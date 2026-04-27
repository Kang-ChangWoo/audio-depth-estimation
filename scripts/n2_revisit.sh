#!/bin/bash
# ============================================================
# n2_revisit.sh — RETRAIN legacy baselines on RADIAL depth GT,
# then test each fresh checkpoint.
#
# Ranges (original exp_id → new exp_id):
#     01–05   baseline (UNet)             → exp350–354
#     06–10   vit (scratch)               → exp355–359
#     11–15   echodiffusion (no wav2vec)  → exp360–364
#     56–60   pretrain_resnet             → exp365–369
#     61–65   pretrain_vit                → exp370–374
#     66–70   echonet                     → exp375–379
#     71–75   batvision                   → exp380–384
#   121–125   echodiffusion + wav2vec     → exp385–389
#
# 40 experiments total (exp350-389), 4 workers × 2 GPUs (0-7)
#
# Log layout:
#     logs/n2_revisit_train/<EXP>.log
#     logs/n2_revisit_test/<EXP>_test.log
#
# Usage:
#     bash scripts/n2_revisit.sh
#     GPU_PAIRS="0,1 2,3 4,5 6,7" bash scripts/n2_revisit.sh
#     DATASET_DIR=/local1/matterport3d_0303renew bash scripts/n2_revisit.sh
#
# DATASET_DIR overrides the dataset_dir baked into each config/*.yaml so
# the same script runs on any node (data lives wherever the node has it
# copied). Leave unset to use the yaml default.
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303renew}"
EPOCHS="${EPOCHS:-40}"
# Lowered from 4 → 2 (2026-04-24). 4 concurrent jobs × 4 workers = 16
# dataloader processes competing for CPU and RAM; that was contributing
# to worker-OOM deaths (see exp363 post-mortem). 4 × 2 = 8 keeps the
# dataloader responsive without overloading.
WORKERS="${WORKERS:-2}"
VIS_PER_SCENE=0

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n2_revisit_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n2_revisit_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

GPU_PAIRS_STR="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
read -r -a GPU_PAIRS <<< "$GPU_PAIRS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

train_and_test() {
    local GPU="$1" CONFIG="$2" EXP="$3" LR="$4" BS="$5"
    shift 5
    local EXTRA=("$@")

    if [ -f "$TEST_LOG_DIR/${EXP}_test.log" ] && \
       grep -q 'ABS_REL:' "$TEST_LOG_DIR/${EXP}_test.log"; then
        echo "[$(date +%H:%M:%S)] [GPU=$GPU] SKIP $EXP (already completed)"
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TRAIN $EXP (cfg=$CONFIG lr=$LR bs=$BS)"

    # Per-job wall-clock budget. 36 000 s = 10 h is enough for even
    # pretrain_vit bs=8 @ 1200 s/epoch × 40 epochs ≈ 13.4 h is NOT enough,
    # so budget per variant: default to JOB_TIMEOUT below (set via env
    # if you want a different cap for this sweep). SIGTERM fires at
    # $JOB_TIMEOUT, Python gets to print a traceback and save state,
    # SIGKILL fires 60 s later as a hard backstop.
    local JOB_TIMEOUT="${JOB_TIMEOUT:-54000}"   # 15 h default
    local ec=0
    CUDA_VISIBLE_DEVICES="$GPU" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        timeout --signal=TERM --kill-after=60 "$JOB_TIMEOUT" \
        "$PYTHON" train.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BS" \
        --num-workers "$WORKERS" \
        --depth-dir "$DEPTH_DIR" \
        "${DATASET_ARGS[@]}" \
        "${EXTRA[@]}" \
        > "$TRAIN_LOG_DIR/${EXP}.log" 2>&1 || ec=$?

    # 124 = timeout hit SIGTERM wall. 137 = SIGKILL backstop. 141 = SIGPIPE.
    if [ "$ec" -eq 124 ] || [ "$ec" -eq 137 ]; then
        echo "  [GPU=$GPU] $EXP TRAIN_TIMEOUT (exit=$ec, budget=${JOB_TIMEOUT}s)"
        tail -3 "$TRAIN_LOG_DIR/${EXP}.log" | sed "s/^/      /"
        return
    fi
    if [ "$ec" -ne 0 ] && [ "$ec" -ne 141 ]; then
        echo "  [GPU=$GPU] $EXP TRAIN_FAILED (exit=$ec)"
        tail -3 "$TRAIN_LOG_DIR/${EXP}.log" | sed "s/^/      /"
        return
    fi
    echo "  [GPU=$GPU] $EXP TRAIN_DONE"

    local CKPT
    CKPT=$(find checkpoints/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "  [GPU=$GPU] $EXP TEST_SKIP (no checkpoint)"
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TEST $EXP"
    ec=0
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        "$PYTHON" -u test.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --checkpoint-path "$CKPT" \
        --eval-on test \
        --batch-size 16 \
        --num-workers "$WORKERS" \
        --vis-per-scene "$VIS_PER_SCENE" \
        --depth-dir "$DEPTH_DIR" \
        "${DATASET_ARGS[@]}" \
        "${EXTRA[@]}" \
        > "$TEST_LOG_DIR/${EXP}_test.log" 2>&1 || ec=$?

    if [ "$ec" -ne 0 ]; then
        echo "  [GPU=$GPU] $EXP TEST_FAILED (exit=$ec)"
    else
        local abs rmse d1
        abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        echo "  [GPU=$GPU] $EXP TEST_OK  ABS=$abs RMSE=$rmse D1=$d1"
    fi
}

# ============================================================
# Experiment definitions: "CONFIG EXP LR BS [EXTRA...]"
# ============================================================
ALL_EXPS=(
    # --- baseline (UNet) → exp350-354 ---
    # "baseline         exp350_baseline_lr1e3_bs16    0.001   16"
    # "baseline         exp351_baseline_lr5e4_bs16    0.0005  16"
    # "baseline         exp352_baseline_lr1e4_bs16    0.0001  16"
    # "baseline         exp353_baseline_lr1e3_bs8     0.001    8"
    # "baseline         exp354_baseline_lr5e4_bs8     0.0005   8"
    # --- vit (scratch) → exp355-359 — all DONE (train+test finished) ---
    # Left at their original bs=8/4 specs so the names match the
    # already-saved checkpoints and test logs (grep ABS_REL in
    # logs/n2_revisit_test/exp35*_test.log).
    # "vit              exp355_vit_lr1e4_bs8          0.0001   8"    # DONE ABS=0.4989 RMSE=1.2621
    # "vit              exp356_vit_lr5e5_bs8          0.00005  8"    # DONE ABS=0.5005 RMSE=1.2631
    # "vit              exp357_vit_lr1e4_bs4          0.0001   4"    # DONE ABS=0.4703 RMSE=1.3107
    # "vit              exp358_vit_lr5e4_bs8          0.0005   8"    # DONE ABS=0.5664 RMSE=1.3783
    # "vit              exp359_vit_lr1e5_bs8          0.00001  8"    # DONE ABS=0.5130 RMSE=1.2593
    # --- echodiffusion (no wav2vec) → exp360-364 ---
    # BS bumped 32→48, 16→32 — pushing toward ~80% of 48 GiB (~40 GiB at
    # steady state). Old bs=16 was only 13 GiB (27%) on-GPU; scaling
    # linearly gives ~39 GiB at bs=48. Leaves ~8 GiB for backward peak.
    # If any of these OOM, cap that row at bs=32 and re-run.
    "echodiffusion    exp360_echodiff_lr1e4_bs48    0.0001   48"
    "echodiffusion    exp361_echodiff_lr5e5_bs48   0.00005  48"
    "echodiffusion    exp362_echodiff_lr1e4_bs32    0.0001   32"
    "echodiffusion    exp363_echodiff_lr5e4_bs48   0.0005   48"
    "echodiffusion    exp364_echodiff_lr1e5_bs48   0.00001  48"
    # --- pretrain_resnet → exp365-369 ---
    # BS bumped further to 160 / 96. At bs=32 the model only used 7 GiB.
    # At 30M params the activation memory dominates, scaling linearly in
    # BS. Target ~37 GiB at steady state (~75% of 48 GiB) — bs=160.
    # "pretrain_resnet  exp365_resnet_lr1e4_bs32      0.0001   32"  # DONE
    "pretrain_resnet  exp366_resnet_lr5e5_bs160     0.00005 160"
    "pretrain_resnet  exp367_resnet_lr5e4_bs160     0.0005  160"
    "pretrain_resnet  exp368_resnet_lr1e4_bs96      0.0001   96"
    "pretrain_resnet  exp369_resnet_lr3e4_bs160     0.0003  160"
    # --- pretrain_vit → exp370-374 ---
    # BS bumped 32→48 / 16→24. 90M ViT was 12 GiB at bs=16 → ~36 GiB at
    # bs=48 (~75%). Further bump needs bf16.
    "pretrain_vit     exp370_previt_lr1e4_bs48      0.0001   48"
    "pretrain_vit     exp371_previt_lr5e5_bs48     0.00005  48"
    "pretrain_vit     exp372_previt_lr5e4_bs48     0.0005   48"
    "pretrain_vit     exp373_previt_lr1e4_bs24      0.0001   24"
    "pretrain_vit     exp374_previt_lr3e5_bs48     0.00003  48"
    # --- echonet → exp375-379 ---
    # BS bumped 32→48 / 16→24. Echonet is 321M with a bilinear fusion so
    # we don't 2× again — 1.5× lands near ~39 GiB (~80%).
    "echonet          exp375_echonet_lr1e3_bs24     0.001    24"
    "echonet          exp376_echonet_lr5e4_bs48     0.0005   48"
    "echonet          exp377_echonet_lr1e4_bs48     0.0001   48"
    "echonet          exp378_echonet_lr1e3_bs48     0.001    48"
    "echonet          exp379_echonet_lr2e3_bs48     0.002    48"
    # --- batvision → exp380-384 ---
    # BS bumped 64→96 / 32→48. UNet-based 54M model. bs=32 was 13 GiB;
    # bs=96 lands near ~39 GiB (~80%).
    "batvision        exp380_batvision_lr1e3_bs96   0.001    96"
    "batvision        exp381_batvision_lr5e4_bs96   0.0005   96"
    "batvision        exp382_batvision_lr1e4_bs96   0.0001   96"
    "batvision        exp383_batvision_lr1e3_bs48   0.001    48"
    "batvision        exp384_batvision_lr2e3_bs96   0.002    96"
    # --- echodiffusion + wav2vec → exp385-389 ---
    # BS bumped 16→32, 32→48, 8→16, 4→8. Wav2vec (+95M frozen) keeps
    # steady-state memory ~15-20% higher than plain echodiffusion at the
    # same BS. Cap at bs=48 to stay under OOM even with wav2vec features
    # peaking during the first few backward steps.
    "echodiffusion    exp385_echodiff_wav2vec_lr1e4_bs32   0.0001   32"
    "echodiffusion    exp386_echodiff_wav2vec_lr5e4_bs32   0.0005   32"
    "echodiffusion    exp387_echodiff_wav2vec_lr1e4_bs48   0.0001   48"
    "echodiffusion    exp388_echodiff_wav2vec_lr5e5_bs16   0.00005  16"
    "echodiffusion    exp389_echodiff_wav2vec_lr1e4_bs4    0.0001    4"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache (avoids 4 workers scanning in parallel)
echo "Pre-warming dataset cache for radial depth..."
"$PYTHON" -c "
from data.dataset import SoundSpacesDataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/baseline.yaml')
cfg.dataset.depth_dir = '$DEPTH_DIR'
if '$DATASET_DIR':
    cfg.dataset.dataset_dir = '$DATASET_DIR'
for split in ['train', 'val', 'test']:
    SoundSpacesDataset(cfg, split=split)
print('Cache warm.')
" 2>&1 | tail -5
echo ""

echo "============================================================"
echo "n2_revisit — $TOTAL experiments on radial depth GT"
echo "  EPOCHS=$EPOCHS  depth_dir=$DEPTH_DIR"
echo "  dataset_dir=${DATASET_DIR:-<yaml default>}"
echo "  $NUM_WORKERS workers × 2 GPUs: ${GPU_PAIRS[*]}"
echo "  $(date)"
echo "============================================================"

run_worker() {
    local WORKER_ID=$1
    local GPU="${GPU_PAIRS[$WORKER_ID]}"

    for (( i=WORKER_ID; i<TOTAL; i+=NUM_WORKERS )); do
        local SPEC="${ALL_EXPS[$i]}"
        local CONFIG=$(echo "$SPEC" | awk '{print $1}')
        local EXP=$(echo    "$SPEC" | awk '{print $2}')
        local LR=$(echo     "$SPEC" | awk '{print $3}')
        local BS=$(echo     "$SPEC" | awk '{print $4}')

        train_and_test "$GPU" "$CONFIG" "$EXP" "$LR" "$BS"
    done
}

PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
    run_worker "$w" &
    PIDS+=($!)
    echo "Worker $w launched (GPU ${GPU_PAIRS[$w]}, PID ${PIDS[-1]})"
done

echo "All $NUM_WORKERS workers running. Waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

echo ""
echo "============================================================"
echo "n2_revisit finished — $(date)"
echo "Workers failed: $FAIL / $NUM_WORKERS"
echo "============================================================"
echo ""
printf "%-45s %8s %8s %8s\n" "Experiment" "ABS_REL" "RMSE" "Delta1"
echo "---------------------------------------------------------------------"

SUCCESS=0
for SPEC in "${ALL_EXPS[@]}"; do
    EXP=$(echo "$SPEC" | awk '{print $2}')
    LOG="$TEST_LOG_DIR/${EXP}_test.log"
    if [ -f "$LOG" ]; then
        abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$LOG" | head -1)
        rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$LOG" | head -1)
        d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$LOG" | head -1)
        if [ -n "$abs" ]; then
            printf "%-45s %8s %8s %8s\n" "$EXP" "$abs" "$rmse" "$d1"
            SUCCESS=$((SUCCESS + 1))
        else
            printf "%-45s %8s\n" "$EXP" "FAILED"
        fi
    else
        printf "%-45s %8s\n" "$EXP" "NO_RESULT"
    fi
done

echo "---------------------------------------------------------------------"
echo "Success: $SUCCESS / $TOTAL"
echo "============================================================"

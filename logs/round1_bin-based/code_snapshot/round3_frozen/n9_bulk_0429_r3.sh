#!/bin/bash
# ============================================================
# n9_bulk_0429_r3.sh — Round 5 (n9): paired 40-epoch baselines + R20 anchors
#
# CONTEXT
# --------
# Companion to n2_bulk_0429_r3.sh. n2 explores soft-quantile × SH ×
# cylindrical at bs=48 / 20 epochs (22 cells). n9 closes the *epoch
# budget* gap that round 4 left open: round-1/2 cells trained 20 epochs
# at bs=16/32; analysis flagged that distribution head may simply
# underfit at 20 epochs and that comparing against round-2 expectation
# best (exp907) requires either a 40-epoch paired baseline or a clear
# gain inside 20 epochs.
#
# n9 cells (10 main):
#   Block F (40-ep paired baseline):
#     exp958  S40 echodiffusion scalar          ← round-1 echodiff exp11 budget match
#     exp959  S40 echorange scalar              ← scalar gap @ 40 ep
#     exp960  R40 expectation                    ← exp907 recipe @ 40 ep
#     exp961  R40 median                         ← exp907 recipe + median output
#   Block G:  exp962  R20 soft-quantile (anchor)
#   Block H:  exp963  R20 SH (L=2 λ=0.02)
#   Block I:  exp964  R20 combo (sq + SH)
#   Block J:  exp965  R20 anchor reproduction (exp907 same-env check)
#             exp966  R20 cylindrical horizontal baseline
#             exp967  R20 cylindrical horizontal + soft-quantile
#
# All R/Cyl cells use the same anchor as exp907:
#   bins=32 log,  r∈[0.1,10],  σ=0.14,
#   cos-lat-weight + far-mask, NLL+BerHu+SILog (silog=0.5).
#
# Hardware constraints (from round-1 n9 driver issue):
#   - Only GPUs 0 and 1 are usable. GPU 2's NVML failure broke NCCL
#     system-wide; even DP[0,1] failed with NCCL Error 2. Only solo
#     single-GPU per process works. Forced GPUS="0 1" below.
#   - 2 workers in parallel, each pinned to its own GPU; cells
#     distribute round-robin.
#
# Time budget (estimated):
#   Block F (4 × 40-ep) ≈ 4 × 5h / 2 workers = 10h
#   Blocks G–J (6 × 20-ep) ≈ 6 × 2.5h / 2 workers = 7.5h
#   Test phase (4 ckpt-tags × 10 cells × ~5min) ≈ 3.3h on 2 workers
#   Total ≈ 21h. skip-if-done lets the next morning wrap up if needed.
#
# Reference baselines (round 1–2):
#   echodiff exp11:   ABS 0.4300  RMSE 1.1060  D1 0.4876   ← strongest
#   softmax exp906:   ABS 0.4814  RMSE 1.2532  D1 0.5079
#   exp907 expect:    ABS 0.481   RMSE 1.25    D1 0.51
#   exp907 median(eval): ABS 0.420  RMSE 1.36     ← best ABS_REL
#   exp912 scalar @bs32: ABS 0.4349  RMSE 1.27
#
# Promote/kill criteria:
#   F1/F2 (S40 scalar) close exp11 within 2 % → epoch budget was THE issue
#                       for scalar; promote 40-epoch as default budget.
#   F3/F4 (R40)  beat exp907 by ≥ 5 % (score) → distribution head's true
#                ceiling needs ≥ 40 ep. Promote R40 as default budget.
#   G/H/I  beat exp907 by ≥ 5 % at 20 ep → aux loss is a free win.
#                Combine with n2's results to pick winning hyperparam.
#   J1 (anchor repro) within 1 % of exp907 → environment same as round 2.
#                Else investigate.
#   J2/J3 (cyl)  match or beat exp907 → cylindrical is viable for R6.
#                Else cylindrical = nice idea, doesn't translate.
#
# Usage:
#   bash scripts/n9_bulk_0429_r3.sh                          # GPUs 0+1 parallel
#   EPOCHS=10        bash scripts/n9_bulk_0429_r3.sh         # quick sweep
#   GPUS="0"         bash scripts/n9_bulk_0429_r3.sh         # single GPU 0
#   EVAL_ALL_BESTS=1 bash scripts/n9_bulk_0429_r3.sh         # test 4 best ckpts (default)
#   EVAL_ALL_BESTS=0 bash scripts/n9_bulk_0429_r3.sh         # only best_score
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
# Node 9 dataset path (the variant WITH underscore).
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303_renew}"
EPOCHS_R20="${EPOCHS_R20:-20}"
EPOCHS_R40="${EPOCHS_R40:-40}"
WORKERS="${WORKERS:-4}"
VIS_PER_SCENE=0
EVAL_ALL_BESTS="${EVAL_ALL_BESTS:-1}"

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: DATASET_DIR not found: $DATASET_DIR" >&2
    echo "       Default for node 9 is /root/local1/changwoo/matterport3d_0303_renew" >&2
    exit 3
fi

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n9_0427_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n9_0427_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

# 2-GPU parallel mode — each worker uses one physical GPU (no NCCL,
# no DataParallel). $GPUS is a space-separated list of GPU indices.
# Defaults to "0 1" (the only working pair on this server).
GPUS_STR="${GPUS:-0 1}"
read -r -a GPU_PAIRS <<< "$GPUS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

# ============================================================
# train_one — train, then evaluate 1-or-4 best checkpoints.
# Same skip-if-done semantics as n2_bulk_0429_r3.sh.
# ============================================================
train_and_test() {
    local GPU="$1" CONFIG="$2" EXP="$3" LR="$4" CELL_BS="$5" CELL_EPOCHS="$6"
    shift 6
    local EXTRA=("$@")

    if [ -f "$TEST_LOG_DIR/${EXP}_test.log" ] && \
       grep -q 'ABS_REL:' "$TEST_LOG_DIR/${EXP}_test.log"; then
        echo "[$(date +%H:%M:%S)] [GPU=$GPU] SKIP $EXP (already completed)"
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TRAIN $EXP (cfg=$CONFIG lr=$LR bs=$CELL_BS ep=$CELL_EPOCHS extra='${EXTRA[*]}')"
    local JOB_TIMEOUT="${JOB_TIMEOUT:-72000}"     # 20 h cap (40-ep cells)
    local ec=0
    CUDA_VISIBLE_DEVICES="$GPU" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        timeout --signal=TERM --kill-after=60 "$JOB_TIMEOUT" \
        "$PYTHON" train.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --epochs "$CELL_EPOCHS" \
        --lr "$LR" \
        --batch-size "$CELL_BS" \
        --num-workers "$WORKERS" \
        --depth-dir "$DEPTH_DIR" \
        "${DATASET_ARGS[@]}" \
        "${EXTRA[@]}" \
        > "$TRAIN_LOG_DIR/${EXP}.log" 2>&1 || ec=$?

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

    local TAGS
    if [ "$EVAL_ALL_BESTS" = "1" ]; then
        TAGS=(score absrel rmse delta1)
    else
        TAGS=(score)
    fi

    for TAG in "${TAGS[@]}"; do
        local CKPT
        if [ "$TAG" = "score" ]; then
            CKPT=$(find checkpoints/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
        else
            CKPT=$(find checkpoints/ -name "best_${TAG}.pth" -path "*${EXP}*" 2>/dev/null | head -1)
        fi
        if [ -z "$CKPT" ]; then
            echo "  [GPU=$GPU] $EXP TEST_SKIP[${TAG}] (no checkpoint)"
            continue
        fi

        local TEST_LOG
        if [ "$TAG" = "score" ]; then
            TEST_LOG="$TEST_LOG_DIR/${EXP}_test.log"
        else
            TEST_LOG="$TEST_LOG_DIR/${EXP}_test_${TAG}.log"
        fi

        echo "[$(date +%H:%M:%S)] [GPU=$GPU] TEST[${TAG}] $EXP"
        ec=0
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
            OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
            "$PYTHON" -u test.py \
            --config "$CONFIG" \
            --experiment-name "$EXP" \
            --checkpoint-path "$CKPT" \
            --eval-on test \
            --batch-size 4 \
            --num-workers "$WORKERS" \
            --vis-per-scene "$VIS_PER_SCENE" \
            --depth-dir "$DEPTH_DIR" \
            "${DATASET_ARGS[@]}" \
            "${EXTRA[@]}" \
            > "$TEST_LOG" 2>&1 || ec=$?

        if [ "$ec" -ne 0 ]; then
            echo "  [GPU=$GPU] $EXP TEST_FAILED[${TAG}] (exit=$ec)"
        else
            local abs rmse d1
            abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG" | head -1)
            rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG" | head -1)
            d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG" | head -1)
            echo "  [GPU=$GPU] $EXP TEST_OK[${TAG}]  ABS=$abs RMSE=$rmse D1=$d1"
        fi
    done
}

# ============================================================
# Anchor base config (round 5): exp907 recipe @ bs=32.
# ============================================================
R_BASE_BS32="--depth-head-type range --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14 --range-output-mode expectation --lambda-range-nll 1.0 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask"

NS40="bs32_r3_ep40"
NS20="bs32_r3"

# ============================================================
# Cell definitions (exp958–967, 10 cells)
# Format: "CONFIG  EXP  LR  CELL_BS  CELL_EPOCHS  [EXTRA…]"
# ============================================================
ALL_EXPS=(
    # ── Block F: 40-epoch paired baselines (4 cells) ───────────────────
    # F1: echodiffusion (original config) — closes the budget gap to
    # round-1 echodiff exp11 best (which trained 40 ep at bs=32).
    "echodiffusion  exp958_R5F_S40_echodiff_${NS40}             0.0001  32  ${EPOCHS_R40}"
    # F2: echorange-scalar — same backbone as echodiff but scalar head
    # via the round-5 cfg path. Splits "config code" from "head code".
    "echorange      exp959_R5F_S40_echorange_scalar_${NS40}     0.0001  32  ${EPOCHS_R40}  --depth-head-type scalar"
    # F3: R40 expectation — the main test of the epoch-budget hypothesis
    # for distribution heads. If exp907 expectation was just under-trained,
    # this should beat round-2 expectation by ≥ 5 % score.
    "echorange      exp960_R5F_R40_expect_${NS40}                0.0001  32  ${EPOCHS_R40}  ${R_BASE_BS32}"
    # F4: R40 median output — pulls in exp907's "best ABS_REL ever" trick
    # under a 40-epoch budget. Hard median is non-differentiable but
    # the train-time NLL pushes the distribution into shape, so 40 ep
    # of NLL + median output may close the gap to soft-quantile.
    "echorange      exp961_R5F_R40_median_${NS40}                0.0001  32  ${EPOCHS_R40}  ${R_BASE_BS32//--range-output-mode expectation/--range-output-mode median}"

    # ── Block G: R20 soft-quantile anchor (1 cell) ─────────────────────
    "echorange      exp962_R5G_R20_sq_q050_${NS20}               0.0001  32  ${EPOCHS_R20}  ${R_BASE_BS32} --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.05"

    # ── Block H: R20 SH anchor (1 cell) ────────────────────────────────
    "echorange      exp963_R5H_R20_sh_L2_l002_${NS20}            0.0001  32  ${EPOCHS_R20}  ${R_BASE_BS32} --lambda-spherical-sh 0.02 --spherical-sh-order 2 --spherical-sh-log-depth"

    # ── Block I: R20 combo anchor (1 cell) ─────────────────────────────
    "echorange      exp964_R5I_R20_combo_${NS20}                 0.0001  32  ${EPOCHS_R20}  ${R_BASE_BS32} --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.05 --lambda-spherical-sh 0.02 --spherical-sh-order 2 --spherical-sh-log-depth"

    # ── Block J: R20 anchor reproduction + cylindrical (3 cells) ──────
    # J1: reproduction of exp907 at bs=32 — sanity-check that round 5
    # code on n9 reproduces round-2 results within 1 %.
    "echorange      exp965_R5J_R20_exp907_repro_${NS20}          0.0001  32  ${EPOCHS_R20}  ${R_BASE_BS32}"
    # J2: cylindrical horizontal baseline at 20 ep.
    "echorange      exp966_R5J_R20_cyl_horiz_base_${NS20}        0.0001  32  ${EPOCHS_R20}  ${R_BASE_BS32} --range-bin-axis horizontal --cyl-min-axis-factor 0.15"
    # J3: cylindrical horizontal + soft-quantile (anchor).
    "echorange      exp967_R5J_R20_cyl_horiz_sq_${NS20}          0.0001  32  ${EPOCHS_R20}  ${R_BASE_BS32} --range-bin-axis horizontal --cyl-min-axis-factor 0.15 --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.05"
)

TOTAL=${#ALL_EXPS[@]}

echo "Pre-warming dataset cache for radial depth on $DATASET_DIR..."
"$PYTHON" -c "
from data.dataset import SoundSpacesDataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/echorange.yaml')
cfg.dataset.depth_dir = '$DEPTH_DIR'
if '$DATASET_DIR':
    cfg.dataset.dataset_dir = '$DATASET_DIR'
for split in ['train', 'val', 'test']:
    SoundSpacesDataset(cfg, split=split)
print('Cache warm.')
" 2>&1 | tail -5
echo ""

echo "============================================================"
echo "n9_bulk_0429_r3 — Round 5 (n9): paired R40 + R20 anchors"
echo "  $TOTAL cells (F=4 R40, G/H/I=3 R20, J=3 anchor+cyl)"
echo "  EPOCHS_R40=$EPOCHS_R40  EPOCHS_R20=$EPOCHS_R20"
echo "  EVAL_ALL_BESTS=$EVAL_ALL_BESTS  (1 = test all 4 best ckpts)"
echo "  dataset_dir=${DATASET_DIR:-<yaml default>}"
echo "  $NUM_WORKERS worker(s) × GPUs: ${GPU_PAIRS[*]}"
echo "  $(date)"
echo "============================================================"

run_train_worker() {
    local WORKER_ID=$1
    local GPU="${GPU_PAIRS[$WORKER_ID]}"
    for (( i=WORKER_ID; i<TOTAL; i+=NUM_WORKERS )); do
        local SPEC="${ALL_EXPS[$i]}"
        read -r -a FIELDS <<< "$SPEC"
        local CONFIG="${FIELDS[0]}"
        local EXP="${FIELDS[1]}"
        local LR="${FIELDS[2]}"
        local CELL_BS="${FIELDS[3]}"
        local CELL_EP="${FIELDS[4]}"
        local EXTRA=("${FIELDS[@]:5}")
        train_and_test "$GPU" "$CONFIG" "$EXP" "$LR" "$CELL_BS" "$CELL_EP" "${EXTRA[@]}"
    done
}

PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
    run_train_worker "$w" &
    PIDS+=($!)
    echo "  Worker $w launched (GPU ${GPU_PAIRS[$w]}, PID ${PIDS[-1]})"
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done
echo "[done] failed workers: $FAIL / $NUM_WORKERS"

echo ""
echo "============================================================"
echo "n9_bulk_0429_r3 finished — $(date)"
echo "============================================================"
echo ""
printf "%-60s %8s %8s %8s | %8s %8s %8s | %8s %8s %8s | %8s %8s %8s\n" \
    "Experiment" "ABS@sc" "RMS@sc" "D1@sc" \
    "ABS@ar" "RMS@ar" "D1@ar" \
    "ABS@rm" "RMS@rm" "D1@rm" \
    "ABS@d1" "RMS@d1" "D1@d1"
echo "------------------------------------------------------------"
SUCCESS=0
for SPEC in "${ALL_EXPS[@]}"; do
    EXP=$(echo "$SPEC" | awk '{print $2}')
    LINE="$(printf '%-60s' "$EXP")"
    HAS_ANY=0
    for TAG in score absrel rmse delta1; do
        if [ "$TAG" = "score" ]; then
            LOG="$TEST_LOG_DIR/${EXP}_test.log"
        else
            LOG="$TEST_LOG_DIR/${EXP}_test_${TAG}.log"
        fi
        if [ -f "$LOG" ]; then
            abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$LOG" | head -1)
            rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$LOG" | head -1)
            d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$LOG" | head -1)
            if [ -n "$abs" ]; then
                LINE="$LINE $(printf '%8s %8s %8s' "$abs" "$rmse" "$d1")"
                HAS_ANY=1
            else
                LINE="$LINE $(printf '%8s %8s %8s' '-' '-' '-')"
            fi
        else
            LINE="$LINE $(printf '%8s %8s %8s' '-' '-' '-')"
        fi
    done
    echo "$LINE"
    [ "$HAS_ANY" -eq 1 ] && SUCCESS=$((SUCCESS + 1))
done
echo "------------------------------------------------------------"
echo "Success: $SUCCESS / $TOTAL"
echo "============================================================"

#!/bin/bash
# ============================================================
# n9_bulk_0429_r4.sh — Round 4 (n9): 40-epoch winners + cosine LR probe.
#
# Concurrent with n9_bulk_0429_r3.sh. r3 is mid-flight as of 2026-04-29:
#   exp958–963 DONE.
#   exp964–967 (all R20 cells) HAVE BEEN MOVED TO n2_bulk_0429_r4.sh
#   so n9 stays focused on the 40-ep block. The user kills the in-flight
#   r3 train.py for exp964/965 before launching r4 on either node.
#
# r3 partial signals to act on:
#   • exp960 R40 expectation regressed against R20 exp907 (ABS 0.4951
#     vs 0.4705). Plausible cause: constant LR + 40 epochs overshoots
#     after the R20 sweet spot. exp981 tests the cosine-LR decay
#     hypothesis directly. exp983 does the same on the scalar baseline.
#   • exp946 SH λ=0.10 was the best R20 aux (ABS 0.4413). exp978–980
#     promote that to R40 — alone, with soft-quantile, and combined.
#   • exp982 stacks SH + soft-quantile + cosine LR at 40 ep — the
#     "max push" cell that decides whether 40-epoch is the right floor
#     for distribution head OR the head ceiling has nothing to do with
#     budget.
#
# Code changes that landed *with* r4 (audited present in master):
#   • Sphere-weighted cos-lat metrics in utils/metrics.py +
#     utils/test_utils.py + test.py (also writes per_sample_<exp>.npz).
#   • train.py learns --lr-schedule=cosine and --lr-warmup-epochs.
#
# Reference baselines (radial uniform metric):
#   echodiff exp11:   ABS 0.4300  RMSE 1.1060  D1 0.4876
#   exp907 expect:    ABS 0.4814  RMSE 1.2532  D1 0.5079  (round-2 bs32)
#   exp907 median:    ABS 0.4202  RMSE 1.36           ← best-ABS-ever
#   exp912 scalar bs32:  ABS 0.4349  RMSE 1.27        D1 0.4831
#   r3 R40 results:   exp958 S40 echodiff  ABS 0.4463 (gap to exp11!)
#                     exp959 S40 echorange ABS 0.4873 (scalar gap re-opens)
#                     exp960 R40 expectation ABS 0.4951 (regression)
#                     exp961 R40 median   ABS 0.4520 (best of R40 family)
#
# Pass / kill criteria:
#   ζ R40    exp978/9/0 ABS ≤ 0.430 OR RMSE ≤ 1.18 → 40 ep is the win
#   ζ LR     exp981 ABS within 1 % of exp907 R20 → constant LR was bad
#                   ABS still ≥ 0.48 → schedule is not the issue,
#                                       distribution head ceiling is real
#   η stack  exp982 ABS ≤ 0.420 OR RMSE ≤ 1.18 → ship as round-5 main
#   θ S40    exp983 closes ≥ 50 % of (exp958 ABS - exp11 ABS) → cosine
#                   LR was the missing ingredient for scalar S40 too
#
# Hardware constraints (from round-1 n9 driver issue):
#   - Only GPUs 0 and 1 are usable (NVML failure on GPU 2 broke NCCL).
#   - 2 workers in parallel, each pinned to one GPU; cells round-robin.
#
# Time budget (estimated):
#   ζ R40        4 × 40-ep ≈ 20 h       → 10 h with 2 workers
#   η stack      1 × 40-ep ≈ 5 h        → solo on the next free worker
#   θ S40        1 × 40-ep ≈ 5 h        → solo on the next free worker
#   Test phase   6 × 4 best-tags × ≈3min → ≈ 1.2 h on 2 workers
#   Total       ≈ 16 h. skip-if-done lets the next morning wrap up.
#
# Usage:
#   bash scripts/n9_bulk_0429_r4.sh                         # GPUs 0+1
#   EPOCHS_R20=10  bash scripts/n9_bulk_0429_r4.sh          # quick sweep
#   GPUS="0"       bash scripts/n9_bulk_0429_r4.sh          # single GPU
#   EVAL_ALL_BESTS=0 bash scripts/n9_bulk_0429_r4.sh        # only score
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
# Node 9 dataset path (the variant WITH underscore).
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303_renew}"
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
GPUS_STR="${GPUS:-0 1}"
read -r -a GPU_PAIRS <<< "$GPUS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

# ============================================================
# train_and_test — same skip-if-done semantics as n9_bulk_0429_r3.sh.
# When EVAL_ALL_BESTS=1, evaluates score / absrel / rmse / delta1 each.
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
# Anchor base config (round 4): exp907 recipe (full ERP) at bs=32.
# All ζ / η / θ cells inherit this. ε carry-over uses the same anchor
# plus the cylindrical bin-axis flag (already in train.py since r3).
# ============================================================
R_BASE_BS32="--depth-head-type range --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14 --range-output-mode expectation --lambda-range-nll 1.0 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask"

NS40="bs32_r4_ep40"

# ============================================================
# Cell definitions (6 cells: 4 R40 winner-aux + 1 R40 stack + 1 S40 LR-decay)
# Format: "CONFIG  EXP  LR  CELL_BS  CELL_EPOCHS  [EXTRA…]"
# ============================================================
ALL_EXPS=(
    # ── Block ζ: 40-ep with r3 winner aux (4 cells) ─────────────────────
    # SH=0.10 was the r3 R20 winner. Soft-quantile q=0.50/τ=0.03/λ=0.25
    # was the r3 sq winner. Combine each at 40 ep.
    "echorange  exp978_R4F_R40_sh_l010_${NS40}                   0.0001  32  ${EPOCHS_R40}  ${R_BASE_BS32} --lambda-spherical-sh 0.10 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp979_R4F_R40_sq_q050_t003_${NS40}              0.0001  32  ${EPOCHS_R40}  ${R_BASE_BS32} --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.03"
    "echorange  exp980_R4F_R40_sh_sq_combo_${NS40}               0.0001  32  ${EPOCHS_R40}  ${R_BASE_BS32} --lambda-spherical-sh 0.10 --spherical-sh-order 2 --spherical-sh-log-depth --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.03"
    "echorange  exp981_R4F_R40_expect_cosineLR_${NS40}           0.0001  32  ${EPOCHS_R40}  ${R_BASE_BS32} --lr-schedule cosine --lr-warmup-epochs 1"

    # ── Block η: max stack at 40 ep (1 cell) ────────────────────────────
    # SH + soft-quantile + cosine LR — the brief's "ABS branch + RMSE
    # branch" combined ceiling probe.
    "echorange  exp982_R4F_R40_full_stack_${NS40}                0.0001  32  ${EPOCHS_R40}  ${R_BASE_BS32} --lambda-spherical-sh 0.10 --spherical-sh-order 2 --spherical-sh-log-depth --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.03 --lr-schedule cosine --lr-warmup-epochs 1"

    # ── Block θ: S40 echodiff + cosine LR (1 cell) ──────────────────────
    # exp958 S40 echodiff was ABS 0.4463 (vs round-1 exp11 ABS 0.4300
    # at 40 ep). The 40 ep budget *should* match exp11 already. Most
    # likely ingredient missing is cosine LR (round-1 may have used a
    # decay schedule). This cell tests that hypothesis directly.
    "echodiffusion  exp983_R4F_S40_echodiff_cosineLR_${NS40}     0.0001  32  ${EPOCHS_R40}  --lr-schedule cosine --lr-warmup-epochs 1"
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
echo "n9_bulk_0429_r4 — Round 4 (n9): R40 winners + LR probe"
echo "  $TOTAL cells (ζ=4 R40, η=1 stack, θ=1 S40)"
echo "  EPOCHS_R40=$EPOCHS_R40"
echo "  EVAL_ALL_BESTS=$EVAL_ALL_BESTS"
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
echo "n9_bulk_0429_r4 finished — $(date)"
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

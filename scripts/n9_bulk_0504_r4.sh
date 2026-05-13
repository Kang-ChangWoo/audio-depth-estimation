#!/bin/bash
# ============================================================
# n9_bulk_0504_r4.sh — Round 4 (n9): SH ablation — L × λ × log/lin sweep
#
# CONTEXT
# --------
# r3 found exp946 (R5B, L=2, λ=0.10, log_depth=true, bs=48) as the only
# 7-metric all-win cell vs exp958 echodiff baseline. But three things are
# undertested in r3:
#   1. SH order L was only swept at L∈{2,3} and only at *low* λ (0.02,0.05).
#      L=2 λ=0.10 (exp946) is a single point with no L-sweep companion.
#      L=0 (mean only), L=1 (dipole only), L=4 are all unsampled.
#   2. λ-curve shows non-monotonicity (0.02 → 0.05 worst → 0.10 best).
#      The right side of the curve (λ > 0.10) is unsampled — sweet spot
#      could be at 0.15, 0.20, or 0.30. No way to tell from r3.
#   3. log_depth=false was tested only at λ=0.02 (exp949), not at λ=0.10.
#      Whether the log-depth advantage is universal or only at high λ is
#      unconfirmed.
#
# This round fills those three gaps in a single 12-cell sweep, all at
# bs=32 / 20 ep on n9. Anchor: R_BASE_BS32 (exp907 recipe) + SH only.
#
# Cells (12 total, exp984–exp995):
#   Block A — L sweep at λ=0.10 fixed (5 cells, "the L curve at the winner λ")
#     exp984  L=0  λ=0.10  log=T   K=1   scene-mean only — falsifies "anisotropy needed"
#     exp985  L=1  λ=0.10  log=T   K=4   dipole only — binaural physics most aligned
#     exp986  L=2  λ=0.10  log=T   K=9   exp946 anchor reproduction at bs=32
#     exp987  L=3  λ=0.10  log=T   K=16  quadrupole — r3 had this only at λ≤0.05
#     exp988  L=4  λ=0.10  log=T   K=25  closed-form max — sanity check at high order
#
#   Block B — λ sweep at L=2 fixed, above 0.10 (3 cells, "right tail of λ-curve")
#     exp989  L=2  λ=0.15  log=T          inverted-U test: peak at 0.10 or higher?
#     exp990  L=2  λ=0.20  log=T          if better than 0.10, sweet spot moved up
#     exp991  L=2  λ=0.30  log=T          aggressive — expected to hurt RMSE if SH dominates
#
#   Block C — loss variations (4 cells)
#     exp992  L=2  λ=0.10  log=F (lin)   fills gap: exp949 only had lin at λ=0.02
#     exp993  L=1  λ=0.30  log=T          dipole-only with strong weight (compensate K=4)
#     exp994  L=4  λ=0.05  log=T          high-order at moderate λ (r3 had L=3 only)
#     exp995  L=3  λ=0.05  log=T          exp948 reproduction at bs=32 (was bs=48)
#
# Decision rules at end of round:
#   • Block A — best L:
#     - argmin(ABS_REL) at L=0 → mean-matching alone explains exp946; SH machinery
#       is over-engineering. Replace with simple cos-lat-weighted mean log-depth loss.
#     - argmin at L=1 → dipole captures audio-recoverable info; L=2 quadrupole
#       contribution is noise. Drop to L=1 default.
#     - argmin at L=2 → exp946 mechanism confirmed (anisotropy useful, audio limit
#       at quadrupole). Promote L=2 as default.
#     - argmin at L≥3 → r3 conclusion of "audio-limited at L=2" is wrong; sweep
#       further (would need closed-form recursion for L≥5).
#   • Block B — λ peak location:
#     - λ=0.10 still best → confirm sweet spot at 0.10, freeze for downstream.
#     - λ=0.15/0.20 best → sweet spot moved up, refine in {0.13, 0.17} next round.
#     - λ=0.30 best → SH dominance is fine, push higher.
#     - All Block B worse than 0.10 → 0.10 is right edge, refine in {0.07, 0.08}.
#   • Block C:
#     - exp992 (L=2 λ=0.10 lin) ≥ exp986 (L=2 λ=0.10 log) → log_depth not necessary
#       at high λ; round-3 §4.3 over-claimed log advantage.
#     - exp993 (L=1 λ=0.30) close to exp986 (L=2 λ=0.10) → L=1 with strong weight
#       is sufficient; no need for K=9.
#     - exp994 (L=4 λ=0.05) ≥ exp995 (L=3 λ=0.05) ≥ exp948 → high order helps at
#       low λ; revisit Block A interpretation.
#
# CAVEATS
# -------
# • Single seed throughout. Multi-seed of the winners belongs in a separate
#   round (R6 Branch B in plan).
# • bs=32 anchor differs from r3's bs=48 R5B. Direct exp946 ↔ exp986 numerical
#   comparison is biased; treat exp986 as the new bs=32 anchor for exp984/5/7/8.
#
# Hardware constraints (inherited from r3):
#   - GPUs 0 and 1 only. NCCL broken on GPU 2.
#   - 2 workers in parallel, each pinned to its own GPU; cells round-robin.
#
# Time budget:
#   12 cells × ~2.5 h / 2 workers = ~15 h.
#   + test phase (4 ckpt-tags × 12 × ~5 min) = ~4 h on 2 workers.
#   Total ≈ 19 h overnight.
#
# Reference baselines (radial uniform metric):
#   echodiff exp11:    ABS 0.4300  RMSE 1.1060  D1 0.4876   ← target
#   exp946 (bs=48):    ABS 0.4413  RMSE 1.2208  D1 0.5019   ← R5B winner
#   exp963 (bs=32 SH L=2 λ=0.02): ABS 0.5221  RMSE 1.2267  D1 0.5010  ← n9 anchor
#   exp958 (bs=32 echodiff 40ep): ABS 0.4463  RMSE 1.2611  D1 0.4983  ← fair baseline
#
# Usage:
#   bash scripts/n9_bulk_0504_r4.sh                          # GPUs 0+1 parallel
#   EPOCHS=10        bash scripts/n9_bulk_0504_r4.sh         # quick sweep
#   GPUS="0"         bash scripts/n9_bulk_0504_r4.sh         # single GPU 0
#   EVAL_ALL_BESTS=0 bash scripts/n9_bulk_0504_r4.sh         # only best_score
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303_renew}"
EPOCHS="${EPOCHS:-20}"
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
# train_and_test — train one cell, then evaluate 1-or-4 best ckpts.
# Same skip-if-done semantics as n9_bulk_0429_r3.sh.
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
    local JOB_TIMEOUT="${JOB_TIMEOUT:-36000}"     # 10 h cap per 20-ep cell
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
# Anchor base config — exp907 recipe @ bs=32, range head, ERP fix on.
# All cells use this and add --lambda-spherical-sh + --spherical-sh-order
# (+ optional --no-spherical-sh-log-depth) to vary the SH branch only.
# ============================================================
R_BASE_BS32="--depth-head-type range --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14 --range-output-mode expectation --lambda-range-nll 1.0 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask"

NS="bs32_r4"

# ============================================================
# Cell definitions (exp984–exp995, 12 cells)
# Format: "CONFIG  EXP  LR  CELL_BS  CELL_EPOCHS  [EXTRA…]"
# ============================================================
ALL_EXPS=(
    # ── Block A: L sweep at λ=0.10, log_depth=T (5 cells) ─────────────
    # The L-curve at the r3 winner λ. Falsifies whether "L=2 sweet spot"
    # claim survives a full sweep at the same λ that produced exp946.
    "echorange  exp984_R4S_sh_L0_l010_logd_${NS}     0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.10 --spherical-sh-order 0 --spherical-sh-log-depth"
    "echorange  exp985_R4S_sh_L1_l010_logd_${NS}     0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.10 --spherical-sh-order 1 --spherical-sh-log-depth"
    "echorange  exp986_R4S_sh_L2_l010_logd_${NS}     0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.10 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp987_R4S_sh_L3_l010_logd_${NS}     0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.10 --spherical-sh-order 3 --spherical-sh-log-depth"
    "echorange  exp988_R4S_sh_L4_l010_logd_${NS}     0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.10 --spherical-sh-order 4 --spherical-sh-log-depth"

    # ── Block B: λ sweep at L=2 fixed, λ > 0.10 (3 cells) ─────────────
    # Right side of the λ-curve. R5B never sampled above 0.10 — sweet
    # spot could be at 0.15+. λ=0.30 expected to hurt as SH dominates.
    "echorange  exp989_R4S_sh_L2_l015_logd_${NS}     0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.15 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp990_R4S_sh_L2_l020_logd_${NS}     0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.20 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp991_R4S_sh_L2_l030_logd_${NS}     0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.30 --spherical-sh-order 2 --spherical-sh-log-depth"

    # ── Block C: loss / depth-space variations (4 cells) ──────────────
    # C1: L=2 λ=0.10 lin — fills the lin × high-λ gap. exp949 was lin × λ=0.02.
    "echorange  exp992_R4S_sh_L2_l010_lin_${NS}      0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.10 --spherical-sh-order 2 --no-spherical-sh-log-depth"
    # C2: L=1 λ=0.30 — strong dipole-only. K=4 means smooth_l1 averaged
    # over fewer modes; compensate with bigger λ to match exp946 effect.
    "echorange  exp993_R4S_sh_L1_l030_logd_${NS}     0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.30 --spherical-sh-order 1 --spherical-sh-log-depth"
    # C3: L=4 λ=0.05 — high-order at moderate λ. r3 only had L=3 at this
    # λ. Tests whether high-order modes can be useful when not weighted hard.
    "echorange  exp994_R4S_sh_L4_l005_logd_${NS}     0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.05 --spherical-sh-order 4 --spherical-sh-log-depth"
    # C4: L=3 λ=0.05 — bs=32 reproduction of exp948 (bs=48). Cross-server
    # control for the "L=3 doesn't beat L=2" claim.
    "echorange  exp995_R4S_sh_L3_l005_logd_${NS}     0.0001  32  ${EPOCHS}  ${R_BASE_BS32} --lambda-spherical-sh 0.05 --spherical-sh-order 3 --spherical-sh-log-depth"
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
echo "n9_bulk_0504_r4 — Round 4 SH ablation: L × λ × log/lin"
echo "  $TOTAL cells (A=5 L-sweep, B=3 λ-sweep, C=4 variations)"
echo "  EPOCHS=$EPOCHS  bs=32"
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
echo "n9_bulk_0504_r4 finished — $(date)"
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

#!/bin/bash
# ============================================================
# n2_bulk_0428.sh — EchoRange round-3 sweep, hazard head + audit fixes
#
# Builds on the round-2 sweep in scripts/n9_bulk_0427.sh (exp907–912).
# Round-2 sweep design: see logs/round1_bin-based/round1/E20260428-*.md
# (round 1) and the in-flight round-2 cells.
#
# This script does THREE things:
#   (A) introduce a new depth head — A-Hazard (first-hit rendering) — and
#       sweep four ablations of it (exp913–916);
#   (B) close two audit gaps from the round-2 critique
#         · exp919 — softmax range head with λ_NLL=0.3 (loss-balance test)
#         · exp920 — second-seed re-train of exp912 for HP-variance baseline
#       (NB: the silog=0.5 / silog=1.0 mismatch in round 2 is fixed at the
#       cell level here — every range/hazard cell in this script uses
#       --lambda-silog 0.5 to match echorange.yaml / echodiffusion.yaml);
#   (C) re-test the round-2 exp907 checkpoint with --range-output-mode
#       median (no retrain, inference-only), to disentangle "training-time
#       median" from "test-time median".
#
# Round 3 (this sweep, 7 jobs):
#   exp913  hazard, full           (λ_hit=0.5 / λ_free=0.10, warmup 3)
#   exp914  hazard, no-free        (--disable-free-loss)
#   exp915  hazard, no-hit         (--disable-hit-loss)
#   exp916  hazard, depth-only     (--hazard-depth-only — BerHu/SILog only)
#   exp917  hazard, stronger sup   (λ_hit=1.0 / λ_free=0.2, warmup 5)
#   exp919  softmax range, λ_NLL=0.3   (audit fix — NLL weight cell)
#   exp920  exp912 echodiffusion seed=2 (audit fix — HP variance baseline)
#   + test_only on exp907 best_model.pth with --range-output-mode median
#
# All training jobs: AdamW, lr=1e-4, bs=32, 20 epochs (round-2 cap),
# range_min/max_depth=0.1/10.0, log spacing, ERP cos-lat + far-mask on,
# --lambda-berhu 1.0 --lambda-silog 0.5 (fix vs round 2's silog=1.0).
#
# Logs land in the SAME directories as round 2 so the round-1/2/3 results
# can be aggregated with a single grep:
#   logs/n9_0427_train/<EXP>.log
#   logs/n9_0427_test/<EXP>_test.log
#
# Hardware: node 2 (8-GPU). Same 2-GPU-per-process DataParallel pattern
# as n2_revisit.sh: each worker receives a comma-separated GPU pair
# (e.g. "0,1") and runs one job under DataParallel across that pair.
# Default GPU_PAIRS="0,1 2,3 4,5 6,7" runs four jobs in parallel;
# with 7 round-3 cells the worst-case load is 2 sequential jobs on
# any one pair.
#
# Batch size: 48 — the cap documented in n2_revisit.sh:208-215 for the
# echodiffusion + wav2vec family (exp385-389), which is the closest
# analog to echorange (also uses CIDE/wav2vec via use_waveform=true).
# Per-GPU under DP this is 24 samples; n2_revisit notes ~80 % of 48 GiB
# at bs=48 on echodiffusion alone. Echorange adds the (B, R=32, h, w)
# hazard tensors (~3 GiB extra) which still fits inside the headroom.
# Drop to bs=32 if any cell OOMs.
#
# Usage:
#   bash scripts/n2_bulk_0428.sh                                 # 4 workers (default)
#   GPU_PAIRS="0,1"               bash scripts/n2_bulk_0428.sh   # 1 worker (sequential)
#   GPU_PAIRS="0,1 2,3"           bash scripts/n2_bulk_0428.sh   # 2 workers
#   GPU_PAIRS="0,1,2,3"           bash scripts/n2_bulk_0428.sh   # 1 worker, 4-GPU DP
#   BS=32                          bash scripts/n2_bulk_0428.sh   # safer batch
#   EPOCHS=10                      bash scripts/n2_bulk_0428.sh   # smoke run
#   SEED=7                         bash scripts/n2_bulk_0428.sh   # all cells
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
# Node 2 dataset path convention is `matterport3d_0303renew` (no
# underscore between 0303 and renew). The n9 server uses the underscored
# variant — do NOT copy that here. See n2_revisit.sh / n2_bulk_0427.sh.
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303renew}"
EPOCHS="${EPOCHS:-20}"
WORKERS="${WORKERS:-2}"   # n2_revisit lowered 4→2 after dataloader OOM (line 40-44)
VIS_PER_SCENE=0
SEED="${SEED:-1}"         # only sets PYTHONHASHSEED. train.py has no torch.manual_seed,
                          # so runs are non-deterministic regardless — exp912 vs exp920
                          # rely on this natural non-determinism for their seed-variance
                          # comparison. Wire torch.manual_seed into train.py to make this
                          # rigorous.
BS="${BS:-48}"            # echodiff+wav2vec cap on this node (n2_revisit:215)

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

# Fail fast if the dataset path doesn't exist on this node — better than
# letting the dataloader die mid-prewarm with a confusing FileNotFoundError.
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: DATASET_DIR not found: $DATASET_DIR" >&2
    echo "       Set DATASET_DIR=<path> on launch or fix the default above." >&2
    echo "       Node 2 default is /root/local1/changwoo/matterport3d_0303renew" >&2
    echo "       (no underscore between 0303 and renew)." >&2
    exit 3
fi

# Round-3 logs land in the round-2 directory so the full bin-based timeline
# (exp9xx) lives in one place — grep / aggregate scripts already exist
# under logs/n9_0427_*.
TRAIN_LOG_DIR="$PROJECT_DIR/logs/n9_0427_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n9_0427_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

# 2-GPU-per-process DataParallel pattern (matches n2_revisit.sh). Each
# worker receives one comma-separated GPU pair (e.g. "0,1") and runs one
# experiment under DP across that pair. Multiple pairs in GPU_PAIRS run
# experiments in parallel — default uses all four pairs on an 8-GPU node
# so the 7 round-3 cells finish in two waves (worst-case 2 sequential
# jobs per pair).
GPU_PAIRS_STR="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
read -r -a GPU_PAIRS <<< "$GPU_PAIRS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

# ------------------------------------------------------------
# Disk-usage sanity guard. echodiff_ambi_cide_train exp711/712 died with
# "No space left on device" while writing val PNGs; with VIS_PER_SCENE=0
# we don't write images, but the eval/soundspaces/test/stats_*.pt files
# and per-experiment results/ dirs still grow. Warn loudly if the project
# disk has less than 5 GiB free before launching anything.
# ------------------------------------------------------------
DF_AVAIL_KB=$(df -k "$PROJECT_DIR" | awk 'NR==2 {print $4}')
DF_AVAIL_GB=$((DF_AVAIL_KB / 1024 / 1024))
echo "[disk] $PROJECT_DIR free: ${DF_AVAIL_GB} GiB"
if [ "$DF_AVAIL_GB" -lt 5 ]; then
    echo "[disk] WARNING: <5 GiB free — clean logs/ eval/ results/ first."
fi

# ============================================================
# Train+test driver — preserves the n9_bulk_0427 contract:
#   • skip if test log already has ABS_REL
#   • SIGTERM at $JOB_TIMEOUT, hard kill 60 s later
#   • test the resulting best_model.pth right after training
# ============================================================
train_and_test() {
    local GPU="$1" CONFIG="$2" EXP="$3" LR="$4" BS="$5"
    shift 5
    local EXTRA=("$@")

    if [ -f "$TEST_LOG_DIR/${EXP}_test.log" ] && \
       grep -q 'ABS_REL:' "$TEST_LOG_DIR/${EXP}_test.log"; then
        echo "[$(date +%H:%M:%S)] [GPU=$GPU] SKIP $EXP (already completed)"
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TRAIN $EXP (cfg=$CONFIG lr=$LR bs=$BS extra='${EXTRA[*]}')"
    local JOB_TIMEOUT="${JOB_TIMEOUT:-54000}"
    local ec=0
    CUDA_VISIBLE_DEVICES="$GPU" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED="$SEED" \
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
        --batch-size 4 \
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
# test_only — for re-evaluating an existing checkpoint with different
# inference-time options (e.g. --range-output-mode median on the exp907
# expectation-trained checkpoint). Looks up best_model.pth by experiment-
# name pattern and writes <EXP>_test.log to the round-2 test dir.
# ============================================================
test_only() {
    local GPU="$1" CONFIG="$2" SRC_EXP="$3" DST_EXP="$4"
    shift 4
    local EXTRA=("$@")

    if [ -f "$TEST_LOG_DIR/${DST_EXP}_test.log" ] && \
       grep -q 'ABS_REL:' "$TEST_LOG_DIR/${DST_EXP}_test.log"; then
        echo "[$(date +%H:%M:%S)] [GPU=$GPU] SKIP test_only $DST_EXP (already completed)"
        return
    fi

    local CKPT
    CKPT=$(find checkpoints/ -name "best_model.pth" -path "*${SRC_EXP}*" 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "  [test_only] $DST_EXP NO_CHECKPOINT (looked for ${SRC_EXP}) — skipping"
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TEST_ONLY $DST_EXP (src=$SRC_EXP, extra='${EXTRA[*]}')"
    local ec=0
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        "$PYTHON" -u test.py \
        --config "$CONFIG" \
        --experiment-name "$DST_EXP" \
        --checkpoint-path "$CKPT" \
        --eval-on test \
        --batch-size 4 \
        --num-workers "$WORKERS" \
        --vis-per-scene "$VIS_PER_SCENE" \
        --depth-dir "$DEPTH_DIR" \
        "${DATASET_ARGS[@]}" \
        "${EXTRA[@]}" \
        > "$TEST_LOG_DIR/${DST_EXP}_test.log" 2>&1 || ec=$?

    if [ "$ec" -ne 0 ]; then
        echo "  [test_only] $DST_EXP FAILED (exit=$ec)"
        tail -3 "$TEST_LOG_DIR/${DST_EXP}_test.log" | sed "s/^/      /"
    else
        local abs rmse d1
        abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${DST_EXP}_test.log" | head -1)
        rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${DST_EXP}_test.log" | head -1)
        d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${DST_EXP}_test.log" | head -1)
        echo "  [test_only] $DST_EXP OK  ABS=$abs RMSE=$rmse D1=$d1"
    fi
}

# ============================================================
# Hazard-head common args (shared bin grid + ERP fix; varied per cell).
# Hazard cells inherit:
#   --depth-head-type hazard   --range-num-bins 32   --range-bin-spacing log
#   --range-min-depth 0.1      --range-max-depth 10.0
#   --hazard-bias-init -4.6
#   --lambda-berhu 1.0   --lambda-silog 0.5     ← audit fix (vs round 2's 1.0)
#   --erp-cos-lat-weight  --erp-far-mask
# Each cell appends its own warmup + hit/free weights / ablation flags.
# ============================================================
HAZ_BASE="--depth-head-type hazard --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --hazard-bias-init -4.6 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask"

# Softmax-range cells used in the audit-fix experiments.
SM_BASE="--depth-head-type range --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14 --range-output-mode expectation --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask"

# ============================================================
# Experiment definitions: "CONFIG  EXP  LR  BS  [EXTRA…]"
# Round 3 — 7 cells (exp913–917 hazard + exp919 audit-NLL + exp920 audit-seed).
# ============================================================
# Experiment-name suffix tracks the actual batch size so checkpoints are
# unambiguous when BS is overridden via the environment.
NS="bs${BS}"

ALL_EXPS=(
    # ── (A) Hazard head sweep ─────────────────────────────────
    # exp913 — full hazard with default warmup schedule.
    #          λ_warmup=(0.3, 0.05) for ep ≤ 3; then λ=(0.5, 0.1).
    "echorange  exp913_echorange_haz32log_d10_full_lr1e4_${NS}           0.0001  ${BS}  ${HAZ_BASE} --hazard-warmup-epochs 3 --lambda-hit-warmup 0.3 --lambda-free-warmup 0.05 --lambda-hit 0.5 --lambda-free 0.1"

    # exp914 — hazard no-free ablation (only L_hit drives bins).
    "echorange  exp914_echorange_haz32log_d10_nofree_lr1e4_${NS}         0.0001  ${BS}  ${HAZ_BASE} --hazard-warmup-epochs 3 --lambda-hit-warmup 0.3 --lambda-free-warmup 0.05 --lambda-hit 0.5 --lambda-free 0.1 --disable-free-loss"

    # exp915 — hazard no-hit ablation (only L_free drives bins).
    "echorange  exp915_echorange_haz32log_d10_nohit_lr1e4_${NS}          0.0001  ${BS}  ${HAZ_BASE} --hazard-warmup-epochs 3 --lambda-hit-warmup 0.3 --lambda-free-warmup 0.05 --lambda-hit 0.5 --lambda-free 0.1 --disable-hit-loss"

    # exp916 — hazard depth-only ablation: BerHu/SILog through the
    # renderer drive the head end-to-end. Tells us whether the hit/free
    # losses are doing useful shaping or just helping the renderer
    # initialize.
    "echorange  exp916_echorange_haz32log_d10_depthonly_lr1e4_${NS}      0.0001  ${BS}  ${HAZ_BASE} --hazard-warmup-epochs 3 --hazard-depth-only"

    # exp917 — stronger / longer hazard supervision. If exp913 underfits
    # the bin distribution (weak L_hit), this checks whether higher
    # weights help. If exp913 already saturates, this should be a no-op.
    "echorange  exp917_echorange_haz32log_d10_strong_lr1e4_${NS}         0.0001  ${BS}  ${HAZ_BASE} --hazard-warmup-epochs 5 --lambda-hit-warmup 0.5 --lambda-free-warmup 0.10 --lambda-hit 1.0 --lambda-free 0.2"

    # ── (B) Audit-fix cells from the round-2 critique ─────────
    # exp919 — softmax range, λ_NLL=0.3 (vs round-2 default 1.0). NLL
    # at 32-bin × log spacing has scale ≈ 3.5 nat; default λ=1 lets it
    # dominate BerHu/SILog. This cell tests "does down-weighting NLL
    # help align gradients with the metric path?".
    "echorange  exp919_echorange_32log_d10_lnll0.3_lr1e4_${NS}           0.0001  ${BS}  ${SM_BASE} --lambda-range-nll 0.3"

    # exp920 — second-seed re-train of the round-2 echodiffusion baseline
    # (cfg matches echodiffusion.yaml; bs comes from BS env var so the
    # gold-baseline cell uses the same batch as the rest of the sweep,
    # giving a fair HP-variance estimate at this exact (lr, bs) cell).
    "echodiffusion  exp920_echodiff_lr1e4_${NS}_seed2                    0.0001  ${BS}"
)

TOTAL=${#ALL_EXPS[@]}

# Pre-warm dataset cache so 4×workers × 4 dataloader procs × 2 jobs don't
# all rescan the dataset directory at the same time.
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
echo "n2_bulk_0428 — ${TOTAL} train cells + 1 test_only re-eval"
echo "  EPOCHS=$EPOCHS  BS=$BS (DataParallel, per-GPU $((BS / 2)))  depth_dir=$DEPTH_DIR  silog=0.5 (audit fix)"
echo "  dataset_dir=${DATASET_DIR:-<yaml default>}"
echo "  $NUM_WORKERS worker(s) × GPU pairs: ${GPU_PAIRS[*]}"
echo "  SEED=$SEED  $(date)"
echo "============================================================"

# ── Phase 1: train+test the new cells in parallel ────────────
run_train_worker() {
    local WORKER_ID=$1
    local GPU="${GPU_PAIRS[$WORKER_ID]}"

    for (( i=WORKER_ID; i<TOTAL; i+=NUM_WORKERS )); do
        local SPEC="${ALL_EXPS[$i]}"
        read -r -a FIELDS <<< "$SPEC"
        local CONFIG="${FIELDS[0]}"
        local EXP="${FIELDS[1]}"
        local LR="${FIELDS[2]}"
        local BS="${FIELDS[3]}"
        local EXTRA=("${FIELDS[@]:4}")
        train_and_test "$GPU" "$CONFIG" "$EXP" "$LR" "$BS" "${EXTRA[@]}"
    done
}

echo "[Phase 1/2] TRAIN+TEST new cells"
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
echo "[Phase 1/2] done — failed workers: $FAIL / $NUM_WORKERS"
echo ""

# ── Phase 2: test-only re-eval of round-2 exp907 with median output ──
# This re-evaluates the EXISTING exp907 best_model.pth (trained with
# expectation) under --range-output-mode median. Lets us decide whether
# round-2 exp910 (median-trained) actually buys anything beyond a
# free inference-time switch on the expectation checkpoint.
echo "[Phase 2/2] test_only — exp907 ckpt re-evaluated as median"
test_only "${GPU_PAIRS[0]}" "echorange" \
    "exp907_echorange_32log_d10_coslat_farmask_lr1e4_bs32" \
    "exp907_echorange_32log_d10_coslat_farmask_TESTmedian" \
    --depth-head-type range --range-num-bins 32 --range-bin-spacing log \
    --range-min-depth 0.1 --range-max-depth 10.0 \
    --range-output-mode median

echo ""
echo "============================================================"
echo "n2_bulk_0428 finished — $(date)"
echo "Workers failed: $FAIL / $NUM_WORKERS"
echo "============================================================"
echo ""
printf "%-58s %8s %8s %8s\n" "Experiment" "ABS_REL" "RMSE" "Delta1"
echo "---------------------------------------------------------------------"
SUCCESS=0
ALL_REPORT=("${ALL_EXPS[@]}" \
    "echorange  exp907_echorange_32log_d10_coslat_farmask_TESTmedian  -  -")
for SPEC in "${ALL_REPORT[@]}"; do
    EXP=$(echo "$SPEC" | awk '{print $2}')
    LOG="$TEST_LOG_DIR/${EXP}_test.log"
    if [ -f "$LOG" ]; then
        abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$LOG" | head -1)
        rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$LOG" | head -1)
        d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$LOG" | head -1)
        if [ -n "$abs" ]; then
            printf "%-58s %8s %8s %8s\n" "$EXP" "$abs" "$rmse" "$d1"
            SUCCESS=$((SUCCESS + 1))
        else
            printf "%-58s %8s\n" "$EXP" "FAILED"
        fi
    else
        printf "%-58s %8s\n" "$EXP" "NO_RESULT"
    fi
done
echo "---------------------------------------------------------------------"
echo "Success: $SUCCESS / $((TOTAL + 1))"
echo "============================================================"

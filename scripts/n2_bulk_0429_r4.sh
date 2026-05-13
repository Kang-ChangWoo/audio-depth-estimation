#!/bin/bash
# ============================================================
# n2_bulk_0429_r4.sh — Round 4 (n2): r3 R20 carry-over + SH refine
#                                     + soft-quantile refine
#                                     + hazard far/free closure.
#
# Carries every UNFINISHED r3 R20 cell (RUN/PARTIAL or MISSING) into
# n2. The user kills the running r3 processes before launching r4 so
# the carry-overs train from scratch on n2's bs=48 / 4-worker setup,
# freeing n9 to run the 40-ep block exclusively.
#
# Carry-over cells (same exp names as r3 → skip-if-done aggregates the
# logs under the existing exp9xx timeline):
#   exp956  R5D cyl_horiz_sh         (was running on n2)
#   exp957  R5E haz_softhit075       (was running on n2)
#   exp964  R5I R20 combo (sq + SH)  (was running on n9)
#   exp965  R5J R20 exp907 repro     (was running on n9)
#   exp966  R5J R20 cyl_horiz_base   (was unstarted on n9)
#   exp967  R5J R20 cyl_horiz_sq     (was unstarted on n9)
#
# Concurrent with n2_bulk_0429_r3.sh — but r3 has already finished its
# R20 done-cells (exp936–955), and the carry-overs above were the only
# r3 cells on either node still in flight. The user will SIGKILL the
# r3 train.py processes before launching r4.
#
# r3 partial winners (used to pick r4 brackets):
#   • Block B SH at L=2: λ=0.02 → ABS 0.4928, λ=0.05 → 0.5222 (noisy),
#                         λ=0.10 → 0.4413  ←  best aux on r3 by ABS_REL
#                                              (beats every soft-quantile
#                                              cell and every combo cell)
#   • Block A soft-quantile: q=0.50/τ=0.03/λ=0.25 → 0.4585  ← best sq cell
#                            q=0.45/τ=0.05/λ=0.25 → 0.5150 (sharper q hurt)
#   • Block C combo: best q=0.55+SH=0.02 → 0.4666 (no synergy at exp946 hp)
#
# r4 plan:
#   Block β  refine around the SH=0.10 winner (5 cells)
#   Block γ  refine around the sq τ=0.03 winner + push λ (3 cells)
#   Block δ  hazard closure: event_nll AND soft_hit + far/free correction
#            with the SH winner kept on (2 cells). Gate: pass = RMSE ≤
#            1.18 AND ABS_REL ≤ 0.445 → keep hazard alive; else archive.
#
# Code changes that landed *with* r4 (audited present in master):
#   • Sphere-weighted cos-lat metrics in utils/metrics.py +
#     utils/test_utils.py + test.py (also writes per_sample_<exp>.npz).
#   • train.py learns --lr-schedule=cosine and --lr-warmup-epochs.
#     (n2 doesn't use them; n9 does for exp981 / exp983.)
#
# Reference baselines (radial uniform metric):
#   echodiff exp11:   ABS 0.4300  RMSE 1.1060  D1 0.4876
#   exp907 expect:    ABS 0.4814  RMSE 1.2532  D1 0.5079  (round-2 bs32)
#   exp907 median:    ABS 0.4202  RMSE 1.36           ← best-ABS-ever
#   exp912 scalar bs32: ABS 0.4349 RMSE 1.27
#   exp946 SH λ=0.10: ABS 0.4413  RMSE ?           ← r3 R20 winner
#
# Pass / kill criteria (per cell):
#   β refine  ABS ≤ 0.435 → promote to R40 on n9 (already covered there)
#   γ refine  ABS ≤ 0.450 AND RMSE ≤ 1.25 → keep soft-quantile alive
#   δ hazard  RMSE ≤ 1.18 AND ABS ≤ 0.445 → keep hazard; else archive
#
# Hardware: node 2, 8-GPU. 4 workers × 2-GPU DataParallel pairs. Same
# anchor as r3 (bs=48, log32, σ=0.14, full ERP cos-lat + far-mask).
#
# Usage:
#   bash scripts/n2_bulk_0429_r4.sh                                # default
#   GPU_PAIRS="0,1"  bash scripts/n2_bulk_0429_r4.sh               # 1 worker
#   BS=32            bash scripts/n2_bulk_0429_r4.sh               # safer
#   EPOCHS=10        bash scripts/n2_bulk_0429_r4.sh               # short
#   EVAL_ALL_BESTS=0 bash scripts/n2_bulk_0429_r4.sh               # only score
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
# Node 2 dataset path (no underscore between 0303 and renew).
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303renew}"
EPOCHS="${EPOCHS:-20}"
WORKERS="${WORKERS:-2}"
VIS_PER_SCENE=0
SEED="${SEED:-1}"
BS="${BS:-48}"
EVAL_ALL_BESTS="${EVAL_ALL_BESTS:-1}"

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: DATASET_DIR not found: $DATASET_DIR" >&2
    echo "       Default for node 2 is /root/local1/changwoo/matterport3d_0303renew" >&2
    exit 3
fi

# Stay in the same log directory so r3 + r4 + earlier rounds aggregate
# under one grep — exp{900..} is one continuous timeline.
TRAIN_LOG_DIR="$PROJECT_DIR/logs/n9_0427_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n9_0427_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

GPU_PAIRS_STR="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
read -r -a GPU_PAIRS <<< "$GPU_PAIRS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

DF_AVAIL_KB=$(df -k "$PROJECT_DIR" | awk 'NR==2 {print $4}')
DF_AVAIL_GB=$((DF_AVAIL_KB / 1024 / 1024))
echo "[disk] $PROJECT_DIR free: ${DF_AVAIL_GB} GiB"
if [ "$DF_AVAIL_GB" -lt 5 ]; then
    echo "[disk] WARNING: <5 GiB free — clean logs/ eval/ results/ first."
fi

# ============================================================
# train_and_test — same contract as r3: skip-if-done, SIGTERM at
# JOB_TIMEOUT, hard kill 60 s later, then test the best checkpoint(s).
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
    local JOB_TIMEOUT="${JOB_TIMEOUT:-54000}"
    local ec=0
    CUDA_VISIBLE_DEVICES="$GPU" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED="$SEED" \
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
# Anchor base config (round 4): exp907 recipe (full ERP) at bs=48.
# All β / γ / δ cells inherit this.
# ============================================================
R_BASE="--depth-head-type range --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14 --range-output-mode expectation --lambda-range-nll 1.0 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask"

# Hazard closure base: exp934 recipe (soft_hit + far/free correction).
# event_nll / soft_hit use far_thresh internally; --hazard-far-thresh
# 9.8 is the same value the audit doc specifies.
HAZ_BASE="--depth-head-type hazard --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --hazard-bias-init -4.6 --hazard-warmup-epochs 3 --hazard-far-thresh 9.8 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask"

NS="bs${BS}_r4"

# ============================================================
# Cell definitions (16 cells: 6 carry-over + exp968–977)
# Format: "CONFIG  EXP  LR  CELL_BS  CELL_EPOCHS  [EXTRA…]"
# ============================================================
# Anchor base for carry-over cells that were on n9 in r3 (bs=32). When
# carried into n2 we still keep their *original* batch size 32 so the
# checkpoint and result are comparable to the round-2 anchor; per-cell
# bs overrides the default $BS.
R_BASE_BS32="--depth-head-type range --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14 --range-output-mode expectation --lambda-range-nll 1.0 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask"

ALL_EXPS=(
    # ── Block α: r3 R20 carry-over (6 cells) ─────────────────────────────
    # Same exp names + flags as r3 so the bin-based timeline (exp9xx) and
    # skip-if-done both still work. bs is fixed *per cell* to match the
    # original r3 settings (956/957 at bs=48, 964–967 at bs=32) so the
    # checkpoints are directly comparable to their r3 cohort even when
    # carried onto a different node.
    "echorange  exp956_R5D_cyl_horiz_sh_bs48_r3                  0.0001  48  20  ${R_BASE} --range-bin-axis horizontal --cyl-min-axis-factor 0.15 --lambda-spherical-sh 0.02 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp957_R5E_haz_softhit075_bs48_r3                0.0001  48  20  --depth-head-type hazard --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --hazard-bias-init -4.6 --hazard-warmup-epochs 3 --hazard-far-thresh 9.8 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask --hazard-aux-mode soft_hit --hazard-soft-hit-target 0.75 --lambda-hit 0.05 --lambda-free 0.02"
    "echorange  exp964_R5I_R20_combo_bs32_r3                     0.0001  32  20  ${R_BASE_BS32} --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.05 --lambda-spherical-sh 0.02 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp965_R5J_R20_exp907_repro_bs32_r3              0.0001  32  20  ${R_BASE_BS32}"
    "echorange  exp966_R5J_R20_cyl_horiz_base_bs32_r3            0.0001  32  20  ${R_BASE_BS32} --range-bin-axis horizontal --cyl-min-axis-factor 0.15"
    "echorange  exp967_R5J_R20_cyl_horiz_sq_bs32_r3              0.0001  32  20  ${R_BASE_BS32} --range-bin-axis horizontal --cyl-min-axis-factor 0.15 --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.05"

    # ── Block β: SH refine around exp946 winner (5 cells) ───────────────
    # exp946 was λ=0.10 / L=2 / log-depth → ABS 0.4413. Bracket it on
    # both sides + add a combo with the soft-quantile winner + an
    # ablation that drops --erp-far-mask (does SH already enforce far
    # layout, making far-mask redundant?).
    "echorange  exp968_R4B_sh_L2_l008_logd_${NS}                0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-spherical-sh 0.08 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp969_R4B_sh_L2_l015_logd_${NS}                0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-spherical-sh 0.15 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp970_R4B_sh_L2_l020_logd_${NS}                0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-spherical-sh 0.20 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp971_R4B_sh_l010_sq_q050_t003_${NS}           0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-spherical-sh 0.10 --spherical-sh-order 2 --spherical-sh-log-depth --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.03"
    "echorange  exp972_R4B_sh_l010_no_farmask_${NS}             0.0001  ${BS}  ${EPOCHS}  --depth-head-type range --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14 --range-output-mode expectation --lambda-range-nll 1.0 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --lambda-spherical-sh 0.10 --spherical-sh-order 2 --spherical-sh-log-depth"

    # ── Block γ: soft-quantile refine around exp939 winner (3 cells) ────
    # exp939 was q=0.50 / τ=0.03 / λ=0.25. Probe sharper τ (0.02), heavier
    # λ (0.50) coupled with the SH winner, and a push λ=1.0 to find where
    # the soft-quantile loss starts to dominate (and likely overfit).
    "echorange  exp973_R4G_sq_q050_t002_l025_${NS}              0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.02"
    "echorange  exp974_R4G_sq_q050_t003_l050_sh010_${NS}        0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.50 --soft-quantile-q 0.50 --soft-quantile-tau 0.03 --lambda-spherical-sh 0.10 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp975_R4G_sq_q050_t003_l100_${NS}              0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 1.0 --soft-quantile-q 0.50 --soft-quantile-tau 0.03"

    # ── Block δ: hazard far/free closure (2 cells) ──────────────────────
    # Brief 80-점 #5: hazard far/free correction is already implemented
    # inside event_nll / soft_hit / survival (audit confirmed). One cell
    # for each of event_nll and soft_hit so we can compare. λ_free 0.02
    # follows the brief's recommendation (the smaller of the two values).
    # SH=0.10 stays on so the comparison is "hazard primary on top of
    # the same SH-shaped distribution head" — what the brief calls the
    # "code-correctness closure for hazard before archival".
    "echorange  exp976_R4H_haz_event_nll_sh010_${NS}            0.0001  ${BS}  ${EPOCHS}  ${HAZ_BASE} --hazard-aux-mode event_nll --lambda-hit 0.05 --lambda-free 0.02 --lambda-spherical-sh 0.10 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp977_R4H_haz_softhit075_sh010_${NS}           0.0001  ${BS}  ${EPOCHS}  ${HAZ_BASE} --hazard-aux-mode soft_hit --hazard-soft-hit-target 0.75 --lambda-hit 0.05 --lambda-free 0.02 --lambda-spherical-sh 0.10 --spherical-sh-order 2 --spherical-sh-log-depth"
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
echo "n2_bulk_0429_r4 — Round 4 (n2): r3 R20 carry-over + SH refine + sq refine + Hfix"
echo "  $TOTAL cells (α=6 carry, β=5, γ=3, δ=2)"
echo "  EPOCHS=$EPOCHS  BS=$BS  EVAL_ALL_BESTS=$EVAL_ALL_BESTS"
echo "  dataset_dir=${DATASET_DIR:-<yaml default>}"
echo "  $NUM_WORKERS worker(s) × GPU pairs: ${GPU_PAIRS[*]}"
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
echo "n2_bulk_0429_r4 finished — $(date)"
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

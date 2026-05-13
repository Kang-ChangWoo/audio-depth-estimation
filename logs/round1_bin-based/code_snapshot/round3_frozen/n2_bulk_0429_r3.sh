#!/bin/bash
# ============================================================
# n2_bulk_0429_r3.sh — Round 5: Spherical Range Posterior + SH + Cylindrical
#
# CONTEXT — what already happened
# --------------------------------
# Round 1–2 (exp900–912) confirmed RangeDepthHead is a viable main path:
#   exp907 (bs=32, log32, sigma=0.14, cos-lat+far-mask, expectation)
#     → ABS 0.4814 RMSE 1.2532, but exp907 *median* hit ABS 0.4202 —
#     stronger than scalar same-env exp912 (ABS 0.4349). Hard median is
#     non-differentiable at train time though, so the gain was only
#     visible at inference.
#
# Round 3–4 (exp913–934) tried hazard heads. Result: hazard never beat
# distribution head. exp915/916 (depth-only / no-hit) reached ABS ~0.48
# RMSE 1.42 — about softmax baseline; exp913/914 (raw α-hit BCE) blew
# up to RMSE 1.9; round-4 rescue cells (exp930–934) stabilised but
# stayed at or below distribution head ceiling. See:
#   logs/round1_bin-based/E20260428-Round4_rigorous_hypothesis_review.md
#   logs/round1_bin-based/E20260428-exp917-934-Round4_hazard_rescue_evaluation.md
#
# THIS ROUND'S GOAL (round 5, 0429 — see plan doc)
# -------------------------------------------------
# Pivot: keep RangeDepthHead as main, add 4 axes that the analysis
# document identified as "highest-leverage next moves":
#   (a) Posterior representative: expectation / MAP / quantile / temp.
#       At eval-time we run a sweep; at train time we add a *soft*
#       quantile loss that keeps gradient alive (replaces hard median).
#   (b) Spherical-harmonic (SH) auxiliary loss on log-depth: matches
#       low-order ERP layout coefficients between pred and gt.
#   (c) Cylindrical bin axis (horizontal ρ_xy). Spherical near the poles
#       is degenerate; horizontal cylinder may be better matched to the
#       dominant scene structure. Cells use clamp_min=0.15 (≈|lat|<81°)
#       to avoid div-by-zero at the poles.
#   (d) 4-best checkpoint save (best_score / best_absrel / best_rmse /
#       best_delta1) so trade-offs aren't masked by a single composite.
#
# Hazard is NOT main this round — closure cell only (R5-E).
#
# n2's role: SOFT-QUANTILE × SH × CYLINDRICAL main sweep (22 cells).
# n9 covers epoch-budget paired baselines (40-epoch S40/R40) on 2 GPUs.
#
# Reference baselines on radial depth (R = ABS_REL / RMSE / δ1):
#   echodiff exp11 best:   R = 0.4300 / 1.1060 / 0.4876
#   softmax exp906 best:   R = 0.4814 / 1.2532 / 0.5079
#   exp907 expectation:    R ≈ 0.481  / 1.25   / 0.51   (round 2 best @bs=32)
#   exp907 median (eval):  R ≈ 0.420  / 1.36   / —     (best ABS_REL ever
#                                                       on this stack)
#   exp912 scalar @bs=32:  R = 0.4349 / 1.27   / —
#   round-4 hazard ok:     RMSE 1.42 (915), 1.43 (916)
#   round-4 hazard fail:   RMSE 1.93 (913), 1.89 (914)
#
# PASS / FAIL CRITERIA — promote / kill decisions
# -------------------------------------------------
# Per-cell smoke:
#   • train depth loss (D = BerHu+0.5·SILog) MUST decrease monotonically
#     through epoch 5. Any upward jump = aux-loss interaction broke
#     training, mark cell for kill.
#   • SH/sq aux losses must stay finite. NaN = bug, kill the whole bulk.
#
# Round-level decisions (after all cells finish):
#                                R5 best cell (val score 0.7·RMSE+0.3·ABS)
#                              ┌────────────────────────────────────────┐
#                              │  beats exp907 by ≥ 5 % (score)         │
#                              │  PROMOTE the winning aux/representive  │
#                              │  to next round main candidate. Re-run  │
#                              │  combo with promoted hp.               │
#                              ├────────────────────────────────────────┤
#                              │  matches exp907 ±2 %                   │
#                              │  Aux is harmless but not a clear win.  │
#                              │  Need n9 R40 paired evidence to decide │
#                              │  whether it's epoch budget vs head.    │
#                              ├────────────────────────────────────────┤
#                              │  worse than exp907 by ≥ 5 %            │
#                              │  KILL that aux direction. Falls back   │
#                              │  to expectation+SH=L2 baseline.        │
#                              └────────────────────────────────────────┘
#
# Best-checkpoint sweep:
#   For *every* cell we save 4 ckpts (best_score / best_rmse /
#   best_absrel / best_delta1). Test pass evaluates ALL FOUR so the
#   trade-off matrix is visible at the end of the night.
#
# Hardware: node 2, 8-GPU. 4 workers × 2-GPU DataParallel pairs.
#   Default GPU_PAIRS="0,1 2,3 4,5 6,7"; cells distribute round-robin.
#
# Batch size: 48 (echodiff+wav2vec cap, same as round 3/4). Per-GPU 24.
#
# Usage:
#   bash scripts/n2_bulk_0429_r3.sh                                # default
#   GPU_PAIRS="0,1"  bash scripts/n2_bulk_0429_r3.sh               # 1 worker
#   BS=32            bash scripts/n2_bulk_0429_r3.sh               # safer
#   EPOCHS=10        bash scripts/n2_bulk_0429_r3.sh               # short
#   EVAL_ALL_BESTS=1 bash scripts/n2_bulk_0429_r3.sh               # 4-best test
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
# Node 2 dataset path (no underscore between 0303 and renew). DO NOT copy
# n9 server's underscore variant here.
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303renew}"
EPOCHS="${EPOCHS:-20}"
WORKERS="${WORKERS:-2}"
VIS_PER_SCENE=0
SEED="${SEED:-1}"
BS="${BS:-48}"
# Round 5: when set to 1, evaluate best_score / best_rmse / best_absrel /
# best_delta1 each. Default 1 — 4× test cost is bounded by short test
# pass and the round 5 plan picks (b) "evaluate all 4 best" (see plan).
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

# Stay in the same log directory so the entire bin-based timeline
# (exp9xx) can be aggregated with one grep.
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
# train_and_test — same contract as r2: skip-if-done, SIGTERM at
# JOB_TIMEOUT, hard kill 60 s later, then test the best checkpoint(s).
# Round-5: $EVAL_ALL_BESTS=1 evaluates all 4 best_<tag>.pth.
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

    # Round-5: evaluate all 4 best ckpts when enabled.
    local TAGS
    if [ "$EVAL_ALL_BESTS" = "1" ]; then
        TAGS=(score absrel rmse delta1)
    else
        TAGS=(score)
    fi

    for TAG in "${TAGS[@]}"; do
        local CKPT
        if [ "$TAG" = "score" ]; then
            # legacy: best_model.pth = best_score.pth alias
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
            TEST_LOG="$TEST_LOG_DIR/${EXP}_test.log"     # legacy filename
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
# Anchor base config (round 5): exp907 recipe + ERP cos-lat + far-mask
# at bs=48, 20 epochs. ALL R-cells inherit this.
# ============================================================
R_BASE="--depth-head-type range --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14 --range-output-mode expectation --lambda-range-nll 1.0 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask"

# Hazard closure base — exp934 recipe (soft_hit + far/free correction).
HAZ_BASE="--depth-head-type hazard --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --hazard-bias-init -4.6 --hazard-warmup-epochs 3 --hazard-far-thresh 9.8 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask"

NS="bs${BS}_r3"

# ============================================================
# Cell definitions (exp936–957, 22 cells)
# Format: "CONFIG  EXP  LR  CELL_BS  CELL_EPOCHS  [EXTRA…]"
# ============================================================
ALL_EXPS=(
    # ── Block A: Soft-quantile sweep (8 cells) ────────────────────────
    # q × τ × λ_sq exploration. Anchor = q=0.50 / τ=0.05 / λ=0.25.
    # The 8 picks below cover all 12 combinations minus 4 (we drop
    # extreme-low (q=0.45,τ=0.03,λ=0.5) and (q=0.55,τ=0.03,λ=0.5)
    # which have heavier interactions). All cells use SH=0, axis=radial.
    "echorange  exp936_R5A_sq_q050_t005_l025_${NS}  0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.05"
    "echorange  exp937_R5A_sq_q045_t005_l025_${NS}  0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.25 --soft-quantile-q 0.45 --soft-quantile-tau 0.05"
    "echorange  exp938_R5A_sq_q055_t005_l025_${NS}  0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.25 --soft-quantile-q 0.55 --soft-quantile-tau 0.05"
    "echorange  exp939_R5A_sq_q050_t003_l025_${NS}  0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.03"
    "echorange  exp940_R5A_sq_q045_t003_l025_${NS}  0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.25 --soft-quantile-q 0.45 --soft-quantile-tau 0.03"
    "echorange  exp941_R5A_sq_q050_t005_l050_${NS}  0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.50 --soft-quantile-q 0.50 --soft-quantile-tau 0.05"
    "echorange  exp942_R5A_sq_q045_t005_l050_${NS}  0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.50 --soft-quantile-q 0.45 --soft-quantile-tau 0.05"
    "echorange  exp943_R5A_sq_q050_t003_l050_${NS}  0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.50 --soft-quantile-q 0.50 --soft-quantile-tau 0.03"

    # ── Block B: SH sweep (6 cells) ───────────────────────────────────
    # L=2 × λ_SH ∈ {0.02, 0.05, 0.10}
    # L=3 × λ_SH ∈ {0.02, 0.05}
    # L=2 λ=0.02 with log_depth=false (sanity ablation)
    "echorange  exp944_R5B_sh_L2_l002_logd_${NS}    0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-spherical-sh 0.02 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp945_R5B_sh_L2_l005_logd_${NS}    0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-spherical-sh 0.05 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp946_R5B_sh_L2_l010_logd_${NS}    0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-spherical-sh 0.10 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp947_R5B_sh_L3_l002_logd_${NS}    0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-spherical-sh 0.02 --spherical-sh-order 3 --spherical-sh-log-depth"
    "echorange  exp948_R5B_sh_L3_l005_logd_${NS}    0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-spherical-sh 0.05 --spherical-sh-order 3 --spherical-sh-log-depth"
    "echorange  exp949_R5B_sh_L2_l002_lin_${NS}     0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-spherical-sh 0.02 --spherical-sh-order 2 --no-spherical-sh-log-depth"

    # ── Block C: Combo (sq + SH) (4 cells) ────────────────────────────
    # anchor sq + L=2 SH at λ ∈ {0.02, 0.05}, plus q=0.45/0.55 sweeps at
    # λ_SH=0.02 to see if sq-sweep stays consistent under SH.
    "echorange  exp950_R5C_combo_q050_sh002_${NS}   0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.05 --lambda-spherical-sh 0.02 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp951_R5C_combo_q050_sh005_${NS}   0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.05 --lambda-spherical-sh 0.05 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp952_R5C_combo_q045_sh002_${NS}   0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.25 --soft-quantile-q 0.45 --soft-quantile-tau 0.05 --lambda-spherical-sh 0.02 --spherical-sh-order 2 --spherical-sh-log-depth"
    "echorange  exp953_R5C_combo_q055_sh002_${NS}   0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --lambda-soft-quantile 0.25 --soft-quantile-q 0.55 --soft-quantile-tau 0.05 --lambda-spherical-sh 0.02 --spherical-sh-order 2 --spherical-sh-log-depth"

    # ── Block D: Cylindrical (3 cells) ────────────────────────────────
    # bin axis = horizontal ρ_xy = D·cos(lat), clamp_min=0.15 → polar
    # pixels (|lat| > 81°) masked from all losses. Pred is projected
    # back to radial inside the train step before BerHu/SILog/SH and
    # inside test_utils before metric so the eval metric stays radial.
    "echorange  exp954_R5D_cyl_horiz_base_${NS}     0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --range-bin-axis horizontal --cyl-min-axis-factor 0.15"
    "echorange  exp955_R5D_cyl_horiz_sq_${NS}       0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --range-bin-axis horizontal --cyl-min-axis-factor 0.15 --lambda-soft-quantile 0.25 --soft-quantile-q 0.50 --soft-quantile-tau 0.05"
    "echorange  exp956_R5D_cyl_horiz_sh_${NS}       0.0001  ${BS}  ${EPOCHS}  ${R_BASE} --range-bin-axis horizontal --cyl-min-axis-factor 0.15 --lambda-spherical-sh 0.02 --spherical-sh-order 2 --spherical-sh-log-depth"

    # ── Block E: Hazard closure (1 cell) ─────────────────────────────
    # exp934-style soft_hit recipe with far/free correction. PASS:
    #   RMSE ≤ 1.18 AND ABS_REL ≤ 0.445  → keep hazard alive
    #   RMSE > 1.18 OR ABS_REL > 0.445  → archive hazard
    "echorange  exp957_R5E_haz_softhit075_${NS}     0.0001  ${BS}  ${EPOCHS}  ${HAZ_BASE} --hazard-aux-mode soft_hit --hazard-soft-hit-target 0.75 --lambda-hit 0.05 --lambda-free 0.02"
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
echo "n2_bulk_0429_r3 — Round 5 (n2): SQ × SH × Cylindrical sweep"
echo "  $TOTAL cells (A=8 sq, B=6 SH, C=4 combo, D=3 cyl, E=1 hazard)"
echo "  EPOCHS=$EPOCHS  BS=$BS (DP, per-GPU $((BS / 2)))  silog=0.5"
echo "  EVAL_ALL_BESTS=$EVAL_ALL_BESTS  (1 = test all 4 best ckpts)"
echo "  dataset_dir=${DATASET_DIR:-<yaml default>}"
echo "  $NUM_WORKERS worker(s) × GPU pairs: ${GPU_PAIRS[*]}"
echo "  SEED=$SEED  $(date)"
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
echo "n2_bulk_0429_r3 finished — $(date)"
echo "============================================================"
echo ""
# Report matrix: rows = cells, cols = best_<tag>.
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

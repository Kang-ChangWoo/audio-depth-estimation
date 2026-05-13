#!/bin/bash
# ============================================================
# n2_bulk_0428_r2.sh — Round 4: Hazard Rescue / Kill Test
#
# CONTEXT — what already happened
# --------------------------------
# Round 3 (n2_bulk_0428.sh, exp913–917) tried an A-Hazard depth head.
# Cells with raw α-hit BCE on (exp913 full, exp914 no-free) blew up to
# val RMSE ≈ 1.9; cells with L_hit off (exp915 no-hit, exp916 depth-only)
# trained cleanly to RMSE ≈ 1.42. The smoking gun was a depth-loss jump
# at epoch 3→4 — exactly when the round-3 warmup λ_hit (0.3 → 0.5) flipped
# discontinuously. Diagnosis: BCE(α, target=1) saturates sigmoid α → 1,
# ∂α/∂logit = α(1−α) → 0, renderer stuck in wrong commit.
#
# (Full diagnosis with code+numbers:
#  logs/n9_0427_train/E20260428-haz_consult_request.md)
#
# THIS ROUND'S GOAL
# ------------------
# Rescue or kill the A-Hazard direction with the smallest possible cell
# set. The hypothesis under test is that the failure is *raw α-hit BCE
# specifically*, not hazard rendering as such — and that supervising
# rendered quantities (w_j or T_j), not raw α_j, dodges the saturation
# trap. Each cell is designed to rule in or rule out one specific cause.
#
# Round 4 cells (6 train + 0 test_only):
#   R4-00  raw_hit reproduction (5 epochs only — failure-signature dump)
#   R4-01  event_nll + weak free        ← MAIN CELL — implement first
#   R4-02  event_nll only (no free)
#   R4-03  survival / ordinal supervision
#   R4-04  soft_hit (target=0.75) + weak free  — α-supervision salvage test
#   R4-05  exp917 retry (raw_hit, λ_hit=1.0/λ_free=0.2, warmup 5).
#          Original round-3 exp917 died at epoch-2 val on the
#          torch.quantile CUDA-element cap inside _hazard_diagnostics
#          (B·R·H·W = 67M > 2^24). Fix landed (subsample-then-quantile);
#          retry here under round-4 raw_hit aux + smooth ramp so the
#          stronger-supervision data point isn't lost from the round.
#
# Common to all 5:
#   • Smooth ramp: λ_aux scales linearly 0 → target over the first
#     `--hazard-warmup-epochs` epochs, then holds. NO discontinuous jump
#     (root-cause fix vs round 3).
#   • Far-censored exclusion: GT ≥ 9.8 m treated as censored — excluded
#     from event/hit/soft_hit (they're not real surfaces). Survival mode
#     uses them as "we know D > r_j for r_j < 9.8" supervision.
#   • Diagnostic dump on first val batch every val pass:
#     alpha p50/p90/p95/p99, frac>{0.5,0.9,0.95,0.99}, bg_weight
#     percentiles, argmax-bin folded histogram, weight entropy, depth-
#     bin sliced ABS_REL/RMSE for [0.1,1) [1,3) [3,6) [6,9.8) [9.8,10.0).
#   • silog 0.5 (yaml default — round-2 audit fix).
#
# Reference baselines on radial depth:
#   echodiff exp11 best:  ABS 0.4300 RMSE 1.1060 δ1 0.4876
#   softmax exp906 best:  ABS 0.4814 RMSE 1.2532 δ1 0.5079
#   round-3 hazard fail:  RMSE 1.93 (913), 1.89 (914)
#   round-3 hazard ok:    RMSE 1.42 (915), 1.43 (916)
#
# PASS / FAIL CRITERIA (decided BEFORE looking at the runs)
# ---------------------------------------------------------
# Per-cell smoke checks (look at the train log):
#   • train depth loss (D = BerHu+0.5·SILog) MUST be monotonically
#     decreasing through epochs 3-4. ANY upward jump at the warmup-end
#     epoch is a failure signature → hazard rescue likely failed for
#     that cell, mark for kill.
#   • frac_alpha_>0.95 must NOT explode in epochs 2-4 from <5 % to
#     >50 %. If it does, we hit the round-3 saturation trap again.
#   • bg_weight_mean must NOT collapse to 0 within the first 3 epochs.
#     A healthy hazard model should keep some background mass for far
#     pixels (since we exclude censored from supervision).
#   • argmax_bin_hist must NOT collapse to a single near-bin or far-bin.
#     Healthy distribution: spread across mid-bins matching GT histogram.
#
# Round-level decision (after all 5 cells finish):
#
#                                R4-01 result (val RMSE)
#                            ┌─────────────────────────────┐
#                            │  ≤ 1.30                     │
#                            │  KEEP HAZARD as candidate.  │
#                            │  Next round: tune sigma,    │
#                            │  add MACV/ToF evidence.     │
#                            ├─────────────────────────────┤
#                            │  1.30–1.42 (≈ exp915)       │
#                            │  EVENT NLL DOES NOT BEAT    │
#                            │  depth-only hazard. Either  │
#                            │  hazard-specific aux is     │
#                            │  redundant, or the head's   │
#                            │  ceiling is set by capacity │
#                            │  / data, not by supervision │
#                            │  design. Compare to R4-03   │
#                            │  before pivoting.           │
#                            ├─────────────────────────────┤
#                            │  > 1.42 OR train D jumps    │
#                            │  KILL HAZARD. Pivot to      │
#                            │  ordinal/quantile head      │
#                            │  (DORN-style).              │
#                            └─────────────────────────────┘
#
# Cross-cell decisions:
#   R4-01 vs R4-02:  same RMSE → free loss is redundant. R4-01 better →
#                    explicit free-space prior helps. R4-02 better → free
#                    actively hurts (over-suppresses near bins).
#   R4-03 ≥ R4-01:   prefer survival/ordinal formulation going forward.
#   R4-04 ≈ R4-01:   raw α-supervision is salvageable with soft target.
#                    Round 3's failure was specifically the target=1 BCE.
#   R4-04 fails alone: confirms α-direct supervision is the broken piece.
#   R4-00 collapses with new logging: failure signature documented for
#                    the report.
#
# Hardware: node 2, 8-GPU. 4 workers × 2-GPU DataParallel pairs.
#   Default GPU_PAIRS="0,1 2,3 4,5 6,7"; 5 cells distribute round-robin
#   so worst-case load is 2 jobs on the worker that owns R4-00 and R4-04.
#
# Batch size: 48 (echodiff+wav2vec cap from n2_revisit:215; Round 3 ran
#   at the same value without OOM). Per-GPU under DP: 24.
#
# Usage:
#   bash scripts/n2_bulk_0428_r2.sh                                # default
#   GPU_PAIRS="0,1"        bash scripts/n2_bulk_0428_r2.sh         # 1 worker
#   BS=32                  bash scripts/n2_bulk_0428_r2.sh         # safer
#   EPOCHS=10              bash scripts/n2_bulk_0428_r2.sh         # short
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
WORKERS="${WORKERS:-2}"      # n2_revisit lowered 4→2 to avoid dataloader OOM
VIS_PER_SCENE=0
SEED="${SEED:-1}"            # PYTHONHASHSEED only — train.py has no
                             # torch.manual_seed yet, so runs are
                             # non-deterministic regardless.
BS="${BS:-48}"               # echodiff+wav2vec cap on this node

DATASET_ARGS=()
if [ -n "$DATASET_DIR" ]; then
    DATASET_ARGS=(--dataset-dir "$DATASET_DIR")
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: DATASET_DIR not found: $DATASET_DIR" >&2
    echo "       Default for node 2 is /root/local1/changwoo/matterport3d_0303renew" >&2
    echo "       (no underscore between 0303 and renew)." >&2
    exit 3
fi

# Stay in the round-2/3 log directory so the entire bin-based timeline
# (exp9xx) can be aggregated with one grep.
TRAIN_LOG_DIR="$PROJECT_DIR/logs/n9_0427_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n9_0427_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

# 2-GPU-per-process DataParallel (matches n2_revisit / n2_bulk_0428).
GPU_PAIRS_STR="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
read -r -a GPU_PAIRS <<< "$GPU_PAIRS_STR"
NUM_WORKERS="${#GPU_PAIRS[@]}"

# Disk-space sanity (round-3 echodiff_ambi_cide died with No space left).
DF_AVAIL_KB=$(df -k "$PROJECT_DIR" | awk 'NR==2 {print $4}')
DF_AVAIL_GB=$((DF_AVAIL_KB / 1024 / 1024))
echo "[disk] $PROJECT_DIR free: ${DF_AVAIL_GB} GiB"
if [ "$DF_AVAIL_GB" -lt 5 ]; then
    echo "[disk] WARNING: <5 GiB free — clean logs/ eval/ results/ first."
fi

# ============================================================
# train_and_test — same contract as n2_bulk_0428: skip-if-done, SIGTERM
# at JOB_TIMEOUT, hard kill 60 s later, test the resulting best ckpt.
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
# Hazard cell base — common to all 5 R4 cells.
# IMPORTANT: --lambda-silog 0.5 (NOT 1.0). Round-2 audit fix.
# IMPORTANT: --hazard-warmup-epochs 3 → smooth ramp 0 → target over
#            epochs 1-3 (replaces round-3's discontinuous jump at ep 4).
# IMPORTANT: --hazard-far-thresh 9.8 → exclude 10 m clamp pixels.
# ============================================================
HAZ_BASE="--depth-head-type hazard --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --hazard-bias-init -4.6 --hazard-warmup-epochs 3 --hazard-far-thresh 9.8 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask"

# Suffix tracks BS so checkpoints don't collide with round-3 (which was
# also bs=48). Round-4 names use _r2 suffix.
NS="bs${BS}_r2"

# ============================================================
# Cell definitions
# Format: "CONFIG  EXP  LR  CELL_BS  CELL_EPOCHS  [EXTRA…]"
# ============================================================
ALL_EXPS=(
    # ── R4-00: raw-hit reproduction with new logging.
    # Goal: confirm round-3 collapse pattern with the new diagnostics.
    # Only 5 epochs — long enough to see the saturation trap fire.
    # Failure signature we expect to see:
    #   • frac_alpha_>0.95 jumps from <5 % at ep 1 to >50 % by ep 3-4.
    #   • bg_weight collapses toward 0.
    #   • argmax_bin_hist concentrates on a single near or far bin.
    #   • train D plateaus around 0.5 (no clean monotonic decrease).
    # NOTE: even though we apply the round-4 smooth ramp here, raw_hit
    #   should still collapse — the saturation is intrinsic to BCE(α,1),
    #   independent of the schedule discontinuity. If R4-00 actually
    #   trains fine, then round-3 failure was the ramp jump alone, not
    #   the loss form, and we should re-evaluate R4-01..R4-04 conclusions.
    "echorange  exp930_haz_R4_00_rawhit_repro_lr1e4_${NS}              0.0001  ${BS}  5   ${HAZ_BASE} --hazard-aux-mode raw_hit --lambda-hit 0.5 --lambda-free 0.10"

    # ── R4-01: MAIN cell. Rendered first-hit weight NLL + weak free.
    # If this trains cleanly with val RMSE ≤ 1.30, hazard direction is
    # rescued and the round-3 failure is confirmed as raw-α specific.
    # If RMSE 1.30–1.42, see R4-02 for whether free is doing anything.
    # If RMSE > 1.42 (worse than depth-only exp916), kill hazard.
    "echorange  exp931_haz_R4_01_eventnll_free_lr1e4_${NS}             0.0001  ${BS}  ${EPOCHS}  ${HAZ_BASE} --hazard-aux-mode event_nll --hazard-event-sigma-bins 1.0 --lambda-hit 0.10 --lambda-free 0.05"

    # ── R4-02: event_nll only. Identical to R4-01 minus L_free.
    # Compare: free helps / hurts / no-effect.
    "echorange  exp932_haz_R4_02_eventnll_only_lr1e4_${NS}             0.0001  ${BS}  ${EPOCHS}  ${HAZ_BASE} --hazard-aux-mode event_nll --hazard-event-sigma-bins 1.0 --lambda-hit 0.10 --disable-free-loss"

    # ── R4-03: survival / ordinal. Different formulation entirely.
    # P(D > r_j) is bounded and monotone-in-α (cumprod), gradient via
    # log_S is well-behaved everywhere. If R4-01 fails but R4-03 works,
    # we should pivot to ordinal-style hazard going forward.
    "echorange  exp933_haz_R4_03_survival_lr1e4_${NS}                  0.0001  ${BS}  ${EPOCHS}  ${HAZ_BASE} --hazard-aux-mode survival --hazard-survival-tau-bins 1.0 --lambda-hit 0.10"

    # ── R4-04: soft-hit α BCE (target=0.75). Salvage test for raw-α.
    # If R4-04 trains and matches R4-01, raw-α supervision was just over-
    # weighted in round 3. If R4-04 alone fails, confirms α-direct
    # supervision is fundamentally the broken piece.
    "echorange  exp934_haz_R4_04_softhit_t075_lr1e4_${NS}              0.0001  ${BS}  ${EPOCHS}  ${HAZ_BASE} --hazard-aux-mode soft_hit --hazard-soft-hit-target 0.75 --lambda-hit 0.05 --lambda-free 0.02"

    # ── R4-05: exp917 retry — raw_hit with stronger supervision
    # (λ_hit=1.0, λ_free=0.2, warmup 5 → overrides HAZ_BASE's warmup 3).
    # Original round-3 run died at epoch-2 val on the torch.quantile
    # element-count cap (now fixed via subsample-then-quantile in
    # _hazard_diagnostics). The _r2 suffix in NS keeps logs/checkpoints
    # disjoint from the failed round-3 run, so reruns don't clobber.
    "echorange  exp917_echorange_haz32log_d10_strong_lr1e4_${NS}       0.0001  ${BS}  ${EPOCHS}  ${HAZ_BASE} --hazard-aux-mode raw_hit --lambda-hit 1.0 --lambda-free 0.2 --hazard-warmup-epochs 5"
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
echo "n2_bulk_0428_r2 — Round 4 Hazard Rescue/Kill (6 cells: 5 R4 + exp917 retry)"
echo "  EPOCHS=$EPOCHS  BS=$BS (DP, per-GPU $((BS / 2)))  silog=0.5"
echo "  far_thresh=9.8  warmup=3 (smooth ramp, NO jump)"
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
echo "n2_bulk_0428_r2 finished — $(date)"
echo "============================================================"
echo ""
printf "%-58s %8s %8s %8s\n" "Experiment" "ABS_REL" "RMSE" "Delta1"
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
echo "Success: $SUCCESS / $TOTAL"
echo "============================================================"

# ============================================================
# ANALYSIS PLAYBOOK — read this AFTER all 5 cells finish
# ============================================================
#
# Step 1 — read each train log's [haz] diagnostic dumps:
#   grep '\[haz\]' logs/n9_0427_train/exp93[0-4]_*.log | head -200
#
#   For each cell, look at how the diagnostics evolve epoch-by-epoch:
#     • alpha p95 trajectory:  healthy < 0.7; collapse risk > 0.95.
#     • frac_alpha_>0.95     : healthy < 10 %; collapse > 50 %.
#     • bg_weight p50         : healthy ~0.05–0.30; collapse near 0.
#     • argmax_bin_hist       : healthy spread mid-bins;
#                               unhealthy concentration in 1-2 bins.
#     • sliced ABS_REL        : near (0.1–1) and mid (1–6) should be
#                               low; far (6–9.8) is the hard slice.
#
# Step 2 — compare to round-3 baselines:
#   exp913 RMSE 1.93  exp915 RMSE 1.42  exp906 RMSE 1.25  exp11 RMSE 1.11
#
#   Decision tree (R4-01 = exp931 = main):
#     RMSE ≤ 1.25 ............... HAZARD WINS, beats softmax baseline.
#                                   Promote to next round main candidate.
#     1.25 < RMSE ≤ 1.30 ........ HAZARD VIABLE. Tune sigma / bins next.
#     1.30 < RMSE ≤ 1.42 ........ AUX SUPERVISION ≈ DEPTH-ONLY. Compare
#                                   to R4-03 (exp933): if survival is
#                                   notably better, switch to it; else
#                                   the head's ceiling is set by data
#                                   /capacity, not aux design — pivot.
#     RMSE > 1.42 OR D-jump ..... HAZARD KILLED. Pivot to ordinal /
#                                   quantile head (DORN-style).
#
# Step 3 — cross-cell ablation reads:
#   R4-00 (exp930): if it collapsed with the new ramp, raw-α BCE is
#       intrinsically broken (independent of schedule). If it trained
#       fine, the round-3 failure was the ramp jump alone — re-evaluate.
#   R4-01 vs R4-02 (exp931 vs exp932): same RMSE → free is redundant;
#       R4-01 better → free helps; R4-02 better → free hurts.
#   R4-03 vs R4-01 (exp933 vs exp931): survival ≥ event → ordinal wins.
#   R4-04 vs round-3 exp915 (exp934 vs round-3 1.42): R4-04 ≪ 1.42 →
#       raw-α supervision is salvageable with a soft target. R4-04 ≈ 1.42
#       or worse → α-direct supervision is the broken piece.
#
# Step 4 — write a short "round 4 verdict" doc next to this script and
#   the round-1 doc. Mention which cell wins, what the diagnostic
#   signature was, and what the next round's candidate cells are.
#
# Step 5 (only if hazard is killed) — first quick pivot to ordinal:
#   add `range_output_mode='ordinal'` to RangeDepthHead (single-bit-per-bin
#   sigmoid head with cumulative training target) and run a 1-cell sanity
#   on softmax exp906's hyperparameters. If it matches exp906's RMSE
#   immediately, pursue ordinal as the new main direction.
# ============================================================

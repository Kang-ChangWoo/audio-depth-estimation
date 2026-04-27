#!/bin/bash
# ============================================================
# n2_bulk.sh — N2 train+test pipeline (exp187-210)
#
# Temporal FOA decomposition: 8 architecture variants testing
# direct ambisonics, temporally binned features, and
# visualization-based inputs for depth estimation.
#
# Node 2: 8 GPUs, 4 workers (2 GPUs each), running in parallel.
# Each worker: train → test → train → test → ... sequentially.
#
# Usage: bash scripts/n2_bulk.sh
# ============================================================
set -euo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

# Activate conda env if not already active
if [ -z "${CONDA_SHLVL:-}" ]; then
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
fi
conda activate shared_audio 2>/dev/null || conda activate base 2>/dev/null || true

WORKERS=2
EPOCHS=40
VIS_PER_SCENE=100

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n2_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n2_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

# ---- Config resolution ----
get_config() {
    local exp="$1"
    case "$exp" in
        *n2_6ch*)       echo "n2_6ch_input" ;;
        *n2_trms_f*)    echo "n2_temporal_rms_film" ;;
        *n2_trms*)      echo "n2_temporal_rms" ;;
        *n2_tenergy*)   echo "n2_temporal_energy" ;;
        *n2_dual*)      echo "n2_dual_enc" ;;
        *n2_stft*)      echo "n2_foa_stft_film" ;;
        *)              echo "UNKNOWN" ;;
    esac
}

# ---- Train one experiment, then test it immediately ----
train_and_test() {
    local GPU="$1"
    local CONFIG="$2"
    local EXP="$3"
    local LR="$4"
    local LSH="$5"

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TRAIN $EXP (config=$CONFIG lr=$LR lsh=$LSH)"

    CUDA_VISIBLE_DEVICES="$GPU" python3 train.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size 128 \
        --num-workers "$WORKERS" \
        --lambda-sh "$LSH" \
        --rotate-canonical \
        > "$TRAIN_LOG_DIR/${EXP}.log" 2>&1

    local TRAIN_EC=$?
    if [ "$TRAIN_EC" -ne 0 ]; then
        echo "  [GPU=$GPU] $EXP TRAIN_FAILED (exit=$TRAIN_EC)"
        return
    fi

    local DONE_LINE=$(grep "^Done\." "$TRAIN_LOG_DIR/${EXP}.log" | tail -1)
    echo "  [GPU=$GPU] $EXP TRAIN_DONE — $DONE_LINE"

    local CKPT_PATH=$(find checkpoints/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
    if [ -z "$CKPT_PATH" ]; then
        echo "  [GPU=$GPU] $EXP TEST_SKIP (no checkpoint)"
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TEST  $EXP"

    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        python3 -u test.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --checkpoint-path "$CKPT_PATH" \
        --eval-on test \
        --batch-size 1 \
        --vis-per-scene "$VIS_PER_SCENE" \
        --rotate-canonical \
        > "$TEST_LOG_DIR/${EXP}_test.log" 2>&1

    local TEST_EC=$?
    if [ "$TEST_EC" -ne 0 ]; then
        echo "  [GPU=$GPU] $EXP TEST_FAILED (exit=$TEST_EC)"
    else
        local ABS=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        local RMSE=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        local D1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        echo "  [GPU=$GPU] $EXP TEST_OK  ABS=$ABS RMSE=$RMSE D1=$D1"
    fi
}

# ---- Test-only: rerun test for an already-trained experiment ----
test_only() {
    local GPU="$1"
    local CONFIG="$2"
    local EXP="$3"

    local CKPT_PATH=$(find checkpoints/ -name "best_model.pth" -path "*${EXP}*" 2>/dev/null | head -1)
    if [ -z "$CKPT_PATH" ]; then
        echo "  [GPU=$GPU] $EXP TEST_SKIP (no checkpoint)"
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TEST  $EXP (retest)"

    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        python3 -u test.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --checkpoint-path "$CKPT_PATH" \
        --eval-on test \
        --batch-size 4 \
        --vis-per-scene "$VIS_PER_SCENE" \
        --rotate-canonical \
        > "$TEST_LOG_DIR/${EXP}_test.log" 2>&1

    local TEST_EC=$?
    if [ "$TEST_EC" -ne 0 ]; then
        echo "  [GPU=$GPU] $EXP TEST_FAILED (exit=$TEST_EC)"
    else
        local ABS=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        local RMSE=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        local D1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$TEST_LOG_DIR/${EXP}_test.log" | head -1)
        echo "  [GPU=$GPU] $EXP TEST_OK  ABS=$ABS RMSE=$RMSE D1=$D1"
    fi
}

# ============================================================
# Experiment definitions (14 experiments, exp187-200):
#   "CONFIG  EXP  LR  LSH"
# ============================================================
ALL_EXPS=(
    # --- E1: 6-channel input (DONE — train 40/40) ---
    # "n2_6ch_input        exp187_n2_6ch_lr1e3_lsh0.1     0.001  0.1"   # DONE 40/40
    # "n2_6ch_input        exp188_n2_6ch_lr5e4_lsh0.1     0.0005 0.1"   # DONE 40/40
    # "n2_6ch_input        exp189_n2_6ch_lr1e3_lsh0.3     0.001  0.3"   # DONE 40/40
    # --- E2: Temporal RMS supervision (12-dim target) ---
    # "n2_temporal_rms     exp190_n2_trms_lr1e3_lsh0.1    0.001  0.1"   # DONE 40/40
    # "n2_temporal_rms     exp191_n2_trms_lr5e4_lsh0.1    0.0005 0.1"   # DONE 40/40
    # --- E3: Temporal energy attention ---
    # "n2_temporal_energy  exp192_n2_tenergy_lr1e3_lsh0.1  0.001  0.1"   # DONE 40/40
    # "n2_temporal_energy  exp193_n2_tenergy_lr5e4_lsh0.1  0.0005 0.1"   # DONE 40/40
    # "n2_temporal_energy  exp194_n2_tenergy_lr1e3_lsh0.3  0.001  0.3"   # DONE 40/40
    # --- E4: Dual encoder (binaural + FOA spec) ---
    # "n2_dual_enc         exp195_n2_dual_lr1e3_lsh0.1    0.001  0.1"   # DONE 40/40
    # "n2_dual_enc         exp196_n2_dual_lr5e4_lsh0.1    0.0005 0.1"   # DONE 40/40
    # --- E5: FOA STFT FiLM ---
    # "n2_foa_stft_film    exp197_n2_stft_lr1e3_lsh0.1    0.001  0.1"   # DONE 40/40
    # "n2_foa_stft_film    exp198_n2_stft_lr5e4_lsh0.1    0.0005 0.1"   # DONE 40/40
    # --- E6: Temporal RMS FiLM (DONE 40/40) ---
    # "n2_temporal_rms_film exp199_n2_trms_film_lr1e3_lsh0.1  0.001  0.1"   # DONE 40/40
    # "n2_temporal_rms_film exp200_n2_trms_film_lr5e4_lsh0.1  0.0005 0.1"   # DONE 40/40
    # --- E7: Temporal energy map input concat (DONE 40/40) ---
    # "n2_temap_input       exp201_n2_temap_lr1e3_lsh0.1      0.001  0.1"   # DONE 40/40
    # "n2_temap_input       exp202_n2_temap_lr5e4_lsh0.1      0.0005 0.1"   # DONE 40/40
    # "n2_temap_input       exp203_n2_temap_lr1e3_lsh0.3      0.001  0.3"   # DONE 40/40
    # --- E8: Temporal bin cross-attention (DONE 40/40) ---
    # "n2_tbin_crossattn    exp204_n2_xattn_lr1e3_lsh0.1      0.001  0.1"   # DONE 40/40
    # "n2_tbin_crossattn    exp205_n2_xattn_lr5e4_lsh0.1      0.0005 0.1"   # DONE 40/40
    # "n2_tbin_crossattn    exp206_n2_xattn_lr1e3_lsh0.3      0.001  0.3"   # DONE 40/40

    # ============================================================
    # exp231-240 — overlapping temporal bins (gradual information)
    # Ref: docs/report_e_0419_analysis.md §12 (esp. §12.5, §12.6)
    #
    # NAMING RULE: first column = CONFIG filename (config/<name>.yaml).
    # Inside each YAML, model.name MUST stay one of the existing
    # _N2_CLASSES keys (n2_temporal_energy, n2_temap_input, ...), else
    # is_n2_model() returns False and the N2 train path is skipped.
    #
    # PREREQS before uncommenting:
    #   (items 1-5 in §12.5 of report_e; summary below)
    #
    #   [items 1+2] required for exp231-235 (dataset + YAML only):
    #     1. data/dataset_n2.py — add to PRESET dict:
    #          BINS_3_OVERLAP = [(0,13000), (2600,18000), (8000,None)]
    #          BINS_4_OVERLAP = [(0,8800), (2600,11000), (5400,15000), (8800,None)]
    #        Accept string keys 'overlap3'/'overlap4' in n_temporal_bins validator.
    #     2. config/*.yaml — new YAMLs with overlap-specific filenames but
    #        canonical model.name. Set dataset.n_temporal_bins: overlap3 (or 4).
    #
    #   [item 3] required for exp236, 238, 239-240 (model code):
    #     3. models/n2_0417/n2_temporal_energy.py — expose:
    #          n_bins (default 3) → parameterize head count + attn_inject_indices
    #          gain_mode ('monotone'|'signed') → h*(1+α*(2*emap-1)) path
    #          cond_source ('bottleneck'|'decoder_level') → energy head input
    #
    #   [item 4] required for exp237 (train-loop code):
    #     4. train.py — energy-loss branch reads cfg.model.lambda_bins (list)
    #        if present; else fall back to scalar lambda_energy.
    #
    # Launch order: 231-235 (items 1+2) → 236-238 (items 3+4) → 239-240 (item 3, n_bins=4).
    # Verify log header prints "N2 dataset: K temporal bins" with K matching cfg.
    # ============================================================

    # --- Layer 1: pure overlap binning (current arch) ---
    "n2_temporal_energy_overlap3   exp231_n2_tenergy_ov3_lr1e3_lsh0.1    0.001  0.1"
    "n2_temporal_energy_overlap3   exp232_n2_tenergy_ov3_lr5e4_lsh0.1    0.0005 0.1"
    "n2_temporal_energy_overlap3   exp233_n2_tenergy_ov3_lr1e3_lsh0.3    0.001  0.3"
    # --- Layer 2: overlap as input concat (no attention) ---
    "n2_temap_input_overlap3       exp234_n2_temap_ov3_lr1e3_lsh0.1      0.001  0.1"
    "n2_temap_input_overlap3       exp235_n2_temap_ov3_lr5e4_lsh0.1      0.0005 0.1"
    # --- Layer 3: overlap + arch fixes ---
    "n2_temporal_energy_overlap3_signed   exp236_n2_tenergy_ov3_signed_lr1e3   0.001  0.1"
    "n2_temporal_energy_overlap3_wloss    exp237_n2_tenergy_ov3_wloss_lr1e3    0.001  0.1"
    "n2_temporal_energy_overlap3_deccond  exp238_n2_tenergy_ov3_deccond_lr1e3  0.001  0.1"
    # --- Layer 4: 4-bin sliding window ---
    "n2_temporal_energy_overlap4   exp239_n2_tenergy_ov4_lr1e3_lsh0.1    0.001  0.1"
    "n2_temporal_energy_overlap4   exp240_n2_tenergy_ov4_lr5e4_lsh0.1    0.0005 0.1"
)

TOTAL=${#ALL_EXPS[@]}

# ============================================================
# Phase 1: test-only (DEACTIVATED — exp187-190 retests skipped)
# ============================================================
# TEST_ONLY_EXPS=(
#     "n2_6ch_input    exp187_n2_6ch_lr1e3_lsh0.1"    # DEACTIVATED
#     "n2_6ch_input    exp188_n2_6ch_lr5e4_lsh0.1"    # DEACTIVATED
#     "n2_6ch_input    exp189_n2_6ch_lr1e3_lsh0.3"    # DEACTIVATED
#     "n2_temporal_rms exp190_n2_trms_lr1e3_lsh0.1"   # DEACTIVATED
# )
TEST_ONLY_EXPS=()

GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
NUM_WORKERS=4

if [ ${#TEST_ONLY_EXPS[@]} -gt 0 ]; then
    echo "============================================================"
    echo "n2_bulk.sh — Phase 1: retest ${#TEST_ONLY_EXPS[@]} done experiments"
    echo "$(date)"
    echo "============================================================"

    TEST_PIDS=()
    for i in "${!TEST_ONLY_EXPS[@]}"; do
        SPEC="${TEST_ONLY_EXPS[$i]}"
        CONFIG=$(echo "$SPEC" | awk '{print $1}')
        EXP=$(echo    "$SPEC" | awk '{print $2}')
        GPU="${GPU_PAIRS[$((i % NUM_WORKERS))]}"
        test_only "$GPU" "$CONFIG" "$EXP" &
        TEST_PIDS+=($!)
    done
    for pid in "${TEST_PIDS[@]}"; do wait "$pid" || true; done
    echo "Phase 1 retests finished — $(date)"
    echo ""
fi

# ============================================================
# Phase 2: train + test for remaining experiments
# ============================================================
echo "============================================================"
echo "n2_bulk.sh — Phase 2: $TOTAL train+test experiments, 4 workers (GPUs 0-7)"
echo "$(date)"
echo "============================================================"

# ============================================================
# Distribute across 4 GPU-pair workers (round-robin)
# Worker 0: GPU 0,1   Worker 1: GPU 2,3
# Worker 2: GPU 4,5   Worker 3: GPU 6,7
# ============================================================

run_worker() {
    local WORKER_ID=$1
    local GPU="${GPU_PAIRS[$WORKER_ID]}"

    for (( i=WORKER_ID; i<TOTAL; i+=NUM_WORKERS )); do
        local SPEC="${ALL_EXPS[$i]}"
        local CONFIG=$(echo "$SPEC" | awk '{print $1}')
        local EXP=$(echo    "$SPEC" | awk '{print $2}')
        local LR=$(echo     "$SPEC" | awk '{print $3}')
        local LSH=$(echo    "$SPEC" | awk '{print $4}')

        train_and_test "$GPU" "$CONFIG" "$EXP" "$LR" "$LSH"
    done
}

PIDS=()
for w in 0 1 2 3; do
    run_worker $w &
    PIDS+=($!)
    echo "Worker $w launched (GPU ${GPU_PAIRS[$w]}, PID ${PIDS[-1]})"
done

echo "All 4 workers running. Waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "n2_bulk.sh finished — $(date)"
echo "Workers failed: $FAIL / $NUM_WORKERS"
echo "============================================================"
echo ""
printf "%-50s %8s %8s %8s\n" "Experiment" "ABS_REL" "RMSE" "Delta1"
echo "--------------------------------------------------------------------------"

SUCCESS=0
FAILURES=0
MISSING=0

# Combine retested + freshly trained experiments in the summary
ALL_SUMMARY_EXPS=("${TEST_ONLY_EXPS[@]}" "${ALL_EXPS[@]}")

for SPEC in "${ALL_SUMMARY_EXPS[@]}"; do
    EXP=$(echo "$SPEC" | awk '{print $2}')
    LOG="$TEST_LOG_DIR/${EXP}_test.log"
    if [ -f "$LOG" ]; then
        ABS=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$LOG" | head -1)
        RMSE=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$LOG" | head -1)
        D1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$LOG" | head -1)
        if [ -n "$ABS" ]; then
            printf "%-50s %8s %8s %8s\n" "$EXP" "$ABS" "$RMSE" "$D1"
            SUCCESS=$((SUCCESS + 1))
        else
            printf "%-50s %8s\n" "$EXP" "FAILED"
            FAILURES=$((FAILURES + 1))
        fi
    else
        printf "%-50s %8s\n" "$EXP" "NO_RESULT"
        MISSING=$((MISSING + 1))
    fi
done

echo "--------------------------------------------------------------------------"
echo "Success: $SUCCESS  Failed: $FAILURES  No result: $MISSING  Total: ${#ALL_SUMMARY_EXPS[@]}"
echo "============================================================"

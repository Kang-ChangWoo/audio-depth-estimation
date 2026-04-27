#!/bin/bash
# ============================================================
# n3_bulk.sh — N3 0419 experiment pipeline (report_d: exp207-220)
#
# Queues the "breakthrough" wave from docs/report_d_0419_breakthough.md,
# aimed at pushing deployable ABS_REL and RMSE below the N3 0417 frontier
# (exp172 RMSE=1.0744, exp171 ABS_REL=0.4218).
#
# Groups:
#   H (exp207-211) — re-tune exp172 (training-side only, no new arch)
#   I (exp212-214) — hybrid architectures (n3_0419 models)
#   J (exp216-220) — ViT backbone ports  [NOT IMPLEMENTED YET — commented out;
#                                         requires new models/pretrain/ code]
#
# Prior runs (exp166-186) are logged in logs/summary_train/ and
# logs/summary_test/; entries in this file are fresh.
#
# SPEC format (space-separated, fixed first 4 fields, optional EXTRA):
#   "CONFIG   EXP   LR   LSH   [extra train.py flags]"
#
# Runtime: 10 GPUs total, 5 workers (2 GPUs each), in parallel.
# Each worker: train → test → train → test → ... sequentially.
# ============================================================
set -euo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

if [ -z "${CONDA_SHLVL:-}" ]; then
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
fi
conda activate shared_audio 2>/dev/null || conda activate base 2>/dev/null || true

WORKERS=2              # dataloader workers per train.py (5 procs × 2 = 10 loaders total)
OMP_THREADS=4          # cap OpenMP/MKL BLAS thread pool per proc so 5 procs
                       # don't spawn 5 × N_CPU threads and saturate the box
EPOCHS=60
BATCH_SIZE=32          # Group H diagnoses whether BS=128 (n3_0417) was over-smoothing
VIS_PER_SCENE=100

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n3_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n3_test"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"

# ---- Train one experiment, then test it immediately ----
train_and_test() {
    local GPU="$1"
    local CONFIG="$2"
    local EXP="$3"
    local LR="$4"
    local LSH="$5"
    shift 5
    local EXTRA=("$@")   # optional extra flags passed verbatim to train.py

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] TRAIN $EXP (config=$CONFIG lr=$LR lsh=$LSH extra='${EXTRA[*]}')"

    CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$OMP_THREADS" \
    OPENBLAS_NUM_THREADS="$OMP_THREADS" NUMEXPR_NUM_THREADS="$OMP_THREADS" \
    python3 train.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$WORKERS" \
        --lambda-sh "$LSH" \
        --rotate-canonical \
        "${EXTRA[@]}" \
        > "$TRAIN_LOG_DIR/${EXP}.log" 2>&1 || true

    local TRAIN_EC=$?
    if [ "$TRAIN_EC" -ne 0 ] && [ "$TRAIN_EC" -ne 141 ]; then
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
        OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
        python3 -u test.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --checkpoint-path "$CKPT_PATH" \
        --eval-on test \
        --batch-size 4 \
        --vis-per-scene "$VIS_PER_SCENE" \
        --rotate-canonical \
        "${EXTRA[@]}" \
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
# Experiment definitions — report_d proposals (exp207-220)
# Format: "CONFIG  EXP  LR  LSH  [extra flags to train.py]"
# Baseline to beat: exp172 RMSE=1.0744, exp171 ABS_REL=0.4218.
# ============================================================
ALL_EXPS=(
    # ============================================================
    # ACTIVE WAVE: Group K (HP sweep around exp171/exp213) + exp215 distill.
    # Baselines to beat (deployable):
    #   exp172  n3_energy_attn           RMSE=1.0744  ABS=0.4792
    #   exp171  n3_multiscale_sh lsh=0.3 RMSE=1.1104  ABS=0.4218  ← best ABS
    #   exp213  n3_mssh_energy_attn      RMSE=1.0777  ABS=0.4645  ← best new hybrid
    # ============================================================
    # --- Group K-A: tune λ_sh and sh_dim on n3_multiscale_sh (exp171 family) ---
    # "n3_multiscale_sh    exp221_n3mssh_lr1e3_lsh0.5               0.001  0.5"                                  # DONE ABS=0.4351 RMSE=1.1052 D1=0.4941
    # "n3_multiscale_sh    exp222_n3mssh_lr1e3_lsh0.7               0.001  0.7"                                  # DONE ABS=0.4546 RMSE=1.0941 D1=0.5022
    # "n3_multiscale_sh    exp223_n3mssh_lr1e3_lsh0.2               0.001  0.2"                                  # DONE ABS=0.4171 RMSE=1.1161 D1=0.4904  ← best ABS
    # "n3_multiscale_sh    exp224_n3mssh_sh9_lr1e3_lsh0.3           0.001  0.3  --sh-dim 9"                      # TRAIN_DONE, TEST failed (sh_dim mismatch) — rerun test manually after test.py --sh-dim fix
    # --- Group K-B: training-side tweaks on exp171 ---
    # "n3_multiscale_sh    exp225_n3mssh_freeze10_lr1e3_lsh0.3      0.001  0.3  --foa-freeze-epochs 10"          # DONE ABS=0.4821 RMSE=1.0811 D1=0.4958
    # "n3_multiscale_sh    exp226_n3mssh_dw2_lr1e3_lsh0.3           0.001  0.3  --depth-weight 2.0"              # DONE ABS=0.4396 RMSE=1.0845 D1=0.4947
    # --- Group K-C: tune exp213 (n3_mssh_energy_attn hybrid) ---
    # "n3_mssh_energy_attn exp227_n3mssh_eattn_lr1e3_lsh0.3         0.001  0.3"                                  # DONE ABS=0.4331 RMSE=1.0895 D1=0.5008
    # "n3_mssh_energy_attn exp228_n3mssh_eattn_lenergy_lr1e3_lsh0.1 0.001  0.1  --lambda-energy 0.3"             # DONE ABS=0.4332 RMSE=1.0898 D1=0.4964
    # "n3_mssh_energy_attn exp229_n3mssh_eattn_sh9_lr1e3_lsh0.3     0.001  0.3  --sh-dim 9"                      # DONE ABS=0.4394 RMSE=1.1013 D1=0.4940
    # --- Group K-D: oracle distillation (exp215) ---
    # "n3_energy_attn_distill exp215_n3eattn_distill_lkd0.5         0.001  0.1  --teacher-ckpt checkpoints/unet_256_soundspaces_BS32_Lr0.001_AdamW_exp180_oracle_nc3_lr1e3_lsh0.3/best_model.pth --lambda-kd 0.5"  # DONE ABS=0.4288 RMSE=1.0853 D1=0.4943

    # ============================================================
    # Group H + I: DONE (exp207-214, train 60/60, tested). None beat the
    # exp172 frontier (RMSE=1.0744, ABS_REL=0.4792). Best of the new wave:
    # exp213 (MSSH+EnergyAttn) at RMSE=1.0777. Kept here as a template —
    # uncomment to rerun.
    # ============================================================
    #
    # --- Group H: re-tune exp172 (training-side only) — DONE ---
    # "n3_energy_attn      exp207_n3eattn_bs32_lr1e3_lsh0.1        0.001  0.1"                                  # DONE RMSE=1.1000 ABS=0.4350
    # "n3_energy_attn      exp208_n3eattn_dw2_lr1e3_lsh0.1         0.001  0.1  --depth-weight 2.0"              # DONE RMSE=1.0813 ABS=0.5003
    # "n3_energy_attn      exp209_n3eattn_freeze15_lr1e3_lsh0.1    0.001  0.1  --foa-freeze-epochs 15"          # DONE RMSE=1.0931 ABS=0.4410
    # "n3_energy_attn      exp210_n3eattn_lenergy0.3_lr1e3_lsh0.1  0.001  0.1  --lambda-energy 0.3"             # DONE RMSE=1.0913 ABS=0.4763
    # "n3_energy_attn      exp211_n3eattn_dw2_lenergy0.3_lr1e3     0.001  0.1  --depth-weight 2.0 --lambda-energy 0.3"  # DONE RMSE=1.1036 ABS=0.4300
    #
    # --- Group I: hybrid architectures (models/n3_0419/) — DONE ---
    # "n3_film_energy_attn exp212_n3film_eattn_lr1e3_lsh0.1        0.001  0.1"                                  # DONE RMSE=1.1025 ABS=0.4307
    # "n3_mssh_energy_attn exp213_n3mssh_eattn_lr1e3_lsh0.1        0.001  0.1"                                  # DONE RMSE=1.0777 ABS=0.4645  ← best of new wave
    # "n3_energy_attn_sh9  exp214_n3eattn_sh9_lr1e3_lsh0.1         0.001  0.1  --sh-dim 9"                      # DONE RMSE=1.0870 ABS=0.4373

    # ============================================================
    # Group L — ViT backbone ports of the n3_0419 wins (exp216-220, 230).
    # Implementation landed 2026-04-20: models/pretrain/pretrained_vit_foa_v6.py
    # + six YAMLs (config/pvitfoa_v3_*.yaml) + routing in utils/train_utils.py
    # (_PVITFOA_AUX_SH_NAMES, _PVITFOA_CLASSES, _PVITFOA_ORACLE_CLASSES).
    #
    # Classes:
    #   pretrained_vit_foa_v6_eattn      — ViT + energy-map head (exp216)
    #   pretrained_vit_foa_v6_mssh       — ViT + multi-scale SH    (exp217)
    #   pretrained_vit_foa_v3            — existing v3 FiLM         (exp218, 220, 230)
    #   pretrained_vit_foa_v6_oracle_nc3 — ViT oracle input_nc=3    (exp219)
    #
    # Note: ViT ports use lr=1e-4 (per pretrain_vit_foa_v3.yaml baseline).
    # The last 4 (exp220) argument deviates from the commented proposal:
    # ViT has no gradual foa_freeze schedule (that hook is UNet-only), so
    # exp220 permanently freezes the encoder via freeze_encoder=true in
    # the YAML — no --foa-freeze-epochs flag needed.
    # ============================================================
    # "pvitfoa_v3_eattn      exp216_pvit_eattn_lr1e4_lsh0.1           0.0001 0.1"                                    # DONE ABS=0.4639 RMSE=1.0863 D1=0.4974
    # "pvitfoa_v3_mssh       exp217_pvit_mssh_lr1e4_lsh0.1            0.0001 0.1"                                    # DONE ABS=0.4656 RMSE=1.1035 D1=0.4860
    # "pvitfoa_v3_film_dw2   exp218_pvit_film_dw2_lr1e4_lsh0.1        0.0001 0.1  --depth-weight 2.0"                # DONE ABS=0.4504 RMSE=1.0805 D1=0.4867
    # "pvitfoa_v3_oracle_nc3 exp219_pvit_oracle_nc3_lr1e4_lsh0.1      0.0001 0.1"                                    # DONE ABS=0.4400 RMSE=1.0612 D1=0.4947  ← best ViT RMSE
    # "pvitfoa_v3_freeze20   exp220_pvit_freeze20_lr1e4_lsh0.1        0.0001 0.1"                                    # DONE 60/60
    # "pvitfoa_v3_distill    exp230_pvit_distill_lkd0.5               0.0001 0.1  --teacher-ckpt checkpoints/unet_256_soundspaces_BS32_Lr0.001_AdamW_exp180_oracle_nc3_lr1e3_lsh0.3/best_model.pth --lambda-kd 0.5"  # 55/60 — deactivated

    # ============================================================
    # exp247-256: Energy-map-only depth estimation (no binaural)
    #
    # Tests: can the FOA energy map alone predict depth?
    # 3 channel adaptor strategies: repeat (1→3), learned conv (1→3),
    # edge-augmented (energy, grad_x, grad_y → 3ch).
    # Models in models/n3_0419/n3_energy_only.py
    # ============================================================
    # --- [1,2,3] UNet + energy-map only (3 adaptor types) ---
    "n3_emap_unet_repeat    exp247_emap_unet_repeat_lr1e3       0.001  0.1"
    "n3_emap_unet_conv      exp248_emap_unet_conv_lr1e3         0.001  0.1"
    "n3_emap_unet_edge      exp249_emap_unet_edge_lr1e3         0.001  0.1"
    # --- [4,5,6] ViT + energy-map only (3 adaptor types) ---
    "n3_emap_vit_repeat     exp250_emap_vit_repeat_lr1e4        0.0001 0.1"
    "n3_emap_vit_conv       exp251_emap_vit_conv_lr1e4          0.0001 0.1"
    "n3_emap_vit_edge       exp252_emap_vit_edge_lr1e4          0.0001 0.1"
    # --- [7,8] Temporal energy (3-bin disjoint) + UNet/ViT ---
    "n3_emap_unet_temporal  exp253_emap_unet_temporal_lr1e3     0.001  0.1"
    "n3_emap_vit_temporal   exp254_emap_vit_temporal_lr1e4      0.0001 0.1"
    # --- [9,10] Temporal energy (3-bin overlap) + UNet/ViT ---
    "n3_emap_unet_temporal_ov exp255_emap_unet_tov_lr1e3        0.001  0.1"
    "n3_emap_vit_temporal_ov  exp256_emap_vit_tov_lr1e4          0.0001 0.1"
)

TOTAL=${#ALL_EXPS[@]}

echo "============================================================"
echo "n3_bulk.sh — $TOTAL experiments, 3 workers (GPUs 4-9), BS=$BATCH_SIZE"
echo "$(date)"
echo "============================================================"

# ============================================================
# 3 GPU-pair workers (round-robin). GPUs 0-3 left for other bulks
# (n1_bulk.sh uses 0,1/3,4/5,6 — when both run concurrently,
# assign n1 to the freed GPUs before launching this one).
#   Worker 0: GPU 4,5   Worker 1: GPU 6,7   Worker 2: GPU 8,9
# ============================================================

GPU_PAIRS=("0,1" "4,5" "6,7" "8,9")
NUM_WORKERS=4

run_worker() {
    local WORKER_ID=$1
    local GPU="${GPU_PAIRS[$WORKER_ID]}"

    for (( i=WORKER_ID; i<TOTAL; i+=NUM_WORKERS )); do
        local SPEC="${ALL_EXPS[$i]}"
        # Split spec into fields; fields 5+ are passed verbatim as extra flags.
        read -r -a FIELDS <<< "$SPEC"
        local CONFIG="${FIELDS[0]}"
        local EXP="${FIELDS[1]}"
        local LR="${FIELDS[2]}"
        local LSH="${FIELDS[3]}"
        local EXTRA=("${FIELDS[@]:4}")

        train_and_test "$GPU" "$CONFIG" "$EXP" "$LR" "$LSH" "${EXTRA[@]}"
    done
}

PIDS=()
for w in 0 1 2 3; do
    run_worker $w &
    PIDS+=($!)
    echo "Worker $w launched (GPU ${GPU_PAIRS[$w]}, PID ${PIDS[-1]})"
done

echo "All 3 workers running. Waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "n3_bulk.sh finished — $(date)"
echo "Workers failed: $FAIL / $NUM_WORKERS"
echo "============================================================"
echo ""
printf "%-50s %8s %8s %8s\n" "Experiment" "ABS_REL" "RMSE" "Delta1"
echo "---------------------------------------------------------------------"

SUCCESS=0
FAILURES=0
MISSING=0

for SPEC in "${ALL_EXPS[@]}"; do
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

echo "---------------------------------------------------------------------"
echo "Success: $SUCCESS  Failed: $FAILURES  No result: $MISSING  Total: $TOTAL"
echo "============================================================"

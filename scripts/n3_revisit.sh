#!/bin/bash
# ============================================================
# n3_revisit.sh — RETRAIN the legacy experiments on RADIAL depth GT,
# then test each fresh checkpoint. Matches n3_bulk.sh rhythm:
#
#     worker: train(exp350) → test(exp350) → train(exp351) → test(exp351) → ...
#
# Enablers (already landed):
#   data/dataset.py     : cfg.dataset.depth_dir override
#   train.py, test.py   : --depth-dir CLI flag
#
# Ranges (original exp_id → new exp_id, starting at EXP_ID_BASE=350):
#     01–05   baseline (UNet)             → exp350–354
#     06–10   vit (scratch)               → exp355–359
#     11–15   echodiffusion (no wav2vec)  → exp360–364
#     56–60   pretrain_resnet             → exp370–374
#     61–65   pretrain_vit                → exp375–379
#     66–70   echonet                     → exp380–384
#     71–75   batvision                   → exp385–389
#   121–125   echodiffusion + wav2vec     → exp390–394
#
# Each experiment's (batch_size, learning_rate) are parsed from the
# ORIGINAL checkpoint directory name so the retrain matches the
# original hyperparameters exactly.
#
# Log layout:
#     logs/n3_revisit_train/<NEW_TAG>.log          training
#     logs/n3_revisit_test/<NEW_TAG>_test.log      post-train test
#     logs/n3_revisit/_summary.tsv                 rollup
#
# Usage:
#     bash scripts/n3_revisit.sh                    # 40 ep, 5 workers
#     EPOCHS=20 bash scripts/n3_revisit.sh
#     GPU_PAIRS="0,1 2,3" bash scripts/n3_revisit.sh
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

# --- Interpreter preflight -------------------------------------------
PYTHON="${PYTHON:-/opt/conda/bin/python3}"
_REQUIRED="torch torchvision torchaudio timm scipy numpy yaml einops"
_MISSING=$("$PYTHON" - <<EOF 2>&1
import importlib
miss=[]
for m in "$_REQUIRED".split():
    try: importlib.import_module(m)
    except Exception as e: miss.append(f"{m}:{e.__class__.__name__}")
print(" ".join(miss) if miss else "OK")
EOF
)
if [ "$_MISSING" != "OK" ]; then
    echo "ERROR: $PYTHON missing: $_MISSING" >&2
    exit 1
fi
echo "Using PYTHON=$PYTHON"

# --- Knobs ------------------------------------------------------------
DEPTH_DIR="erp_depth_radial"
EPOCHS="${EPOCHS:-40}"
NUM_DL_WORKERS="${NUM_DL_WORKERS:-4}"
EXP_ID_BASE="${EXP_ID_BASE:-350}"

TRAIN_LOG_DIR="$PROJECT_DIR/logs/n3_revisit_train"
TEST_LOG_DIR="$PROJECT_DIR/logs/n3_revisit_test"
SUMMARY_DIR="$PROJECT_DIR/logs/n3_revisit"
mkdir -p "$TRAIN_LOG_DIR" "$TEST_LOG_DIR" "$SUMMARY_DIR"

# GPU-pair fan-out. Each worker gets 2 GPUs.
GPU_PAIRS_STR="${GPU_PAIRS:-0,1 2,3 4,5 6,7 8,9}"
read -r -a GPU_PAIRS <<< "$GPU_PAIRS_STR"
NUM_WORKERS="${NUM_WORKERS:-${#GPU_PAIRS[@]}}"
echo "GPU workers (${NUM_WORKERS} × 2 GPU): ${GPU_PAIRS[*]}"

# --- Hard GPU preflight ---------------------------------------------
_FIRST_PAIR="${GPU_PAIRS[0]}"
_GPU_OK=$(CUDA_VISIBLE_DEVICES="$_FIRST_PAIR" "$PYTHON" - <<'PYEOF' 2>&1
try:
    import torch
    print(f"OK {torch.cuda.device_count()}" if torch.cuda.is_available() else "NO_CUDA")
except Exception as e:
    print(f"ERR {e}")
PYEOF
)
if [[ "$_GPU_OK" != OK\ * ]]; then
    echo "ERROR: CUDA not usable on GPU pair $_FIRST_PAIR ($_GPU_OK)" >&2
    exit 2
fi
echo "CUDA preflight: $_GPU_OK on first pair ($_FIRST_PAIR)"

# ============================================================
# SPEC TABLE — (config, regex, excl-substr-or-"-")
# Each matched checkpoint dir is a template: we parse (BS, Lr) from
# its name and retrain with those values + --depth-dir radial +
# --experiment-name exp<NEW_ID>_...
# ============================================================
SPECS=(
    "baseline         exp0[1-5]_baseline_          -"
    "vit              exp(0[6-9]|10)_vit_          -"
    "echodiffusion    exp1[1-5]_echodiff_          wav2vec"
    "foa_v2           exp(5[6-9]|60)_foav2_        -"
    "pretrain_resnet  exp(5[6-9]|60)_resnet_       -"
    "pretrain_vit     exp6[1-5]_vit_               -"
    "echonet          exp(6[6-9]|70)_echonet_      -"
    "batvision        exp7[1-5]_batvision_         -"
    "echodiffusion    exp12[1-5]_echodiff_wav2vec_ -"
)

# ---- Resolve: for each matched OLD checkpoint dir, emit a training task.
declare -a TASK_TAGS TASK_CONFIGS TASK_LR TASK_BS TASK_OLD
_EXP_ID=$EXP_ID_BASE
for spec in "${SPECS[@]}"; do
    read -r CONFIG PATTERN EXCL <<< "$spec"
    while IFS= read -r dir; do
        [ -z "$dir" ] && continue
        BASE=$(basename "$dir")
        if [ "$EXCL" != "-" ] && [[ "$BASE" == *"$EXCL"* ]]; then continue; fi
        [ ! -f "$dir/best_model.pth" ] && continue

        # Parse BS and Lr from the checkpoint dir:
        #   <generator>_<dataset>_BS<bs>_Lr<lr>_AdamW_exp<NN>_<rest>
        OLD_BS=$(echo "$BASE" | sed -nE 's/.*_BS([0-9]+)_Lr.*/\1/p')
        OLD_LR=$(echo "$BASE" | sed -nE 's/.*_BS[0-9]+_Lr([^_]+)_AdamW_.*/\1/p')
        OLD_TAG=$(echo "$BASE" | sed -E 's/^.+_BS[0-9]+_Lr[^_]+_AdamW_//')
        REST=$(echo "$OLD_TAG" | sed -E 's/^exp[0-9]+_//')
        NEW_TAG="exp${_EXP_ID}_${REST}"

        TASK_TAGS+=("$NEW_TAG")
        TASK_CONFIGS+=("$CONFIG")
        TASK_LR+=("$OLD_LR")
        TASK_BS+=("$OLD_BS")
        TASK_OLD+=("$OLD_TAG")
        _EXP_ID=$((_EXP_ID + 1))
    done < <(find "$PROJECT_DIR/checkpoints" -maxdepth 1 -type d \
                  -regextype posix-extended \
                  -regex ".*${PATTERN}.*" 2>/dev/null | sort)
done

TOTAL=${#TASK_TAGS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: no experiments resolved" >&2; exit 3
fi

echo "============================================================"
echo "n3_revisit — RETRAIN $TOTAL experiments on radial depth GT"
echo "            EPOCHS=$EPOCHS  dl_workers=$NUM_DL_WORKERS"
echo "            depth_dir=$DEPTH_DIR"
echo "            fan-out across $NUM_WORKERS GPU pairs"
echo "            logs: $TRAIN_LOG_DIR  +  $TEST_LOG_DIR"
echo "============================================================"

# ---- train_and_test: one full cycle per experiment --------------
train_and_test() {
    local WORKER_ID=$1 TAG=$2 CONFIG=$3 LR=$4 BS=$5 OLD=$6
    local GPU="${GPU_PAIRS[$WORKER_ID]}"
    local tlog="$TRAIN_LOG_DIR/${TAG}.log"
    local xlog="$TEST_LOG_DIR/${TAG}_test.log"

    echo "[$(date +%H:%M:%S)] [w$WORKER_ID/GPU=$GPU] TRAIN $TAG  (cfg=$CONFIG lr=$LR bs=$BS, was $OLD)"

    local ec=0
    CUDA_VISIBLE_DEVICES="$GPU" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
        "$PYTHON" train.py \
        --config "$CONFIG" \
        --experiment-name "$TAG" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BS" \
        --num-workers "$NUM_DL_WORKERS" \
        --rotate-canonical \
        --depth-dir "$DEPTH_DIR" \
        > "$tlog" 2>&1 || ec=$?

    if [ "$ec" -ne 0 ] && [ "$ec" -ne 141 ]; then
        echo "  [w$WORKER_ID] $TAG  TRAIN_FAILED (exit=$ec)"
        tail -3 "$tlog" | sed "s/^/      /"
        return
    fi
    local DONE_LINE=$(grep "^Done\." "$tlog" | tail -1)
    echo "  [w$WORKER_ID] $TAG  TRAIN_DONE — $DONE_LINE"

    local CKPT
    CKPT=$(find "$PROJECT_DIR/checkpoints" -name "best_model.pth" -path "*${TAG}*" 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "  [w$WORKER_ID] $TAG  TEST_SKIP (no best_model.pth)"
        return
    fi

    echo "[$(date +%H:%M:%S)] [w$WORKER_ID/GPU=$GPU] TEST  $TAG"
    ec=0
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        "$PYTHON" -u test.py \
        --config "$CONFIG" \
        --experiment-name "$TAG" \
        --checkpoint-path "$CKPT" \
        --eval-on test \
        --batch-size 16 \
        --num-workers "$NUM_DL_WORKERS" \
        --vis-per-scene 0 \
        --rotate-canonical \
        --depth-dir "$DEPTH_DIR" \
        > "$xlog" 2>&1 || ec=$?

    if [ "$ec" -ne 0 ]; then
        echo "  [w$WORKER_ID] $TAG  TEST_FAILED (exit=$ec)"
        tail -3 "$xlog" | sed "s/^/      /"
    else
        local abs rmse d1
        abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$xlog" | head -1)
        rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$xlog" | head -1)
        d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$xlog" | head -1)
        echo "  [w$WORKER_ID] $TAG  TEST_OK  ABS=$abs RMSE=$rmse D1=$d1"
    fi
}

run_worker() {
    local WORKER_ID=$1
    for (( i=WORKER_ID; i<TOTAL; i+=NUM_WORKERS )); do
        train_and_test "$WORKER_ID" \
            "${TASK_TAGS[$i]}" "${TASK_CONFIGS[$i]}" \
            "${TASK_LR[$i]}"   "${TASK_BS[$i]}"   "${TASK_OLD[$i]}"
    done
}

# ---- Launch ---------------------------------------------------
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

# ---- Post-hoc summary rollup -------------------------------------
SUMMARY="$SUMMARY_DIR/_summary.tsv"
printf "new_exp\told_exp\tconfig\tlr\tbs\tABS_REL\tRMSE\tDelta1\tMAE\tFOA_L1\n" > "$SUMMARY"
SUCCESS=0
for i in $(seq 0 $((TOTAL - 1))); do
    tag="${TASK_TAGS[$i]}"
    old="${TASK_OLD[$i]}"
    cfg="${TASK_CONFIGS[$i]}"
    lr="${TASK_LR[$i]}"
    bs="${TASK_BS[$i]}"
    xlog="$TEST_LOG_DIR/${tag}_test.log"
    abs="-"; rmse="-"; d1="-"; mae="-"; foa="-"
    if [ -f "$xlog" ]; then
        abs=$(grep -oP 'ABS_REL:\s*\K[0-9.]+' "$xlog" | head -1); abs="${abs:--}"
        rmse=$(grep -oP '^\s*RMSE:\s*\K[0-9.]+' "$xlog" | head -1); rmse="${rmse:--}"
        d1=$(grep -oP 'Delta1:\s*\K[0-9.]+' "$xlog" | head -1); d1="${d1:--}"
        mae=$(grep -oP '\sMAE:\s*\K[0-9.]+' "$xlog" | head -1); mae="${mae:--}"
        foa=$(grep -oP 'FOA_L1:\s*\K[0-9.]+' "$xlog" | head -1); foa="${foa:--}"
        [ "$abs" != "-" ] && SUCCESS=$((SUCCESS + 1))
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$tag" "$old" "$cfg" "$lr" "$bs" "$abs" "$rmse" "$d1" "$mae" "$foa" >> "$SUMMARY"
done

echo ""
echo "============================================================"
echo "n3_revisit finished — success=$SUCCESS / $TOTAL   worker-failures=$FAIL"
echo "summary TSV: $SUMMARY"
echo "============================================================"
printf "\n%-40s %-28s %-14s %6s %6s %8s %8s %8s\n" \
    "new_exp" "old_exp" "config" "lr" "bs" "ABS_REL" "RMSE" "Delta1"
tail -n +2 "$SUMMARY" | sort -t$'\t' -k7,7g | \
    awk -F'\t' '{printf "%-40s %-28s %-14s %6s %6s %8s %8s %8s\n",
                         $1, $2, $3, $4, $5, $6, $7, $8}'

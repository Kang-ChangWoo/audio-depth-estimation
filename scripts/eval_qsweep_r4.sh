#!/bin/bash
# ============================================================
# eval_qsweep_r4.sh — Round 4 inference-only output-mode sweep.
#
# For a small set of high-information r3 checkpoints, run test.py with
# every available `--range-eval-mode` representative:
#     expectation, map, q25, q35, q45, q50, q55, q65, q75,
#     temp05, temp075, temp15
# and (optionally) for every saved best-tag (score/absrel/rmse/delta1).
#
# Output naming:
#     logs/n9_0427_test/qsweep_r4/{exp}_{tag}_{mode}.log
#
# Sample-level npzs land in `eval/<dataset>/<eval_on>/per_sample_<exp>.npz`
# but the npz filename is `{exp_name}` only — to keep separate per-mode
# npzs we re-run with --experiment-name suffixed by `__qsweep_{tag}_{mode}`.
# That preserves backward compat for the legacy stats save and lets the
# downstream paired_bootstrap.py work directly.
#
# Usage:
#   bash scripts/eval_qsweep_r4.sh                  # default: built-in list
#   EXPS="exp946,exp939" bash scripts/eval_qsweep_r4.sh
#   GPUS="0 1" bash scripts/eval_qsweep_r4.sh        # parallel test, 2 GPUs
#   TAGS="score absrel"  bash scripts/eval_qsweep_r4.sh    # only some tags
# ============================================================
set -eo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
DEPTH_DIR="erp_depth_radial"
DATASET_DIR="${DATASET_DIR:-/root/local1/changwoo/matterport3d_0303_renew}"
WORKERS="${WORKERS:-2}"

if [ ! -d "$DATASET_DIR" ]; then
    # n2 default path (no underscore)
    DATASET_DIR="/root/local1/changwoo/matterport3d_0303renew"
fi
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: DATASET_DIR not found, tried both 0303renew and 0303_renew" >&2
    exit 3
fi

# Output log directory
TEST_LOG_DIR="$PROJECT_DIR/logs/n9_0427_test/qsweep_r4"
mkdir -p "$TEST_LOG_DIR"

# Modes to sweep
MODES_DEFAULT="expectation map q25 q35 q45 q50 q55 q65 q75 temp05 temp075 temp15"
MODES="${MODES:-$MODES_DEFAULT}"
TAGS_DEFAULT="score absrel rmse delta1"
TAGS="${TAGS:-$TAGS_DEFAULT}"

# Round-5 R_BASE — the set of model-construction flags that EVERY range
# cell shares (must match training-time so the checkpoint loads). Cells
# that deviate (e.g. cylindrical exp954/955) need their own row below.
R_BASE_RADIAL="--depth-head-type range --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14 --range-output-mode expectation --erp-cos-lat-weight --erp-far-mask"

R_BASE_CYL="--depth-head-type range --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14 --range-output-mode expectation --erp-cos-lat-weight --erp-far-mask --range-bin-axis horizontal --cyl-min-axis-factor 0.15"

# Cells to sweep. Format: "FULL_EXP_NAME  CONFIG  BASE_FLAGS_KEY"
# BASE_FLAGS_KEY ∈ {radial, cyl}.
ALL_TARGETS_DEFAULT=(
    # Round-2 anchors
    "exp907_bs32_ERPfix_distribution                          echorange  radial"
    # Round-3 r3 winners (R20)
    "exp946_R5B_sh_L2_l010_logd_bs48_r3                       echorange  radial"
    "exp939_R5A_sq_q050_t003_l025_bs48_r3                     echorange  radial"
    "exp953_R5C_combo_q055_sh002_bs48_r3                      echorange  radial"
    # Round-3 R40 paired (n9)
    "exp960_R5F_R40_expect_bs32_r3_ep40                       echorange  radial"
    "exp961_R5F_R40_median_bs32_r3_ep40                       echorange  radial"
    # Round-2 ERP ablation
    "exp908_bs32_cosLat_only                                  echorange  radial"
    "exp909_bs32_farMask_only                                 echorange  radial"
)

# Comma-separated EXPS env override: matches by prefix against ALL_TARGETS.
if [ -n "$EXPS" ]; then
    IFS=',' read -r -a wanted <<< "$EXPS"
    TARGETS=()
    for spec in "${ALL_TARGETS_DEFAULT[@]}"; do
        ename=$(echo "$spec" | awk '{print $1}')
        for w in "${wanted[@]}"; do
            if [[ "$ename" == "$w"* ]]; then
                TARGETS+=("$spec")
                break
            fi
        done
    done
else
    TARGETS=("${ALL_TARGETS_DEFAULT[@]}")
fi

# GPU pool — comma-list per worker, e.g. "0 1" → 2 single-GPU workers.
GPUS_STR="${GPUS:-0}"
read -r -a GPUS_ARR <<< "$GPUS_STR"
NUM_WORKERS="${#GPUS_ARR[@]}"

# Build a flat job list: (exp, config, base, tag, mode)
JOBS=()
for spec in "${TARGETS[@]}"; do
    read -r -a F <<< "$spec"
    EXP="${F[0]}"
    CONFIG="${F[1]}"
    BASE_KEY="${F[2]}"
    case "$BASE_KEY" in
        radial) BASE_FLAGS="$R_BASE_RADIAL" ;;
        cyl)    BASE_FLAGS="$R_BASE_CYL"   ;;
        *) echo "[skip] $EXP — unknown base key '$BASE_KEY'" >&2; continue ;;
    esac
    for TAG in $TAGS; do
        # ckpt path: best_model.pth for score, best_<tag>.pth otherwise.
        if [ "$TAG" = "score" ]; then
            CKPT_NAME="best_model.pth"
        else
            CKPT_NAME="best_${TAG}.pth"
        fi
        CKPT=$(find checkpoints/ -name "$CKPT_NAME" -path "*${EXP}*" 2>/dev/null | head -1)
        if [ -z "$CKPT" ]; then
            continue
        fi
        for MODE in $MODES; do
            JOBS+=("$EXP|$CONFIG|$BASE_FLAGS|$TAG|$MODE|$CKPT")
        done
    done
done

TOTAL=${#JOBS[@]}
echo "============================================================"
echo "eval_qsweep_r4 — $TOTAL jobs across ${#TARGETS[@]} targets"
echo "  TAGS=$TAGS"
echo "  MODES=$MODES"
echo "  GPUs: ${GPUS_ARR[*]}"
echo "============================================================"

run_one() {
    local GPU="$1" SPEC="$2"
    IFS='|' read -r EXP CONFIG BASE_FLAGS TAG MODE CKPT <<< "$SPEC"
    local TAG_EXP="${EXP}__qsweep_${TAG}_${MODE}"
    local LOG="$TEST_LOG_DIR/${EXP}_${TAG}_${MODE}.log"

    if [ -f "$LOG" ] && grep -q 'ABS_REL:' "$LOG"; then
        return
    fi

    echo "[$(date +%H:%M:%S)] [GPU=$GPU] $EXP tag=$TAG mode=$MODE"
    local ec=0
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        "$PYTHON" -u test.py \
        --config "$CONFIG" \
        --experiment-name "$TAG_EXP" \
        --checkpoint-path "$CKPT" \
        --eval-on test \
        --batch-size 4 \
        --num-workers "$WORKERS" \
        --vis-per-scene 0 \
        --depth-dir "$DEPTH_DIR" \
        --dataset-dir "$DATASET_DIR" \
        $BASE_FLAGS \
        --range-eval-mode "$MODE" \
        > "$LOG" 2>&1 || ec=$?
    if [ "$ec" -ne 0 ]; then
        echo "  [GPU=$GPU] $EXP $TAG $MODE FAILED (exit=$ec)"
        tail -3 "$LOG" | sed 's/^/      /'
    fi
}

run_worker() {
    local W="$1"
    local GPU="${GPUS_ARR[$W]}"
    for ((i=W; i<TOTAL; i+=NUM_WORKERS)); do
        run_one "$GPU" "${JOBS[$i]}"
    done
}

PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
    run_worker "$w" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done

echo ""
echo "============================================================"
echo "eval_qsweep_r4 done — $(date)"
echo "  output logs: $TEST_LOG_DIR"
echo "  per-sample npzs: eval/soundspaces/test/per_sample_*.npz"
echo "============================================================"

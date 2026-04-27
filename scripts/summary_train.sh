#!/bin/bash
# ============================================================
# summary_train.sh — Consolidated training script for ALL completed experiments
#
# This script contains every experiment that satisfies:
#   (1) Training log completed (has "Done." marker)
#   (2) best_model.pth exists in checkpoints/
#   (3) Had an execution script
#
# Excludes: JS-related experiments, incomplete/stalled runs
#
# Source scripts: bulk0407, bulk0408, bulk0410_train_41,
#   bulk0415_train_foa25, bulk0416_train_vit, train_CW
#
# Usage: GPUS="0,1" bash summary_train.sh [GROUP]
#   GROUP: all (default), baseline, vit, echodiff, foa, foa_variant,
#          foav2, resnet, previt, echonet, batvision, foa0415, pvitvoa, cw,
#          n3
# ============================================================
set -euo pipefail

PROJECT_DIR="/root/storage/implementation/shared_audio/baseline"
cd "$PROJECT_DIR"

GPUS="${GPUS:-0,1}"
GROUP="${1:-all}"
WORKERS="${WORKERS:-8}"

run_train() {
    local CONFIG="$1"
    local EXP="$2"
    local EPOCHS="$3"
    local LR="$4"
    local BS="$5"
    shift 5
    local EXTRA_ARGS="$@"

    echo "[$(date +%H:%M:%S)] Training $EXP (config=$CONFIG, lr=$LR, bs=$BS, epochs=$EPOCHS)"

    CUDA_VISIBLE_DEVICES="$GPUS" python3 train.py \
        --config "$CONFIG" \
        --experiment-name "$EXP" \
        --epochs "$EPOCHS" \
        --learning-rate "$LR" \
        --batch-size "$BS" \
        --num-workers "$WORKERS" \
        $EXTRA_ARGS \
        2>&1 | tee "logs/summary_train/${EXP}.log"
}

mkdir -p logs/summary_train

# ============================================================
# GROUP: baseline (5 experiments, 40 epochs)
# ============================================================
train_baseline() {
    run_train baseline exp01_baseline_lr1e3_bs32 40 0.001 32
    run_train baseline exp02_baseline_lr5e4_bs32 40 0.0005 32
    run_train baseline exp03_baseline_lr1e4_bs32 40 0.0001 32
    run_train baseline exp04_baseline_lr1e3_bs16 40 0.001 16
    run_train baseline exp05_baseline_lr5e4_bs16 40 0.0005 16
}

# ============================================================
# GROUP: vit — AudioDepthViT from scratch (5 experiments, 40 epochs)
# ============================================================
train_vit() {
    run_train vit exp06_vit_lr1e4_bs32 40 0.0001 32
    run_train vit exp07_vit_lr5e5_bs32 40 0.00005 32
    run_train vit exp08_vit_lr1e4_bs16 40 0.0001 16
    run_train vit exp09_vit_lr5e4_bs32 40 0.0005 32
    run_train vit exp10_vit_lr1e5_bs32 40 0.00001 32
}

# ============================================================
# GROUP: echodiff — EchoDiffusion + Wav2Vec (10 experiments, 40 epochs)
# ============================================================
train_echodiff() {
    run_train echodiffusion exp11_echodiff_lr1e4_bs32 40 0.0001 32
    run_train echodiffusion exp12_echodiff_lr5e5_bs32 40 0.00005 32
    run_train echodiffusion exp13_echodiff_lr1e4_bs16 40 0.0001 16
    run_train echodiffusion exp14_echodiff_lr5e4_bs32 40 0.0005 32
    run_train echodiffusion exp15_echodiff_lr1e5_bs32 40 0.00001 32
    run_train echodiffusion exp121_echodiff_wav2vec_lr1e4_bs16 40 0.0001 16
    run_train echodiffusion exp122_echodiff_wav2vec_lr5e4_bs16 40 0.0005 16
    run_train echodiffusion exp123_echodiff_wav2vec_lr1e4_bs32 40 0.0001 32
    run_train echodiffusion exp124_echodiff_wav2vec_lr5e5_bs16 40 0.00005 16
    run_train echodiffusion exp125_echodiff_wav2vec_lr1e4_bs8 40 0.0001 8
}

# ============================================================
# GROUP: foa_variant — CrossAttn/FeatBank/MSAttn/ChannelAttn (30 experiments)
#   First batch: no KL (exp16-35), Second batch: +KL (exp76-95)
# ============================================================
train_foa_variant() {
    # CrossAttn (no KL)
    for args in "1e-3 0.1 exp16" "5e-4 0.1 exp17" "1e-4 0.1 exp18" "1e-3 0.2 exp19" "5e-4 0.2 exp20"; do
        set -- $args; run_train foa_crossattn "${3}_crossattn_lr${1}_fw${2}" 40 "$1" 32
    done
    # FeatBank (no KL)
    for args in "1e-3 0.1 exp21" "5e-4 0.1 exp22" "1e-4 0.1 exp23" "1e-3 0.2 exp24" "5e-4 0.2 exp25"; do
        set -- $args; run_train foa_featbank "${3}_featbank_lr${1}_fw${2}" 40 "$1" 32
    done
    # MSAttn (no KL)
    for args in "1e-3 0.1 exp26" "5e-4 0.1 exp27" "1e-4 0.1 exp28" "1e-3 0.2 exp29" "5e-4 0.2 exp30"; do
        set -- $args; run_train foa_msattn "${3}_msattn_lr${1}_fw${2}" 40 "$1" 32
    done
    # ChannelAttn (no KL)
    for args in "1e-3 0.1 exp31" "5e-4 0.1 exp32" "1e-4 0.1 exp33" "1e-3 0.2 exp34" "5e-4 0.2 exp35"; do
        set -- $args; run_train foa_channelattn "${3}_channelattn_lr${1}_fw${2}" 40 "$1" 32
    done
    # CrossAttn +KL
    run_train foa_crossattn exp76_crossattn_lr1e3_fw0.1_kl0.02 40 0.001 32
    run_train foa_crossattn exp77_crossattn_lr5e4_fw0.2_kl0.005 40 0.0005 32
    run_train foa_crossattn exp79_crossattn_lr5e4_fw0.1_hw0.2_kl0.01 40 0.0005 32
    run_train foa_crossattn exp80_crossattn_lr1e3_fw0.3_kl0.01 40 0.001 32
    # FeatBank +KL
    run_train foa_featbank exp81_featbank_lr1e3_fw0.1_kl0.02 40 0.001 32
    run_train foa_featbank exp83_featbank_lr1e3_fw0.05_kl0.01 40 0.001 32
    run_train foa_featbank exp84_featbank_lr5e4_fw0.1_hw0.2_kl0.01 40 0.0005 32
    run_train foa_featbank exp85_featbank_lr1e3_fw0.3_kl0.01 40 0.001 32
    # MSAttn +KL
    run_train foa_msattn exp87_msattn_lr5e4_fw0.2_kl0.005 40 0.0005 32
    run_train foa_msattn exp88_msattn_lr1e3_fw0.05_kl0.01 40 0.001 32
    run_train foa_msattn exp89_msattn_lr5e4_fw0.1_hw0.2_kl0.01 40 0.0005 32
    # ChannelAttn +KL
    run_train foa_channelattn exp91_channelattn_lr1e3_fw0.1_kl0.02 40 0.001 32
    run_train foa_channelattn exp92_channelattn_lr5e4_fw0.2_kl0.005 40 0.0005 32
    run_train foa_channelattn exp93_channelattn_lr1e3_fw0.05_kl0.01 40 0.001 32
    run_train foa_channelattn exp95_channelattn_lr1e3_fw0.3_kl0.01 40 0.001 32
}

# ============================================================
# GROUP: foa — FOA Original (39 experiments, 40 epochs)
# ============================================================
train_foa() {
    run_train foa exp36_foa_lr1e3_dw1.0_fw0.1_hw0.1 40 0.001 32
    run_train foa exp37_foa_lr5e4_dw1.0_fw0.1_hw0.1 40 0.0005 32
    run_train foa exp38_foa_lr1e4_dw1.0_fw0.1_hw0.1 40 0.0001 32
    run_train foa exp39_foa_lr1e3_bs16_dw1.0_fw0.1_hw0.1 40 0.001 16
    run_train foa exp40_foa_lr5e4_bs16_dw1.0_fw0.1_hw0.1 40 0.0005 16
    run_train foa exp41_foa_lr1e3_dw1.0_fw0.2_hw0.1 40 0.001 32
    run_train foa exp42_foa_lr5e4_dw1.0_fw0.2_hw0.1 40 0.0005 32
    run_train foa exp43_foa_lr1e3_dw1.0_fw0.1_hw0.2 40 0.001 32
    run_train foa exp44_foa_lr5e4_dw1.0_fw0.1_hw0.2 40 0.0005 32
    run_train foa exp45_foa_lr1e3_dw1.0_fw0.2_hw0.2 40 0.001 32
    run_train foa exp46_foa_lr1e3_dw0.5_fw0.1_hw0.1 40 0.001 32
    run_train foa exp47_foa_lr1e3_dw2.0_fw0.1_hw0.1 40 0.001 32
    run_train foa exp48_foa_lr1e3_dw1.0_fw0.05_hw0.1 40 0.001 32
    run_train foa exp49_foa_lr1e3_dw1.0_fw0.1_hw0.05 40 0.001 32
    run_train foa exp50_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze5 40 0.001 32
    run_train foa exp51_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze10 40 0.001 32
    run_train foa exp52_foa_lr5e4_dw1.0_fw0.2_hw0.2 40 0.0005 32
    run_train foa exp53_foa_lr5e4_dw0.5_fw0.2_hw0.1 40 0.0005 32
    run_train foa exp54_foa_lr1e4_dw1.0_fw0.2_hw0.1 40 0.0001 32
    run_train foa exp55_foa_lr1e4_bs16_dw1.0_fw0.1_hw0.1 40 0.0001 16
    # Extended sweep (bulk0408)
    run_train foa exp96_foa_lr2e4_dw1.0_fw0.1_hw0.1 40 0.0002 32
    run_train foa exp97_foa_lr3e4_dw1.0_fw0.1_hw0.1 40 0.0003 32
    run_train foa exp99_foa_lr1e3_dw1.5_fw0.1_hw0.1 40 0.001 32
    run_train foa exp100_foa_lr1e3_fw0.15_hw0.1 40 0.001 32
    run_train foa exp101_foa_lr1e3_fw0.1_hw0.15 40 0.001 32
    run_train foa exp103_foa_lr5e4_fw0.15_hw0.1 40 0.0005 32
    run_train foa exp104_foa_lr5e4_fw0.1_hw0.15 40 0.0005 32
    run_train foa exp105_foa_lr1e3_fw0.3_hw0.1 40 0.001 32
    run_train foa exp107_foa_lr5e4_fw0.3_hw0.1 40 0.0005 32
    run_train foa exp108_foa_lr5e4_dw2.0_fw0.1_hw0.1 40 0.0005 32
    run_train foa exp109_foa_lr1e3_dw1.0_fw0.2_hw0.2_freeze5 40 0.001 32
    run_train foa exp111_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze15 40 0.001 32
    run_train foa exp112_foa_lr5e4_dw1.0_fw0.1_hw0.1_freeze10 40 0.0005 32
    run_train foa exp113_foa_lr1e3_dw1.0_fw0.05_hw0.05 40 0.001 32
    run_train foa exp115_foa_lr1e3_dw2.0_fw0.2_hw0.1 40 0.001 32
    run_train foa exp116_foa_lr3e4_fw0.2_hw0.1 40 0.0003 32
    run_train foa exp117_foa_lr2e4_dw1.0_fw0.2_hw0.2 40 0.0002 32
    run_train foa exp119_foa_lr1e3_fw0.1_hw0.2_freeze3 40 0.001 32
    run_train foa exp120_foa_lr5e4_fw0.05_hw0.1_freeze5 40 0.0005 32
}

# ============================================================
# GROUP: foav2 — FOA v2 (5 experiments, 40 epochs)
# ============================================================
train_foav2() {
    run_train foa_v2 exp56_foav2_lr1e3_dw1.0_fw0.1_hw0.1 40 0.001 32
    run_train foa_v2 exp57_foav2_lr5e4_dw1.0_fw0.1_hw0.1 40 0.0005 32
    run_train foa_v2 exp58_foav2_lr1e4_dw1.0_fw0.1_hw0.1 40 0.0001 32
    run_train foa_v2 exp59_foav2_lr1e3_dw1.0_fw0.2_hw0.1 40 0.001 32
    run_train foa_v2 exp60_foav2_lr5e4_dw1.0_fw0.2_hw0.2 40 0.0005 32
}

# ============================================================
# GROUP: resnet — Pretrained ResNet-50 (5 experiments, 40 epochs)
# ============================================================
train_resnet() {
    run_train pretrain_resnet exp56_resnet_lr1e4_bs32 40 0.0001 32
    run_train pretrain_resnet exp57_resnet_lr5e5_bs32 40 0.00005 32
    run_train pretrain_resnet exp58_resnet_lr5e4_bs32 40 0.0005 32
    run_train pretrain_resnet exp59_resnet_lr1e4_bs16 40 0.0001 16
    run_train pretrain_resnet exp60_resnet_lr3e4_bs32 40 0.0003 32
}

# ============================================================
# GROUP: previt — Pretrained ViT-B/16 (5 experiments, 40 epochs)
# ============================================================
train_previt() {
    run_train pretrain_vit exp61_vit_lr1e4_bs16 40 0.0001 16
    run_train pretrain_vit exp62_vit_lr5e5_bs16 40 0.00005 16
    run_train pretrain_vit exp63_vit_lr5e4_bs16 40 0.0005 16
    run_train pretrain_vit exp64_vit_lr1e4_bs8 40 0.0001 8
    run_train pretrain_vit exp65_vit_lr3e5_bs16 40 0.00003 16
}

# ============================================================
# GROUP: echonet (1 completed experiment, 40 epochs)
# ============================================================
train_echonet() {
    run_train echonet exp70_echonet_lr2e3_bs16 40 0.002 16
}

# ============================================================
# GROUP: batvision (5 experiments, 40 epochs)
# ============================================================
train_batvision() {
    run_train batvision exp71_batvision_lr1e3_bs32 40 0.001 32
    run_train batvision exp72_batvision_lr5e4_bs32 40 0.0005 32
    run_train batvision exp73_batvision_lr1e4_bs32 40 0.0001 32
    run_train batvision exp74_batvision_lr1e3_bs16 40 0.001 16
    run_train batvision exp75_batvision_lr2e3_bs32 40 0.002 32
}

# ============================================================
# GROUP: foa0415 — FOA 0415 v1-v5 with canonical rotation (25 experiments, 60 epochs)
# ============================================================
train_foa0415() {
    # v1: exp130-134
    run_train foa_0415_v1 exp130_foa0415v1_lr1e3_lsh0.1 60 0.001  32 --rotate-canonical
    run_train foa_0415_v1 exp131_foa0415v1_lr1e3_lsh0.3 60 0.001  32 --rotate-canonical
    run_train foa_0415_v1 exp132_foa0415v1_lr5e4_lsh0.1 60 0.0005 32 --rotate-canonical
    run_train foa_0415_v1 exp133_foa0415v1_lr5e4_lsh0.5 60 0.0005 32 --rotate-canonical
    run_train foa_0415_v1 exp134_foa0415v1_lr1e4_lsh0.1 60 0.0001 32 --rotate-canonical
    # v2: exp135-139
    run_train foa_0415_v2 exp135_foa0415v2_lr1e3_lsh0.1 60 0.001  32 --rotate-canonical
    run_train foa_0415_v2 exp136_foa0415v2_lr1e3_lsh0.3 60 0.001  32 --rotate-canonical
    run_train foa_0415_v2 exp137_foa0415v2_lr5e4_lsh0.1 60 0.0005 32 --rotate-canonical
    run_train foa_0415_v2 exp138_foa0415v2_lr5e4_lsh0.5 60 0.0005 32 --rotate-canonical
    run_train foa_0415_v2 exp139_foa0415v2_lr1e4_lsh0.1 60 0.0001 32 --rotate-canonical
    # v3: exp140-144
    run_train foa_0415_v3 exp140_foa0415v3_lr1e3_lsh0.1 60 0.001  32 --rotate-canonical
    run_train foa_0415_v3 exp141_foa0415v3_lr1e3_lsh0.3 60 0.001  32 --rotate-canonical
    run_train foa_0415_v3 exp142_foa0415v3_lr5e4_lsh0.1 60 0.0005 32 --rotate-canonical
    run_train foa_0415_v3 exp143_foa0415v3_lr5e4_lsh0.5 60 0.0005 32 --rotate-canonical
    run_train foa_0415_v3 exp144_foa0415v3_lr1e4_lsh0.1 60 0.0001 32 --rotate-canonical
    # v4: exp145-149
    run_train foa_0415_v4 exp145_foa0415v4_lr1e3_lsh0.1 60 0.001  32 --rotate-canonical
    run_train foa_0415_v4 exp146_foa0415v4_lr1e3_lsh0.3 60 0.001  32 --rotate-canonical
    run_train foa_0415_v4 exp147_foa0415v4_lr5e4_lsh0.1 60 0.0005 32 --rotate-canonical
    run_train foa_0415_v4 exp148_foa0415v4_lr5e4_lsh0.5 60 0.0005 32 --rotate-canonical
    run_train foa_0415_v4 exp149_foa0415v4_lr1e4_lsh0.1 60 0.0001 32 --rotate-canonical
    # v5: exp150-154
    run_train foa_0415_v5 exp150_foa0415v5_lr1e3_lsh0.1 60 0.001  32 --rotate-canonical
    run_train foa_0415_v5 exp151_foa0415v5_lr1e3_lsh0.3 60 0.001  32 --rotate-canonical
    run_train foa_0415_v5 exp152_foa0415v5_lr5e4_lsh0.1 60 0.0005 32 --rotate-canonical
    run_train foa_0415_v5 exp153_foa0415v5_lr5e4_lsh0.5 60 0.0005 32 --rotate-canonical
    run_train foa_0415_v5 exp154_foa0415v5_lr1e4_lsh0.1 60 0.0001 32 --rotate-canonical
}

# ============================================================
# GROUP: pvitvoa — Pretrained ViT + FOA v1-v3 (6 experiments, 40 epochs)
# ============================================================
train_pvitvoa() {
    run_train pretrain_vit_foa   exp160_pvitfoav1_lr1e4_w0.1 40 0.0001 16 --rotate-canonical
    run_train pretrain_vit_foa   exp161_pvitfoav1_lr5e5_w0.3 40 0.00005 16 --rotate-canonical
    run_train pretrain_vit_foa_v2 exp162_pvitfoav2_lr1e4_w0.1 40 0.0001 16 --rotate-canonical
    run_train pretrain_vit_foa_v2 exp163_pvitfoav2_lr5e5_w0.3 40 0.00005 16 --rotate-canonical
    run_train pretrain_vit_foa_v3 exp164_pvitfoav3_lr1e4_w0.1 40 0.0001 16 --rotate-canonical
    run_train pretrain_vit_foa_v3 exp165_pvitfoav3_lr5e5_w0.3 40 0.00005 16 --rotate-canonical
}

# ============================================================
# GROUP: cw — Canonical rotation standalone (6 experiments, 40 epochs)
# ============================================================
train_cw() {
    run_train foa             CW_rot_foa             40 0.001 32 --rotate-canonical
    run_train foa_crossattn   CW_rot_foa_crossattn   40 0.001 32 --rotate-canonical
    run_train foa_featbank    CW_rot_foa_featbank    40 0.001 32 --rotate-canonical
    run_train foa_msattn      CW_rot_foa_msattn      40 0.001 32 --rotate-canonical
    run_train foa_channelattn CW_rot_foa_channelattn 40 0.001 32 --rotate-canonical
    run_train foa_v2          CW_rot_foa_v2          40 0.001 32 --rotate-canonical
}

# ============================================================
# GROUP: n3 — N3 energy-aware UNet variants + oracle (21 experiments, 60 epochs, BS=128)
#   Source script: scripts/n3_bulk.sh (exp166-186, now template-only).
#   All use --rotate-canonical. λ_sh passed via --lambda-sh.
# ============================================================
train_n3() {
    # FiLM
    run_train n3_film            exp166_n3film_lr1e3_lsh0.1    60 0.001  128 --lambda-sh 0.1 --rotate-canonical
    run_train n3_film            exp167_n3film_lr5e4_lsh0.1    60 0.0005 128 --lambda-sh 0.1 --rotate-canonical
    run_train n3_film            exp168_n3film_lr1e3_lsh0.3    60 0.001  128 --lambda-sh 0.3 --rotate-canonical
    # Multi-scale SH
    run_train n3_multiscale_sh   exp169_n3mssh_lr1e3_lsh0.1    60 0.001  128 --lambda-sh 0.1 --rotate-canonical
    run_train n3_multiscale_sh   exp170_n3mssh_lr5e4_lsh0.1    60 0.0005 128 --lambda-sh 0.1 --rotate-canonical
    run_train n3_multiscale_sh   exp171_n3mssh_lr1e3_lsh0.3    60 0.001  128 --lambda-sh 0.3 --rotate-canonical
    # Energy attention (incl. exp186 ablation λ_sh=0.5)
    run_train n3_energy_attn     exp172_n3eattn_lr1e3_lsh0.1   60 0.001  128 --lambda-sh 0.1 --rotate-canonical
    run_train n3_energy_attn     exp173_n3eattn_lr5e4_lsh0.1   60 0.0005 128 --lambda-sh 0.1 --rotate-canonical
    run_train n3_energy_attn     exp174_n3eattn_lr1e3_lsh0.3   60 0.001  128 --lambda-sh 0.3 --rotate-canonical
    run_train n3_energy_attn     exp186_n3eattn_lr1e3_lsh0.5   60 0.001  128 --lambda-sh 0.5 --rotate-canonical
    # Temporal window
    run_train n3_temporal_window exp175_n3twin_lr1e3_lsh0.1    60 0.001  128 --lambda-sh 0.1 --rotate-canonical
    run_train n3_temporal_window exp176_n3twin_lr5e4_lsh0.1    60 0.0005 128 --lambda-sh 0.1 --rotate-canonical
    run_train n3_temporal_window exp177_n3twin_lr1e3_lsh0.3    60 0.001  128 --lambda-sh 0.3 --rotate-canonical
    # Oracle nc3 (binaural + GT energy map)
    run_train foa_oracle_nc3     exp178_oracle_nc3_lr1e3_lsh0.1   60 0.001  128 --lambda-sh 0.1 --rotate-canonical
    run_train foa_oracle_nc3     exp179_oracle_nc3_lr5e4_lsh0.1   60 0.0005 128 --lambda-sh 0.1 --rotate-canonical
    run_train foa_oracle_nc3     exp180_oracle_nc3_lr1e3_lsh0.3   60 0.001  128 --lambda-sh 0.3 --rotate-canonical
    # Oracle nc1 (GT energy map only)
    run_train foa_oracle_nc1     exp181_oracle_nc1_lr1e3_lsh0.1   60 0.001  128 --lambda-sh 0.1 --rotate-canonical
    run_train foa_oracle_nc1     exp182_oracle_nc1_lr5e4_lsh0.1   60 0.0005 128 --lambda-sh 0.1 --rotate-canonical
    run_train foa_oracle_nc1     exp183_oracle_nc1_lr1e3_lsh0.3   60 0.001  128 --lambda-sh 0.3 --rotate-canonical
    # foa_0415_v1 baseline (Group G)
    run_train foa_0415_v1        exp184_v1base_lr1e3_lsh0.1       60 0.001  128 --lambda-sh 0.1 --rotate-canonical
    run_train foa_0415_v1        exp185_v1base_lr5e4_lsh0.1       60 0.0005 128 --lambda-sh 0.1 --rotate-canonical
}

# ============================================================
# DISPATCHER
# ============================================================
case "$GROUP" in
    all)
        train_baseline; train_vit; train_echodiff; train_foa_variant
        train_foa; train_foav2; train_resnet; train_previt
        train_echonet; train_batvision; train_foa0415; train_pvitvoa; train_cw
        train_n3
        ;;
    baseline)     train_baseline ;;
    vit)          train_vit ;;
    echodiff)     train_echodiff ;;
    foa_variant)  train_foa_variant ;;
    foa)          train_foa ;;
    foav2)        train_foav2 ;;
    resnet)       train_resnet ;;
    previt)       train_previt ;;
    echonet)      train_echonet ;;
    batvision)    train_batvision ;;
    foa0415)      train_foa0415 ;;
    pvitvoa)      train_pvitvoa ;;
    cw)           train_cw ;;
    n3)           train_n3 ;;
    *)            echo "Unknown group: $GROUP"; exit 1 ;;
esac

echo "============================================================"
echo "summary_train.sh ($GROUP) finished — $(date)"
echo "============================================================"

#!/bin/bash
# Base 실험 - train.py 사용
# SI loss (Combined) + Mel Spectrogram이 디폴트로 적용됨

echo "=========================================="
echo "Base Experiment (train.py)"
echo "SI loss + Mel Spectrogram (Default)"
echo "=========================================="
echo ""

# 기본 실험 - config 파일의 디폴트 설정 사용
# - audio_format: mel_spectrogram (디폴트)
# - criterion: Combined (L1 + SI loss, 디폴트)
# - silog_weight: 0.637, l1_weight: 0.237 (디폴트)
# - silog_lambda: 0.869 (디폴트)
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --experiment_name base_default

echo ""
echo "=========================================="
echo "실험 완료!"
echo "=========================================="
echo ""
echo "결과 확인:"
echo "  - W&B: https://wandb.ai/"
echo "  - Checkpoints: ./checkpoints/unet_baseline_batvisionv2_BS256_Lr0.002_AdamW_base_default/"
echo "  - Results: ./results/unet_baseline_batvisionv2_BS256_Lr0.002_AdamW_base_default/"
echo ""

#!/bin/bash
# Base 실험 - Batvision V1
# SI loss (Combined) + Spectrogram

echo "=========================================="
echo "Base Experiment - Batvision V1"
echo "SI loss + Spectrogram"
echo "=========================================="
echo ""
echo "Note: BatvisionV1 설정"
echo "  - Audio Format: spectrogram (mel_spectrogram 미지원)"
echo "  - Max Depth: 12.0m"
echo "  - Learning Rate: 0.001 (paper default)"
echo "  - Batch Size: 128 (paper default)"
echo ""

# BatvisionV1 실험
# - audio_format: spectrogram (디폴트, mel은 V1에서 미지원)
# - criterion: Combined (L1 + SI loss, 디폴트)
# - learning_rate: 0.001 (paper default for V1)
# - batch_size: 128 (paper default for V1)
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name base_v1_default

echo ""
echo "=========================================="
echo "실험 완료!"
echo "=========================================="
echo ""
echo "결과 확인:"
echo "  - W&B: https://wandb.ai/"
echo "  - Checkpoints: ./checkpoints/unet_baseline_batvisionv1_BS128_Lr0.001_AdamW_base_v1_default/"
echo "  - Results: ./results/unet_baseline_batvisionv1_BS128_Lr0.001_AdamW_base_v1_default/"
echo ""







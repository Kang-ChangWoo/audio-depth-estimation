#!/bin/bash

# AdaBins Knowledge Distillation - Example Commands
# RGB → Audio Transfer Learning

echo "======================================"
echo "AdaBins Distillation Training Examples"
echo "======================================"

# ==========================================
# Example 1: Basic Distillation
# ==========================================
echo ""
echo "Example 1: Basic Distillation"
echo "------------------------------"
echo "python train_adabins_distillation.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 64 \\"
echo "  --learning_rate 0.001 \\"
echo "  --n_bins 128 \\"
echo "  --temperature 4.0 \\"
echo "  --use_wandb \\"
echo "  --experiment_name basic_distill"
echo ""

# ==========================================
# Example 2: Adaptive Curriculum Learning (RECOMMENDED)
# ==========================================
echo "Example 2: Adaptive Curriculum Learning (RECOMMENDED)"
echo "------------------------------------------------------"
echo "python train_adabins_distillation.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 64 \\"
echo "  --learning_rate 0.001 \\"
echo "  --n_bins 128 \\"
echo "  --use_adaptive_loss \\"
echo "  --temperature 5.0 \\"
echo "  --use_wandb \\"
echo "  --experiment_name adaptive_distill_v1"
echo ""

# ==========================================
# Example 3: Frozen Teacher (Fast Training)
# ==========================================
echo "Example 3: Frozen Teacher (Fast Training)"
echo "------------------------------------------"
echo "python train_adabins_distillation.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 64 \\"
echo "  --freeze_rgb \\"
echo "  --temperature 6.0 \\"
echo "  --lambda_response 0.8 \\"
echo "  --use_wandb \\"
echo "  --experiment_name frozen_teacher"
echo ""

# ==========================================
# Example 4: Custom Loss Weights
# ==========================================
echo "Example 4: Custom Loss Weights"
echo "--------------------------------"
echo "python train_adabins_distillation.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 64 \\"
echo "  --lambda_task 1.0 \\"
echo "  --lambda_response 0.7 \\"
echo "  --lambda_feature 0.5 \\"
echo "  --lambda_bin 0.3 \\"
echo "  --lambda_sparse 0.15 \\"
echo "  --use_wandb \\"
echo "  --experiment_name custom_weights"
echo ""

# ==========================================
# Example 5: High Resolution with More Bins
# ==========================================
echo "Example 5: High Resolution with More Bins"
echo "-------------------------------------------"
echo "python train_adabins_distillation.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 32 \\"
echo "  --learning_rate 0.0005 \\"
echo "  --n_bins 256 \\"
echo "  --base_channels 64 \\"
echo "  --temperature 4.0 \\"
echo "  --use_wandb \\"
echo "  --experiment_name high_res_256bins"
echo ""

# ==========================================
# Example 6: Quick Debug (Small Model)
# ==========================================
echo "Example 6: Quick Debug (Small Model)"
echo "--------------------------------------"
echo "python train_adabins_distillation.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 16 \\"
echo "  --learning_rate 0.001 \\"
echo "  --n_bins 64 \\"
echo "  --base_channels 32 \\"
echo "  --nb_epochs 20 \\"
echo "  --experiment_name debug_small"
echo ""

# ==========================================
# Example 7: Multi-GPU Training
# ==========================================
echo "Example 7: Multi-GPU Training"
echo "------------------------------"
echo "python train_adabins_distillation.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 256 \\"
echo "  --learning_rate 0.002 \\"
echo "  --gpu_ids 0,1,2,3 \\"
echo "  --use_wandb \\"
echo "  --experiment_name multi_gpu"
echo ""

# ==========================================
# Example 8: BatvisionV1 (No RGB Available)
# ==========================================
echo "Example 8: BatvisionV1 (Limited Distillation)"
echo "-----------------------------------------------"
echo "# Note: BV1 has no RGB, so distillation is limited"
echo "python train_adabins_distillation.py \\"
echo "  --dataset batvisionv1 \\"
echo "  --batch_size 64 \\"
echo "  --learning_rate 0.001 \\"
echo "  --lambda_task 1.0 \\"
echo "  --lambda_response 0.0 \\"
echo "  --lambda_feature 0.0 \\"
echo "  --use_wandb \\"
echo "  --experiment_name bv1_audio_only"
echo ""

echo "======================================"
echo "To run an example, copy and paste the command"
echo "Or modify the parameters as needed"
echo "======================================"


#!/bin/bash

# Binaural Attention Training Examples
# Choose the appropriate command for your use case

echo "====================================================================="
echo "Binaural Attention Depth Estimation - Training Examples"
echo "====================================================================="
echo ""

# ============================================
# Example 1: Quick Test (Fast Training)
# ============================================
echo "Example 1: Quick Test"
echo "  - Small model for fast iteration"
echo "  - Fewer attention levels"
echo "  - Good for debugging and hyperparameter search"
echo ""
echo "Command:"
echo "python train_binaural_attention.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 128 \\"
echo "  --base_channels 32 \\"
echo "  --attention_levels 3 4 5 \\"
echo "  --nb_epochs 100 \\"
echo "  --learning_rate 0.001 \\"
echo "  --use_wandb \\"
echo "  --experiment_name binaural_quick_test"
echo ""
echo "-------------------------------------------------------------------"
echo ""

# ============================================
# Example 2: Standard Training (Recommended)
# ============================================
echo "Example 2: Standard Training (RECOMMENDED)"
echo "  - Default configuration"
echo "  - Good balance of speed and performance"
echo "  - Standard loss with fixed weights"
echo ""
echo "Command:"
echo "python train_binaural_attention.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 64 \\"
echo "  --base_channels 64 \\"
echo "  --attention_levels 2 3 4 5 \\"
echo "  --learning_rate 0.001 \\"
echo "  --nb_epochs 200 \\"
echo "  --optimizer AdamW \\"
echo "  --scheduler cosine \\"
echo "  --use_wandb \\"
echo "  --experiment_name binaural_standard_v1"
echo ""
echo "-------------------------------------------------------------------"
echo ""

# ============================================
# Example 3: Adaptive Loss (Best Performance)
# ============================================
echo "Example 3: Adaptive Loss (BEST PERFORMANCE)"
echo "  - Curriculum learning with adaptive weights"
echo "  - Starts with reconstruction, gradually adds edge/smooth"
echo "  - Most stable training"
echo ""
echo "Command:"
echo "python train_binaural_attention.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 64 \\"
echo "  --base_channels 64 \\"
echo "  --attention_levels 2 3 4 5 \\"
echo "  --learning_rate 0.001 \\"
echo "  --use_adaptive_loss \\"
echo "  --nb_epochs 200 \\"
echo "  --optimizer AdamW \\"
echo "  --use_wandb \\"
echo "  --experiment_name binaural_adaptive_v1"
echo ""
echo "-------------------------------------------------------------------"
echo ""

# ============================================
# Example 4: Maximum Quality (Slow but Best)
# ============================================
echo "Example 4: Maximum Quality"
echo "  - Largest model with all attention levels"
echo "  - Adaptive loss for stable training"
echo "  - Best possible performance (but slow)"
echo ""
echo "Command:"
echo "python train_binaural_attention.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 32 \\"
echo "  --base_channels 96 \\"
echo "  --attention_levels 1 2 3 4 5 \\"
echo "  --learning_rate 0.0005 \\"
echo "  --use_adaptive_loss \\"
echo "  --nb_epochs 250 \\"
echo "  --optimizer AdamW \\"
echo "  --weight_decay 0.01 \\"
echo "  --scheduler cosine \\"
echo "  --use_wandb \\"
echo "  --experiment_name binaural_max_quality"
echo ""
echo "-------------------------------------------------------------------"
echo ""

# ============================================
# Example 5: Resume Training
# ============================================
echo "Example 5: Resume Training"
echo "  - Continue from a saved checkpoint"
echo "  - Useful if training was interrupted"
echo ""
echo "Command:"
echo "python train_binaural_attention.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 64 \\"
echo "  --checkpoints 100 \\"
echo "  --experiment_name binaural_adaptive_v1 \\"
echo "  --use_wandb"
echo ""
echo "-------------------------------------------------------------------"
echo ""

# ============================================
# Example 6: Custom Loss Weights
# ============================================
echo "Example 6: Custom Loss Weights"
echo "  - Fine-tune loss component weights"
echo "  - Higher edge weight for sharper boundaries"
echo ""
echo "Command:"
echo "python train_binaural_attention.py \\"
echo "  --dataset batvisionv2 \\"
echo "  --batch_size 64 \\"
echo "  --lambda_recon 1.0 \\"
echo "  --lambda_edge 0.3 \\"
echo "  --lambda_smooth 0.15 \\"
echo "  --learning_rate 0.001 \\"
echo "  --use_wandb \\"
echo "  --experiment_name binaural_custom_loss"
echo ""
echo "-------------------------------------------------------------------"
echo ""

# ============================================
# Example 7: BatvisionV1 Dataset
# ============================================
echo "Example 7: BatvisionV1 Dataset"
echo "  - Train on BatvisionV1 instead of V2"
echo "  - V1 has different audio format"
echo ""
echo "Command:"
echo "python train_binaural_attention.py \\"
echo "  --dataset batvisionv1 \\"
echo "  --batch_size 64 \\"
echo "  --base_channels 64 \\"
echo "  --use_adaptive_loss \\"
echo "  --use_wandb \\"
echo "  --experiment_name binaural_batvision_v1"
echo ""
echo "-------------------------------------------------------------------"
echo ""

echo "====================================================================="
echo "To run an example, copy-paste the command above"
echo "====================================================================="
echo ""
echo "Tips:"
echo "  - Start with Example 2 (Standard Training) for initial experiments"
echo "  - Use Example 3 (Adaptive Loss) for best results"
echo "  - Monitor training on W&B: https://wandb.ai"
echo "  - Check results in: results/<experiment_name>/"
echo ""
echo "For more details, see: BINAURAL_ATTENTION_GUIDE.md"
echo ""


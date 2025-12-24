#!/bin/bash
# =============================================================================
# RGB Depth Estimation Training Examples
# =============================================================================
# 
# This script provides example training configurations for RGB-based
# depth estimation. The trained RGB model can serve as a teacher for
# knowledge distillation to audio-based models.
#
# Usage:
#   bash run_rgb_depth_examples.sh [example_number]
#
# Examples:
#   bash run_rgb_depth_examples.sh 1    # Run example 1
#   bash run_rgb_depth_examples.sh      # Interactive selection
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Example 1: Basic RGB Training (BatvisionV2)
# =============================================================================
example_1() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Example 1: Basic RGB Training${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  - Dataset: BatvisionV2"
    echo "  - Batch size: 64"
    echo "  - Learning rate: 0.0001"
    echo "  - Base channels: 64"
    echo "  - Optimizer: AdamW"
    echo ""
    
    python train_rgb_depth.py \
        --dataset batvisionv2 \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --base_channels 64 \
        --optimizer AdamW \
        --nb_epochs 200 \
        --save_frequency 2 \
        --experiment_name rgb_depth_basic_v1
}

# =============================================================================
# Example 2: RGB Training with W&B Logging
# =============================================================================
example_2() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Example 2: RGB Training with W&B${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  - Dataset: BatvisionV2"
    echo "  - Batch size: 64"
    echo "  - Learning rate: 0.0001"
    echo "  - W&B logging: Enabled"
    echo ""
    
    python train_rgb_depth.py \
        --dataset batvisionv2 \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --use_wandb \
        --wandb_project batvision-rgb-depth \
        --experiment_name rgb_depth_wandb_v1
}

# =============================================================================
# Example 3: Lightweight RGB Model (for faster training)
# =============================================================================
example_3() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Example 3: Lightweight RGB Model${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  - Base channels: 32 (~5M params)"
    echo "  - Batch size: 128"
    echo "  - Learning rate: 0.0002"
    echo "  - Faster training with smaller model"
    echo ""
    
    python train_rgb_depth.py \
        --dataset batvisionv2 \
        --base_channels 32 \
        --batch_size 128 \
        --learning_rate 0.0002 \
        --optimizer AdamW \
        --experiment_name rgb_depth_lightweight_v1
}

# =============================================================================
# Example 4: High-Capacity RGB Model
# =============================================================================
example_4() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Example 4: High-Capacity Model${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  - Base channels: 64 (~20M params)"
    echo "  - Batch size: 32 (larger model needs smaller batch)"
    echo "  - Learning rate: 0.00005"
    echo "  - Better performance, slower training"
    echo ""
    
    python train_rgb_depth.py \
        --dataset batvisionv2 \
        --base_channels 64 \
        --batch_size 32 \
        --learning_rate 0.00005 \
        --optimizer AdamW \
        --weight_decay 0.01 \
        --scheduler cosine \
        --experiment_name rgb_depth_highcap_v1
}

# =============================================================================
# Example 5: Resume Training from Checkpoint
# =============================================================================
example_5() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Example 5: Resume from Checkpoint${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  - Resume from epoch 50"
    echo "  - Continue training existing model"
    echo ""
    
    read -p "Enter experiment name to resume: " exp_name
    read -p "Enter epoch to resume from: " epoch_num
    
    python train_rgb_depth.py \
        --dataset batvisionv2 \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --experiment_name "$exp_name" \
        --checkpoints "$epoch_num"
}

# =============================================================================
# Example 6: Train Teacher Model for Distillation
# =============================================================================
example_6() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Example 6: Teacher Model for KD${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  - High-capacity teacher model"
    echo "  - Base channels: 64"
    echo "  - Train to convergence (200 epochs)"
    echo "  - Save for distillation to audio model"
    echo ""
    
    python train_rgb_depth.py \
        --dataset batvisionv2 \
        --base_channels 64 \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --optimizer AdamW \
        --scheduler cosine \
        --nb_epochs 200 \
        --save_frequency 5 \
        --use_wandb \
        --experiment_name rgb_teacher_for_distillation_v1
}

# =============================================================================
# Example 7: Custom Loss Weights
# =============================================================================
example_7() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Example 7: Custom Loss Weights${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  - λ_l1: 1.0 (reconstruction)"
    echo "  - λ_smooth: 0.2 (smoothness)"
    echo "  - Emphasize smooth depth maps"
    echo ""
    
    python train_rgb_depth.py \
        --dataset batvisionv2 \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --lambda_l1 1.0 \
        --lambda_smooth 0.2 \
        --experiment_name rgb_depth_smooth_v1
}

# =============================================================================
# Example 8: Quick Test Run (2 epochs)
# =============================================================================
example_8() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Example 8: Quick Test Run${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  - 2 epochs only"
    echo "  - Test pipeline and data loading"
    echo "  - Small batch size for quick iteration"
    echo ""
    
    python train_rgb_depth.py \
        --dataset batvisionv2 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --nb_epochs 2 \
        --save_frequency 1 \
        --experiment_name rgb_depth_test
}

# =============================================================================
# Main Menu
# =============================================================================
show_menu() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}RGB Depth Training Examples${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "Available examples:"
    echo ""
    echo "  1) Basic RGB Training"
    echo "  2) RGB Training with W&B Logging"
    echo "  3) Lightweight Model (Fast)"
    echo "  4) High-Capacity Model (Accurate)"
    echo "  5) Resume from Checkpoint"
    echo "  6) Train Teacher Model for Distillation"
    echo "  7) Custom Loss Weights"
    echo "  8) Quick Test Run (2 epochs)"
    echo ""
    echo "  0) Exit"
    echo ""
}

# =============================================================================
# Main Script
# =============================================================================
main() {
    # Check if example number provided as argument
    if [ $# -eq 1 ]; then
        case $1 in
            1) example_1 ;;
            2) example_2 ;;
            3) example_3 ;;
            4) example_4 ;;
            5) example_5 ;;
            6) example_6 ;;
            7) example_7 ;;
            8) example_8 ;;
            *)
                echo -e "${RED}Invalid example number: $1${NC}"
                echo "Usage: bash run_rgb_depth_examples.sh [1-8]"
                exit 1
                ;;
        esac
        exit 0
    fi
    
    # Interactive mode
    while true; do
        show_menu
        read -p "Select an example (0-8): " choice
        echo ""
        
        case $choice in
            1) example_1 ;;
            2) example_2 ;;
            3) example_3 ;;
            4) example_4 ;;
            5) example_5 ;;
            6) example_6 ;;
            7) example_7 ;;
            8) example_8 ;;
            0)
                echo -e "${YELLOW}Exiting...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please select 0-8.${NC}"
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
        clear
    done
}

# Run main
main "$@"







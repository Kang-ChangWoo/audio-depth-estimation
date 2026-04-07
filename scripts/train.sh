#!/bin/bash
# Training script for AudioDepthFOA UNet
# Usage: bash scripts/train.sh

cd "$(dirname "$0")/.."

python train.py \
    --batch-size 32 \
    --epochs 40 \
    --lr 0.001 \
    --optimizer AdamW \
    --criterion L1 \
    --experiment-name default

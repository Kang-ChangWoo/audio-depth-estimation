#!/bin/bash
# Testing script for AudioDepthFOA UNet
# Usage: bash scripts/test.sh

cd "$(dirname "$0")/.."

python test.py \
    --eval-on test \
    --checkpoints best \
    --experiment-name default

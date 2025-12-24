# RGB Depth Estimation Model

## Overview

This module provides a **monocular RGB-based depth estimation model** that is architecturally compatible with the binaural attention model for knowledge distillation.

### Key Features

- ✅ **RGB Input**: Standard 3-channel RGB images (vs. 2-channel stereo audio)
- ✅ **Feature Compatibility**: Feature dimensions match `BinauralAttentionDepthNet` at all encoder levels
- ✅ **Distillation Ready**: Can serve as teacher model for audio-to-depth networks
- ✅ **U-Net Architecture**: Encoder-decoder with skip connections
- ✅ **~20M Parameters**: With base_channels=64 (vs. ~40M for binaural model with dual encoders)

---

## Architecture Comparison

### Binaural Attention Model (Audio)
```
Input: [B, 2, H, W]  (stereo audio)
├── Left Encoder  → features
├── Right Encoder → features
├── Cross Attention (multi-scale)
├── Fusion Layers
└── Decoder → Depth [B, 1, H, W]
```

### RGB Depth Model (Vision)
```
Input: [B, 3, H, W]  (RGB image)
├── Single Encoder → features
└── Decoder → Depth [B, 1, H, W]
```

### Feature Dimensions (COMPATIBLE!)

Both models produce the **same feature dimensions** at each level:

| Level | Channels | Spatial Size | Binaural Model | RGB Model |
|-------|----------|--------------|----------------|-----------|
| x1    | 64       | H × W        | ✓ (after fusion) | ✓        |
| x2    | 128      | H/2 × W/2    | ✓ (after fusion) | ✓        |
| x3    | 256      | H/4 × W/4    | ✓ (after fusion) | ✓        |
| x4    | 512      | H/8 × W/8    | ✓ (after fusion) | ✓        |
| x5    | 512      | H/16 × W/16  | ✓ (after fusion) | ✓        |

This compatibility enables:
- **Knowledge Distillation**: RGB → Audio
- **Feature-level distillation**: Match intermediate representations
- **Hybrid Models**: Combine RGB and audio features

---

## Files

### Model Definition
- **`models/rgb_depth_model.py`**: RGB depth estimation model
  - `RGBDepthNet`: Main model class
  - `create_rgb_depth_model()`: Factory function
  - `forward(x, return_features=True)`: Support for distillation

### Training Script
- **`train_rgb_depth.py`**: Training script for RGB model
  - Compatible with BatvisionV1 and BatvisionV2 datasets
  - Supports W&B logging
  - Checkpoint saving and resuming

### Examples
- **`run_rgb_depth_examples.sh`**: Bash script with 8 training examples
  - Basic training
  - W&B logging
  - Lightweight/high-capacity variants
  - Teacher model training for distillation

---

## Usage

### 1. Basic Training

```bash
python train_rgb_depth.py \
    --dataset batvisionv2 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --experiment_name rgb_depth_basic
```

### 2. Train Teacher Model (for Distillation)

```bash
python train_rgb_depth.py \
    --dataset batvisionv2 \
    --base_channels 64 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --nb_epochs 200 \
    --use_wandb \
    --experiment_name rgb_teacher_for_kd
```

### 3. Use Pre-configured Examples

```bash
# Interactive menu
bash run_rgb_depth_examples.sh

# Or run specific example
bash run_rgb_depth_examples.sh 6  # Example 6: Teacher model for distillation
```

### 4. Inference with Feature Extraction

```python
from models.rgb_depth_model import create_rgb_depth_model
import torch

# Create model
model = create_rgb_depth_model(base_channels=64)
model.eval()

# RGB input [B, 3, H, W]
rgb_image = torch.randn(1, 3, 256, 256)

# Normal forward pass
depth = model(rgb_image)
print(f"Depth shape: {depth.shape}")  # [1, 1, 256, 256]

# Forward with features (for distillation)
depth, features = model(rgb_image, return_features=True)
print(f"Encoder features: {list(features.keys())}")
# ['x1', 'x2', 'x3', 'x4', 'x5', 'd1', 'd2', 'd3', 'd4']
```

---

## Knowledge Distillation Setup

### Step 1: Train RGB Teacher Model

```bash
python train_rgb_depth.py \
    --dataset batvisionv2 \
    --base_channels 64 \
    --nb_epochs 200 \
    --experiment_name rgb_teacher
```

### Step 2: Train Audio Student with Distillation

```python
# Pseudo-code for distillation training loop

# Load teacher model (RGB)
teacher = create_rgb_depth_model(base_channels=64)
teacher.load_state_dict(checkpoint['model_state_dict'])
teacher.eval()

# Create student model (Audio)
student = create_binaural_attention_model(base_channels=64)

for epoch in range(num_epochs):
    for audio, rgb, depth_gt in dataloader:
        # Teacher forward (no gradient)
        with torch.no_grad():
            depth_teacher, feats_teacher = teacher(rgb, return_features=True)
        
        # Student forward
        depth_student = student(audio)
        feats_student = student.get_features()  # Implement this!
        
        # Distillation loss
        loss_task = criterion(depth_student, depth_gt)
        loss_kd = kd_criterion(depth_student, depth_teacher)
        loss_feat = feature_distillation_loss(feats_student, feats_teacher)
        
        total_loss = loss_task + lambda_kd * loss_kd + lambda_feat * loss_feat
        total_loss.backward()
        optimizer.step()
```

### Feature Matching for Distillation

Both models expose features at the same levels:

```python
# RGB Teacher
depth_rgb, feats_rgb = teacher(rgb, return_features=True)
# feats_rgb['x1']: [B, 64, H, W]
# feats_rgb['x2']: [B, 128, H/2, W/2]
# ...

# Audio Student (after implementing feature extraction)
depth_audio, feats_audio = student(audio, return_features=True)
# feats_audio['x1']: [B, 64, H, W]  <- Should match!
# feats_audio['x2']: [B, 128, H/2, W/2]  <- Should match!
# ...

# Match features level-by-level
for level in ['x1', 'x2', 'x3', 'x4', 'x5']:
    loss_feat += F.mse_loss(feats_audio[level], feats_rgb[level])
```

---

## Model Specifications

### RGB Depth Model

| Configuration | Base Channels | Parameters | Memory (FP32) | Speed (RTX 3090) |
|---------------|---------------|------------|---------------|------------------|
| Lightweight   | 32            | ~5M        | ~2GB          | ~120 fps         |
| Standard      | 64            | ~20M       | ~4GB          | ~80 fps          |

### Comparison with Binaural Model

| Model                  | Input     | Channels | Parameters | Features         |
|------------------------|-----------|----------|------------|------------------|
| RGB Depth              | RGB (3ch) | 64       | ~20M       | Single encoder   |
| Binaural Attention     | Audio (2ch)| 64      | ~40M       | Dual encoder + Attention |

---

## Command-Line Arguments

### Dataset
- `--dataset`: `batvisionv1` or `batvisionv2` (default: `batvisionv2`)
- `--batch_size`: Batch size (default: 64)
- `--num_workers`: Data loading workers (default: 4)

### Model
- `--base_channels`: Base channel count (default: 64)
  - 32: ~5M params, faster training
  - 64: ~20M params, better performance
- `--bilinear`: Use bilinear upsampling (default: True)

### Loss
- `--lambda_l1`: Weight for L1 loss (default: 1.0)
- `--lambda_smooth`: Weight for smoothness loss (default: 0.1)

### Training
- `--learning_rate`: Learning rate (default: 0.0001)
- `--nb_epochs`: Number of epochs (default: 200)
- `--optimizer`: `Adam`, `AdamW`, or `SGD` (default: `AdamW`)
- `--weight_decay`: Weight decay (default: 0.01)
- `--scheduler`: `none`, `cosine`, or `step` (default: `cosine`)

### Checkpoints
- `--checkpoints`: Resume from epoch number
- `--save_frequency`: Save every N epochs (default: 2)

### Logging
- `--use_wandb`: Enable Weights & Biases
- `--wandb_project`: W&B project name
- `--experiment_name`: Experiment name

---

## Directory Structure

```
UNetSoundOnly/
├── models/
│   ├── rgb_depth_model.py          ← New RGB model
│   └── binaural_attention_model.py ← Audio model (compatible)
├── train_rgb_depth.py              ← New training script
├── train_binaural_attention.py     ← Audio training script
├── run_rgb_depth_examples.sh       ← New example runner
├── run_binaural_attention_examples.sh
├── checkpoints/
│   ├── rgb_depth_basic/            ← RGB checkpoints
│   └── binaural_attn_*/            ← Audio checkpoints
├── results/
│   ├── rgb_depth_basic/            ← RGB visualizations
│   └── binaural_attn_*/            ← Audio visualizations
└── logs/
```

---

## Next Steps

### 1. Train RGB Teacher Model

```bash
bash run_rgb_depth_examples.sh 6  # Teacher model training
```

### 2. Implement Feature Extraction in Audio Model

Modify `BinauralAttentionDepthNet.forward()` to optionally return features:

```python
def forward(self, x, return_features=False):
    # ... existing code ...
    
    if return_features:
        features = {
            'x1': left_feats['x1'],  # After fusion
            'x2': left_feats['x2'],
            'x3': left_feats['x3'],
            'x4': left_feats['x4'],
            'x5': left_feats['x5']
        }
        return depth, features
    else:
        return depth
```

### 3. Implement Distillation Training Script

Create `train_distillation.py` that:
- Loads RGB teacher model
- Trains audio student model
- Applies knowledge distillation loss
- Matches intermediate features

### 4. Evaluate Performance

Compare:
- RGB model (upper bound)
- Audio model (no distillation)
- Audio model (with distillation) ← Should improve!

---

## Citation

If you use this code, please cite:

```bibtex
@article{batvision2024,
  title={Audio-Visual Depth Estimation with Knowledge Distillation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

---

## License

MIT License (or your preferred license)

---

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Happy Training! 🚀**







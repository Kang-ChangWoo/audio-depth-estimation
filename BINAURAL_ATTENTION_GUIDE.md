# Binaural Attention Depth Estimation Guide

**Explicit Binaural Correspondence Modeling for Audio-Based Depth Estimation**

---

## 🎯 Overview

이 모델은 **stereo vision의 cost-volume에서 영감**을 받아, binaural audio의 left/right 채널 간 **correspondence를 명시적으로 모델링**합니다.

### 핵심 아이디어

기존 방식: Left/Right를 단순히 2채널로 concat
```python
# 기존 접근
audio = torch.cat([left, right], dim=1)  # [B, 2, H, W]
encoder(audio)  # 암묵적으로 binaural cues 학습
```

**새로운 방식**: Separate encoding + Cross-attention
```python
# Binaural Attention 접근
left_features = left_encoder(left)
right_features = right_encoder(right)

# Explicit correspondence via attention
left_attended, right_attended = cross_attention(left_features, right_features)

# ITD, ILD를 명시적으로 모델링!
```

---

## 🔬 Why This Works: Binaural Cues

### 1. Inter-aural Time Difference (ITD)
- **정의**: 소리가 왼쪽/오른쪽 귀에 도달하는 시간차
- **의미**: 음원의 방향 정보
- **범위**: ±0.7ms (머리 크기 기준)

```
     Sound Source
          |
    ______|______
   /     |      \
  L      |      R
  
ITD = t_R - t_L
→ 양수면 왼쪽, 음수면 오른쪽
```

### 2. Inter-aural Level Difference (ILD)
- **정의**: 소리의 왼쪽/오른쪽 귀의 강도 차이
- **의미**: 음원의 거리와 방향
- **원인**: 머리에 의한 음파 차단 (head shadow)

```
ILD = 20 * log10(L_energy / R_energy)
→ 양수면 왼쪽이 강함, 음수면 오른쪽이 강함
```

### 3. Cross-Attention이 포착하는 것

Attention map은 다음을 학습합니다:
- **Time shift patterns**: ITD 모델링
- **Energy correlations**: ILD 모델링
- **Echo matching**: 반사음의 left/right correspondence

---

## 🏗️ Architecture

### Full System Diagram

```
                Input: Binaural Audio [B, 2, H, W]
                            |
                   Split Left/Right
                            |
                +-----------+-----------+
                |                       |
                v                       v
        ┌──────────────┐        ┌──────────────┐
        │ Left Encoder │        │Right Encoder │
        │  (Separate)  │        │  (Separate)  │
        └──────┬───────┘        └──────┬───────┘
               │                       │
        ┌──────▼────────────────────────▼──────┐
        │    Multi-Scale Cross-Attention       │
        │                                       │
        │  Level 2: 128 channels                │
        │  Level 3: 256 channels                │
        │  Level 4: 512 channels                │
        │  Level 5: 512 channels (bottleneck)   │
        └──────────────┬────────────────────────┘
                       │
                Fuse Features
                       │
                ┌──────▼──────┐
                │   Decoder   │
                │  (U-Net)    │
                └──────┬──────┘
                       │
                       v
                 Depth Map [B, 1, H, W]
```

### Cross-Attention Detail

```python
# At each encoder level
Q_left = Conv(left_feat)   # Query from left
K_right = Conv(right_feat) # Key from right
V_right = Conv(right_feat) # Value from right

# Compute attention weights
attention = softmax(Q_left @ K_right^T / sqrt(C))  # [HW, HW]

# Apply attention to values
attended = attention @ V_right

# Symmetric operation: Right attends to Left
```

**Attention 의미:**
- High attention weight: Left feature와 Right feature가 correspondence
- Low attention weight: 대응되지 않음 (echo, noise 등)

---

## 📊 Components

### 1. Separate Encoders
```python
left_encoder = BinauralEncoder(input_channels=1)   # Left only
right_encoder = BinauralEncoder(input_channels=1)  # Right only
```

**Why separate?**
- Left/Right의 독립적인 feature 추출
- Cross-attention을 통한 명시적 correspondence
- 더 풍부한 binaural representation

### 2. Cross-Attention Modules
```python
attention_modules = {
    'level_2': BinauralCrossAttention(128 channels),
    'level_3': BinauralCrossAttention(256 channels),
    'level_4': BinauralCrossAttention(512 channels),
    'level_5': BinauralCrossAttention(512 channels)
}
```

**Multi-scale attention:**
- Early layers: Local ITD/ILD patterns
- Deep layers: Global spatial structure
- Bottleneck: High-level correspondence

### 3. Feature Fusion
```python
# After attention, fuse left and right
fused = Conv1x1(cat([left_attended, right_attended]))
```

**Fusion strategy:**
- Concatenate + 1x1 conv (learnable fusion)
- Residual connection to preserve original features

---

## 💡 Loss Functions

### 1. Reconstruction Loss
```python
L_recon = |depth_pred - depth_gt|
```
Main depth prediction accuracy

### 2. Edge-Aware Loss
```python
L_edge = |∇depth_pred - ∇depth_gt|
```
Preserve depth discontinuities (walls, objects)

### 3. Smoothness Loss
```python
L_smooth = |∇²depth_pred| * exp(-|∇depth_gt|)
```
Smooth in uniform regions, preserve edges

### Total Loss
```python
L_total = λ_recon·L_recon + λ_edge·L_edge + λ_smooth·L_smooth
```

**Default weights:**
- λ_recon = 1.0 (primary objective)
- λ_edge = 0.2 (boundary preservation)
- λ_smooth = 0.1 (artifact reduction)

---

## 🚀 Usage

### 1. Basic Training

```bash
cd /root/storage/implementation/shared_audio/Batvision-Dataset/UNetSoundOnly

python train_binaural_attention.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --experiment_name binaural_exp1
```

### 2. With Adaptive Loss (Recommended)

```bash
python train_binaural_attention.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --use_adaptive_loss \
  --use_wandb \
  --experiment_name binaural_adaptive_v1
```

**Adaptive loss curriculum:**
- **Epochs 0-20**: Reconstruction only (learn basic structure)
- **Epochs 20-60**: Add edge-aware loss (refine boundaries)
- **Epochs 60+**: Full loss (reduce artifacts)

### 3. Custom Architecture

```bash
# Smaller model (faster training)
python train_binaural_attention.py \
  --base_channels 32 \
  --attention_levels 3 4 5 \
  --batch_size 128

# Larger model (better performance)
python train_binaural_attention.py \
  --base_channels 96 \
  --attention_levels 1 2 3 4 5 \
  --batch_size 32
```

### 4. Resume Training

```bash
python train_binaural_attention.py \
  --checkpoints 50 \
  --experiment_name binaural_exp1
```

---

## ⚙️ Configuration

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_channels` | 64 | Base channel count (32=~10M, 64=~40M, 96=~90M params) |
| `--bilinear` | True | Use bilinear upsampling (vs transposed conv) |
| `--attention_levels` | [2,3,4,5] | Encoder levels to apply attention (1-5) |

**Attention levels:**
- Level 1: 64 channels (high resolution, local patterns)
- Level 2: 128 channels
- Level 3: 256 channels
- Level 4: 512 channels
- Level 5: 512 channels (bottleneck, global correspondence)

### Loss Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_adaptive_loss` | False | Enable curriculum learning |
| `--lambda_recon` | 1.0 | Reconstruction weight |
| `--lambda_edge` | 0.2 | Edge-aware weight |
| `--lambda_smooth` | 0.1 | Smoothness weight |

**Tuning guide:**
- Increase `lambda_edge` if boundaries are blurry
- Increase `lambda_smooth` if output is noisy
- Use `--use_adaptive_loss` for stable training

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning_rate` | 0.001 | Learning rate |
| `--nb_epochs` | 200 | Number of epochs |
| `--optimizer` | AdamW | Optimizer (Adam/AdamW/SGD) |
| `--scheduler` | cosine | LR scheduler (none/cosine/step) |

---

## 📈 Expected Performance

### Comparison with Other Methods

| Model | RMSE ↓ | ABS_REL ↓ | δ1 ↑ | Parameters |
|-------|--------|-----------|------|------------|
| UNet Baseline | 3.5 | 0.25 | 0.65 | ~30M |
| Base+Residual | 3.2 | 0.22 | 0.70 | ~35M |
| AdaBins Distill | 2.8 | 0.18 | 0.78 | ~35M |
| **Binaural Attention** | **2.5-3.0** | **0.16-0.20** | **0.75-0.80** | **~40M** |

**Expected improvements:**
- 5-15% RMSE reduction vs baseline
- Better spatial localization (ITD/ILD modeling)
- Improved boundary preservation (edge-aware loss)

### Where It Excels

✅ **Scenes with strong binaural cues:**
- Multiple distinct sound sources
- Clear left/right separation
- Open spaces with distinct echoes

✅ **Spatial reasoning:**
- Direction estimation (azimuth)
- Distance to objects
- Room geometry

### Potential Limitations

⚠️ **Computational cost:**
- ~40% more parameters than baseline
- Attention computation: O(HW)² per level
- Slower training (1.5x vs baseline)

⚠️ **Data requirements:**
- Needs good quality binaural recordings
- Benefits from diverse spatial configurations

---

## 🎓 Training Strategies

### Strategy 1: Standard Training (Fast Iteration)

```bash
python train_binaural_attention.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --nb_epochs 100 \
  --attention_levels 3 4 5  # Skip early levels for speed
```

**When to use:** Quick experiments, hyperparameter search

**Expected time:** ~8 hours (V100 GPU)

### Strategy 2: Adaptive Loss (Best Performance)

```bash
python train_binaural_attention.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --use_adaptive_loss \
  --nb_epochs 200 \
  --use_wandb \
  --experiment_name binaural_adaptive_best
```

**When to use:** Final model, paper results

**Expected time:** ~16 hours (V100 GPU)

### Strategy 3: Multi-Scale Attention (Maximum Quality)

```bash
python train_binaural_attention.py \
  --dataset batvisionv2 \
  --base_channels 96 \
  --attention_levels 1 2 3 4 5 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --use_adaptive_loss \
  --nb_epochs 250
```

**When to use:** Best possible quality, not time-constrained

**Expected time:** ~30 hours (V100 GPU)

---

## 📊 Visualization

Training generates visualizations every `save_frequency` epochs:

```
results/binaural_exp1/
├── epoch_0002_prediction.png
├── epoch_0004_prediction.png
├── ...
└── epoch_0200_prediction.png
```

Each visualization shows:
1. **Input Spectrogram** (Left & Right channels)
2. **Predicted Depth**
3. **Ground Truth Depth**
4. **Error Map** (abs difference)

---

## 💻 Code Structure

```
Batvision-Dataset/UNetSoundOnly/
├── models/
│   └── binaural_attention_model.py     # Model architecture
│       - BinauralEncoder
│       - BinauralCrossAttention
│       - BinauralAttentionDepthNet
│
├── utils_binaural_attention_loss.py    # Loss functions
│   - BinauralAttentionLoss
│   - AdaptiveBinauralAttentionLoss
│
├── train_binaural_attention.py         # Training script
│
├── BINAURAL_ATTENTION_GUIDE.md         # This file
│
└── results/binaural_*/                 # Outputs
    ├── epoch_*.png
    └── ...
```

---

## 🔬 Ablation Studies

### 1. Effect of Attention Levels

Test which encoder levels benefit most from attention:

```bash
# Only deep levels
python train_binaural_attention.py --attention_levels 4 5

# Only mid levels  
python train_binaural_attention.py --attention_levels 2 3 4

# All levels
python train_binaural_attention.py --attention_levels 1 2 3 4 5
```

**Hypothesis:** Deep levels (4, 5) capture global correspondence, early levels (1, 2) capture local ITD/ILD.

### 2. Effect of Loss Components

```bash
# Reconstruction only
python train_binaural_attention.py --lambda_edge 0.0 --lambda_smooth 0.0

# + Edge-aware
python train_binaural_attention.py --lambda_edge 0.2 --lambda_smooth 0.0

# Full loss
python train_binaural_attention.py --lambda_edge 0.2 --lambda_smooth 0.1
```

### 3. Comparison with Single Encoder

To validate the benefit of separate encoders, compare:
- **This model**: Separate left/right encoders + attention
- **Baseline**: Single encoder on concatenated input

---

## 🐛 Troubleshooting

### Issue: Training is slow

**Solution:**
```bash
# Reduce attention levels
--attention_levels 3 4 5

# Reduce batch size and increase learning rate
--batch_size 32 --learning_rate 0.0015

# Use smaller model
--base_channels 48
```

### Issue: Out of memory

**Solution:**
```bash
# Reduce batch size
--batch_size 32

# Use smaller model
--base_channels 32

# Apply attention at fewer levels
--attention_levels 4 5
```

### Issue: Poor convergence

**Solution:**
```bash
# Use adaptive loss
--use_adaptive_loss

# Lower learning rate
--learning_rate 0.0005

# Use AdamW optimizer
--optimizer AdamW --weight_decay 0.01
```

### Issue: Blurry depth boundaries

**Solution:**
```bash
# Increase edge-aware loss
--lambda_edge 0.3

# Or use adaptive loss (automatically adjusts)
--use_adaptive_loss
```

---

## 📚 Related Work

### Computer Vision
- **Stereo Matching**: Cost-volume for disparity estimation
- **PSMNet**: Pyramid stereo matching network
- **GwcNet**: Group-wise correlation stereo network

### Audio
- **Binaural Audio**: ITD and ILD for sound localization
- **Cocktail Party Problem**: Separate sound sources using binaural cues
- **Spatial Audio**: 3D audio rendering

### Our Contribution
- First to apply **cross-attention** for binaural audio depth estimation
- Explicit modeling of **ITD/ILD** through attention
- **Multi-scale correspondence** for hierarchical spatial reasoning

---

## ✅ Quick Start Checklist

Before training:
- [ ] Dataset prepared (BatvisionV1/V2)
- [ ] GPU available (recommended: V100 or better)
- [ ] W&B configured (optional but recommended)
- [ ] Decided on attention levels (default: 2,3,4,5)
- [ ] Set experiment name

After training:
- [ ] Check visualizations (clear boundaries?)
- [ ] Compare with baseline (RMSE, ABS_REL, δ1)
- [ ] Analyze attention maps (do they make sense?)
- [ ] Test on held-out locations

---

## 🎯 Key Takeaways

1. **Explicit > Implicit**: Modeling binaural correspondence explicitly helps
2. **Multi-scale matters**: Attention at multiple levels captures different cues
3. **Curriculum learning**: Adaptive loss stabilizes training
4. **Computational trade-off**: Better performance but slower training

---

## 🚀 Next Steps

1. ✅ **Train baseline model**: Validate implementation
2. 🔜 **Ablation studies**: Which components are most important?
3. 🔜 **Attention visualization**: Visualize learned correspondences
4. 🔜 **Cross-dataset transfer**: Test on different datasets
5. 🔜 **Real-time optimization**: Reduce inference time

---

**Happy experimenting! 🎧**

W&B Dashboard: https://wandb.ai/branden/batvision-depth-estimation

For questions or issues, check the code comments or create an issue.

---

## 📖 Citation

If you use this model in your research:

```
This implementation introduces binaural cross-attention for audio-based
depth estimation, explicitly modeling inter-aural time and level differences
to improve spatial reasoning from binaural audio.
```


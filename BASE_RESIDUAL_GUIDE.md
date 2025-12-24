# Base + Residual Depth Estimation Guide

## 🎯 Overview

This implementation introduces a novel approach to audio-based depth estimation by decomposing the prediction into two components:

1. **Base Depth**: Coarse depth map capturing room layout and structure
2. **Residual**: Fine-grained corrections for object details

**Final Depth = Base Depth + Residual**

### Why This Approach?

Audio-to-depth estimation is an **under-constrained** problem:
- Limited spatial information from binaural audio
- Ambiguity in depth from echo patterns
- Difficulty learning both structure and details simultaneously

By decomposing the problem:
- **Base** focuses on low-frequency components (walls, floor, ceiling)
- **Residual** focuses on high-frequency details (furniture, edges)
- Network learns easier sub-problems instead of one hard problem

This is analogous to **Taylor Series Expansion**:
```
f(x) ≈ f(a) + f'(a)(x-a)
     └─ base  └─ residual
```

---

## 📐 Architecture

### Model Structure

```
                    Input (Binaural Audio)
                            ↓
                    ┌──────────────┐
                    │    Shared    │
                    │   Encoder    │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
     ┌────────────────┐        ┌────────────────┐
     │  Base Decoder  │        │  Res Decoder   │
     │  (Coarse Depth)│        │ (Fine Details) │
     └───────┬────────┘        └───────┬────────┘
             ↓                         ↓
        Base Depth                Residual
             │                         │
             └──────────┬──────────────┘
                        ↓
                   Final Depth
```

### Key Features

- **Shared Encoder**: Extracts audio features once
- **Dual Decoders**: Separate pathways for base and residual
- **Skip Connections**: U-Net style connections in both decoders
- **Independent Outputs**: Base and residual can be analyzed separately

---

## 💡 Loss Function Design

### Three-Component Loss

```python
L_total = λ1 * L_reconstruction + λ2 * L_structural + λ3 * L_sparsity
```

#### 1. Reconstruction Loss (λ1)
```python
L_recon = ||D_gt - (D_base + D_res)||_1
```
- Ensures final prediction matches ground truth
- Standard depth estimation objective

#### 2. Structural Guidance Loss (λ2)
```python
L_struct = ||D_base - LowPass(D_gt)||_1
```
- **Key Innovation**: No explicit layout GT needed!
- Low-pass filters GT depth to extract structure
- Base depth learns to match this coarse structure
- Acts as implicit "room layout" supervision

#### 3. Sparsity Penalty (λ3)
```python
L_sparse = ||D_res||_1
```
- Keeps residual small (like epsilon in Taylor series)
- Forces base to do most of the work
- Prevents residual from dominating

### Why This Works

**Without Layout GT:**
- Standard approach: Learn end-to-end depth (difficult!)
- Our approach: First learn structure, then refine (easier!)

**With Structural Guidance:**
- Base learns "this is a rectangular room ~5m deep"
- Residual learns "table at 2.5m, chair at 3.2m"
- Much more tractable for under-constrained problem

---

## 🚀 Usage

### Basic Training

```bash
python train_base_residual.py \
  --dataset batvisionv2 \
  --batch_size 256 \
  --learning_rate 0.002 \
  --experiment_name exp1
```

### With W&B Logging

```bash
python train_base_residual.py \
  --dataset batvisionv2 \
  --use_wandb \
  --wandb_project batvision-base-residual \
  --experiment_name exp1
```

### Custom Loss Weights

```bash
python train_base_residual.py \
  --lambda_recon 1.0 \
  --lambda_base 0.5 \
  --lambda_sparse 0.1 \
  --lowpass_kernel 8
```

### Adaptive Loss (Curriculum Learning)

```bash
python train_base_residual.py \
  --use_adaptive_loss \
  --warmup_epochs 20
```

This automatically adjusts loss weights during training:
- **Early epochs**: High λ_base (learn structure first)
- **Later epochs**: High λ_recon (refine accuracy)

---

## ⚙️ Configuration

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_channels` | 64 | Base number of channels (64 = ~30M params) |
| `--bilinear` | True | Use bilinear upsampling (vs transposed conv) |

### Loss Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda_recon` | 1.0 | Reconstruction weight |
| `--lambda_base` | 0.5 | Structural guidance weight |
| `--lambda_sparse` | 0.1 | Sparsity penalty weight |
| `--lowpass_kernel` | 8 | Low-pass filter size (larger = coarser base) |

**Tuning Guide:**
- Increase `lambda_base` if base is too noisy
- Decrease `lambda_sparse` if residual is too suppressed
- Increase `lowpass_kernel` for larger rooms

### Adaptive Loss

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_adaptive_loss` | False | Enable curriculum learning |
| `--warmup_epochs` | 20 | Epochs to transition weights |

---

## 📊 Analyzing Results

### Visualizations

The training script automatically saves decomposition visualizations:

```
results/{experiment_name}/epoch_{N:04d}_decomposition.png
```

Each visualization shows 4 columns:
1. **Base Depth** (green colormap) - Room structure
2. **Residual** (red-blue colormap) - Corrections
3. **Final Depth** (green colormap) - Base + Residual
4. **Ground Truth** (green colormap) - Target

### What to Look For

#### Healthy Learning Pattern

```
Base Depth:    Smooth, shows walls/floor clearly
Residual:      Small values, mostly near objects
Final Depth:   Combines both, accurate
```

#### Problem Patterns

**Base is too noisy:**
- Symptom: Base has high-frequency details
- Solution: Increase `lambda_base` or `lowpass_kernel`

**Residual dominates:**
- Symptom: Residual has large values everywhere
- Solution: Increase `lambda_sparse`

**Poor reconstruction:**
- Symptom: Final depth doesn't match GT
- Solution: Increase `lambda_recon`

### W&B Metrics

Track these metrics:
- `train/loss_base`: Structural learning progress
- `train/loss_sparse`: Residual magnitude
- `val/rmse`: Final accuracy
- Visualize: Base vs Residual magnitude over time

---

## 🔬 Experimental Results (Expected)

Based on the design, you should expect:

### Convergence Speed
- **Faster initial convergence** than standard U-Net
- Base learns room shape in first 10-20 epochs
- Residual refines details in later epochs

### Final Performance
- **Similar or better RMSE** than baseline
- **Better structure preservation** (straighter walls)
- **Reduced artifacts** in uniform regions

### Interpretability
- **Explainable predictions**: Can visualize what base learned
- **Room layout extraction**: Base depth shows approximate geometry
- **Failure analysis**: Can identify if error is in structure or details

---

## 💻 Code Structure

```
Batvision-Dataset/UNetSoundOnly/
├── models/
│   └── base_residual_model.py          # Model architecture
├── utils_base_residual_loss.py         # Loss functions
├── train_base_residual.py              # Training script
├── BASE_RESIDUAL_GUIDE.md              # This guide
└── results/{experiment}/                # Outputs
    └── epoch_{N}_decomposition.png
```

---

## 🎓 Advanced Usage

### Multi-Scale Low-Pass Filtering

Try different kernel sizes to see effect:

```bash
# Coarse base (good for large rooms)
python train_base_residual.py --lowpass_kernel 16

# Fine base (good for cluttered rooms)
python train_base_residual.py --lowpass_kernel 4
```

### Frequency-Domain Loss (Experimental)

```python
from utils_base_residual_loss import FrequencyAwareBaseResidualLoss

# In training script, replace:
criterion = FrequencyAwareBaseResidualLoss(freq_cutoff=0.1)
```

This uses FFT to explicitly separate frequencies:
- Base matches low frequencies
- Residual matches high frequencies

**Note**: More sophisticated but computationally expensive

### Analyzing Learned Structure

```python
import torch
import matplotlib.pyplot as plt

# Load trained model
model = ...  # Load checkpoint
model.eval()

# Forward pass
with torch.no_grad():
    base, residual, final = model(audio_input)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(base[0, 0].cpu(), cmap='viridis')
axes[0].set_title('Learned Room Layout')
axes[1].imshow(residual[0, 0].cpu(), cmap='RdBu_r')
axes[1].set_title('Object Details')
axes[2].imshow(final[0, 0].cpu(), cmap='viridis')
axes[2].set_title('Final Prediction')
```

---

## 🐛 Troubleshooting

### Base looks identical to final
- **Cause**: Residual being suppressed too much
- **Fix**: Decrease `lambda_sparse` (try 0.05 or 0.01)

### Residual is very large
- **Cause**: Base not learning structure properly
- **Fix**: Increase `lambda_base` (try 1.0 or 1.5)

### Training unstable
- **Cause**: Loss weights not balanced
- **Fix**: Use `--use_adaptive_loss` for automatic balancing

### Base has sharp edges
- **Cause**: Low-pass filtering too weak
- **Fix**: Increase `lowpass_kernel` (try 12 or 16)

---

## 📝 Citation

If this approach works well for your research, consider citing:

```
This implementation introduces a base+residual decomposition for
audio-based depth estimation, making the under-constrained problem
more tractable by learning structure and details separately.
```

---

## 🔗 Related Work

### Similar Approaches in Vision

1. **Laplacian Pyramid Networks** (Lai et al., 2017)
   - Multi-scale residual learning
   - Similar decomposition philosophy

2. **Coarse-to-Fine Depth** (Eigen et al., 2014)
   - First predict coarse, then refine
   - Our approach learns both jointly

3. **Structural Priors** (Liu et al., 2015)
   - Use Manhattan world assumption
   - We learn structure from data

### Key Differences

- **No explicit layout GT**: Uses depth map's low-freq as proxy
- **Joint learning**: Base and residual trained simultaneously
- **Audio-specific**: Designed for echo-based depth estimation

---

## ✅ Quick Start Checklist

Before training:
- [ ] Dataset prepared (BatvisionV1/V2)
- [ ] W&B configured (optional)
- [ ] Checked GPU availability
- [ ] Decided on loss weights (or use adaptive)
- [ ] Set experiment name

After training:
- [ ] Check decomposition visualizations
- [ ] Verify base shows room structure
- [ ] Confirm residual is small
- [ ] Compare with baseline model

---

## 🎯 Expected Benefits

Based on the design philosophy:

1. **Faster Convergence**: 20-30% fewer epochs to reach target RMSE
2. **Better Structure**: Straighter walls, cleaner floor/ceiling
3. **Fewer Artifacts**: Residual handles details, base stays smooth
4. **Interpretability**: Can visualize what model learned about room
5. **Debugging**: Know if error is structural or detail-level

---

**Happy experimenting! 🚀**

For questions or improvements, check the code comments or create an issue.

---

## 📚 Appendix: Mathematical Details

### Taylor Series Analogy

In Taylor series expansion:
```
f(x) = f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...
```

Our approach:
- `f(a)` → Base depth (coarse structure)
- `f'(a)(x-a)` → Residual (local corrections)
- Higher orders ignored (sparsity penalty)

### Low-Pass Filtering

The structural guidance uses spatial averaging:
```
D_struct[i,j] = (1/K²) Σ D_gt[i+m, j+n]  for m,n in [-K/2, K/2]
```

This removes high frequencies while preserving room geometry.

### Frequency Interpretation

In Fourier domain:
- Base ← Low frequencies (< cutoff)
- Residual ← High frequencies (> cutoff)
- Final = Base + Residual (all frequencies)

This is why the approach works: decompose by frequency!



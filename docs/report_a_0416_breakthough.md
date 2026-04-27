# Performance Improvement Report — FOA-Depth Baseline
**Date:** 2026-04-16  
**Scope:** Full codebase audit of `/root/storage/implementation/shared_audio/baseline/`  
**Objective:** Identify concrete, actionable strategies to push depth estimation performance beyond current SOTA (RMSE 1.0803) using Ambisonics

---

## 1. Current State of Affairs

### 1.1 Best Results (Test Set — 9 scenes, 3192 samples)

| Rank | Model | Exp | RMSE | ABS_REL | Delta1 | Key Config |
|------|-------|-----|------|---------|--------|------------|
| 1 | FOA (UNet) | exp49 | **1.0803** | 0.4631 | 0.5023 | lr=1e-3, dw=1.0, fw=0.1, hw=0.05 |
| 2 | Baseline UNet | exp01 | 1.0817 | 0.4553 | 0.5031 | lr=1e-3, bs=32 |
| 3 | Pretrained ViT-B/16 | exp65 | 1.0818 | 0.4467 | 0.4959 | lr=3e-5, bs=16 |
| 4 | BatVision | exp74 | 1.0817 | 0.4652 | 0.4944 | lr=1e-3, bs=16 |
| 5 | PreViT+FOA v3 | exp164 | 1.0929 | 0.4372 | 0.4972 | lr=1e-4, lambda_sh=0.1 |
| 6 | PreViT+FOA v1 | exp160 | 1.0820 | 0.4619 | 0.4948 | lr=1e-4, lambda_sh=0.1 |

### 1.2 Key Observations from 83 Experiments

1. **Top 3 models are within 0.0015 RMSE** — the ceiling is real with current approach.
2. **FOA guidance helps marginally** (FOA exp49 beats baseline by 0.0014 RMSE). The auxiliary SH branch is validated but the gain is tiny.
3. **25 of 83 experiments failed** — all advanced FOA variants (CrossAttn, FeatBank, MSAttn, ChannelAttn, FOAv2) crashed due to dimension mismatches. These architectures were never properly evaluated.
4. **No learning rate schedule used** — constant LR throughout training. Massive missed opportunity.
5. **No data augmentation at all** — no time/frequency masking, no noise injection, no channel swaps.
6. **FOA is underutilized** — only 4-dim RMS vector supervised; the 256x512 energy map is used only in histogram alignment loss, never as direct spatial input.
7. **PreViT+FOA v3 (FiLM conditioning) achieves best ABS_REL (0.4369-0.4372)** despite higher RMSE, suggesting directional cues help relative accuracy.

---

## 2. Root Cause Analysis — Why Performance Plateaus

### 2.1 Information Bottleneck in Audio Representation

**Problem:** The binaural spectrogram (2, 256, 512) is a low-information-density input. Two-channel audio from a single emission point captures limited geometric detail compared to visual inputs.

**Evidence:**
- Baseline UNet (no FOA, no pretraining) already achieves RMSE 1.0817
- Adding FOA as auxiliary loss only reduces RMSE by 0.0014
- PreViT with ImageNet pretraining achieves RMSE 1.0818 — transfer learning adds almost nothing

**Diagnosis:** The encoder is already extracting most recoverable information from the 2-channel spectrogram. The bottleneck is **input information**, not model capacity.

### 2.2 FOA Used as Regularizer, Not as Input

**Problem:** FOA energy is only used to regularize the latent space (auxiliary loss). The spatial structure in Ambisonics is never fed as direct input to the network.

**Current pipeline:**
```
Binaural spectrogram (2ch) → Encoder → Bottleneck → Decoder → Depth
                                          ↓
                                    SH Head → FOA loss (aux only)
```

**What's missing:** The 4-channel FOA IR and the directional energy map contain spatial information that the binaural spectrogram alone cannot provide — specifically, elevation cues (Z channel) and omnidirectional energy (W channel) that binaural recordings lose.

### 2.3 Training Infrastructure Gaps

| Issue | Impact | Evidence |
|-------|--------|----------|
| No LR schedule | Training plateau at epoch 12-16 | Validation loss flattens in all >40ep runs |
| No augmentation | Overfitting to acoustic conditions | Val-test gap of ~0.15 RMSE (favorable) suggests low variance but limited generalization |
| Constant 40 epochs | Undertraining some models | 60-epoch FOA 0415 runs showed continued improvement |
| No gradient accumulation | Small effective BS for ViT | ViT limited to BS=16 by memory |
| BerHu threshold c=0.2*max | Threshold adapts to outliers, unstable | max_error varies per batch |

---

## 3. Recommended Improvements — Prioritized

### Priority 1: FOA as Direct Input (Expected Impact: RMSE -0.05 to -0.10)

**The single highest-impact change.** Instead of using FOA only as auxiliary supervision, feed the ambisonic information directly into the network.

#### 3.1a Multi-Channel Input Fusion

Replace the 2-channel binaural input with a 6-channel tensor:

```python
# Current: input = spectrogram(binaural)  →  (B, 2, 256, 512)
# Proposed: input = concat(spectrogram(binaural), energy_map, foa_channels)
#   → (B, 2 + 1 + 3, 256, 512) = (B, 6, 256, 512)
```

Where:
- Channels 0-1: Binaural spectrogram (existing)
- Channel 2: Covariance-based energy map (directional energy, already computed in dataset.py)
- Channels 3-5: Spatial FOA spectrograms from X, Y, Z channels

**Implementation:** Modify `UnetGenerator.__init__` to accept `input_nc=6` and compute per-channel spectrograms of the FOA IR in `dataset.py.__getitem__`.

**Why this works:** The energy map is a 256x512 spatial prior that tells the network *where* sound energy comes from. This is exactly the directional information that depth correlates with — nearby surfaces reflect more energy.

#### 3.1b FOA Spectrogram Input (4-Channel)

Compute full spectrograms from FOA channels instead of just RMS:

```python
# In dataset.py, alongside binaural spectrogram:
foa_ir = np.load(ambi_path)  # (4, T)
foa_spec = []
for ch in range(4):
    spec = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=160)(
        torch.from_numpy(foa_ir[ch:ch+1])
    )
    foa_spec.append(spec)
foa_spec = torch.cat(foa_spec, dim=0)  # (4, F, T)
# Resize to (4, 256, 512)
```

Feed as 6-channel input: `(2 binaural + 4 FOA) = 6 channels`  
Or separate dual-encoder: binaural encoder + FOA encoder → fusion at bottleneck.

#### 3.1c Dual-Encoder Architecture

```
Binaural (2ch) → Encoder_A → Features_A ─┐
                                           ├→ Fusion → Decoder → Depth
FOA spec (4ch) → Encoder_B → Features_B ─┘
```

This is architecturally similar to the RGB-teacher approach (`foa_v2_js_rgb`) but replaces the RGB encoder with a FOA encoder that runs at both train and test time.

**Key advantage over current approach:** The FOA encoder captures *spectral* directional information, not just RMS statistics. The current 4-dim FOA target (RMS per channel) collapses all temporal/frequency information into a single scalar per channel.

---

### Priority 2: Training Recipe Improvements (Expected Impact: RMSE -0.02 to -0.05)

#### 3.2a Learning Rate Schedule

**Current:** Constant LR throughout training.  
**Proposed:** Cosine annealing with warmup.

```python
# In train.py, after optimizer creation:
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=epochs - 5, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])
```

**Evidence:** All training runs show validation plateau at epoch 12-16. A decaying LR would allow finer convergence in later epochs instead of oscillating around the minimum.

#### 3.2b Data Augmentation for Audio

Currently there is **zero augmentation**. Add:

1. **SpecAugment** (frequency/time masking):
   ```python
   # Mask random frequency bands (imitates room mode variation)
   freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=30)
   time_mask = torchaudio.transforms.TimeMasking(time_mask_param=50)
   ```

2. **Channel swap** (left-right flip with depth horizontal flip):
   ```python
   if random.random() > 0.5:
       audio = audio.flip(0)       # swap L/R channels
       depth = depth.flip(-1)      # horizontal flip depth
       # Also negate Y channel in FOA (lateral dipole)
   ```

3. **Gaussian noise injection** (SNR 20-40 dB):
   ```python
   noise = torch.randn_like(audio) * audio.std() * 10**(-snr_db/20)
   audio = audio + noise
   ```

4. **Random gain** (amplitude scaling ±3 dB):
   ```python
   gain = 10 ** (random.uniform(-3, 3) / 20)
   audio = audio * gain
   ```

#### 3.2c Longer Training with Early Stopping

- Increase to **80-100 epochs** with cosine LR
- Implement proper early stopping (patience=15 on validation score)
- The 60-epoch FOA 0415 runs showed continued improvement over 40-epoch runs

#### 3.2d Gradient Accumulation for Larger Effective Batch Size

ViT models are limited to BS=16 due to GPU memory. Use gradient accumulation:

```python
accumulation_steps = 4  # effective BS = 16 * 4 = 64
for i, batch in enumerate(loader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### Priority 3: Architectural Improvements (Expected Impact: RMSE -0.02 to -0.04)

#### 3.3a Fix Failed FOA Variants

25 experiments failed due to dimension mismatches. The architectures (CrossAttn, FeatBank, MSAttn, ChannelAttn, FOAv2) were never evaluated. These should be debugged and tested — they may contain the best ideas.

**Specific failures to fix:**
- `foa_crossattn`: Gradient flow issues in cross-attention between FOA tokens and depth features
- `foa_v2`: BatchNorm dimension mismatch in FiLM conditioning layer
- `foa_msattn`, `foa_featbank`, `foa_channelattn`: Feature dimension incompatibility

#### 3.3b FiLM Conditioning (ViT FOA v3 Style) on UNet

The PreViT+FOA v3 architecture (early-layer SH tap + FiLM conditioning) achieved the **best ABS_REL (0.4369)** across all experiments. Port this to the UNet backbone:

```python
# After SH head predicts FOA coefficients from bottleneck:
gamma, beta = self.film_proj(foa_latent).chunk(2, dim=-1)  # (B, C)
# Modulate decoder features at each skip connection:
h = h * (1 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
```

This injects directional information back into the depth decoder, unlike the current auxiliary-only approach.

#### 3.3c Multi-Scale SH Tapping (ViT FOA v4 Style)

Extract SH predictions from multiple encoder depths, not just the bottleneck:

```python
# Tap features at encoder layers 2, 4, 6, 8 (UNet) or blocks 2, 5, 8, 11 (ViT)
sh_preds = [sh_head_i(features_i) for i, features_i in enumerate(tapped_features)]
sh_final = self.sh_mix(torch.cat(sh_preds, dim=-1))  # learnable blend
```

**Rationale:** Early layers capture low-level acoustic cues (ILD, phase), while deeper layers capture semantic structure. Multi-scale SH prediction captures both.

#### 3.3d Encoder Upgrade: Replace UNet with Pretrained Audio Encoder

The current UNet encoder trains from scratch on ~23K samples. Use a pretrained audio backbone:

1. **AudioMAE** (masked autoencoder pretrained on AudioSet): Provides strong spectrogram features
2. **BEATs** (audio pre-training with acoustic tokenizers): SOTA on audio understanding
3. **AST** (Audio Spectrogram Transformer): Directly processes spectrograms with ViT

```python
# Replace UNet encoder with frozen/finetuned AudioMAE:
class AudioMAEDepthDecoder(nn.Module):
    def __init__(self):
        self.encoder = AudioMAE.from_pretrained("audiomae-base")
        self.decoder = DepthDecoder(embed_dim=768)  # reuse existing decoder
```

**Why:** ImageNet-pretrained ViT (exp65) already matches the UNet despite domain mismatch. An *audio*-pretrained encoder should transfer far better.

---

### Priority 4: Ambisonics-Specific Innovations (Expected Impact: RMSE -0.03 to -0.08)

#### 3.4a Higher-Order Ambisonics (HOA) Input

Currently only FOA (order 1, 4 channels) is used. The SoundSpaces simulator can provide higher-order Ambisonics:

| SH Order | Channels | Angular Resolution |
|----------|----------|--------------------|
| 1 (FOA) | 4 | ~90 deg |
| 2 | 9 | ~45 deg |
| 3 | 16 | ~30 deg |
| 4 | 25 | ~22.5 deg |
| 5 | 36 | ~18 deg |

**Action:** Generate HOA impulse responses from SoundSpaces (if available) or compute from the Room Impulse Response (RIR). The existing `sh_basis.py` already supports up to order 5.

FOA 0415 v5 (sh_dim=36) already predicts order-5 SH from the bottleneck — but only the first 4 dims are supervised. With HOA ground truth, all 36 coefficients can be supervised, providing much richer directional guidance.

#### 3.4b Intensity Vector Input

Compute the acoustic intensity vector from FOA channels:

```python
# Active intensity vector (direction of energy flow):
W, Y, Z, X = foa_ir[0], foa_ir[1], foa_ir[2], foa_ir[3]
Ix = W * X  # front-back intensity
Iy = W * Y  # left-right intensity  
Iz = W * Z  # up-down intensity

# Direction of Arrival (DoA):
doa_azimuth = np.arctan2(Iy, Ix)
doa_elevation = np.arctan2(Iz, np.sqrt(Ix**2 + Iy**2))
```

This produces per-time-frame directional estimates that can be:
1. Converted to a DoA spectrogram (azimuth/elevation vs. time/frequency)
2. Used as additional input channels
3. Used as auxiliary supervision targets

**Key insight:** The intensity vector encodes *where* reflections come from at each frequency. This is a much richer signal than the 4-dim RMS vector currently used.

#### 3.4c Directional Energy Map as Spatial Attention

Use the covariance-based energy map (already computed in `dataset.py`) as a spatial attention prior:

```python
# energy_map: (B, 1, 256, 512) — where sound energy comes from
# Use as attention weight on decoder features:
spatial_attention = torch.sigmoid(self.attn_conv(energy_map))  # (B, 1, H, W)
decoder_out = decoder_out * spatial_attention + decoder_out  # residual attention
```

**Why:** The energy map is the most geometrically informative signal — it directly indicates which directions have reflecting surfaces (= close geometry). Currently it's only used in a scalar histogram alignment loss, wasting its spatial structure.

#### 3.4d Ambisonics-Conditioned Diffusion

Combine EchoDiffusion with FOA conditioning:

```python
# During diffusion denoising, condition on FOA latent:
class FOAConditionedDiffusion(EchoDiffusion):
    def forward(self, x_t, t, audio_feat, foa_feat):
        # Inject FOA features via cross-attention at each denoising step
        cond = self.foa_proj(foa_feat)  # (B, D)
        return self.denoise(x_t, t, audio_feat, cond)
```

EchoDiffusion (exp14, RMSE=1.0908) underperforms the UNet baseline, likely because it lacks directional priors. Adding FOA conditioning to the denoising process could improve spatial coherence.

---

### Priority 5: Loss Function Improvements (Expected Impact: RMSE -0.01 to -0.03)

#### 3.5a Edge-Aware Depth Loss

Add a gradient-matching term to preserve depth discontinuities:

```python
def gradient_loss(pred, gt):
    pred_dx = pred[:,:,:,1:] - pred[:,:,:,:-1]
    pred_dy = pred[:,:,1:,:] - pred[:,:,:-1,:]
    gt_dx = gt[:,:,:,1:] - gt[:,:,:,:-1]
    gt_dy = gt[:,:,1:,:] - gt[:,:,:-1,:]
    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)
```

**Note:** `foa_v2.py` already has gradient consistency loss between depth and FOA energy, but it crashed before being evaluated. The idea is sound — implement it properly.

#### 3.5b Multi-Scale SSIM Loss

Add structural similarity at multiple scales for perceptual depth quality:

```python
from pytorch_msssim import ms_ssim
L_msssim = 1 - ms_ssim(pred_depth, gt_depth, data_range=1.0, size_average=True)
L_total = L_depth + 0.1 * L_msssim
```

#### 3.5c Uncertainty-Weighted Multi-Task Loss

Instead of hand-tuning depth_weight, foa_weight, hist_weight, learn task weights:

```python
# Kendall et al., "Multi-Task Learning Using Uncertainty"
log_var_depth = nn.Parameter(torch.zeros(1))
log_var_foa = nn.Parameter(torch.zeros(1))
L = (1 / (2 * torch.exp(log_var_depth))) * L_depth + log_var_depth / 2 \
  + (1 / (2 * torch.exp(log_var_foa))) * L_foa + log_var_foa / 2
```

**Evidence:** 20 FOA experiments swept dw/fw/hw manually. The best (exp49, hw=0.05) was found by grid search. Learned weights could find better balances automatically.

---

## 4. Ambisonics-Specific Research Directions

### 4.1 Time-Frequency FOA Analysis

**Current limitation:** FOA targets are computed as per-channel RMS — a single scalar per channel. This collapses all time-frequency information.

**Proposed:** Compute full FOA spectrograms (4 channels x F x T) and use them as:
1. **Input channels** (Priority 1 above)
2. **Supervision targets** — predict per-frequency FOA coefficients, not just RMS
3. **Cross-modal attention keys** — FOA spectrogram provides spatial grounding for each time-frequency bin

### 4.2 Canonical Frame Alignment Analysis

`SoundSpacesDatasetRotated` applies yaw rotation to align FOA to ego-centric frame (view_mod = idx % 4). This is critical but:
- Only affects Y and X channels (azimuth)
- Z channel (elevation) is untouched — correct for yaw-only rotation
- **Question:** Are there pitch/roll variations in the dataset? If so, full 3D rotation is needed.

The FOA 0415 series uses `rotate_canonical=true` and achieves strong validation scores (ABS_REL ~0.38), confirming rotation helps. **All future experiments should use canonical rotation.**

### 4.3 Reflection-Aware Temporal Windowing

**Current:** Audio is truncated to first 20m of sound travel (`cut = int((2 * 20.0 / 340) * sr)`).

**Proposed:** Use multiple temporal windows to capture different reflection orders:
- **0-2ms:** Direct sound (source position)
- **2-10ms:** Early reflections (nearest surfaces)
- **10-50ms:** Late reflections (room geometry)

```python
windows = [(0, 96), (96, 480), (480, 2400)]  # samples at 48kHz
multi_spec = []
for start, end in windows:
    windowed = audio[:, start:end]
    spec = spectrogram_transform(windowed)
    multi_spec.append(spec)
input = torch.cat(multi_spec, dim=0)  # (6, F, T) for 2ch * 3 windows
```

Early reflections carry the most geometric information. Late reverb carries room-level statistics. Separating them lets the network weight appropriately.

### 4.4 Spatial Coherence Loss

Enforce that predicted depth is spatially consistent with FOA intensity direction:

```python
# If FOA says energy comes from direction (az, el),
# the depth at that pixel should be small (close surface)
intensity_direction = compute_intensity_vector(foa_ir)
depth_at_direction = sample_depth_at_direction(pred_depth, intensity_direction)
# Loss: energy-weighted depth should be low
L_coherence = (energy_weights * depth_at_direction).mean()
```

This is a physics-informed loss that directly encodes the relationship: **strong reflections come from nearby surfaces**.

---

## 5. Implementation Priority Roadmap

### Phase 1: Quick Wins (1-2 days)

| Action | Expected RMSE Gain | Effort |
|--------|-------------------|--------|
| Add cosine annealing LR scheduler | -0.01 to -0.02 | 10 lines in train.py |
| Add SpecAugment (freq/time masking) | -0.005 to -0.01 | 15 lines in dataset.py |
| Add channel-swap augmentation | -0.005 to -0.01 | 10 lines in dataset.py |
| Increase training to 80 epochs | -0.005 to -0.01 | Config change |
| Fix FOAv2 BatchNorm dimension bug | Unlocks FOAv2 evaluation | Debug models/foa_v2.py |

### Phase 2: FOA as Input (3-5 days)

| Action | Expected RMSE Gain | Effort |
|--------|-------------------|--------|
| Compute FOA spectrograms in dataset | Foundation for all below | dataset.py modification |
| 6-channel input (binaural + FOA specs) | -0.03 to -0.05 | Model input_nc change |
| Energy map as spatial attention | -0.02 to -0.04 | New attention module |
| FiLM conditioning on UNet decoder | -0.01 to -0.03 | Port from pretrained_vit_foa_v3.py |

### Phase 3: Architecture Upgrade (1-2 weeks)

| Action | Expected RMSE Gain | Effort |
|--------|-------------------|--------|
| Dual-encoder (binaural + FOA) | -0.03 to -0.06 | New model class |
| AudioMAE/BEATs pretrained encoder | -0.03 to -0.05 | Integration + finetuning |
| Multi-scale SH tapping on UNet | -0.01 to -0.02 | Port from pretrained_vit_foa_v4.py |
| Intensity vector as input/supervision | -0.02 to -0.04 | Dataset + loss modification |

### Phase 4: Advanced (2-4 weeks)

| Action | Expected RMSE Gain | Effort |
|--------|-------------------|--------|
| HOA generation (order 3-5) | -0.02 to -0.05 | Data pipeline + SH supervision |
| Temporal windowing (early/late reflections) | -0.02 to -0.04 | Dataset modification |
| FOA-conditioned diffusion | -0.02 to -0.04 | New model architecture |
| Learned multi-task loss weights | -0.005 to -0.01 | Loss function modification |

---

## 6. Experiment Design Recommendations

### 6.1 Suggested Next Experiments (exp170+)

```
exp170: Baseline UNet + cosine LR + SpecAugment + 80ep
exp171: FOA UNet (exp49 config) + cosine LR + SpecAugment + 80ep
exp172: FOA UNet + 6ch input (binaural + FOA specs)
exp173: FOA UNet + 6ch input + energy map attention
exp174: FOA UNet + 6ch input + FiLM decoder conditioning
exp175: Dual encoder (binaural UNet + FOA UNet → fusion)
exp176: PreViT + FOA 6ch input
exp177: PreViT + FOA v3 FiLM + cosine LR + 80ep
exp178: AudioMAE encoder + depth decoder
exp179: AudioMAE encoder + FOA auxiliary + FiLM
```

### 6.2 Ablation Controls

Every experiment should report:
1. Standard depth metrics (RMSE, ABS_REL, Delta1/2/3, Log10, MAE)
2. FOA metrics (FOA_L1, FOA_COS, FOA_DIR)
3. Training curve: validation RMSE per epoch (watch for plateau timing)
4. Per-scene breakdown (some scenes are inherently harder)

### 6.3 Hyperparameter Ranges

Based on 83 experiments, the optimal ranges are:

| Parameter | Optimal Range | Evidence |
|-----------|--------------|----------|
| Learning rate (UNet) | 5e-4 to 1e-3 | exp01, exp02, exp49 all in this range |
| Learning rate (ViT) | 3e-5 to 1e-4 | exp61, exp65 |
| Batch size | 16-32 | BS=32 generally better for UNet |
| depth_weight | 1.0 | exp47 (dw=2.0) ≈ exp36 (dw=1.0) |
| foa_weight | 0.05-0.1 | exp48 (fw=0.05) competitive |
| hist_weight | 0.05 | exp49 (hw=0.05) best; hw=0.1+ hurts |
| lambda_sh (0415) | 0.1-0.3 | exp160-165 |
| Epochs | 60-100 | 40 undertrained; 60 better; 80+ with LR schedule |

---

## 7. Summary

The current FOA-Depth system achieves RMSE 1.0803, barely edging out the vanilla baseline (1.0817). The **fundamental limitation** is that Ambisonics information is used only as auxiliary supervision on a 4-dimensional RMS vector — discarding 99.9% of the spatial-spectral information in the FOA impulse response.

The path to significant improvement:

1. **Feed FOA spectrograms as direct input** — the single most impactful change
2. **Fix training recipe** — LR scheduling, augmentation, longer training
3. **Use spatial information structurally** — energy maps as attention, FiLM conditioning, intensity vectors
4. **Leverage pretrained audio encoders** — AudioMAE/BEATs instead of training from scratch

The gap between using FOA as a 4-dim regularizer vs. using it as a rich spatial input signal is where the next 0.05-0.10 RMSE improvement lives.

---

*Report generated from analysis of 83 experiments, 292 log files, 25 config files, and full codebase review.*

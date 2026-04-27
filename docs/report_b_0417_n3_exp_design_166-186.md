# N3 Experiment Design: FOA Representation & Oracle Analysis
**Date:** 2026-04-17  
**Experiments:** exp166-186 (20 trials, 60 epochs each)  
**Scripts:** `n3_bulk_train.sh`, `n3_bulk_test.sh`  
**Models:** `models/n3_0417/`

---

## 1. Prioritized Recommendation

### Run Order (highest information gain first)

| Priority | Exp | Group | Architecture | Why First |
|----------|-----|-------|-------------|-----------|
| 1 | 178-180 | **D1** | Oracle: binaural+GT energy | Answers "does energy map help at all?" — gates all C experiments |
| 2 | 181-183 | **D5** | Oracle: GT energy only | Answers "can energy alone predict depth?" — fundamental signal question |
| 3 | 184-185 | **D/baseline** | foa_0415_v1 at 60ep | Fair comparison baseline for all N3 models |
| 4 | 166-168 | **C2** | FiLM conditioning | Simplest architectural change, minimal risk |
| 5 | 172-174 | **C3** | Energy attention | Tests spatial use of FOA without changing encoder |
| 6 | 169-171 | **C1** | Multi-scale SH | Tests if early encoder layers carry directional signal |
| 7 | 175-177 | **C5** | Temporal windowing | Tests reflection structure, 3x encoder cost |
| 8 | 186 | **C3+** | Energy attn strong SH | Ablation on SH weight |

**Rationale:** Group D experiments (oracle) should run FIRST because they answer a gating question: if GT energy map doesn't help, then the problem is not "how to inject FOA" but "FOA doesn't contain enough depth signal." This changes the entire research direction.

---

## 2. Experiment Details

### Group C: Deployable (binaural-only inference)

#### C2: FiLM Conditioning (exp166-168)

**Model:** `N3FiLMGenerator` (`models/n3_0417/n3_film.py`)  
**Config:** `config/n3_film.yaml`

**Motivation:** Current auxiliary SH supervision is "fire and forget" — the predicted SH never feeds back into depth decoding. FiLM injects the SH prediction as a channel-wise affine modulation on the first decoder feature, creating a direct information pathway from directional cues to depth.

**Architecture change vs v1:**
```
v1:  bottleneck → SH head → pred_sh (loss only, no decoder feedback)
                → decoder → pred_depth

FiLM: bottleneck → SH head → pred_sh → FiLMProjector → (gamma, beta)
                 → decoder[0] → h * (1+gamma) + beta → rest of decoder → pred_depth
```

**New module:** `FiLMProjector(sh_dim=4, film_hidden=128, feat_channels=512)`
- Parameters: 4→128→1024 = ~132K params (0.2% of total)
- Applied after first decoder step where features are (B, 512, 2, 4)

**Expected effect:** RMSE -0.01 to -0.03 if SH carries useful signal  
**Likely failure mode:** If SH prediction is noisy early in training, FiLM may destabilize decoder. Mitigated by (1+gamma) residual form.  
**Ablation:** Compare exp166 vs exp184 (same LR, with/without FiLM)

| Exp | LR | lambda_sh | Compare to |
|-----|-----|-----------|------------|
| 166 | 1e-3 | 0.1 | exp184 (v1 baseline) |
| 167 | 5e-4 | 0.1 | exp185 (v1 baseline) |
| 168 | 1e-3 | 0.3 | exp166 (stronger SH → better FiLM?) |

---

#### C1: Multi-Scale SH Heads (exp169-171)

**Model:** `N3MultiScaleSHGenerator` (`models/n3_0417/n3_multiscale_sh.py`)  
**Config:** `config/n3_multiscale_sh.yaml`

**Motivation:** The bottleneck-only SH prediction may miss spatial cues that are better captured at earlier encoder stages (e.g., ILD at lower resolutions, phase patterns at higher resolutions).

**Architecture change vs v1:**
```
v1:  enc[0..6] → bottleneck → SHHead → pred_sh

MSSH: enc[2] → SHHead_0 → sh_0  ─┐
      enc[4] → SHHead_1 → sh_1   ├→ cat → Linear → pred_sh
      enc[6] → SHHead_2 → sh_2   │
      bottleneck → SHHead_3 → sh_3 ─┘
```

**Tap channels:** enc[2]=256, enc[4]=512, enc[6]=512, bottleneck=512  
**Fusion:** `Linear(sh_dim*4, sh_dim)` = 64 extra params  
**New params:** 4 SHHeads (vs 1 in v1) ≈ +500K params

**Expected effect:** RMSE -0.005 to -0.02 — modest but interpretable  
**Likely failure mode:** If directional signal is only in bottleneck, extra heads add noise  
**Ablation:** Compare exp169 vs exp184. Also inspect per-scale SH accuracy (model outputs `pred_sh_scales`)

---

#### C3: Predicted Energy Attention (exp172-174)

**Model:** `N3EnergyAttnGenerator` (`models/n3_0417/n3_energy_attn.py`)  
**Config:** `config/n3_energy_attn.yaml`

**Motivation:** The energy map is the most geometrically informative FOA-derived signal (it shows WHERE sound energy comes from). Instead of using it only as a loss target, predict it from binaural features and use as spatial attention.

**Architecture change vs v1:**
```
v1:  bottleneck → decoder → pred_depth

EATTN: bottleneck → EnergyHead → pred_energy (B,1,H,W)
       bottleneck → decoder[0..4] → h
       h = h * (1 + energy_attn)  ← resize pred_energy to match h
       decoder[5..] → pred_depth
```

**EnergyHead:** 3 ConvTranspose2d layers (512→256→128→1, Sigmoid)  
**Attention form:** residual multiplicative `h * (1 + attn)` — safe, cannot zero out features  
**Applied at:** decoder step 5 (ngf*2=128 channels, spatial ≈64x128)

**Expected effect:** RMSE -0.01 to -0.03 if energy map is learnable from binaural  
**Likely failure mode:** If EnergyHead produces uniform maps, attention has no effect (safe failure)  
**Key diagnostic:** Visualize `pred_energy` — does it correlate with GT energy map?  
**Future:** Can explicitly supervise pred_energy with GT energy_map (exp186 tests stronger lambda_sh)

---

#### C5: Temporal Windowed Input (exp175-177)

**Model:** `N3TemporalWindowGenerator` (`models/n3_0417/n3_temporal_window.py`)  
**Config:** `config/n3_temporal_window.yaml`

**Motivation:** Different temporal segments of the impulse response encode different geometric information:
- Early (0-33%): direct sound + first reflections → nearest surfaces
- Mid (17-67%): early reverb → room geometry
- Late (50-100%): diffuse field → room volume

**Architecture change vs v1:**
```
v1:  x → encoder → bottleneck → decoder → pred_depth

TWIN: x → split into [w0, w1, w2] (overlapping, zero-padded to 512)
      w0 → shared_encoder → bn0 ─┐
      w1 → shared_encoder → bn1  ├→ cat → Linear(512*3, 512) → fused_bn
      w2 → shared_encoder → bn2 ─┘
      fused_bn → SH head → pred_sh
      fused_bn → decoder (skips from w0) → pred_depth
```

**Windows:** [0:170], [85:341], [256:512] — overlapping, zero-padded to maintain shape  
**Bottleneck fusion:** `Linear(512*3, 512)` applied per spatial position  
**Skip connections:** from window 0 only (most information-dense)

**Expected effect:** RMSE -0.01 to -0.03 if reflection separation helps  
**Likely failure mode:** 3x encoder cost. Zero-padding may dilute signal. Shared encoder may not specialize.  
**Ablation:** Compare exp175 vs exp184 (same LR)  
**Cost:** ~3x forward time (3 encoder passes), same decoder cost

---

### Group D: Oracle / Upper-Bound

#### D1: Binaural + GT Energy Map (exp178-180)

**Model:** `FOAOracleGenerator` with `input_nc=3`  
**Config:** `config/foa_oracle_nc3.yaml`

**Input:** `cat(binaural_spec [2ch], gt_energy_map [1ch]) = [3, 256, 512]`

**The critical question:** If we hand the model a perfect spatial energy prior, does depth improve significantly over binaural-only?

**Interpretation:**
- If RMSE drops >0.03: **Energy map is highly informative** → invest in better energy prediction (C3) or energy-guided architectures
- If RMSE drops 0.01-0.03: **Energy helps modestly** → useful as auxiliary signal, not transformative
- If RMSE drops <0.01: **Energy map is redundant** with binaural features → don't invest in energy-based approaches

---

#### D5: GT Energy Map Only (exp181-183)

**Model:** `FOAOracleGenerator` with `input_nc=1`  
**Config:** `config/foa_oracle_nc1.yaml`

**Input:** `gt_energy_map [1ch, 256, 512]` — NO binaural audio

**The diagnostic question:** Can the directional energy distribution alone predict depth?

**Interpretation:**
- If RMSE close to binaural baseline (~1.08): **Energy is informationally equivalent** to binaural for depth → implies FOA captures the essential geometry signal
- If RMSE significantly worse (>1.15): **Energy alone is insufficient** → binaural carries information (timing, spectrum) that energy map doesn't
- If RMSE better than binaural: **Unlikely** but would mean energy map is a superior representation

---

#### Baseline Controls (exp184-185)

**Model:** `FOA0415V1Generator` (unchanged v1)  
**Purpose:** Fair 60-epoch comparison at same settings as all N3 experiments

Existing v1 results (exp130-134) used 60 epochs but different LR/lambda_sh grid. These two runs at lr=1e-3/lambda_sh=0.1 and lr=5e-4/lambda_sh=0.1 provide direct comparison points.

---

## 3. Architecture Guidance

### Tensor Shapes (all models, BS=32)

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input (C models) | (32, 2, 256, 512) | Binaural spectrogram |
| Input (D1 oracle) | (32, 3, 256, 512) | Binaural + GT energy |
| Input (D5 oracle) | (32, 1, 256, 512) | GT energy only |
| Encoder features | [(32,64,128,256), ..., (32,512,2,4)] | 8 levels |
| Bottleneck | (32, 512, 1, 2) | After enc_inner |
| pred_sh | (32, 4) | SH coefficients |
| FiLM gamma/beta | (32, 512, 1, 1) | Broadcast spatially |
| pred_energy | (32, 1, 256, 512) | Energy attention map |
| pred_depth | (32, 1, 256, 512) | Final output |

### What to Keep Fixed for Fair Comparison

- Encoder: same UNet 8-level, ngf=64 (all models)
- Decoder: same skip-connection structure
- Depth loss: BerHu + SILog (w_berhu=1.0, w_silog=1.0)
- SH loss: L1 on first 4 dims of pred_sh
- Dataset: SoundSpaces with rotate_canonical=true
- Training: 60 epochs, AdamW, BS=32, validation every 4 epochs

---

## 4. Implementation Plan

### Files Created

| File | Purpose |
|------|---------|
| `models/n3_0417/__init__.py` | Module init |
| `models/n3_0417/n3_film.py` | FiLM conditioning (C2) |
| `models/n3_0417/n3_multiscale_sh.py` | Multi-scale SH (C1) |
| `models/n3_0417/n3_energy_attn.py` | Energy attention (C3) |
| `models/n3_0417/n3_temporal_window.py` | Temporal windowing (C5) |
| `models/n3_0417/n3_oracle.py` | Oracle model (D1/D5) |
| `config/n3_film.yaml` | Config for FiLM |
| `config/n3_multiscale_sh.yaml` | Config for multi-scale SH |
| `config/n3_energy_attn.yaml` | Config for energy attention |
| `config/n3_temporal_window.yaml` | Config for temporal window |
| `config/foa_oracle_nc3.yaml` | Config for D1 oracle |
| `config/foa_oracle_nc1.yaml` | Config for D5 oracle |
| `scripts/n3_bulk_train.sh` | Training script |
| `scripts/n3_bulk_test.sh` | Testing script |

### Files Modified

| File | Change |
|------|--------|
| `models/__init__.py` | Added N3 model imports |
| `utils/train_utils.py` | Added N3 models to `_FOA_0415_CLASSES`, oracle classes |
| `train.py` | Added `_train_step_oracle`, oracle dispatch |

### What Was NOT Modified

- `models/foa_0415_v1.py` through `v5.py` — original placeholders restored
- `data/dataset.py` — no changes (dataset returns needed 4-tuple already)
- `models/losses.py` — no changes
- `test.py` — no changes (oracle test uses same checkpoint path)

---

## 5. Interpretation Guide

### How to Read Group C vs Group D

```
If D1 (oracle) >> C_best:
  → FOA energy is useful but model can't infer it well from binaural
  → Invest in better energy prediction (improve EnergyHead, add explicit supervision)

If D1 (oracle) ≈ C_best ≈ baseline:
  → Energy map doesn't help even when given for free
  → FOA-based approaches may be at ceiling for this dataset

If C_best > baseline but D1 >> C_best:
  → C models partially extract FOA signal; room for improvement
  → Try combining best C variant with explicit energy supervision
```

### How to Compare Across C Variants

| Comparison | Interpretation |
|------------|----------------|
| FiLM > v1 baseline | SH → decoder feedback helps |
| MultiScale > v1 | Directional cues exist at multiple encoder depths |
| EnergyAttn > v1 | Spatial attention from predicted energy useful |
| TemporalWindow > v1 | Reflection separation helps (but check cost) |
| FiLM > EnergyAttn | Channel-wise modulation > spatial attention |
| EnergyAttn > FiLM | Spatial information > global directional summary |

### How to Compare Within Each Variant (3 settings per arch)

| Setting | Purpose |
|---------|---------|
| lr=1e-3, lsh=0.1 | Standard (matches best existing FOA settings) |
| lr=5e-4, lsh=0.1 | Tests if lower LR helps new modules converge |
| lr=1e-3, lsh=0.3 | Tests if stronger SH supervision improves auxiliary quality |

If lsh=0.3 >> lsh=0.1: the SH branch was under-supervised before.  
If lsh=0.3 << lsh=0.1: too much SH supervision hurts depth.

---

## 6. Experiments NOT Included (and Why)

| Idea from request | Status | Reason |
|-------------------|--------|--------|
| C4: Intensity vector supervision | Deferred | Requires new dataset preprocessing (compute intensity per frame); high implementation cost, medium information gain |
| C6: Frequency-wise modulation | Deferred | Interesting but lower priority than spatial approaches; add after C2/C3 results known |
| C7: Uncertainty-weighted loss | Deferred | Generic training trick, not FOA-specific; add later as orthogonal improvement |
| D2: Binaural + GT FOA spectrogram | Deferred | Requires computing per-channel spectrograms from FOA IR in dataset; implement after D1 results known |
| D3: D2 + GT energy attention | Deferred | Depends on D2 |
| D4: Dual encoder | Deferred | Complex architecture; D1/D5 results will indicate if it's worth the cost |

These are explicitly deferred, not rejected. If D1 shows large gains, D2/D3/D4 become high priority.

---

## 7. Execution

```bash
# Train all 20 experiments sequentially on GPUs 0,1
GPUS="0,1" bash scripts/n3_bulk_train.sh

# Or train groups separately
GPUS="0,1" bash scripts/n3_bulk_train.sh groupD   # oracle first (priority 1)
GPUS="2,3" bash scripts/n3_bulk_train.sh groupC   # deployable in parallel

# Test after training
GPUS="0,1" bash scripts/n3_bulk_test.sh
```

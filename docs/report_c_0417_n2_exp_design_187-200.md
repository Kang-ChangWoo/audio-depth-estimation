# N2 Experiment Design: Temporal FOA Decomposition for Depth Estimation
**Date:** 2026-04-17 (updated 2026-04-18)  
**Experiments:** exp187-206 (20 trials, 40 epochs each)  
**Script:** `scripts/n2_bulk.sh`  
**Models:** `models/n2_0417/`  
**Node:** N2 (8 GPUs, 4 workers × 2 GPUs, BS=128, num_workers=16)

---

## 0. Hypothesis Under Test

Ambisonics from Habitat-Sim should not be treated as a single static spatial map. The raw FOA impulse response is a time-resolved spherical sound-field signal, and different temporal segments encode different geometric information:

- **Early** (0–59 ms, ≤2600 samples @44.1 kHz): direct sound + first reflections → nearest surfaces
- **Mid** (59–295 ms): early reverberations → room geometry
- **Late** (295 ms+): diffuse field → room volume

Accumulating over the full IR averages out directional structure. This experiment set tests whether temporal decomposition of FOA — and direct use of FOA as input — improves depth estimation.

Three representation strategies are compared:

| Strategy | Variants | What it tests |
|----------|----------|---------------|
| **(1) Direct Ambisonics** | E1 (6ch input), E4 (dual enc), E5 (STFT FiLM) | Let the network learn FOA features end-to-end |
| **(2) Temporal-bin features** | E2 (temporal RMS), E3 (temporal energy attn), E6 (temporal RMS FiLM), E7 (temporal energy input), E8 (cross-attention) | Explicit early/mid/late decomposition |
| **(3) Visualization-based** | E7 (temporal energy input), E3 (temporal energy attn) | Hand-crafted energy maps per temporal bin |

---

## 1. Prioritized Recommendation

### Run Order (highest information gain first)

| Priority | Exp | Variant | Architecture | Why First |
|----------|-----|---------|-------------|-----------|
| 1 | 187-189 | **E1** | 6ch input | Simplest direct FOA test — zero new modules, just input_nc=6 |
| 2 | 190-191 | **E2** | Temporal RMS | Simplest temporal decomposition — only changes supervision target |
| 3 | 192-194 | **E3** | Temporal energy attn | Spatial + temporal: predicted energy maps from 3 bins as attention |
| 4 | 195-196 | **E4** | Dual encoder | Tests whether FOA deserves its own encoder pathway |
| 5 | 197-198 | **E5** | FOA STFT FiLM | Richest direct FOA: learned features → FiLM at 3 decoder levels |
| 6 | 199-200 | **E6** | Temporal RMS FiLM | Combines temporal supervision (E2) + FiLM injection (E5 idea) |
| 7 | 201-203 | **E7** | Temporal energy input | Simplest visualization-based temporal: energy maps as extra input channels |
| 8 | 204-206 | **E8** | Temporal bin cross-attn | Most expressive: cross-attention over per-bin energy features at bottleneck |

**Rationale:** E1 and E7 are the simplest gating experiments — E1 tests "does raw FOA help?" and E7 tests "do temporal energy maps help as input?" If both fail, the temporal hypothesis is weakened. E8 is the most ambitious and should only be prioritized if E3/E7 show temporal energy maps carry useful signal.

---

## 2. Experiment Details

### E1: 6-Channel Input (exp187-189)

**Model:** `FOA0415V1Generator` with `input_nc=6`  
**Config:** `config/n2_6ch_input.yaml`

**Input:** `cat(binaural_spec [2ch], foa_spectrogram [4ch]) = [6, 256, 512]`

The FOA spectrogram is computed from the first ~5176 samples (117 ms) of the rotated FOA IR using the same STFT parameters as the binaural path (n_fft=512, hop=160, win=400), then resized to (4, 256, 512).

**What this tests:** Can the network learn useful features from raw FOA input? No hand-crafted statistics (RMS, energy map) — the 4 FOA spectral channels are given directly alongside binaural.

**Architecture change vs v1:**
```
v1:  (2, 256, 512) → enc0(in=2) → ... → pred_depth
E1:  (6, 256, 512) → enc0(in=6) → ... → pred_depth
```

**New params:** ~768 extra params in enc0 (4 extra input channels × 64 filters × 3×3... actually 4×4 kernel). Negligible.

**Expected effect:** RMSE -0.01 to -0.04 if FOA carries geometric signal beyond what binaural captures  
**Failure mode:** If FOA spectrogram is too noisy or dominated by W channel (omnidirectional), the extra channels add noise. Safe failure — model can learn to ignore them.

**Key diagnostic:** Compare enc0 filter activations for the binaural vs FOA channels.

| Exp | LR | lambda_sh | Compare to |
|-----|-----|-----------|------------|
| 187 | 1e-3 | 0.1 | exp184 (v1 baseline, N3) |
| 188 | 5e-4 | 0.1 | exp185 (v1 baseline, N3) |
| 189 | 1e-3 | 0.3 | exp187 (stronger SH) |

---

### E2: Temporal RMS Supervision (exp190-191)

**Model:** `FOA0415V1Generator` with `sh_dim=12`  
**Config:** `config/n2_temporal_rms.yaml`

**Supervision target:** Instead of global RMS (4-dim), the SH head predicts 12 dims = concat(early_rms[4], mid_rms[4], late_rms[4]).

**What this tests:** Does breaking the RMS target into temporal bins preserve more geometric structure? This is the minimal temporal decomposition — same model architecture, only the supervision target changes.

**Architecture change vs v1:**
```
v1:  sh_head → (B, 4)   supervised by global_rms
E2:  sh_head → (B, 12)  supervised by temporal_rms = [early|mid|late]
```

**New params:** SHHead output layer: 4→12 adds ~2K params. Negligible.

**Expected effect:** If the early bin RMS strongly correlates with depth structure while the late bin is noise, the network can learn to weight them differently. RMSE -0.005 to -0.02.  
**Failure mode:** If per-bin RMS is too noisy (short segments reduce RMS accuracy), the 12-dim target may be harder to learn than 4-dim. Check per-bin SH L1 to diagnose.

| Exp | LR | lambda_sh | Compare to |
|-----|-----|-----------|------------|
| 190 | 1e-3 | 0.1 | exp184 (v1 baseline) |
| 191 | 5e-4 | 0.1 | exp185 (v1 baseline) |

---

### E3: Temporal Energy Attention (exp192-194)

**Model:** `N2TemporalEnergyGenerator` (`models/n2_0417/n2_temporal_energy.py`)  
**Config:** `config/n2_temporal_energy.yaml`

**Architecture:** The model predicts 3 spatial energy maps (one per temporal bin) from the bottleneck, each applied as residual multiplicative attention at different decoder levels:

```
bottleneck → EnergyHead_0 → energy_early  → attn at decoder[4] (32×64)
           → EnergyHead_1 → energy_mid    → attn at decoder[5] (64×128)
           → EnergyHead_2 → energy_late   → attn at decoder[6] (128×256)
```

**Attention form:** `h = h * (1 + energy_map)` — residual, cannot zero out features.

**Loss:** `L = L_depth + λ_sh * L1(pred_sh, gt_foa) + λ_energy * Σ L1(pred_energy_k, gt_energy_k)`

**What this tests:** The full hypothesis: spatial-temporal decomposition of FOA, with each temporal bin's energy map guiding a different decoder resolution. Early reflections modulate low-res features (depth structure), late reverb modulates high-res features (fine detail).

**Expected effect:** RMSE -0.01 to -0.04 if temporal energy maps are learnable from binaural.  
**Failure mode:** EnergyHeads predict uniform maps → attention has no effect (safe). Or: predicted maps don't match GT temporal energies → noisy attention.

**Key diagnostic:** Visualize pred_energy_k vs GT temporal_energy_k per bin. If early-bin maps are sharper than late-bin maps, the temporal decomposition is justified.

| Exp | LR | lambda_sh | lambda_energy | Compare to |
|-----|-----|-----------|---------------|------------|
| 192 | 1e-3 | 0.1 | 0.1 | exp172-174 (N3 energy attn, global) |
| 193 | 5e-4 | 0.1 | 0.1 | exp192 (LR ablation) |
| 194 | 1e-3 | 0.3 | 0.1 | exp192 (SH weight ablation) |

---

### E4: Dual Encoder (exp195-196)

**Model:** `N2DualEncGenerator` (`models/n2_0417/n2_dual_enc.py`)  
**Config:** `config/n2_dual_enc.yaml`

**Architecture:** Two independent UNet encoders:
- Binaural encoder: (2, 256, 512) → bottleneck (512, 1, 2)
- FOA encoder: (4, 256, 512) → bottleneck (512, 1, 2)

Bottlenecks are concatenated and linearly projected to (512, 1, 2). Decoder uses skip connections from the binaural encoder only.

```
binaural → bin_enc → bin_bn ─┐
                              ├→ cat → Linear(1024, 512) → fused_bn
FOA spec → foa_enc → foa_bn ─┘
fused_bn → SH head → pred_sh
fused_bn → decoder (skips from bin_enc) → pred_depth
```

**What this tests:** Does FOA benefit from dedicated feature extraction? The binaural encoder learns echo timing/ILD; the FOA encoder can specialize in directional-temporal patterns.

**New params:** ~2× encoder (~15M → ~30M total). Most expensive variant.  
**Expected effect:** RMSE -0.02 to -0.05 if FOA features complement binaural features.  
**Failure mode:** FOA encoder learns redundant features → bottleneck fusion wastes capacity. Also: 2× encoder means slower convergence for the same epoch count.

| Exp | LR | lambda_sh | Compare to |
|-----|-----|-----------|------------|
| 195 | 1e-3 | 0.1 | exp187 (6ch input, same data, single encoder) |
| 196 | 5e-4 | 0.1 | exp188 (6ch input, same data, single encoder) |

---

### E5: FOA STFT FiLM (exp197-198)

**Model:** `N2FOASTFTFiLMGenerator` (`models/n2_0417/n2_foa_stft_film.py`)  
**Config:** `config/n2_foa_stft_film.yaml`

**Architecture:** A lightweight ConvNet processes the FOA spectrogram (4, 256, 512) into a compact feature vector (512,), which is injected via FiLM conditioning at 3 decoder levels:

```
FOA spec → FOAFeatureExtractor → foa_feat (512,)
           ↓ FiLM at decoder[0]  (h * (1+γ) + β)
           ↓ FiLM at decoder[4]
           ↓ FiLM at decoder[5]
```

**What this tests:** Direct ambisonics feature learning (no hand-crafted statistics), with the learned features modulating the depth decoder. Lighter than dual encoder (shared UNet encoder for binaural; small ConvNet for FOA).

**FOAFeatureExtractor:** 4 strided convs (4→32→64→128→256) + AdaptiveAvgPool + Linear(256, 512). ~1M extra params.

**Expected effect:** RMSE -0.01 to -0.03.  
**Failure mode:** ConvNet may not learn useful features from FOA spec alone (only 4 channels, noisy). FiLM may be too global to capture spatial structure.

| Exp | LR | lambda_sh | Compare to |
|-----|-----|-----------|------------|
| 197 | 1e-3 | 0.1 | exp166 (N3 FiLM, 4-dim SH → FiLM) |
| 198 | 5e-4 | 0.1 | exp167 (N3 FiLM, lower LR) |

---

### E6: Temporal RMS FiLM (exp199-200)

**Model:** `N2TemporalRMSFiLMGenerator` (`models/n2_0417/n2_temporal_rms_film.py`)  
**Config:** `config/n2_temporal_rms_film.yaml`

**Architecture:** Like N3 FiLM, but the FiLM conditioning vector is the predicted 12-dim temporal RMS (3 bins × 4 channels) instead of global 4-dim RMS. Also supervised against 12-dim temporal_rms target.

```
bottleneck → SHHead(sh_dim=12) → pred_sh (12,) ─── loss: L1 vs temporal_rms
                                                └→ FiLMProjector → (γ, β)
                                                   decoder[0] → h*(1+γ)+β → ...
```

**What this tests:** Combines temporal decomposition (E2) with decoder feedback (FiLM). If E2 shows temporal RMS is more informative than global RMS, does injecting that richer signal back into the decoder help further?

**Expected effect:** If E2 > v1 and N3 FiLM > v1, then E6 should beat both.  
**Failure mode:** 12-dim FiLM may overfit compared to 4-dim FiLM. FiLMProjector is small (12→128→1024) so this risk is low.

| Exp | LR | lambda_sh | Compare to |
|-----|-----|-----------|------------|
| 199 | 1e-3 | 0.1 | exp166 (N3 FiLM, 4-dim), exp190 (temporal RMS, no FiLM) |
| 200 | 5e-4 | 0.1 | exp167, exp191 |

---

### E7: Temporal Energy Map Input (exp201-203)

**Model:** `N2TemapInputGenerator` (`models/n2_0417/n2_temap_input.py`)  
**Config:** `config/n2_temap_input.yaml`

**Input:** `cat(binaural_spec [2ch], temporal_energies [3ch]) = [5, 256, 512]`

The 3 temporal energy maps (one per bin: direct, early reverb, late diffuse) are concatenated with the binaural spectrogram at the input level. The SH head predicts 12-dim temporal_rms as auxiliary supervision.

**What this tests:** The simplest visualization-based temporal variant. Each energy map is a physics-grounded spatial projection (y(Ω)ᵀ R_k y(Ω)) of the per-bin covariance. If the temporal hypothesis is correct, the early-bin map should show sharper directional structure aligned with nearby surfaces.

**Architecture change vs v1:**
```
v1:  (2, 256, 512) → enc0(in=2) → ... → sh_head(4) → pred_depth
E7:  (5, 256, 512) → enc0(in=5) → ... → sh_head(12) → pred_depth
```

**New params:** ~576 extra params in enc0 + ~2K in SH head. Negligible (54.5M total).

**Expected effect:** RMSE -0.01 to -0.03. If temporal energy maps carry geometry signal, the model gets a free spatial prior at the input level.  
**Failure mode:** If energy maps are noisy or redundant with binaural, model ignores extra channels. HIGH robustness — safe fallback to 2-channel behavior.

| Exp | LR | lambda_sh | Compare to |
|-----|-----|-----------|------------|
| 201 | 1e-3 | 0.1 | exp187 (E1, FOA spec input), exp192 (E3, energy attn) |
| 202 | 5e-4 | 0.1 | exp188, exp193 |
| 203 | 1e-3 | 0.3 | exp201 (stronger SH) |

---

### E8: Temporal Bin Cross-Attention (exp204-206)

**Model:** `N2TBinCrossAttnGenerator` (`models/n2_0417/n2_tbin_crossattn.py`)  
**Config:** `config/n2_tbin_crossattn.yaml`

**Architecture:** The binaural spectrogram is encoded by the main UNet encoder. Each temporal bin's energy map is independently processed by a **shared** lightweight conv encoder (1→32→64→128, adaptive pool to 4×8). At the bottleneck, multi-head cross-attention lets binaural features (queries) attend to temporal-bin energy features (keys/values):

```
binaural → UNet encoder → bottleneck (512, h_bn, w_bn) → Q
temporal_energies[:, 0] → shared_bin_enc → feat_0 (128, 4, 8) ─┐
temporal_energies[:, 1] → shared_bin_enc → feat_1 (128, 4, 8)  ├→ K, V
temporal_energies[:, 2] → shared_bin_enc → feat_2 (128, 4, 8) ─┘
                                                                 ↓
                                            CrossAttention(Q, [K,V]) + residual
                                                                 ↓
                                            → SH head → pred_sh
                                            → decoder → pred_depth
```

**Cross-attention:** 4-head, d_model=512, bin_dim=128. With LayerNorm and residual connection. Total cross-attention params: ~0.8M.

**What this tests:** The most explicit temporal factoring. The model can learn which temporal bin is most relevant for each spatial location. For example, a corner might attend strongly to bin 0 (direct reflections off nearby walls), while open areas attend more to bin 1 (early reverb encodes farther surfaces).

**Expected effect:** RMSE -0.02 to -0.05 if temporal bins carry complementary geometric signals.  
**Failure mode:** If all bins provide similar information, cross-attention learns uniform weights — degrades to bottleneck averaging (MEDIUM robustness). Also: ~55.4M params, slightly larger than base model.

| Exp | LR | lambda_sh | Compare to |
|-----|-----|-----------|------------|
| 204 | 1e-3 | 0.1 | exp192 (E3, energy attn), exp201 (E7, energy input) |
| 205 | 5e-4 | 0.1 | exp193, exp202 |
| 206 | 1e-3 | 0.3 | exp204 (stronger SH) |

---

## 3. Which Representation to Rotate

All experiments use `rotate_canonical=true` and `use_n2_features=true`. The canonical-frame FOA rotation is applied at the IR level before any statistic is computed (see `data/dataset_rotated.py`). Therefore:

- FOA spectrogram: computed from rotated IR → ego-centric
- Temporal RMS: computed from rotated IR → ego-centric  
- Temporal energy maps: computed from rotated IR → ego-centric

No additional rotation is needed in the model or training step.

---

## 4. Tensor Shapes (BS=32)

| Tensor | Shape | Used by |
|--------|-------|---------|
| audio (binaural spec) | (32, 2, 256, 512) | All |
| gt_depth | (32, 1, 256, 512) | All |
| foa_target (global RMS) | (32, 4) | E1, E3, E4, E5 |
| energy_map (global) | (32, 1, 256, 512) | (unused in N2, carried for compat) |
| foa_spec | (32, 4, 256, 512) | E1, E4, E5 |
| temporal_rms | (32, 12) | E2, E6 |
| temporal_energies | (32, 3, 256, 512) | E3, E7, E8 |

---

## 5. Comparison Framework

### Direct FOA vs Temporal-Bin vs Visualization-Based

```
Compare across representation strategies:
  E1 (6ch)   vs  N3 oracle_nc3  → direct FOA input vs GT energy input
  E1 (6ch)   vs  E2 (temp RMS)  → direct vs temporal-bin (input vs supervision)
  E3 (t-energy) vs N3 energy_attn → temporal vs global energy attention
  E5 (STFT FiLM) vs N3 FiLM     → learned FOA features vs SH-derived FiLM
```

### Cross-variant Analysis

| Comparison | Interpretation |
|------------|----------------|
| E1 > v1 | Raw FOA spectrogram adds useful information beyond binaural |
| E2 > v1 | Temporal-bin RMS is more informative than global RMS |
| E3 > N3 energy_attn | Temporal energy decomposition > global energy |
| E4 > E1 | Dedicated FOA encoder extracts better features than shared encoder |
| E5 > N3 FiLM | Learned FOA features > hand-crafted SH statistics for FiLM |
| E6 > E2 and E6 > N3 FiLM | Temporal RMS + FiLM > either alone |
| E7 > v1 | Temporal energy maps as input carry spatial-temporal geometry |
| E7 > E1 | Visualization-based (energy maps) > direct FOA spectrogram |
| E8 > E3 | Cross-attention over bins > fixed-layer attention mapping |
| E8 > E7 | Learned attention over bins > naive input concatenation |

### Robustness Assessment

If the physical interpretation is only partially correct:
- **E1** is robust: just adds channels — if FOA is useless, model ignores them
- **E2** is robust: if temporal bins are unhelpful, 12-dim target is ~3× harder but not harmful
- **E3** is moderately robust: residual attention (1+x form) degrades gracefully
- **E4** is fragile: wasted encoder capacity if FOA is uninformative
- **E5** is moderately robust: ConvNet can learn to produce a null feature if FOA is useless
- **E6** is robust: FiLM residual form + shared v1 encoder backbone
- **E7** is HIGH robustness: extra input channels are free, standard UNet can ignore them
- **E8** is MEDIUM robustness: cross-attention with residual connection, but complex module adds risk

---

## 6. Implementation

### Files Created

| File | Purpose |
|------|---------|
| `data/dataset_n2.py` | Dataset returning 7-tuple with temporal FOA features |
| `models/n2_0417/__init__.py` | Module init |
| `models/n2_0417/n2_temporal_energy.py` | E3: temporal energy attention |
| `models/n2_0417/n2_dual_enc.py` | E4: dual encoder |
| `models/n2_0417/n2_foa_stft_film.py` | E5: FOA STFT FiLM |
| `models/n2_0417/n2_temporal_rms_film.py` | E6: temporal RMS FiLM |
| `config/n2_6ch_input.yaml` | E1 config |
| `config/n2_temporal_rms.yaml` | E2 config |
| `config/n2_temporal_energy.yaml` | E3 config |
| `config/n2_dual_enc.yaml` | E4 config |
| `config/n2_foa_stft_film.yaml` | E5 config |
| `config/n2_temporal_rms_film.yaml` | E6 config |
| `models/n2_0417/n2_temap_input.py` | E7: temporal energy map input (subclass of FOA0415V1) |
| `models/n2_0417/n2_tbin_crossattn.py` | E8: temporal bin cross-attention |
| `config/n2_temap_input.yaml` | E7 config (input_nc=5, sh_dim=12) |
| `config/n2_tbin_crossattn.yaml` | E8 config (input_nc=2, sh_dim=4) |
| `scripts/n2_bulk.sh` | Train+test all 20 experiments |

### Files Modified

| File | Change |
|------|--------|
| `models/__init__.py` | Added N2 model imports (incl. E7, E8) |
| `models/n2_0417/__init__.py` | Added E7, E8 exports |
| `utils/train_utils.py` | Added `_N2_CLASSES` entries for E7, E8 |
| `train.py` | Added routing for `n2_temap_input`, `n2_tbin_crossattn` in `_train_step_n2()` |
| `data/dataloader.py` | Added `SoundSpacesDatasetN2` routing via `use_n2_features` |

### What Was NOT Modified

- Existing N3 models, configs, scripts — untouched
- `data/dataset.py`, `data/dataset_rotated.py`, `data/dataset_n2.py` — untouched
- `test.py` — N2 models use existing foa0415 test path (batch[0:4] suffice for eval)

---

## 7. Temporal Bins

| Bin | Sample range | Time range | Physical content |
|-----|-------------|-----------|-----------------|
| 0 (early) | 0–2600 | 0–59 ms | Direct sound + first-order reflections |
| 1 (mid) | 2600–13000 | 59–295 ms | Early reverberation (room modes) |
| 2 (late) | 13000–55043 | 295 ms–1.25 s | Diffuse field / late reverb |

The 2600-sample boundary corresponds to the 10 m round-trip distance (2×10m / 340 m/s × 44100 Hz ≈ 2594 samples), matching the user's identified critical cutoff.

---

## 8. Execution

```bash
# Run all 20 experiments on Node 2 (8 GPUs)
bash scripts/n2_bulk.sh

# Expected runtime: ~20 experiments / 4 parallel workers = 5 rounds
# Each round: ~2.5h train (40 epochs, BS=128) + ~30min test = ~3h
# Total: ~15 hours
```

### Worker distribution (round-robin):
```
Worker 0 (GPU 0,1): exp187, exp191, exp195, exp199, exp203
Worker 1 (GPU 2,3): exp188, exp192, exp196, exp200, exp204
Worker 2 (GPU 4,5): exp189, exp193, exp197, exp201, exp205
Worker 3 (GPU 6,7): exp190, exp194, exp198, exp202, exp206
```

---

## 9. Experiments NOT Included (Deferred)

| Idea | Reason deferred |
|------|----------------|
| FOA IR as raw waveform input | Requires separate waveform encoder (Wav2Vec2-style); high compute |
| Higher-order ambisonics (HOA) | Dataset only provides FOA (order 1) |
| Learnable temporal bin boundaries | Requires differentiable window function; add if fixed bins help |
| Frequency-dependent temporal analysis | Interesting but multiplies complexity; add after E1-E6 results |
| Per-temporal-bin FOA spectrograms | Would need (K*4, 256, 512) per sample — ~2GB/batch at BS=32; use energy maps instead (E7, E8) |
| Learnable temporal bin boundaries | Requires differentiable window function; add if fixed bins help |

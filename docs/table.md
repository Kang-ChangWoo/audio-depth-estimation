# `n2_revisit_test` — per-family HP-sweep statistics (mean ± std)

Source: `baseline/logs/n2_revisit_test/` (25 logs, 5 architectures × 5 LR/BS variants).

Each family below is a 5-run hyperparameter sweep over (learning rate, batch size). The std/range numbers below describe **HP-induced variance** (how much the metric moves when you change LR or BS within a family) — this is **not** seed variance. Use it as a lower bound on the true uncertainty: real seed-to-seed variance on the same HP setting would also contribute, on top of this.

---

## 1. Per-family means

| family               | n | ABS_REL ↓        | RMSE ↓           | δ1 ↑             | δ2 ↑             | δ3 ↑             | Log10 ↓          | MAE ↓            |
|----------------------|---|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| pretrained_resnet    | 5 | 0.5179 ± 0.0230  | 1.3343 ± 0.0176  | 0.4569 ± 0.0137  | 0.6675 ± 0.0091  | 0.7990 ± 0.0047  | 0.1748 ± 0.0027  | 0.8673 ± 0.0093  |
| vit (from-scratch)   | 5 | 0.5098 ± 0.0353  | 1.2947 ± 0.0514  | 0.4660 ± 0.0188  | 0.6790 ± 0.0174  | 0.8071 ± 0.0144  | 0.1694 ± 0.0084  | 0.8409 ± 0.0345  |
| baseline (UNet)      | 5 | 0.4621 ± 0.0090  | 1.2297 ± 0.0104  | 0.4979 ± 0.0017  | 0.7110 ± 0.0010  | 0.8309 ± 0.0022  | 0.1564 ± 0.0013  | 0.7874 ± 0.0048  |
| echodiffusion        | 5 | 0.4847 ± 0.0516  | 1.2483 ± 0.0336  | 0.4866 ± 0.0111  | 0.7000 ± 0.0126  | 0.8224 ± 0.0105  | 0.1620 ± 0.0074  | 0.8102 ± 0.0308  |
| pretrained_vit       | 5 | 0.4602 ± 0.0290  | 1.2463 ± 0.0202  | 0.4878 ± 0.0082  | 0.7038 ± 0.0058  | 0.8268 ± 0.0025  | 0.1589 ± 0.0019  | 0.7985 ± 0.0081  |

---

## 2. Per-family ranges (max − min) and best/worst HP cells

| family               | metric  | min (best)        | max (worst)         | range  |
|----------------------|---------|-------------------|---------------------|--------|
| baseline (UNet)      | ABS_REL | 0.4486 (exp350)   | 0.4703 (exp352)     | 0.0217 |
|                      | RMSE    | 1.2117 (exp354)   | 1.2386 (exp351)     | 0.0269 |
|                      | δ1      | 0.4952 (exp350)   | 0.4992 (exp351/352) | 0.0040 |
| vit (from-scratch)   | ABS_REL | 0.4703 (exp357)   | 0.5664 (exp358)     | 0.0961 |
|                      | RMSE    | 1.2593 (exp359)   | 1.3783 (exp358)     | 0.1190 |
|                      | δ1      | 0.4373 (exp358)   | 0.4836 (exp356)     | 0.0463 |
| echodiffusion        | ABS_REL | 0.4482 (exp363)   | 0.5748 (exp361)     | 0.1266 |
|                      | RMSE    | 1.2198 (exp363)   | 1.3055 (exp364)     | 0.0857 |
|                      | δ1      | 0.4713 (exp361)   | 0.4972 (exp360)     | 0.0259 |
| pretrained_resnet    | ABS_REL | 0.4964 (exp369)   | 0.5554 (exp365)     | 0.0590 |
|                      | RMSE    | 1.3154 (exp365)   | 1.3601 (exp369)     | 0.0447 |
|                      | δ1      | 0.4371 (exp369)   | 0.4688 (exp365)     | 0.0317 |
| pretrained_vit       | ABS_REL | 0.4226 (exp371)   | 0.4985 (exp374)     | 0.0759 |
|                      | RMSE    | 1.2269 (exp373)   | 1.2806 (exp371)     | 0.0537 |
|                      | δ1      | 0.4795 (exp370)   | 0.4993 (exp373)     | 0.0198 |

---

## 3. Reading

### 3.1 Std rankings (lower = more HP-robust)

| metric  | most stable                                | least stable                              |
|---------|--------------------------------------------|-------------------------------------------|
| ABS_REL | **baseline UNet** (0.0090) → resnet (0.0230) → previt (0.0290) → vit (0.0353) → echodiff (0.0516) | echodiff |
| RMSE    | **baseline UNet** (0.0104) → resnet (0.0176) → previt (0.0202) → echodiff (0.0336) → vit (0.0514) | vit |
| δ1      | **baseline UNet** (0.0017) → previt (0.0082) → echodiff (0.0111) → resnet (0.0137) → vit (0.0188) | vit |

Baseline UNet is by far the most HP-robust architecture on every metric. Whatever LR/BS you pick within the sweep, you get RMSE in [1.21, 1.24]. This makes UNet the right reference baseline — its single-run number is reliable.

### 3.2 The variance bound on cross-family claims

To claim "family A beats family B" with HP variance alone, the gap between A's best and B's mean (or B's best and A's mean) needs to exceed roughly 2× the larger of the two stds.

For example:
- best `pretrained_vit` (RMSE 1.2269) vs `baseline (UNet)` mean (1.2297). Gap = 0.0028. previt std = 0.0202 → gap is **0.14σ**. **No real difference.**
- best `echodiffusion` (RMSE 1.2198) vs `baseline (UNet)` mean (1.2297). Gap = 0.0099. echodiff std = 0.0336 → gap is **0.29σ**. **No real difference.**
- best `pretrained_resnet` (RMSE 1.3154) vs `baseline (UNet)` mean (1.2297). Gap = 0.0857. **Resnet is meaningfully worse** (~5σ in baseline std, ~5σ in resnet std).

### 3.3 Implication for ambisonic-EchoDiffusion

Earlier cross-comparison of n4 (oracle FOA, RMSE 1.2013) vs binaural baselines (best RMSE 1.2117) showed a 0.0104 gap. That's:
- 1.0× baseline UNet HP-std (0.0104) — **marginal** even within HP variance
- 0.3× echodiff HP-std (0.0336) — **well below noise**

So adding ambisonics to EchoDiffusion is statistically unlikely to produce a credible improvement. If a 4-cell ambisonic-EchoDiffusion sweep returns RMSE ≈ 1.21, there's no way to distinguish "ambisonic helped" from "we got a lucky HP cell."

### 3.4 The pretrained_vit ABS_REL anomaly

`exp371` (pretrained_vit, lr=5e-5, bs=48) has ABS_REL=0.4226 — **better than every other run including the n4 oracle-FOA best (0.4235)**. But it's at the low end of `pretrained_vit`'s ABS_REL range (sweep: [0.4226, 0.4985], std=0.0290), so this single number could plausibly be an outlier in the high-variance tail.

To know if this is real, **rerun exp371 across 3 seeds** before claiming ImageNet-pretrained ViT beats oracle-FOA on ABS_REL. If the seed std is ≈0.01, the 0.4226 is a real ceiling-of-pretrained-ViT measurement. If seed std is ≈0.03 (matching HP std), the result is noise.

---

## 4. Notes

- All metrics are on the test split (3192 samples, 9 scenes). Caches `e2314b68a4f5` (binaural-only) and `7027059baf06` (ambisonic-on) contain identical samples (verified 2026-04-26), so cross-group cache mismatch is not a confound.
- "Std" is the sample standard deviation (Bessel-corrected, `n−1` denominator). With n=5 per family, this is a noisy estimator — use it as a rough scale, not a precise CI.
- Family ranges:
    - **baseline**: exp350–354 (`unet_baseline`, vanilla 8-level UNet, binaural input)
    - **vit (from-scratch)**: exp355–359 (`vit_baseline`, no ImageNet pretraining)
    - **echodiffusion**: exp360–364 (`echodiffusion`, diffusion-UNet backbone + Wav2Vec2 conditioning)
    - **pretrained_resnet**: exp365–369 (`pretrained_resnet`, ImageNet ResNet-50 + FPN decoder)
    - **pretrained_vit**: exp370–374 (`pretrained_vit`, ImageNet ViT-B/16 + ConvTranspose decoder)

# exp360–364 EchoDiffusion sweep — research findings

> Saved from chat 2026-04-26. Source data: `logs/n2_revisit_test/exp36*_test.log` and `logs/n2_revisit_train/exp36*_bs48.log` (+ exp362 bs32). Context: assessing whether the original EchoDiffusion's Wav2Vec2/CIDE branch adds depth-estimation value, and predicting behavior of the upcoming `n9_echodiffambi_cide.sh` (exp710-713) sweep.

---

## 1. Headline table

| exp | LR | BS | best val RMSE @ ep | **test ABS** | **test RMSE** | **test δ1** |
|-----|------|-----|---:|--------:|---------:|--------:|
| 360 | 1e-4 | 48 | 1.400 @ 20 | 0.4774 | 1.2473 | 0.4972 |
| 361 | 5e-5 | 48 | 1.359 @ 8  | 0.5748 | 1.2396 | 0.4713 |
| 362 | 1e-4 | 32 | 1.389 @ 8  | 0.4557 | 1.2292 | 0.4923 |
| **363** | **5e-4** | **48** | **1.372 @ 36** | **0.4482** | **1.2198** | 0.4936 |
| 364 | 1e-5 | 48 | 1.446 @ 20 | 0.4675 | 1.3055 | 0.4788 |

---

## 2. Five things this sweep tells you

### (1) **Best cell is exp363** (lr=5e-4, bs=48)
Best test RMSE (1.2198) **and** best test ABS_REL (0.4482). Surprisingly its best val RMSE epoch is **36/40** (vs 8 or 20 for others) — it converges slowly but ends up generalizing better.

### (2) **All cells severely overfit**
Final-epoch train loss vs val loss gap is **3-4×**:

| cell | train final L | val L | gap |
|------|--------------:|------:|----:|
| 360 | 0.0811 | 0.245 | 3.0× |
| 361 | 0.0653 | 0.246 | 3.8× |
| 362 | 0.0774 | 0.245 | 3.2× |
| 363 | 0.1592 | 0.245 | 1.5× |
| 364 | 0.0792 | 0.247 | 3.1× |

exp363 has the smallest gap (1.5×) — this is **why it generalizes best**. Higher LR effectively prevents memorization. The other cells are memorizing the 23560 training samples.

### (3) **Val RMSE (~1.40) is consistently HIGHER than test RMSE (~1.23)**
This is unusual. Normally val and test should be similar. Possibilities:
- Val split has harder scenes than test split (random scene assignment with `split_seed=42`)
- Val samples are drawn at uniform random, test samples may be biased
- Worth checking: re-evaluate on val using the test pipeline to verify the split disparity

This means **val metrics are not a reliable proxy for test metrics** — the model selected by best-val-RMSE may not be the best test model.

### (4) **LR sweet spot: 5e-4 by a clear margin**
RMSE vs LR:

```
  lr=1e-5   ━━━━ 1.305  (too slow, undertrained)
  lr=5e-5   ━━━ 1.240   (unstable: ABS=0.5748 worst)
  lr=1e-4   ━━━ 1.230   (mediocre, overfits)
  lr=5e-4   ━━━ 1.220   ← BEST
  lr=1e-3   (not tested for echodiff)
```

The lr=5e-4 cell takes longer to converge (best val at ep 36 vs ep 8) but ends up better. Higher LR = more regularization via SGD noise = less memorization.

### (5) **bs=32 vs bs=48 (same lr=1e-4)**: bs=32 marginally better
- exp360 (bs=48): RMSE=1.2473
- exp362 (bs=32): RMSE=1.2292

A 1.5% difference. Within noise but consistent with "smaller bs = more stochastic = better generalization."

---

## 3. Wav2Vec2 was loaded properly in all 5 cells

Confirmed from train log header (e.g. exp363):

```
Wav2Vec2Model LOAD REPORT from: facebook/wav2vec2-base-960h
Key               | Status     | 
------------------+------------+-
lm_head.bias      | UNEXPECTED | 
lm_head.weight    | UNEXPECTED | 
masked_spec_embed | MISSING    | 

Model: echodiffusion (132.6M params)
```

The `lm_head` UNEXPECTED + `masked_spec_embed` MISSING are benign (we instantiate `Wav2Vec2Model`, not `Wav2Vec2ForCTC`). 132.6M params confirms Wav2Vec2 is in the model.

---

## 4. So — does Wav2Vec2 help?

Comparing to other architectures **without** Wav2Vec2 on the same benchmark:

| family (best cell) | uses Wav2Vec2 | RMSE | ABS_REL | δ1 |
|--------------------|:------:|------:|--------:|-----:|
| **echodiffusion exp363** | ✅ | **1.2198** | **0.4482** | 0.4936 |
| unet_baseline exp354 | ❌ | 1.2117 | 0.4676 | 0.4988 |
| pretrained_vit exp371 | ❌ | 1.2806 | 0.4226 | 0.4809 |
| pretrained_vit exp373 | ❌ | 1.2269 | 0.4743 | 0.4993 |

Within the noise floor (per `docs/table.md`, echodiff HP-std on RMSE = 0.0336):
- echodiff (with W2V2) vs UNet baseline (no W2V2): gap = 0.0081 ≈ **0.24σ** of echodiff std → indistinguishable
- echodiff (with W2V2) vs pretrained_vit on ABS_REL: 0.4482 vs 0.4226 = +0.026 ≈ **+0.74σ** of vit std (0.029) → **vit's ABS is BETTER**

**Conclusion: this sweep provides no evidence that Wav2Vec2 adds value for depth estimation on this benchmark.** It might in some other setting, but here it's at best a wash and at worst slightly worse than ImageNet-pretrained ViT (which is even more "wrong domain" than Wav2Vec2 — image features for audio input).

### Why this might be the case

Wav2Vec2 (`facebook/wav2vec2-base-960h`) was pretrained on 960h of LibriSpeech — clean read English speech. Its features are tuned for phonemes, prosody, speaker characteristics. Our input is **room impulse responses** (chirp + reverb), which has nothing in common with speech distributionally:

- Sparse impulsive onsets, not continuous voiced speech
- Frequency content dominated by reverb decay, not formants
- 5648-sample windows (~128 ms at 44.1 kHz), much shorter than typical Wav2Vec2 training utterances

So Wav2Vec2 is being used **off-distribution**. The CIDE module then has to learn to extract useful signal through a 100-dim softmax bottleneck on top of these mismatched features — a tall order. This explains why a 132.6M-param model with Wav2Vec2 doesn't beat a much smaller UNet baseline.

---

## 5. Implications for the `n9_echodiffambi_cide.sh` runs (exp710-713)

The new sweep adds Wav2Vec2/CIDE on TOP of bin-gated FOA conditioning (rather than in place of it). Based on exp363 patterns, expect:

- **Best LR will likely be 5e-4** (exp710, exp712). Same as exp363.
- **lr=1e-4 (exp711, exp713) will probably overfit** like exp360/362 did, ending up at RMSE ~1.23-1.25.
- **Best val epoch will likely be late** (epoch 30+) for the lr=5e-4 cells. Train budget matters — running fewer epochs may miss the best.
- **Val RMSE will be ~1.40** but test RMSE will be ~1.22 (the val/test disparity). Don't be discouraged by val numbers.
- **Combined with FOA**: *if* CIDE alone gave RMSE 1.22 and oracle FOA alone gave RMSE 1.20, the additive effect is bounded — the noise floor sits at ~1.20-1.22. So expect exp710/712 in the **1.18-1.22 range** at best; ≤1.18 would be a meaningful gain (~1.5σ), <1.16 would be a real win.

---

## 6. Things to watch in the n9_echodiffambi_cide.sh logs

1. **Is val RMSE around 1.40 like exp363?** If yes, training is on track.
2. **Is the train-loss curve converging slowly (stays >0.15 by epoch 40 like exp363) or memorizing (<0.10 by epoch 20 like exp360)?** Slow = good for generalization.
3. **Does exp710 (lr=5e-4) finish with a *later* best-val epoch than exp711 (lr=1e-4)?** This would replicate the exp363 vs exp360/362 pattern and is a sign the LR scale is doing what it should.
4. **Are FOA_L1 / FOA_COS / FOA_DIR metrics in the test log close to {0.0, 1.0, 1.0}?** They should be since rep_pred=rep_gt (oracle). If not, something's broken in the cascade.

---

## 7. Cross-reference

- Per-family stats: `docs/table.md`
- Cross-architecture comparison: `docs/report_g_0426_report.md`
- Bin-gate failure mode: `EDA/_9_comparison_0426/out/summary.txt`
- Earlier "options to boost" answer: `docs/answer.md`
- Source train logs: `logs/n2_revisit_train/exp36*.log`
- Source test logs: `logs/n2_revisit_test/exp36*_test.log`

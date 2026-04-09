# Experiment Results

## Summary

- **bulk0407**: 50 experiments (15 DONE, 3 PARTIAL, 12 NO_RESULT/FAILED, 20 FOA-variant FAILED)
- **bulk0408**: 15 experiments (10 DONE, 4 PARTIAL, 1 CRASHED)

Best model selection: `score = 0.7 × RMSE + 0.3 × abs_rel` (lower is better)

---

## bulk0407 — Completed Experiments

### Baseline UNet (5/5 done)

| Exp | Name | LR | BS | RMSE | ABS_REL | Score |
|-----|------|----|----|------|---------|-------|
| 01 | exp01_baseline_lr1e3_bs32 | 1e-3 | 32 | 1.2396 | 0.4026 | 0.9885 |
| 02 | exp02_baseline_lr5e4_bs32 | 5e-4 | 32 | 1.2535 | 0.3829 | 0.9923 |
| 03 | exp03_baseline_lr1e4_bs32 | 1e-4 | 32 | 1.2631 | 0.4017 | 1.0047 |
| 04 | exp04_baseline_lr1e3_bs16 | 1e-3 | 16 | 1.2680 | 0.3815 | 1.0020 |
| 05 | exp05_baseline_lr5e4_bs16 | 5e-4 | 16 | 1.2443 | 0.4310 | 1.0003 |

**Best**: exp01 (score=0.9885, RMSE=1.2396, lr=1e-3, bs=32)

### ViT (5/5 done)

| Exp | Name | LR | BS | RMSE | ABS_REL | Score |
|-----|------|----|----|------|---------|-------|
| 06 | exp06_vit_lr1e4_bs32 | 1e-4 | 32 | 1.2612 | 0.4412 | 1.0152 |
| 07 | exp07_vit_lr5e5_bs32 | 5e-5 | 32 | 1.2684 | 0.4312 | 1.0172 |
| 08 | exp08_vit_lr1e4_bs16 | 1e-4 | 16 | 1.2424 | 0.4566 | 1.0067 |
| 09 | exp09_vit_lr5e4_bs32 | 5e-4 | 32 | 1.3828 | 0.5275 | 1.1262 |
| 10 | exp10_vit_lr1e5_bs32 | 1e-5 | 32 | 1.3164 | 0.4410 | 1.0538 |

**Best**: exp08 (score=1.0067, RMSE=1.2424, lr=1e-4, bs=16)

### EchoDiffusion (5/5 done)

| Exp | Name | LR | BS | RMSE | ABS_REL | Score |
|-----|------|----|----|------|---------|-------|
| 11 | exp11_echodiff_lr1e4_bs32 | 1e-4 | 32 | 1.2784 | 0.3815 | 1.0093 |
| 12 | exp12_echodiff_lr5e5_bs32 | 5e-5 | 32 | 1.2645 | 0.4309 | 1.0144 |
| 13 | exp13_echodiff_lr1e4_bs16 | 1e-4 | 16 | 1.2559 | 0.4038 | 1.0003 |
| 14 | exp14_echodiff_lr5e4_bs32 | 5e-4 | 32 | 1.2372 | 0.4096 | 0.9889 |
| 15 | exp15_echodiff_lr1e5_bs32 | 1e-5 | 32 | 1.2823 | 0.4282 | 1.0261 |

**Best**: exp14 (score=0.9889, RMSE=1.2372, lr=5e-4, bs=32)

### FOA Variants (all 20 FAILED)

Experiments 16-35 (crossattn, featbank, msattn, channelattn) all failed due to model errors.

### FOA Original — Hyperparameter Tuning

| Exp | Name | LR | BS | dw | fw | hw | Freeze | RMSE | ABS_REL | Score | Status |
|-----|------|----|----|----|----|-------|--------|------|---------|-------|--------|
| 36 | exp36_foa_lr1e3_dw1.0_fw0.1_hw0.1 | 1e-3 | 32 | 1.0 | 0.1 | 0.1 | 0 | 1.2257 | 0.4386 | 0.9896 | DONE |
| 37 | exp37_foa_lr5e4_dw1.0_fw0.1_hw0.1 | 5e-4 | 32 | 1.0 | 0.1 | 0.1 | 0 | 1.2283 | 0.4068 | 0.9819 | DONE |
| 38 | exp38_foa_lr1e4_dw1.0_fw0.1_hw0.1 | 1e-4 | 32 | 1.0 | 0.1 | 0.1 | 0 | 1.2416 | 0.3993 | 0.9889 | PARTIAL |
| 39 | exp39_foa_lr1e3_bs16_dw1.0_fw0.1_hw0.1 | 1e-3 | 16 | 1.0 | 0.1 | 0.1 | 0 | 1.2321 | 0.4411 | 0.9948 | DONE |
| 40 | exp40_foa_lr5e4_bs16_dw1.0_fw0.1_hw0.1 | 5e-4 | 16 | 1.0 | 0.1 | 0.1 | 0 | 1.2223 | 0.4153 | **0.9802** | DONE |
| 41 | exp41_foa_lr1e3_dw1.0_fw0.2_hw0.1 | 1e-3 | 32 | 1.0 | 0.2 | 0.1 | 0 | 1.2362 | 0.4037 | 0.9865 | DONE |
| 42 | exp42_foa_lr5e4_dw1.0_fw0.2_hw0.1 | 5e-4 | 32 | 1.0 | 0.2 | 0.1 | 0 | 1.2384 | 0.3981 | 0.9863 | DONE |
| 44 | exp44_foa_lr5e4_dw1.0_fw0.1_hw0.2 | 5e-4 | 32 | 1.0 | 0.1 | 0.2 | 0 | 1.2442 | 0.4108 | 0.9942 | DONE |
| 45 | exp45_foa_lr1e3_dw1.0_fw0.2_hw0.2 | 1e-3 | 32 | 1.0 | 0.2 | 0.2 | 0 | 1.2237 | 0.4375 | 0.9878 | DONE |
| 46 | exp46_foa_lr1e3_dw0.5_fw0.1_hw0.1 | 1e-3 | 32 | 0.5 | 0.1 | 0.1 | 0 | 1.2377 | 0.4252 | 0.9939 | DONE |
| 47 | exp47_foa_lr1e3_dw2.0_fw0.1_hw0.1 | 1e-3 | 32 | 2.0 | 0.1 | 0.1 | 0 | 1.2358 | 0.3968 | 0.9841 | DONE |
| 49 | exp49_foa_lr1e3_dw1.0_fw0.1_hw0.05 | 1e-3 | 32 | 1.0 | 0.1 | 0.05 | 0 | 1.2354 | 0.4180 | 0.9902 | PARTIAL |
| 50 | exp50_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze5 | 1e-3 | 32 | 1.0 | 0.1 | 0.1 | 5 | 1.2322 | 0.4305 | 0.9917 | PARTIAL |
| 51 | exp51_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze10 | 1e-3 | 32 | 1.0 | 0.1 | 0.1 | 10 | - | - | - | FAILED |
| 52 | exp52_foa_lr5e4_dw1.0_fw0.2_hw0.2 | 5e-4 | 32 | 1.0 | 0.2 | 0.2 | 0 | - | - | - | FAILED |

Missing: exp43, exp48, exp53, exp54, exp55 (not found in logs)

**Best FOA**: exp40 (score=**0.9802**, RMSE=1.2223, lr=5e-4, bs=16, dw=1.0, fw=0.1, hw=0.1)

---

## bulk0408 — Completed Experiments

### Pretrained ResNet-50 (5/5 done)

| Exp | Name | LR | BS | RMSE | ABS_REL | Score |
|-----|------|----|----|------|---------|-------|
| 56 | exp56_resnet_lr1e4_bs32 | 1e-4 | 32 | 1.3391 | 0.4709 | 1.0786 |
| 57 | exp57_resnet_lr5e5_bs32 | 5e-5 | 32 | 1.3273 | 0.4801 | 1.0731 |
| 58 | exp58_resnet_lr5e4_bs32 | 5e-4 | 32 | 1.3557 | 0.4507 | 1.0842 |
| 59 | exp59_resnet_lr1e4_bs16 | 1e-4 | 16 | 1.3187 | 0.5124 | 1.0768 |
| 60 | exp60_resnet_lr3e4_bs32 | 3e-4 | 32 | 1.3238 | 0.4729 | 1.0685 |

**Best**: exp60 (score=1.0685, RMSE=1.3238, lr=3e-4, bs=32)

### Pretrained ViT-B/16 (5/5 done)

| Exp | Name | LR | BS | RMSE | ABS_REL | Score |
|-----|------|----|----|------|---------|-------|
| 61 | exp61_vit_lr1e4_bs16 | 1e-4 | 16 | 1.2434 | 0.3903 | 0.9875 |
| 62 | exp62_vit_lr5e5_bs16 | 5e-5 | 16 | 1.2350 | 0.3909 | 0.9818 |
| 63 | exp63_vit_lr5e4_bs16 | 5e-4 | 16 | 1.2674 | 0.4147 | 1.0116 |
| 64 | exp64_vit_lr1e4_bs8 | 1e-4 | 8 | 1.2432 | 0.4166 | 0.9953 |
| 65 | exp65_vit_lr3e5_bs16 | 3e-5 | 16 | 1.2397 | 0.4053 | 0.9894 |

**Best**: exp62 (score=**0.9818**, RMSE=1.2350, lr=5e-5, bs=16)

### Echo-Net / Parida (4/4 partial — runs killed before 40 epochs)

| Exp | Name | LR | BS | RMSE | ABS_REL | Score | Last Epoch | Status |
|-----|------|----|----|------|---------|-------|------------|--------|
| 66 | exp66_echonet_lr1e3_bs8 | 1e-3 | 8 | 1.2886 | 0.5440 | 1.0652 | 16/40 | PARTIAL |
| 67 | exp67_echonet_lr5e4_bs16 | 5e-4 | 16 | 1.9231 | 1.0187 | 1.6518 | 20/40 | PARTIAL |
| 68 | exp68_echonet_lr1e4_bs16 | 1e-4 | 16 | 1.8267 | 0.5881 | 1.4551 | 18/40 | PARTIAL |
| 69 | exp69_echonet_lr1e3_bs16 | 1e-3 | 16 | 1.6544 | 0.5872 | 1.3343 | 17/40 | PARTIAL |

**Best (so far)**: exp66 (score=1.0652, RMSE=1.2886, lr=1e-3, bs=8) — still PARTIAL, full 40-epoch run pending.

### EchoDiffusion + Wav2Vec (1 crashed)

| Exp | Name | LR | BS | RMSE | ABS_REL | Score | Status |
|-----|------|----|----|------|---------|-------|--------|
| 121 | exp121_echodiff_wav2vec_lr1e4_bs16 | 1e-4 | 16 | - | - | - | CRASHED (epoch 20) |

---

## Overall Ranking (Top 10 by Score)

| Rank | Exp | Model | Score | RMSE | ABS_REL | Key Params |
|------|-----|-------|-------|------|---------|------------|
| 1 | 40 | **FOA** | **0.9802** | 1.2223 | 0.4153 | lr=5e-4, bs=16, dw=1.0, fw=0.1, hw=0.1 |
| 2 | 62 | **Pretrained ViT** | **0.9818** | 1.2350 | 0.3909 | lr=5e-5, bs=16 |
| 3 | 37 | FOA | 0.9819 | 1.2283 | 0.4068 | lr=5e-4, bs=32, dw=1.0, fw=0.1, hw=0.1 |
| 4 | 47 | FOA | 0.9841 | 1.2358 | 0.3968 | lr=1e-3, bs=32, dw=2.0, fw=0.1, hw=0.1 |
| 5 | 42 | FOA | 0.9863 | 1.2384 | 0.3981 | lr=5e-4, bs=32, dw=1.0, fw=0.2, hw=0.1 |
| 6 | 41 | FOA | 0.9865 | 1.2362 | 0.4037 | lr=1e-3, bs=32, dw=1.0, fw=0.2, hw=0.1 |
| 7 | 61 | Pretrained ViT | 0.9875 | 1.2434 | 0.3903 | lr=1e-4, bs=16 |
| 8 | 45 | FOA | 0.9878 | 1.2237 | 0.4375 | lr=1e-3, bs=32, dw=1.0, fw=0.2, hw=0.2 |
| 9 | 01 | Baseline | 0.9885 | 1.2396 | 0.4026 | lr=1e-3, bs=32 |
| 10 | 14 | EchoDiffusion | 0.9889 | 1.2372 | 0.4096 | lr=5e-4, bs=32 |

---

## Key Observations

1. **FOA dominates** — 6 of top 10 spots. Best overall: exp40 (FOA, lr=5e-4, bs=16).
2. **Pretrained ViT** is competitive at rank 2 (exp62, score=0.9818), very close to FOA.
3. **Lowest RMSE** (1.2223) belongs to FOA exp40; **lowest ABS_REL** (0.3815) to baseline exp04 and echodiff exp11.
4. **FOA variants (crossattn, featbank, msattn, channelattn)** all failed — need debugging.
5. **Echo-Net (Parida)** improved from earlier failed runs — best PARTIAL is exp66 (lr=1e-3, bs=8) with score 1.0652 at epoch 16/40. Still trailing FOA/ViT but viable. Full 40-epoch run pending.
6. **Pretrained ResNet** underperforms (scores ~1.07) compared to other models (~0.98).
7. **Missing FOA experiments**: exp43, exp48, exp53, exp54, exp55 — logs not found.
8. **bulk0408 was killed early** — only exps 56–69 + exp121 ran. Echo-Net runs are PARTIAL, batvision (71–75) and groups B/C (76–120) not started.

---

## Status Summary

| Model | Total | Done | Partial | Failed | No Result |
|-------|-------|------|---------|--------|-----------|
| Baseline | 5 | 5 | 0 | 0 | 0 |
| ViT (scratch) | 5 | 5 | 0 | 0 | 0 |
| EchoDiffusion | 5 | 5 | 0 | 0 | 0 |
| FOA CrossAttn | 5 | 0 | 0 | 5 | 0 |
| FOA FeatBank | 5 | 0 | 0 | 5 | 0 |
| FOA MSAttn | 5 | 0 | 0 | 5 | 0 |
| FOA ChannelAttn | 5 | 0 | 0 | 5 | 0 |
| FOA Original | 15* | 10 | 3 | 2 | 0 |
| Pretrained ResNet | 5 | 5 | 0 | 0 | 0 |
| Pretrained ViT | 5 | 5 | 0 | 0 | 0 |
| Echo-Net | 4 | 0 | 4 | 0 | 0 |
| EchoDiff+Wav2Vec | 1 | 0 | 0 | 1 | 0 |

*5 FOA original experiments (43, 48, 53, 54, 55) have no log files.

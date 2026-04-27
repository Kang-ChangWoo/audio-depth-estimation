# Training Experiment Report
**Last updated:** 2026-04-18  
**Dataset:** Matterport3D / SoundSpaces (72 train / 9 val / 9 test scenes)  
**Best model selection:** `score = 0.7 * RMSE + 0.3 * ABS_REL` (lower is better)  
**Validation metrics logged at best epoch:** ABS_REL, RMSE, Score (Delta2/3, Log10, MAE only available from test evaluation)

---

## Summary

| Metric | Count |
|--------|-------|
| Total experiments | 210 |
| Training DONE | 169 |
| Training ONGOING | 12 |
| Training STALLED | 1 |
| Training NEVER STARTED | 26 |
| Checkpoints with best_model.pth | 185 |

*Delta since 2026-04-16*: +41 experiments from Bulk0417 — N3 (exp166–186): 17 DONE / 4 ONGOING; N2 (exp187–206): 4 DONE / 16 QUEUED on a second server.


---

## Experiment Index

### Bulk0407 — Baseline + First FOA Sweep (exp01-60, 40 epochs)

#### Baseline UNet (5/5 DONE)

| Exp | Config | LR | BS | Epochs | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|--------|-------------|----------|-----------|--------|--------|
| 01 | baseline | 1e-3 | 32 | 40/40 | 0.4026 | 1.2396 | 0.9885 | 16 | DONE |
| 02 | baseline | 5e-4 | 32 | 40/40 | 0.3829 | 1.2535 | 0.9923 | 12 | DONE |
| 03 | baseline | 1e-4 | 32 | 40/40 | 0.4017 | 1.2631 | 1.0047 | 8 | DONE |
| 04 | baseline | 1e-3 | 16 | 40/40 | **0.3815** | 1.2680 | 1.0020 | 12 | DONE |
| 05 | baseline | 5e-4 | 16 | 40/40 | 0.4310 | 1.2443 | 1.0003 | 8 | DONE |

#### AudioDepthViT (5/5 DONE)

| Exp | Config | LR | BS | Epochs | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|--------|-------------|----------|-----------|--------|--------|
| 06 | vit | 1e-4 | 32 | 40/40 | 0.4412 | 1.2612 | 1.0152 | 12 | DONE |
| 07 | vit | 5e-5 | 32 | 40/40 | 0.4312 | 1.2684 | 1.0172 | 16 | DONE |
| 08 | vit | 1e-4 | 16 | 40/40 | 0.4566 | 1.2424 | 1.0067 | 32 | DONE |
| 09 | vit | 5e-4 | 32 | 40/40 | 0.5275 | 1.3828 | 1.1262 | 16 | DONE |
| 10 | vit | 1e-5 | 32 | 40/40 | 0.4410 | 1.3164 | 1.0538 | 20 | DONE |

#### EchoDiffusion (5/5 DONE)

| Exp | Config | LR | BS | Epochs | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|--------|-------------|----------|-----------|--------|--------|
| 11 | echodiffusion | 1e-4 | 32 | 40/40 | **0.3815** | 1.2784 | 1.0093 | 8 | DONE |
| 12 | echodiffusion | 5e-5 | 32 | 40/40 | 0.4309 | 1.2645 | 1.0144 | 16 | DONE |
| 13 | echodiffusion | 1e-4 | 16 | 40/40 | 0.4038 | 1.2559 | 1.0003 | 20 | DONE |
| 14 | echodiffusion | 5e-4 | 32 | 40/40 | 0.4096 | 1.2372 | 0.9889 | 28 | DONE |
| 15 | echodiffusion | 1e-5 | 32 | 40/40 | 0.4282 | 1.2823 | 1.0261 | 8 | DONE |

#### FOA CrossAttn (5/5 DONE)

| Exp | Config | LR | fw | Epochs | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|--------|-------------|----------|-----------|--------|--------|
| 16 | foa_crossattn | 1e-3 | 0.1 | 40/40 | 0.4206 | 1.2355 | 0.9911 | 16 | DONE |
| 17 | foa_crossattn | 5e-4 | 0.1 | 40/40 | 0.3975 | 1.2509 | 0.9949 | 16 | DONE |
| 18 | foa_crossattn | 1e-4 | 0.1 | 40/40 | 0.4280 | **1.2198** | **0.9822** | 12 | DONE |
| 19 | foa_crossattn | 1e-3 | 0.2 | 40/40 | 0.4126 | 1.2350 | 0.9883 | 20 | DONE |
| 20 | foa_crossattn | 5e-4 | 0.2 | 40/40 | 0.4161 | 1.2466 | 0.9975 | 16 | DONE |

#### FOA FeatBank (5/5 DONE)

| Exp | Config | LR | fw | Epochs | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|--------|-------------|----------|-----------|--------|--------|
| 21 | foa_featbank | 1e-3 | 0.1 | 40/40 | 0.4326 | 1.2283 | 0.9895 | 20 | DONE |
| 22 | foa_featbank | 5e-4 | 0.1 | 40/40 | 0.4161 | 1.2391 | 0.9922 | 20 | DONE |
| 23 | foa_featbank | 1e-4 | 0.1 | 40/40 | 0.4287 | 1.2426 | 0.9984 | 16 | DONE |
| 24 | foa_featbank | 1e-3 | 0.2 | 40/40 | 0.4169 | 1.2471 | 0.9980 | 20 | DONE |
| 25 | foa_featbank | 5e-4 | 0.2 | 40/40 | **0.3953** | 1.2409 | 0.9872 | 16 | DONE |

#### FOA MSAttn (5/5 DONE)

| Exp | Config | LR | fw | Epochs | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|--------|-------------|----------|-----------|--------|--------|
| 26 | foa_msattn | 1e-3 | 0.1 | 40/40 | 0.3949 | 1.2506 | 0.9939 | 20 | DONE |
| 27 | foa_msattn | 5e-4 | 0.1 | 40/40 | 0.4028 | 1.2317 | 0.9830 | 12 | DONE |
| 28 | foa_msattn | 1e-4 | 0.1 | 40/40 | 0.4368 | **1.2196** | **0.9848** | 16 | DONE |
| 29 | foa_msattn | 1e-3 | 0.2 | 40/40 | 0.4222 | 1.2365 | 0.9922 | 16 | DONE |
| 30 | foa_msattn | 5e-4 | 0.2 | 40/40 | 0.4033 | 1.2494 | 0.9956 | 16 | DONE |

#### FOA ChannelAttn (5/5 DONE)

| Exp | Config | LR | fw | Epochs | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|--------|-------------|----------|-----------|--------|--------|
| 31 | foa_channelattn | 1e-3 | 0.1 | 40/40 | **0.3989** | 1.2393 | 0.9872 | 16 | DONE |
| 32 | foa_channelattn | 5e-4 | 0.1 | 40/40 | 0.4035 | 1.2375 | 0.9873 | 16 | DONE |
| 33 | foa_channelattn | 1e-4 | 0.1 | 40/40 | 0.3980 | 1.2588 | 1.0006 | 16 | DONE |
| 34 | foa_channelattn | 1e-3 | 0.2 | 40/40 | 0.4011 | 1.2478 | 0.9938 | 16 | DONE |
| 35 | foa_channelattn | 5e-4 | 0.2 | 40/40 | 0.4180 | 1.2364 | 0.9909 | 16 | DONE |

#### FOA Original (20/20 DONE)

| Exp | LR | BS | dw | fw | hw | Frz | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|----|----|----|----|-----|-----|-------------|----------|-----------|--------|--------|
| 36 | 1e-3 | 32 | 1.0 | 0.1 | 0.1 | 0 | 0.4386 | 1.2257 | 0.9896 | 16 | DONE |
| 37 | 5e-4 | 32 | 1.0 | 0.1 | 0.1 | 0 | 0.4068 | 1.2283 | 0.9819 | 12 | DONE |
| 38 | 1e-4 | 32 | 1.0 | 0.1 | 0.1 | 0 | 0.3993 | 1.2416 | 0.9889 | 12 | DONE |
| 39 | 1e-3 | 16 | 1.0 | 0.1 | 0.1 | 0 | 0.4411 | 1.2321 | 0.9948 | 20 | DONE |
| 40 | 5e-4 | 16 | 1.0 | 0.1 | 0.1 | 0 | 0.4153 | 1.2223 | **0.9802** | 16 | DONE |
| 41 | 1e-3 | 32 | 1.0 | 0.2 | 0.1 | 0 | 0.4037 | 1.2362 | 0.9865 | 16 | DONE |
| 42 | 5e-4 | 32 | 1.0 | 0.2 | 0.1 | 0 | 0.3981 | 1.2384 | 0.9863 | 20 | DONE |
| 43 | 1e-3 | 32 | 1.0 | 0.1 | 0.2 | 0 | 0.4161 | 1.2451 | 0.9964 | 20 | DONE |
| 44 | 5e-4 | 32 | 1.0 | 0.1 | 0.2 | 0 | 0.4108 | 1.2442 | 0.9942 | 16 | DONE |
| 45 | 1e-3 | 32 | 1.0 | 0.2 | 0.2 | 0 | 0.4375 | 1.2237 | 0.9878 | 16 | DONE |
| 46 | 1e-3 | 32 | 0.5 | 0.1 | 0.1 | 0 | 0.4252 | 1.2377 | 0.9939 | 20 | DONE |
| 47 | 1e-3 | 32 | 2.0 | 0.1 | 0.1 | 0 | **0.3968** | 1.2358 | 0.9841 | 12 | DONE |
| 48 | 1e-3 | 32 | 1.0 | 0.05 | 0.1 | 0 | 0.4176 | 1.2270 | 0.9842 | 16 | DONE |
| 49 | 1e-3 | 32 | 1.0 | 0.1 | 0.05 | 0 | 0.4180 | 1.2354 | 0.9902 | 16 | DONE |
| 50 | 1e-3 | 32 | 1.0 | 0.1 | 0.1 | 5 | 0.4305 | 1.2322 | 0.9917 | 12 | DONE |
| 51 | 1e-3 | 32 | 1.0 | 0.1 | 0.1 | 10 | 0.4342 | 1.2302 | 0.9914 | 16 | DONE |
| 52 | 5e-4 | 32 | 1.0 | 0.2 | 0.2 | 0 | 0.4251 | 1.2404 | 0.9958 | 16 | DONE |
| 53 | 5e-4 | 32 | 0.5 | 0.2 | 0.1 | 0 | 0.4147 | 1.2248 | 0.9818 | 16 | DONE |
| 54 | 1e-4 | 32 | 1.0 | 0.2 | 0.1 | 0 | 0.4095 | 1.2432 | 0.9931 | 16 | DONE |
| 55 | 1e-4 | 16 | 1.0 | 0.1 | 0.1 | 0 | 0.4165 | 1.2258 | 0.9830 | 12 | DONE |

#### FOA v2 (5/5 DONE)

| Exp | Config | LR | dw | fw | hw | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|----|----|-------------|----------|-----------|--------|--------|
| 56 | foa_v2 | 1e-3 | 1.0 | 0.1 | 0.1 | 0.4117 | 1.2490 | 0.9978 | 20 | DONE |
| 57 | foa_v2 | 5e-4 | 1.0 | 0.1 | 0.1 | 0.4190 | 1.2416 | 0.9948 | 20 | DONE |
| 58 | foa_v2 | 1e-4 | 1.0 | 0.1 | 0.1 | 0.4146 | 1.2348 | 0.9888 | 16 | DONE |
| 59 | foa_v2 | 1e-3 | 1.0 | 0.2 | 0.1 | 0.4107 | 1.2384 | 0.9901 | 24 | DONE |
| 60 | foa_v2 | 5e-4 | 1.0 | 0.2 | 0.2 | 0.4201 | 1.2424 | 0.9957 | 24 | DONE |

---

### Bulk0408 — Pretrained Backbones & Extended Sweeps (exp56r-125, 40 epochs)

#### Pretrained ResNet-50 (5/5 DONE)

| Exp | Config | LR | BS | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|-------------|----------|-----------|--------|--------|
| 56r | pretrain_resnet | 1e-4 | 32 | 0.4709 | 1.3391 | 1.0786 | 4 | DONE |
| 57r | pretrain_resnet | 5e-5 | 32 | 0.4801 | 1.3273 | 1.0731 | 8 | DONE |
| 58r | pretrain_resnet | 5e-4 | 32 | 0.4507 | 1.3557 | 1.0842 | 20 | DONE |
| 59r | pretrain_resnet | 1e-4 | 16 | 0.5124 | 1.3187 | 1.0768 | 4 | DONE |
| 60r | pretrain_resnet | 3e-4 | 32 | 0.4729 | 1.3238 | 1.0685 | 8 | DONE |

*Note: exp56-60 have dual naming — FOA v2 and ResNet share the same exp IDs in different checkpoint directories.*

#### Pretrained ViT-B/16 (5/5 DONE)

| Exp | Config | LR | BS | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|-------------|----------|-----------|--------|--------|
| 61 | pretrain_vit | 1e-4 | 16 | **0.3903** | 1.2434 | 0.9875 | 12 | DONE |
| 62 | pretrain_vit | 5e-5 | 16 | 0.3909 | **1.2350** | **0.9818** | 8 | DONE |
| 63 | pretrain_vit | 5e-4 | 16 | 0.4147 | 1.2674 | 1.0116 | 32 | DONE |
| 64 | pretrain_vit | 1e-4 | 8 | 0.4166 | 1.2432 | 0.9953 | 12 | DONE |
| 65 | pretrain_vit | 3e-5 | 16 | 0.4053 | 1.2397 | 0.9894 | 8 | DONE |

#### Echo-Net (1/5 DONE, 4/5 ONGOING)

| Exp | Config | LR | BS | Epochs | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|--------|-------------|----------|-----------|--------|--------|
| 66 | echonet | 1e-3 | 8 | 17/40 | — | 1.2886 | — | — | ONGOING |
| 67 | echonet | 5e-4 | 16 | 20/40 | — | 1.9231 | — | — | ONGOING |
| 68 | echonet | 1e-4 | 16 | 18/40 | — | 1.8267 | — | — | ONGOING |
| 69 | echonet | 1e-3 | 16 | 17/40 | — | 1.6544 | — | — | ONGOING |
| 70 | echonet | 2e-3 | 16 | 40/40 | 0.4587 | 1.2946 | 1.0438 | 24 | DONE |

#### BatVision (5/5 DONE)

| Exp | Config | LR | BS | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|-------------|----------|-----------|--------|--------|
| 71 | batvision | 1e-3 | 32 | 0.4062 | 1.2543 | 0.9999 | 12 | DONE |
| 72 | batvision | 5e-4 | 32 | 0.4256 | **1.2273** | **0.9868** | 12 | DONE |
| 73 | batvision | 1e-4 | 32 | 0.4085 | 1.2464 | 0.9950 | 12 | DONE |
| 74 | batvision | 1e-3 | 16 | 0.4237 | 1.2341 | 0.9910 | — | DONE |
| 75 | batvision | 2e-3 | 32 | 0.4153 | 1.2358 | 0.9897 | 20 | DONE |

#### FOA CrossAttn + KL (4/5 DONE, 1/5 STALLED)

| Exp | Config | LR | fw | kl | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|-----|-------------|----------|-----------|--------|--------|
| 76 | foa_crossattn | 1e-3 | 0.1 | 0.02 | 0.4173 | 1.2360 | 0.9904 | 16 | DONE |
| 77 | foa_crossattn | 5e-4 | 0.2 | 0.005 | 0.4415 | **1.2254** | **0.9902** | 20 | DONE |
| 78 | foa_crossattn | 1e-3 | 0.05 | 0.01 | — | — | — | — | STALLED (4/40) |
| 79 | foa_crossattn | 5e-4 | 0.1 | 0.01 | 0.4063 | 1.2502 | 0.9970 | 16 | DONE |
| 80 | foa_crossattn | 1e-3 | 0.3 | 0.01 | 0.4200 | 1.2368 | 0.9918 | 20 | DONE |

#### FOA FeatBank + KL (4/5 DONE, 1/5 NEVER STARTED)

| Exp | Config | LR | fw | kl | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|-----|-------------|----------|-----------|--------|--------|
| 81 | foa_featbank | 1e-3 | 0.1 | 0.02 | **0.3986** | 1.2437 | 0.9902 | 16 | DONE |
| 82 | foa_featbank | 5e-4 | 0.2 | 0.005 | — | — | — | — | NEVER STARTED |
| 83 | foa_featbank | 1e-3 | 0.05 | 0.01 | 0.4127 | 1.2370 | 0.9897 | 16 | DONE |
| 84 | foa_featbank | 5e-4 | 0.1 | 0.01 | 0.4138 | 1.2361 | 0.9894 | 16 | DONE |
| 85 | foa_featbank | 1e-3 | 0.3 | 0.01 | 0.4256 | 1.2369 | 0.9935 | 16 | DONE |

#### FOA MSAttn + KL (3/5 DONE, 2/5 NEVER STARTED)

| Exp | Config | LR | fw | kl | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|-----|-------------|----------|-----------|--------|--------|
| 86 | foa_msattn | 1e-3 | 0.1 | 0.02 | — | — | — | — | NEVER STARTED |
| 87 | foa_msattn | 5e-4 | 0.2 | 0.005 | 0.4201 | 1.2282 | 0.9858 | 12 | DONE |
| 88 | foa_msattn | 1e-3 | 0.05 | 0.01 | 0.4228 | 1.2370 | 0.9928 | 24 | DONE |
| 89 | foa_msattn | 5e-4 | 0.1 | 0.01 | 0.4118 | 1.2451 | 0.9951 | 16 | DONE |
| 90 | foa_msattn | 1e-3 | 0.3 | 0.01 | — | — | — | — | NEVER STARTED |

#### FOA ChannelAttn + KL (4/5 DONE, 1/5 NEVER STARTED)

| Exp | Config | LR | fw | kl | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|-----|-------------|----------|-----------|--------|--------|
| 91 | foa_channelattn | 1e-3 | 0.1 | 0.02 | 0.4044 | 1.2387 | 0.9884 | 12 | DONE |
| 92 | foa_channelattn | 5e-4 | 0.2 | 0.005 | **0.3948** | 1.2524 | 0.9951 | 20 | DONE |
| 93 | foa_channelattn | 1e-3 | 0.05 | 0.01 | 0.4177 | 1.2320 | 0.9877 | 20 | DONE |
| 94 | foa_channelattn | 5e-4 | 0.1 | 0.01 | — | — | — | — | NEVER STARTED |
| 95 | foa_channelattn | 1e-3 | 0.3 | 0.01 | 0.4100 | 1.2513 | 0.9989 | 20 | DONE |

#### FOA Extended Sweep (19/26 DONE, 7/26 NEVER STARTED)

| Exp | LR | dw | fw | hw | Frz | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|----|----|----|-----|-----|-------------|----------|-----------|--------|--------|
| 96 | 2e-4 | 1.0 | 0.1 | 0.1 | 0 | 0.4236 | **1.2210** | **0.9817** | 12 | DONE |
| 97 | 3e-4 | 1.0 | 0.1 | 0.1 | 0 | **0.3949** | 1.2453 | 0.9902 | 12 | DONE |
| 98 | 7e-4 | 1.0 | 0.1 | 0.1 | 0 | — | — | — | — | NEVER STARTED |
| 99 | 1e-3 | 1.5 | 0.1 | 0.1 | 0 | 0.4333 | 1.2428 | 1.0000 | 20 | DONE |
| 100 | 1e-3 | 1.0 | 0.15 | 0.1 | 0 | 0.4357 | 1.2303 | 0.9919 | 12 | DONE |
| 101 | 1e-3 | 1.0 | 0.1 | 0.15 | 0 | 0.3995 | 1.2369 | 0.9857 | 16 | DONE |
| 102 | 5e-4 | 1.5 | 0.1 | 0.1 | 0 | — | — | — | — | NEVER STARTED |
| 103 | 5e-4 | 1.0 | 0.15 | 0.1 | 0 | 0.4166 | 1.2264 | 0.9834 | 12 | DONE |
| 104 | 5e-4 | 1.0 | 0.1 | 0.15 | 0 | 0.4138 | 1.2278 | 0.9836 | 16 | DONE |
| 105 | 1e-3 | 1.0 | 0.3 | 0.1 | 0 | 0.4137 | 1.2193 | **0.9776** | 16 | DONE |
| 106 | 1e-3 | 1.0 | 0.1 | 0.3 | 0 | — | — | — | — | NEVER STARTED |
| 107 | 5e-4 | 1.0 | 0.3 | 0.1 | 0 | 0.4144 | 1.2318 | 0.9866 | 16 | DONE |
| 108 | 5e-4 | 2.0 | 0.1 | 0.1 | 0 | 0.4552 | 1.2468 | 1.0093 | 16 | DONE |
| 109 | 1e-3 | 1.0 | 0.2 | 0.2 | 5 | 0.4359 | 1.2309 | 0.9924 | — | DONE |
| 110 | 5e-4 | 1.0 | 0.2 | 0.1 | 5 | — | — | — | — | NEVER STARTED |
| 111 | 1e-3 | 1.0 | 0.1 | 0.1 | 15 | 0.4144 | 1.2390 | 0.9916 | 16 | DONE |
| 112 | 5e-4 | 1.0 | 0.1 | 0.1 | 10 | 0.3996 | 1.2480 | 0.9935 | — | DONE |
| 113 | 1e-3 | 1.0 | 0.05 | 0.05 | 0 | 0.4095 | 1.2370 | 0.9887 | — | DONE |
| 114 | 5e-4 | 0.5 | 0.1 | 0.2 | 0 | — | — | — | — | NEVER STARTED |
| 115 | 1e-3 | 2.0 | 0.2 | 0.1 | 0 | 0.3964 | 1.2245 | 0.9761 | — | DONE |
| 116 | 3e-4 | 1.0 | 0.2 | 0.1 | 0 | 0.3910 | 1.2474 | 0.9905 | — | DONE |
| 117 | 2e-4 | 1.0 | 0.2 | 0.2 | 0 | 0.4234 | 1.2235 | 0.9835 | — | DONE |
| 118 | 7e-4 | 1.0 | 0.15 | 0.15 | 0 | — | — | — | — | NEVER STARTED |
| 119 | 1e-3 | 1.0 | 0.1 | 0.2 | 3 | 0.4090 | 1.2455 | 0.9945 | — | DONE |
| 120 | 5e-4 | 1.0 | 0.05 | 0.1 | 5 | 0.4063 | 1.2339 | 0.9856 | — | DONE |

#### EchoDiffusion + Wav2Vec (5/5 DONE)

| Exp | Config | LR | BS | Val ABS_REL | Val RMSE | Val Score | BestEp | Status |
|-----|--------|----|----|-------------|----------|-----------|--------|--------|
| 121 | echodiff+wav2vec | 1e-4 | 16 | 0.4059 | 1.2441 | 0.9926 | 8 | DONE |
| 122 | echodiff+wav2vec | 5e-4 | 16 | 0.4258 | 1.2527 | 1.0046 | 28 | DONE |
| 123 | echodiff+wav2vec | 1e-4 | 32 | **0.3863** | 1.2768 | 1.0096 | 12 | DONE |
| 124 | echodiff+wav2vec | 5e-5 | 16 | 0.4398 | 1.2479 | 1.0055 | 8 | DONE |
| 125 | echodiff+wav2vec | 1e-4 | 8 | 0.4273 | **1.2315** | **0.9902** | 20 | DONE |

---

### Bulk0415 — FOA 0415 Variants with Canonical Rotation (exp130-154, 60 epochs)

All use `rotate_canonical=true`, `w_silog=1.0`, BS=32.

#### FOA 0415 v1 — sh_dim=4, head_hidden=256 (5/5 DONE)

| Exp | LR | lambda_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|-----|-----------|--------|-------------|----------|-----------|--------|
| 130 | 1e-3 | 0.1 | 60/60 | **0.3841** | 1.2600 | 0.9972 | DONE |
| 131 | 1e-3 | 0.3 | 60/60 | 0.4094 | 1.2485 | 0.9968 | DONE |
| 132 | 5e-4 | 0.1 | 60/60 | 0.4054 | 1.2547 | 0.9999 | DONE |
| 133 | 5e-4 | 0.5 | 60/60 | 0.3882 | 1.2549 | 0.9949 | DONE |
| 134 | 1e-4 | 0.1 | 60/60 | 0.4183 | **1.2388** | **0.9927** | DONE |

#### FOA 0415 v2 — sh_dim=4, head_hidden=512 (5/5 DONE)

| Exp | LR | lambda_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|-----|-----------|--------|-------------|----------|-----------|--------|
| 135 | 1e-3 | 0.1 | 60/60 | 0.3929 | 1.2396 | 0.9856 | DONE |
| 136 | 1e-3 | 0.3 | 60/60 | 0.4339 | 1.2470 | 1.0031 | DONE |
| 137 | 5e-4 | 0.1 | 60/60 | 0.4073 | **1.2358** | **0.9872** | DONE |
| 138 | 5e-4 | 0.5 | 60/60 | **0.3834** | 1.2613 | 0.9980 | DONE |
| 139 | 1e-4 | 0.1 | 60/60 | 0.3892 | 1.2545 | 0.9949 | DONE |

#### FOA 0415 v3 — sh_dim=16, head_hidden=256 (5/5 DONE)

| Exp | LR | lambda_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|-----|-----------|--------|-------------|----------|-----------|--------|
| 140 | 1e-3 | 0.1 | 60/60 | 0.4173 | 1.2440 | 0.9960 | DONE |
| 141 | 1e-3 | 0.3 | 60/60 | 0.3971 | 1.2394 | 0.9867 | DONE |
| 142 | 5e-4 | 0.1 | 60/60 | 0.4177 | **1.2374** | **0.9915** | DONE |
| 143 | 5e-4 | 0.5 | 60/60 | **0.3970** | 1.2531 | 0.9963 | DONE |
| 144 | 1e-4 | 0.1 | 60/60 | 0.3974 | 1.2511 | 0.9950 | DONE |

#### FOA 0415 v4 — sh_dim=25, head_hidden=512 (5/5 DONE)

| Exp | LR | lambda_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|-----|-----------|--------|-------------|----------|-----------|--------|
| 145 | 1e-3 | 0.1 | 60/60 | 0.4257 | 1.2507 | 1.0032 | DONE |
| 146 | 1e-3 | 0.3 | 60/60 | 0.4052 | 1.2440 | 0.9924 | DONE |
| 147 | 5e-4 | 0.1 | 60/60 | 0.4110 | 1.2526 | 1.0001 | DONE |
| 148 | 5e-4 | 0.5 | 60/60 | **0.4020** | 1.2361 | **0.9859** | DONE |
| 149 | 1e-4 | 0.1 | 60/60 | 0.4333 | **1.2275** | 0.9892 | DONE |

#### FOA 0415 v5 — sh_dim=36, head_hidden=512 (5/5 DONE)

| Exp | LR | lambda_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|-----|-----------|--------|-------------|----------|-----------|--------|
| 150 | 1e-3 | 0.1 | 60/60 | 0.4074 | 1.2392 | 0.9897 | DONE |
| 151 | 1e-3 | 0.3 | 60/60 | 0.3969 | 1.2561 | 0.9983 | DONE |
| 152 | 5e-4 | 0.1 | 60/60 | 0.4005 | 1.2511 | 0.9959 | DONE |
| 153 | 5e-4 | 0.5 | 60/60 | 0.4103 | **1.2347** | **0.9874** | DONE |
| 154 | 1e-4 | 0.1 | 60/60 | **0.4021** | 1.2533 | 0.9979 | DONE |

---

### Bulk0416 — Pretrained ViT + FOA (exp160-169, 40 epochs)

All use `rotate_canonical=true`, BS=16.

#### Pretrained ViT FOA v1 (2/2 DONE)

| Exp | Config | LR | lambda_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|-----------|--------|-------------|----------|-----------|--------|
| 160 | pretrain_vit_foa | 1e-4 | 0.1 | 40/40 | 0.4211 | **1.2219** | **0.9817** | DONE |
| 161 | pretrain_vit_foa | 5e-5 | 0.3 | 40/40 | 0.4074 | 1.2356 | 0.9872 | DONE |

#### Pretrained ViT FOA v2 — Histogram Alignment (2/2 DONE)

| Exp | Config | LR | lambda_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|-----------|--------|-------------|----------|-----------|--------|
| 162 | pretrain_vit_foa_v2 | 1e-4 | 0.1 | 40/40 | 0.4200 | 1.2542 | 1.0039 | DONE |
| 163 | pretrain_vit_foa_v2 | 5e-5 | 0.3 | 40/40 | 0.4426 | 1.3577 | 1.0832 | DONE |

#### Pretrained ViT FOA v3 — FiLM Conditioning (2/2 DONE)

| Exp | Config | LR | lambda_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|-----------|--------|-------------|----------|-----------|--------|
| 164 | pretrain_vit_foa_v3 | 1e-4 | 0.1 | 40/40 | 0.3934 | 1.2418 | 0.9873 | DONE |
| 165 | pretrain_vit_foa_v3 | 5e-5 | 0.3 | 40/40 | **0.3817** | 1.2392 | **0.9819** | DONE |

#### Pretrained ViT FOA v4 — Multi-Scale SH (2/2 ONGOING)

| Exp | Config | LR | lambda_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|-----------|--------|-------------|----------|-----------|--------|
| 166 | pretrain_vit_foa_v4 | 1e-4 | 0.1 | 6/40 | — | 1.2680 | — | ONGOING |
| 167 | pretrain_vit_foa_v4 | 5e-5 | 0.3 | 4/40 | — | — | — | ONGOING |

#### Pretrained ViT FOA v5 — Cross-Attention (0/2 NEVER STARTED)

| Exp | Config | LR | lambda_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|-----------|--------|-------------|----------|-----------|--------|
| 168 | pretrain_vit_foa_v5 | 1e-4 | 0.1 | — | — | — | — | NEVER STARTED |
| 169 | pretrain_vit_foa_v5 | 5e-5 | 0.3 | — | — | — | — | NEVER STARTED |

> Note: exp IDs 166–169 were later reassigned to the N3 bulk (below). The PreViT v4/v5 runs above were killed or never started and the IDs recycled.

---

### Bulk0417 N3 — Energy-aware UNet Variants + Oracle (exp166–186, 60 epochs)

All use `rotate_canonical=true`, BS=32 (exp166–177) / BS=128 (exp178–186), AdamW.

#### N3 FiLM — energy-map FiLM conditioning (3/3 DONE)

| Exp | Config | LR | λ_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|------|--------|-------------|----------|-----------|--------|
| 166 | n3_film | 1e-3 | 0.1 | 60/60 | 0.3878 | 1.2525 | 0.9931 | DONE |
| 167 | n3_film | 5e-4 | 0.1 | 60/60 | 0.4152 | 1.2502 | 0.9997 | DONE |
| 168 | n3_film | 1e-3 | 0.3 | 60/60 | 0.4037 | 1.2560 | 1.0003 | DONE |

#### N3 Multi-Scale SH (3/3 DONE)

| Exp | Config | LR | λ_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|------|--------|-------------|----------|-----------|--------|
| 169 | n3_multiscale_sh | 1e-3 | 0.1 | 60/60 | 0.4092 | 1.2452 | 0.9944 | DONE |
| 170 | n3_multiscale_sh | 5e-4 | 0.1 | 60/60 | 0.4083 | **1.2392** | **0.9899** | DONE |
| 171 | n3_multiscale_sh | 1e-3 | 0.3 | 60/60 | **0.3878** | 1.2591 | 0.9977 | DONE |

#### N3 Energy Attention (3/3 DONE, 1 ONGOING)

| Exp | Config | LR | λ_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|------|--------|-------------|----------|-----------|--------|
| 172 | n3_energy_attn | 1e-3 | 0.1 | 60/60 | 0.4187 | **1.2217** | **0.9808** | DONE |
| 173 | n3_energy_attn | 5e-4 | 0.1 | 60/60 | 0.4049 | 1.2361 | 0.9867 | DONE |
| 174 | n3_energy_attn | 1e-3 | 0.3 | 60/60 | 0.4214 | 1.2391 | 0.9938 | DONE |
| 186 | n3_energy_attn | 1e-3 | 0.5 | 37/60 | 0.4524 | 1.2860 | — | ONGOING |

#### N3 Temporal Window (3/3 DONE)

| Exp | Config | LR | λ_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|------|--------|-------------|----------|-----------|--------|
| 175 | n3_temporal_window | 1e-3 | 0.1 | 60/60 | 0.4576 | 1.4040 | 1.1201 | DONE |
| 176 | n3_temporal_window | 5e-4 | 0.1 | 60/60 | 0.4802 | 1.3995 | 1.1237 | DONE |
| 177 | n3_temporal_window | 1e-3 | 0.3 | 60/60 | 0.4712 | 1.3353 | 1.0761 | DONE |

#### Oracle nc3 — binaural + GT energy map (3/3 DONE)

| Exp | Config | LR | λ_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|------|--------|-------------|----------|-----------|--------|
| 178 | foa_oracle_nc3 | 1e-3 | 0.1 | 60/60 | 0.3756 | 1.2181 | 0.9653 | DONE |
| 179 | foa_oracle_nc3 | 5e-4 | 0.1 | 60/60 | 0.4159 | **1.1886** | **0.9568** | DONE |
| 180 | foa_oracle_nc3 | 1e-3 | 0.3 | 60/60 | **0.3739** | 1.2033 | 0.9545 | DONE |

#### Oracle nc1 — GT energy map only (2/2 DONE, 1 ONGOING)

| Exp | Config | LR | λ_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|------|--------|-------------|----------|-----------|--------|
| 181 | foa_oracle_nc1 | 1e-3 | 0.1 | 60/60 | 0.4631 | 1.4557 | 1.1580 | DONE |
| 182 | foa_oracle_nc1 | 5e-4 | 0.1 | 60/60 | 0.4824 | 1.4427 | 1.1546 | DONE |
| 183 | foa_oracle_nc1 | 1e-3 | 0.3 | 38/60 | 0.4856 | 1.4684 | — | ONGOING |

#### Baseline foa_0415_v1 + ablation (0/3 DONE — all ONGOING)

| Exp | Config | LR | λ_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|------|--------|-------------|----------|-----------|--------|
| 184 | foa_0415_v1 | 1e-3 | 0.1 | 38/60 | 0.4179 | 1.3215 | — | ONGOING |
| 185 | foa_0415_v1 | 5e-4 | 0.1 | 37/60 | 0.4322 | 1.2877 | — | ONGOING |

---

### Bulk0417 N2 — Temporal FOA Decomposition (exp187–206, 40 epochs)

All use `rotate_canonical=true`, BS=128, AdamW. exp191–206 pending on a second server.

#### N2 E1 — 6-channel input (concat binaural + FOA spec, 3/3 DONE)

| Exp | Config | LR | λ_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|------|--------|-------------|----------|-----------|--------|
| 187 | n2_6ch_input | 1e-3 | 0.1 | 40/40 | 0.4134 | 1.2417 | 0.9932 | DONE |
| 188 | n2_6ch_input | 5e-4 | 0.1 | 40/40 | 0.3927 | 1.2499 | 0.9927 | DONE |
| 189 | n2_6ch_input | 1e-3 | 0.3 | 40/40 | **0.3785** | **1.2426** | **0.9834** | DONE |

#### N2 E2 — Temporal RMS supervision (1/2 DONE)

| Exp | Config | LR | λ_sh | Epochs | Val ABS_REL | Val RMSE | Val Score | Status |
|-----|--------|----|------|--------|-------------|----------|-----------|--------|
| 190 | n2_temporal_rms | 1e-3 | 0.1 | 40/40 | 0.3852 | 1.2805 | 1.0119 | DONE |
| 191 | n2_temporal_rms | 5e-4 | 0.1 | — | — | — | — | NEVER STARTED |

#### N2 E3–E8 — pending (exp192–206, 15 experiments)

Training queued via `scripts/n2_bulk.sh` on a second server. Families: E3 temporal_energy (×3), E4 dual_enc (×2), E5 foa_stft_film (×2), E6 temporal_rms_film (×2), E7 temap_input (×3), E8 tbin_crossattn (×3).

---

## Training Status Summary

| Model Family | Total | DONE | ONGOING | NEVER STARTED | Best Val ABS_REL | Best Val RMSE | Best Val Score |
|--------------|-------|------|---------|---------------|------------------|---------------|----------------|
| Baseline UNet | 5 | 5 | 0 | 0 | 0.3815 (exp04) | 1.2228 (exp02) | 0.9885 (exp01) |
| AudioDepthViT | 5 | 5 | 0 | 0 | 0.4312 (exp07) | 1.2424 (exp08) | 1.0067 (exp08) |
| EchoDiffusion | 5 | 5 | 0 | 0 | 0.3815 (exp11) | 1.2304 (exp14) | 0.9889 (exp14) |
| EchoDiff+Wav2Vec | 5 | 5 | 0 | 0 | 0.3863 (exp123) | 1.2315 (exp125) | 0.9902 (exp125) |
| FOA CrossAttn | 10 | 9 | 1 | 0 | 0.3975 (exp17) | 1.2198 (exp18) | 0.9822 (exp18) |
| FOA FeatBank | 10 | 9 | 0 | 1 | 0.3953 (exp25) | 1.2281 (exp25) | 0.9872 (exp25) |
| FOA MSAttn | 10 | 8 | 0 | 2 | 0.3949 (exp26) | 1.2196 (exp28) | 0.9830 (exp27) |
| FOA ChannelAttn | 10 | 9 | 0 | 1 | 0.3948 (exp92) | 1.2320 (exp93) | 0.9872 (exp31) |
| FOA Original | 39 | 39 | 0 | 0 | 0.3949 (exp97) | **1.2193 (exp105)** | **0.9761 (exp115)** |
| FOA v2 | 5 | 5 | 0 | 0 | 0.4107 (exp59) | 1.2348 (exp58) | 0.9888 (exp58) |
| Pretrained ResNet | 5 | 5 | 0 | 0 | 0.4507 (exp58r) | 1.3142 (exp59r) | 1.0685 (exp60r) |
| Pretrained ViT | 5 | 5 | 0 | 0 | 0.3903 (exp61) | 1.2350 (exp62) | 0.9818 (exp62) |
| Echo-Net | 5 | 1 | 4 | 0 | 0.4587 (exp70) | 1.2946 (exp70) | 1.0438 (exp70) |
| BatVision | 5 | 5 | 0 | 0 | 0.4062 (exp71) | 1.2273 (exp72) | 0.9868 (exp72) |
| FOA 0415 v1-v5 | 25 | 25 | 0 | 0 | 0.3834 (exp138) | 1.2275 (exp149) | 0.9856 (exp135) |
| PreViT+FOA v1-v5 | 10 | 6 | 2 | 2 | **0.3817 (exp165)** | **1.2219 (exp160)** | **0.9817 (exp160)** |
| **TOTAL** | **169** | **148** | **8** | **6** | | | |

---

## Top 10 by Validation Score (lower is better)

★ = oracle (GT FOA info at inference — not comparable to deployable models).

| Rank | Exp | Model | Val Score | Val ABS_REL | Val RMSE | BestEp |
|------|-----|-------|-----------|-------------|----------|--------|
| 1 ★ | 180 | Oracle nc3 (λ_sh=0.3) | **0.9545** | 0.3739 | 1.2033 | — |
| 2 ★ | 179 | Oracle nc3 (lr=5e-4) | 0.9568 | 0.4159 | **1.1886** | — |
| 3 ★ | 178 | Oracle nc3 (lr=1e-3) | 0.9653 | 0.3756 | 1.2181 | — |
| 4 | 115 | FOA (dw=2.0,fw=0.2) | 0.9761 | 0.3964 | 1.2245 | — |
| 5 | 105 | FOA (fw=0.3) | 0.9776 | 0.4137 | 1.2193 | 16 |
| 6 | 40 | FOA (bs=16) | 0.9802 | 0.4153 | 1.2223 | 16 |
| 7 | 172 | N3 energy_attn | 0.9808 | 0.4187 | 1.2217 | — |
| 8 | 96 | FOA (lr=2e-4) | 0.9817 | 0.4236 | 1.2210 | 12 |
| 9 | 160 | PreViT+FOA v1 | 0.9817 | 0.4211 | 1.2219 | — |
| 10 | 53 | FOA (dw=0.5,fw=0.2) | 0.9818 | 0.4147 | 1.2248 | 16 |

Deployable-only leader: **exp172 (N3 energy_attn, Val Score 0.9808)** — edges out exp115 on val RMSE (1.2217 vs 1.2245).

---

## Top 10 by Validation ABS_REL (lower is better)

★ = oracle (GT FOA info at inference — not comparable to deployable models).

| Rank | Exp | Model | Val ABS_REL | Val RMSE | Val Score |
|------|-----|-------|-------------|----------|-----------|
| 1 ★ | 180 | Oracle nc3 (λ_sh=0.3) | **0.3739** | 1.2033 | 0.9545 |
| 2 ★ | 178 | Oracle nc3 (lr=1e-3) | 0.3756 | 1.2181 | 0.9653 |
| 3 | 189 | N2 6ch (λ_sh=0.3) | 0.3785 | 1.2426 | 0.9834 |
| 4 | 04 | Baseline | 0.3815 | 1.2680 | 1.0020 |
| 5 | 11 | EchoDiffusion | 0.3815 | 1.2784 | 1.0093 |
| 6 | 165 | PreViT+FOA v3 | 0.3817 | 1.2392 | 0.9819 |
| 7 | 02 | Baseline | 0.3829 | 1.2535 | 0.9923 |
| 8 | 138 | FOA0415 v2 | 0.3834 | 1.2613 | 0.9980 |
| 9 | 130 | FOA0415 v1 | 0.3841 | 1.2600 | 0.9972 |
| 10 | 190 | N2 temporal_rms | 0.3852 | 1.2805 | 1.0119 |

Deployable-only leader on ABS_REL: **exp189 (N2 6ch λ_sh=0.3, 0.3785)** — also the best deployable Val Score in the non-oracle family.

---

## Never-Started Experiments (10 total)

All killed before reaching them during `bulk0408_65exps.sh`:

| Exp | Config | Reason |
|-----|--------|--------|
| 82 | foa_featbank | Killed (bulk0408) |
| 86 | foa_msattn | Killed (bulk0408) |
| 90 | foa_msattn | Killed (bulk0408) |
| 94 | foa_channelattn | Killed (bulk0408) |
| 98 | foa | Killed (bulk0408) |
| 102 | foa | Killed (bulk0408) |
| 106 | foa | Killed (bulk0408) |
| 110 | foa | Killed (bulk0408) |
| 114 | foa | Killed (bulk0408) |
| 118 | foa | Killed (bulk0408) |


---

## Master Ranking — all trained experiments (updated 2026-04-21)

Single source of truth. Every experiment with a train log in
`logs/{summary_train,n1_train,n2_train,n3_train}/` is listed exactly once.
DONE rows are ranked by the final `Best score` (lower is better: combined
depth+foa+val metric reported at the end of training). IN_PROGRESS rows are
listed separately with their latest epoch.

Note: indices 56–60 intentionally appear twice — two separate waves reused
those numbers (resnet family vs. foav2 family). Rows keep the full
experiment name so the two series are distinguishable.

Legend: Rank · Idx · Experiment name · Status · Best score ↓ · RMSE · ABS_REL · Dir

### Completed training runs (ranked by Best score ↓)

| Rank | Idx | Experiment | Status | Score | RMSE | ABS_REL | Dir |
|---|---|---|---|---|---|---|---|
| 1 | 219 | exp219_pvit_oracle_nc3_lr1e4_lsh0.1 | DONE | 0.9511 | 1.1994 | 0.3715 | n3_train |
| 2 | 180 | exp180_oracle_nc3_lr1e3_lsh0.3 | DONE | 0.9545 | 1.2033 | 0.3739 | summary_train |
| 3 | 179 | exp179_oracle_nc3_lr5e4_lsh0.1 | DONE | 0.9568 | 1.1886 | 0.4159 | summary_train |
| 4 | 202 | exp202_n2_temap_lr5e4_lsh0.1 | DONE | 0.9582 | 1.2036 | 0.3855 | n2_train |
| 5 | 201 | exp201_n2_temap_lr1e3_lsh0.1 | DONE | 0.9585 | 1.2022 | 0.3897 | n2_train |
| 6 | 241 | exp241_n1_pvit_temap_lr1e4_lsh0.1 | DONE | 0.9607 | 1.1960 | 0.4118 | n1_train |
| 7 | 243 | exp243_n1_pvit_temap_lr1e4_lsh0.3 | DONE | 0.9631 | 1.2066 | 0.3949 | n1_train |
| 8 | 178 | exp178_oracle_nc3_lr1e3_lsh0.1 | DONE | 0.9653 | 1.2181 | 0.3756 | summary_train |
| 9 | 203 | exp203_n2_temap_lr1e3_lsh0.3 | DONE | 0.9661 | 1.2096 | 0.3978 | n2_train |
| 10 | 115 | exp115_foa_lr1e3_dw2.0_fw0.2_hw0.1 | DONE | 0.9761 | 1.2245 | 0.3964 | summary_train |
| 11 | 242 | exp242_n1_pvit_temap_lr5e5_lsh0.1 | DONE | 0.9765 | 1.2168 | 0.4158 | n1_train |
| 12 | 105 | exp105_foa_lr1e3_fw0.3_hw0.1 | DONE | 0.9776 | 1.2193 | 0.4137 | summary_train |
| 13 | 215 | exp215_n3eattn_distill_lkd0.5 | DONE | 0.9790 | 1.2280 | 0.3979 | n3_train |
| 14 | 216 | exp216_pvit_eattn_lr1e4_lsh0.1 | DONE | 0.9796 | 1.2285 | 0.3987 | n3_train |
| 15 | 40 | exp40_foa_lr5e4_bs16_dw1.0_fw0.1_hw0.1 | DONE | 0.9802 | 1.2223 | 0.4153 | summary_train |
| 16 | 230 | exp230_pvit_distill_lkd0.5 | DONE | 0.9806 | 1.2207 | 0.4206 | n3_train |
| 17 | 172 | exp172_n3eattn_lr1e3_lsh0.1 | DONE | 0.9808 | 1.2217 | 0.4187 | summary_train |
| 18 | 160 | exp160_pvitfoav1_lr1e4_w0.1 | DONE | 0.9817 | 1.2219 | 0.4211 | summary_train |
| 19 | 96 | exp96_foa_lr2e4_dw1.0_fw0.1_hw0.1 | DONE | 0.9817 | 1.2210 | 0.4236 | summary_train |
| 20 | 53 | exp53_foa_lr5e4_dw0.5_fw0.2_hw0.1 | DONE | 0.9818 | 1.2248 | 0.4147 | summary_train |
| 21 | 62 | exp62_vit_lr5e5_bs16 | DONE | 0.9818 | 1.2350 | 0.3909 | summary_train |
| 22 | 165 | exp165_pvitfoav3_lr5e5_w0.3 | DONE | 0.9819 | 1.2392 | 0.3817 | summary_train |
| 23 | 37 | exp37_foa_lr5e4_dw1.0_fw0.1_hw0.1 | DONE | 0.9819 | 1.2283 | 0.4068 | summary_train |
| 24 | 18 | exp18_crossattn_lr1e4_fw0.1 | DONE | 0.9822 | 1.2198 | 0.4280 | summary_train |
| 25 | 27 | exp27_msattn_lr5e4_fw0.1 | DONE | 0.9830 | 1.2317 | 0.4028 | summary_train |
| 26 | 55 | exp55_foa_lr1e4_bs16_dw1.0_fw0.1_hw0.1 | DONE | 0.9830 | 1.2258 | 0.4165 | summary_train |
| 27 | 103 | exp103_foa_lr5e4_fw0.15_hw0.1 | DONE | 0.9834 | 1.2264 | 0.4166 | summary_train |
| 28 | 189 | exp189_n2_6ch_lr1e3_lsh0.3 | DONE | 0.9834 | 1.2426 | 0.3785 | n2_train |
| 29 | 117 | exp117_foa_lr2e4_dw1.0_fw0.2_hw0.2 | DONE | 0.9835 | 1.2235 | 0.4234 | summary_train |
| 30 | 104 | exp104_foa_lr5e4_fw0.1_hw0.15 | DONE | 0.9836 | 1.2278 | 0.4138 | summary_train |
| 31 | 47 | exp47_foa_lr1e3_dw2.0_fw0.1_hw0.1 | DONE | 0.9841 | 1.2358 | 0.3968 | summary_train |
| 32 | 48 | exp48_foa_lr1e3_dw1.0_fw0.05_hw0.1 | DONE | 0.9842 | 1.2270 | 0.4176 | summary_train |
| 33 | 218 | exp218_pvit_film_dw2_lr1e4_lsh0.1 | DONE | 0.9843 | 1.2365 | 0.3958 | n3_train |
| 34 | 28 | exp28_msattn_lr1e4_fw0.1 | DONE | 0.9848 | 1.2196 | 0.4368 | summary_train |
| 35 | 120 | exp120_foa_lr5e4_fw0.05_hw0.1_freeze5 | DONE | 0.9856 | 1.2339 | 0.4063 | summary_train |
| 36 | 135 | exp135_foa0415v2_lr1e3_lsh0.1 | DONE | 0.9856 | 1.2396 | 0.3929 | summary_train |
| 37 | 101 | exp101_foa_lr1e3_fw0.1_hw0.15 | DONE | 0.9857 | 1.2369 | 0.3995 | summary_train |
| 38 | 87 | exp87_msattn_lr5e4_fw0.2_kl0.005 | DONE | 0.9858 | 1.2282 | 0.4201 | summary_train |
| 39 | 148 | exp148_foa0415v4_lr5e4_lsh0.5 | DONE | 0.9859 | 1.2361 | 0.4020 | summary_train |
| 40 | 184 | exp184_v1base_lr1e3_lsh0.1 | DONE | 0.9863 | 1.2276 | 0.4231 | summary_train |
| 41 | 42 | exp42_foa_lr5e4_dw1.0_fw0.2_hw0.1 | DONE | 0.9863 | 1.2384 | 0.3981 | summary_train |
| 42 | 222 | exp222_n3mssh_lr1e3_lsh0.7 | DONE | 0.9864 | 1.2346 | 0.4074 | n3_train |
| 43 | 41 | exp41_foa_lr1e3_dw1.0_fw0.2_hw0.1 | DONE | 0.9865 | 1.2362 | 0.4037 | summary_train |
| 44 | 107 | exp107_foa_lr5e4_fw0.3_hw0.1 | DONE | 0.9866 | 1.2318 | 0.4144 | summary_train |
| 45 | 141 | exp141_foa0415v3_lr1e3_lsh0.3 | DONE | 0.9867 | 1.2394 | 0.3971 | summary_train |
| 46 | 173 | exp173_n3eattn_lr5e4_lsh0.1 | DONE | 0.9867 | 1.2361 | 0.4049 | summary_train |
| 47 | 229 | exp229_n3mssh_eattn_sh9_lr1e3_lsh0.3 | DONE | 0.9868 | 1.2374 | 0.4022 | n3_train |
| 48 | 72 | exp72_batvision_lr5e4_bs32 | DONE | 0.9868 | 1.2273 | 0.4256 | summary_train |
| 49 | 225 | exp225_n3mssh_freeze10_lr1e3_lsh0.3 | DONE | 0.9869 | 1.2287 | 0.4227 | n3_train |
| 50 | 200 | exp200_n2_trms_film_lr5e4_lsh0.1 | DONE | 0.9870 | 1.2362 | 0.4055 | n2_train |
| 51 | 137 | exp137_foa0415v2_lr5e4_lsh0.1 | DONE | 0.9872 | 1.2358 | 0.4073 | summary_train |
| 52 | 161 | exp161_pvitfoav1_lr5e5_w0.3 | DONE | 0.9872 | 1.2356 | 0.4074 | summary_train |
| 53 | 217 | exp217_pvit_mssh_lr1e4_lsh0.1 | DONE | 0.9872 | 1.2358 | 0.4073 | n3_train |
| 54 | 25 | exp25_featbank_lr5e4_fw0.2 | DONE | 0.9872 | 1.2409 | 0.3953 | summary_train |
| 55 | 31 | exp31_channelattn_lr1e3_fw0.1 | DONE | 0.9872 | 1.2393 | 0.3989 | summary_train |
| 56 | 164 | exp164_pvitfoav3_lr1e4_w0.1 | DONE | 0.9873 | 1.2418 | 0.3934 | summary_train |
| 57 | 32 | exp32_channelattn_lr5e4_fw0.1 | DONE | 0.9873 | 1.2375 | 0.4035 | summary_train |
| 58 | 153 | exp153_foa0415v5_lr5e4_lsh0.5 | DONE | 0.9874 | 1.2347 | 0.4103 | summary_train |
| 59 | 61 | exp61_vit_lr1e4_bs16 | DONE | 0.9875 | 1.2434 | 0.3903 | summary_train |
| 60 | 93 | exp93_channelattn_lr1e3_fw0.05_kl0.01 | DONE | 0.9877 | 1.2320 | 0.4177 | summary_train |
| 61 | 192 | exp192_n2_tenergy_lr1e3_lsh0.1 | DONE | 0.9878 | 1.2305 | 0.4214 | n2_train |
| 62 | 45 | exp45_foa_lr1e3_dw1.0_fw0.2_hw0.2 | DONE | 0.9878 | 1.2237 | 0.4375 | summary_train |
| 63 | 19 | exp19_crossattn_lr1e3_fw0.2 | DONE | 0.9883 | 1.2350 | 0.4126 | summary_train |
| 64 | 91 | exp91_channelattn_lr1e3_fw0.1_kl0.02 | DONE | 0.9884 | 1.2387 | 0.4044 | summary_train |
| 65 | 01 | exp01_baseline_lr1e3_bs32 | DONE | 0.9885 | 1.2396 | 0.4026 | summary_train |
| 66 | 208 | exp208_n3eattn_dw2_lr1e3_lsh0.1 | DONE | 0.9886 | 1.2210 | 0.4463 | n3_train |
| 67 | 113 | exp113_foa_lr1e3_dw1.0_fw0.05_hw0.05 | DONE | 0.9887 | 1.2370 | 0.4095 | summary_train |
| 68 | 58 | exp58_foav2_lr1e4_dw1.0_fw0.1_hw0.1 | DONE | 0.9888 | 1.2348 | 0.4146 | summary_train |
| 69 | 14 | exp14_echodiff_lr5e4_bs32 | DONE | 0.9889 | 1.2372 | 0.4096 | summary_train |
| 70 | 38 | exp38_foa_lr1e4_dw1.0_fw0.1_hw0.1 | DONE | 0.9889 | 1.2416 | 0.3993 | summary_train |
| 71 | 149 | exp149_foa0415v4_lr1e4_lsh0.1 | DONE | 0.9892 | 1.2275 | 0.4333 | summary_train |
| 72 | 195 | exp195_n2_dual_lr1e3_lsh0.1 | DONE | 0.9893 | 1.2411 | 0.4018 | n2_train |
| 73 | 194 | exp194_n2_tenergy_lr1e3_lsh0.3 | DONE | 0.9894 | 1.2337 | 0.4193 | n2_train |
| 74 | 214 | exp214_n3eattn_sh9_lr1e3_lsh0.1 | DONE | 0.9894 | 1.2475 | 0.3870 | n3_train |
| 75 | 65 | exp65_vit_lr3e5_bs16 | DONE | 0.9894 | 1.2397 | 0.4053 | summary_train |
| 76 | 84 | exp84_featbank_lr5e4_fw0.1_hw0.2_kl0.01 | DONE | 0.9894 | 1.2361 | 0.4138 | summary_train |
| 77 | 21 | exp21_featbank_lr1e3_fw0.1 | DONE | 0.9895 | 1.2283 | 0.4326 | summary_train |
| 78 | 36 | exp36_foa_lr1e3_dw1.0_fw0.1_hw0.1 | DONE | 0.9896 | 1.2257 | 0.4386 | summary_train |
| 79 | 150 | exp150_foa0415v5_lr1e3_lsh0.1 | DONE | 0.9897 | 1.2392 | 0.4074 | summary_train |
| 80 | 75 | exp75_batvision_lr2e3_bs32 | DONE | 0.9897 | 1.2358 | 0.4153 | summary_train |
| 81 | 83 | exp83_featbank_lr1e3_fw0.05_kl0.01 | DONE | 0.9897 | 1.2370 | 0.4127 | summary_train |
| 82 | 170 | exp170_n3mssh_lr5e4_lsh0.1 | DONE | 0.9899 | 1.2392 | 0.4083 | summary_train |
| 83 | 59 | exp59_foav2_lr1e3_dw1.0_fw0.2_hw0.1 | DONE | 0.9901 | 1.2384 | 0.4107 | summary_train |
| 84 | 125 | exp125_echodiff_wav2vec_lr1e4_bs8 | DONE | 0.9902 | 1.2315 | 0.4273 | summary_train |
| 85 | 49 | exp49_foa_lr1e3_dw1.0_fw0.1_hw0.05 | DONE | 0.9902 | 1.2354 | 0.4180 | summary_train |
| 86 | 77 | exp77_crossattn_lr5e4_fw0.2_kl0.005 | DONE | 0.9902 | 1.2254 | 0.4415 | summary_train |
| 87 | 81 | exp81_featbank_lr1e3_fw0.1_kl0.02 | DONE | 0.9902 | 1.2437 | 0.3986 | summary_train |
| 88 | 97 | exp97_foa_lr3e4_dw1.0_fw0.1_hw0.1 | DONE | 0.9902 | 1.2453 | 0.3949 | summary_train |
| 89 | 213 | exp213_n3mssh_eattn_lr1e3_lsh0.1 | DONE | 0.9903 | 1.2379 | 0.4126 | n3_train |
| 90 | 76 | exp76_crossattn_lr1e3_fw0.1_kl0.02 | DONE | 0.9904 | 1.2360 | 0.4173 | summary_train |
| 91 | 116 | exp116_foa_lr3e4_fw0.2_hw0.1 | DONE | 0.9905 | 1.2474 | 0.3910 | summary_train |
| 92 | 207 | exp207_n3eattn_bs32_lr1e3_lsh0.1 | DONE | 0.9905 | 1.2434 | 0.4005 | n3_train |
| 93 | 35 | exp35_channelattn_lr5e4_fw0.2 | DONE | 0.9909 | 1.2364 | 0.4180 | summary_train |
| 94 | 227 | exp227_n3mssh_eattn_lr1e3_lsh0.3 | DONE | 0.9910 | 1.2442 | 0.4003 | n3_train |
| 95 | 74 | exp74_batvision_lr1e3_bs16 | DONE | 0.9910 | 1.2341 | 0.4237 | summary_train |
| 96 | 16 | exp16_crossattn_lr1e3_fw0.1 | DONE | 0.9911 | 1.2355 | 0.4206 | summary_train |
| 97 | 199 | exp199_n2_trms_film_lr1e3_lsh0.1 | DONE | 0.9912 | 1.2446 | 0.4001 | n2_train |
| 98 | 51 | exp51_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze10 | DONE | 0.9914 | 1.2302 | 0.4342 | summary_train |
| 99 | 142 | exp142_foa0415v3_lr5e4_lsh0.1 | DONE | 0.9915 | 1.2374 | 0.4177 | summary_train |
| 100 | 111 | exp111_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze15 | DONE | 0.9916 | 1.2390 | 0.4144 | summary_train |
| 101 | 226 | exp226_n3mssh_dw2_lr1e3_lsh0.3 | DONE | 0.9917 | 1.2451 | 0.4003 | n3_train |
| 102 | 50 | exp50_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze5 | DONE | 0.9917 | 1.2322 | 0.4305 | summary_train |
| 103 | 80 | exp80_crossattn_lr1e3_fw0.3_kl0.01 | DONE | 0.9918 | 1.2368 | 0.4200 | summary_train |
| 104 | 100 | exp100_foa_lr1e3_fw0.15_hw0.1 | DONE | 0.9919 | 1.2303 | 0.4357 | summary_train |
| 105 | 22 | exp22_featbank_lr5e4_fw0.1 | DONE | 0.9922 | 1.2391 | 0.4161 | summary_train |
| 106 | 29 | exp29_msattn_lr1e3_fw0.2 | DONE | 0.9922 | 1.2365 | 0.4222 | summary_train |
| 107 | 02 | exp02_baseline_lr5e4_bs32 | DONE | 0.9923 | 1.2535 | 0.3829 | summary_train |
| 108 | 109 | exp109_foa_lr1e3_dw1.0_fw0.2_hw0.2_freeze5 | DONE | 0.9924 | 1.2309 | 0.4359 | summary_train |
| 109 | 146 | exp146_foa0415v4_lr1e3_lsh0.3 | DONE | 0.9924 | 1.2440 | 0.4052 | summary_train |
| 110 | 186 | exp186_n3eattn_lr1e3_lsh0.5 | DONE | 0.9925 | 1.2446 | 0.4043 | summary_train |
| 111 | 121 | exp121_echodiff_wav2vec_lr1e4_bs16 | DONE | 0.9926 | 1.2441 | 0.4059 | summary_train |
| 112 | 134 | exp134_foa0415v1_lr1e4_lsh0.1 | DONE | 0.9927 | 1.2388 | 0.4183 | summary_train |
| 113 | 188 | exp188_n2_6ch_lr5e4_lsh0.1 | DONE | 0.9927 | 1.2499 | 0.3927 | n2_train |
| 114 | 88 | exp88_msattn_lr1e3_fw0.05_kl0.01 | DONE | 0.9928 | 1.2370 | 0.4228 | summary_train |
| 115 | 166 | exp166_n3film_lr1e3_lsh0.1 | DONE | 0.9931 | 1.2525 | 0.3878 | summary_train |
| 116 | 54 | exp54_foa_lr1e4_dw1.0_fw0.2_hw0.1 | DONE | 0.9931 | 1.2432 | 0.4095 | summary_train |
| 117 | 187 | exp187_n2_6ch_lr1e3_lsh0.1 | DONE | 0.9932 | 1.2417 | 0.4134 | n2_train |
| 118 | 112 | exp112_foa_lr5e4_dw1.0_fw0.1_hw0.1_freeze10 | DONE | 0.9935 | 1.2480 | 0.3996 | summary_train |
| 119 | 85 | exp85_featbank_lr1e3_fw0.3_kl0.01 | DONE | 0.9935 | 1.2369 | 0.4256 | summary_train |
| 120 | 209 | exp209_n3eattn_freeze15_lr1e3_lsh0.1 | DONE | 0.9937 | 1.2471 | 0.4023 | n3_train |
| 121 | 174 | exp174_n3eattn_lr1e3_lsh0.3 | DONE | 0.9938 | 1.2391 | 0.4214 | summary_train |
| 122 | 221 | exp221_n3mssh_lr1e3_lsh0.5 | DONE | 0.9938 | 1.2519 | 0.3915 | n3_train |
| 123 | 34 | exp34_channelattn_lr1e3_fw0.2 | DONE | 0.9938 | 1.2478 | 0.4011 | summary_train |
| 124 | 26 | exp26_msattn_lr1e3_fw0.1 | DONE | 0.9939 | 1.2506 | 0.3949 | summary_train |
| 125 | 46 | exp46_foa_lr1e3_dw0.5_fw0.1_hw0.1 | DONE | 0.9939 | 1.2377 | 0.4252 | summary_train |
| 126 | 44 | exp44_foa_lr5e4_dw1.0_fw0.1_hw0.2 | DONE | 0.9942 | 1.2442 | 0.4108 | summary_train |
| 127 | 169 | exp169_n3mssh_lr1e3_lsh0.1 | DONE | 0.9944 | 1.2452 | 0.4092 | summary_train |
| 128 | 119 | exp119_foa_lr1e3_fw0.1_hw0.2_freeze3 | DONE | 0.9945 | 1.2455 | 0.4090 | summary_train |
| 129 | 196 | exp196_n2_dual_lr5e4_lsh0.1 | DONE | 0.9946 | 1.2384 | 0.4258 | n2_train |
| 130 | 210 | exp210_n3eattn_lenergy0.3_lr1e3_lsh0.1 | DONE | 0.9947 | 1.2404 | 0.4214 | n3_train |
| 131 | 39 | exp39_foa_lr1e3_bs16_dw1.0_fw0.1_hw0.1 | DONE | 0.9948 | 1.2321 | 0.4411 | summary_train |
| 132 | 57 | exp57_foav2_lr5e4_dw1.0_fw0.1_hw0.1 | DONE | 0.9948 | 1.2416 | 0.4190 | summary_train |
| 133 | 133 | exp133_foa0415v1_lr5e4_lsh0.5 | DONE | 0.9949 | 1.2549 | 0.3882 | summary_train |
| 134 | 139 | exp139_foa0415v2_lr1e4_lsh0.1 | DONE | 0.9949 | 1.2545 | 0.3892 | summary_train |
| 135 | 17 | exp17_crossattn_lr5e4_fw0.1 | DONE | 0.9949 | 1.2509 | 0.3975 | summary_train |
| 136 | 144 | exp144_foa0415v3_lr1e4_lsh0.1 | DONE | 0.9950 | 1.2511 | 0.3974 | summary_train |
| 137 | 73 | exp73_batvision_lr1e4_bs32 | DONE | 0.9950 | 1.2464 | 0.4085 | summary_train |
| 138 | 89 | exp89_msattn_lr5e4_fw0.1_hw0.2_kl0.01 | DONE | 0.9951 | 1.2451 | 0.4118 | summary_train |
| 139 | 92 | exp92_channelattn_lr5e4_fw0.2_kl0.005 | DONE | 0.9951 | 1.2524 | 0.3948 | summary_train |
| 140 | 64 | exp64_vit_lr1e4_bs8 | DONE | 0.9953 | 1.2432 | 0.4166 | summary_train |
| 141 | 30 | exp30_msattn_lr5e4_fw0.2 | DONE | 0.9956 | 1.2494 | 0.4033 | summary_train |
| 142 | 60 | exp60_foav2_lr5e4_dw1.0_fw0.2_hw0.2 | DONE | 0.9957 | 1.2424 | 0.4201 | summary_train |
| 143 | 52 | exp52_foa_lr5e4_dw1.0_fw0.2_hw0.2 | DONE | 0.9958 | 1.2404 | 0.4251 | summary_train |
| 144 | 152 | exp152_foa0415v5_lr5e4_lsh0.1 | DONE | 0.9959 | 1.2511 | 0.4005 | summary_train |
| 145 | 140 | exp140_foa0415v3_lr1e3_lsh0.1 | DONE | 0.9960 | 1.2440 | 0.4173 | summary_train |
| 146 | 198 | exp198_n2_stft_lr5e4_lsh0.1 | DONE | 0.9962 | 1.2428 | 0.4209 | n2_train |
| 147 | 143 | exp143_foa0415v3_lr5e4_lsh0.5 | DONE | 0.9963 | 1.2531 | 0.3970 | summary_train |
| 148 | 43 | exp43_foa_lr1e3_dw1.0_fw0.1_hw0.2 | DONE | 0.9964 | 1.2451 | 0.4161 | summary_train |
| 149 | 131 | exp131_foa0415v1_lr1e3_lsh0.3 | DONE | 0.9968 | 1.2485 | 0.4094 | summary_train |
| 150 | 211 | exp211_n3eattn_dw2_lenergy0.3_lr1e3 | DONE | 0.9969 | 1.2525 | 0.4005 | n3_train |
| 151 | 79 | exp79_crossattn_lr5e4_fw0.1_hw0.2_kl0.01 | DONE | 0.9970 | 1.2502 | 0.4063 | summary_train |
| 152 | 130 | exp130_foa0415v1_lr1e3_lsh0.1 | DONE | 0.9972 | 1.2600 | 0.3841 | summary_train |
| 153 | 20 | exp20_crossattn_lr5e4_fw0.2 | DONE | 0.9975 | 1.2466 | 0.4161 | summary_train |
| 154 | 171 | exp171_n3mssh_lr1e3_lsh0.3 | DONE | 0.9977 | 1.2591 | 0.3878 | summary_train |
| 155 | 56 | exp56_foav2_lr1e3_dw1.0_fw0.1_hw0.1 | DONE | 0.9978 | 1.2490 | 0.4117 | summary_train |
| 156 | 154 | exp154_foa0415v5_lr1e4_lsh0.1 | DONE | 0.9979 | 1.2533 | 0.4021 | summary_train |
| 157 | 138 | exp138_foa0415v2_lr5e4_lsh0.5 | DONE | 0.9980 | 1.2613 | 0.3834 | summary_train |
| 158 | 24 | exp24_featbank_lr1e3_fw0.2 | DONE | 0.9980 | 1.2471 | 0.4169 | summary_train |
| 159 | 205 | exp205_n2_xattn_lr5e4_lsh0.1 | DONE | 0.9981 | 1.2601 | 0.3867 | n2_train |
| 160 | 151 | exp151_foa0415v5_lr1e3_lsh0.3 | DONE | 0.9983 | 1.2561 | 0.3969 | summary_train |
| 161 | 23 | exp23_featbank_lr1e4_fw0.1 | DONE | 0.9984 | 1.2426 | 0.4287 | summary_train |
| 162 | 185 | exp185_v1base_lr5e4_lsh0.1 | DONE | 0.9989 | 1.2477 | 0.4184 | summary_train |
| 163 | 95 | exp95_channelattn_lr1e3_fw0.3_kl0.01 | DONE | 0.9989 | 1.2513 | 0.4100 | summary_train |
| 164 | 167 | exp167_n3film_lr5e4_lsh0.1 | DONE | 0.9997 | 1.2502 | 0.4152 | summary_train |
| 165 | 132 | exp132_foa0415v1_lr5e4_lsh0.1 | DONE | 0.9999 | 1.2547 | 0.4054 | summary_train |
| 166 | 71 | exp71_batvision_lr1e3_bs32 | DONE | 0.9999 | 1.2543 | 0.4062 | summary_train |
| 167 | 99 | exp99_foa_lr1e3_dw1.5_fw0.1_hw0.1 | DONE | 1.0000 | 1.2428 | 0.4333 | summary_train |
| 168 | 147 | exp147_foa0415v4_lr5e4_lsh0.1 | DONE | 1.0001 | 1.2526 | 0.4110 | summary_train |
| 169 | 05 | exp05_baseline_lr5e4_bs16 | DONE | 1.0003 | 1.2443 | 0.4310 | summary_train |
| 170 | 13 | exp13_echodiff_lr1e4_bs16 | DONE | 1.0003 | 1.2559 | 0.4038 | summary_train |
| 171 | 168 | exp168_n3film_lr1e3_lsh0.3 | DONE | 1.0003 | 1.2560 | 0.4037 | summary_train |
| 172 | 33 | exp33_channelattn_lr1e4_fw0.1 | DONE | 1.0006 | 1.2588 | 0.3980 | summary_train |
| 173 | 206 | exp206_n2_xattn_lr1e3_lsh0.3 | DONE | 1.0010 | 1.2537 | 0.4115 | n2_train |
| 174 | 193 | exp193_n2_tenergy_lr5e4_lsh0.1 | DONE | 1.0016 | 1.2559 | 0.4080 | n2_train |
| 175 | 212 | exp212_n3film_eattn_lr1e3_lsh0.1 | DONE | 1.0017 | 1.2617 | 0.3950 | n3_train |
| 176 | 04 | exp04_baseline_lr1e3_bs16 | DONE | 1.0020 | 1.2680 | 0.3815 | summary_train |
| 177 | 223 | exp223_n3mssh_lr1e3_lsh0.2 | DONE | 1.0029 | 1.2698 | 0.3800 | n3_train |
| 178 | 228 | exp228_n3mssh_eattn_lenergy_lr1e3_lsh0.1 | DONE | 1.0029 | 1.2582 | 0.4072 | n3_train |
| 179 | 136 | exp136_foa0415v2_lr1e3_lsh0.3 | DONE | 1.0031 | 1.2470 | 0.4339 | summary_train |
| 180 | 145 | exp145_foa0415v4_lr1e3_lsh0.1 | DONE | 1.0032 | 1.2507 | 0.4257 | summary_train |
| 181 | 162 | exp162_pvitfoav2_lr1e4_w0.1 | DONE | 1.0039 | 1.2542 | 0.4200 | summary_train |
| 182 | 224 | exp224_n3mssh_sh9_lr1e3_lsh0.3 | DONE | 1.0043 | 1.2636 | 0.3994 | n3_train |
| 183 | 122 | exp122_echodiff_wav2vec_lr5e4_bs16 | DONE | 1.0046 | 1.2527 | 0.4258 | summary_train |
| 184 | 03 | exp03_baseline_lr1e4_bs32 | DONE | 1.0047 | 1.2631 | 0.4017 | summary_train |
| 185 | 204 | exp204_n2_xattn_lr1e3_lsh0.1 | DONE | 1.0050 | 1.2554 | 0.4209 | n2_train |
| 186 | 124 | exp124_echodiff_wav2vec_lr5e5_bs16 | DONE | 1.0055 | 1.2479 | 0.4398 | summary_train |
| 187 | 08 | exp08_vit_lr1e4_bs16 | DONE | 1.0067 | 1.2424 | 0.4566 | summary_train |
| 188 | 191 | exp191_n2_trms_lr5e4_lsh0.1 | DONE | 1.0076 | 1.2731 | 0.3880 | n2_train |
| 189 | 108 | exp108_foa_lr5e4_dw2.0_fw0.1_hw0.1 | DONE | 1.0093 | 1.2468 | 0.4552 | summary_train |
| 190 | 11 | exp11_echodiff_lr1e4_bs32 | DONE | 1.0093 | 1.2784 | 0.3815 | summary_train |
| 191 | 123 | exp123_echodiff_wav2vec_lr1e4_bs32 | DONE | 1.0096 | 1.2768 | 0.3863 | summary_train |
| 192 | 197 | exp197_n2_stft_lr1e3_lsh0.1 | DONE | 1.0111 | 1.2756 | 0.3940 | n2_train |
| 193 | 63 | exp63_vit_lr5e4_bs16 | DONE | 1.0116 | 1.2674 | 0.4147 | summary_train |
| 194 | 190 | exp190_n2_trms_lr1e3_lsh0.1 | DONE | 1.0119 | 1.2805 | 0.3852 | n2_train |
| 195 | 12 | exp12_echodiff_lr5e5_bs32 | DONE | 1.0144 | 1.2645 | 0.4309 | summary_train |
| 196 | 06 | exp06_vit_lr1e4_bs32 | DONE | 1.0152 | 1.2612 | 0.4412 | summary_train |
| 197 | 07 | exp07_vit_lr5e5_bs32 | DONE | 1.0172 | 1.2684 | 0.4312 | summary_train |
| 198 | 220 | exp220_pvit_freeze20_lr1e4_lsh0.1 | DONE | 1.0193 | 1.2805 | 0.4099 | n3_train |
| 199 | 15 | exp15_echodiff_lr1e5_bs32 | DONE | 1.0261 | 1.2823 | 0.4282 | summary_train |
| 200 | 70 | exp70_echonet_lr2e3_bs16 | DONE | 1.0438 | 1.2946 | 0.4587 | summary_train |
| 201 | 10 | exp10_vit_lr1e5_bs32 | DONE | 1.0538 | 1.3164 | 0.4410 | summary_train |
| 202 | 60 | exp60_resnet_lr3e4_bs32 | DONE | 1.0685 | 1.3238 | 0.4729 | summary_train |
| 203 | 57 | exp57_resnet_lr5e5_bs32 | DONE | 1.0731 | 1.3273 | 0.4801 | summary_train |
| 204 | 177 | exp177_n3twin_lr1e3_lsh0.3 | DONE | 1.0761 | 1.3353 | 0.4712 | summary_train |
| 205 | 59 | exp59_resnet_lr1e4_bs16 | DONE | 1.0768 | 1.3187 | 0.5124 | summary_train |
| 206 | 56 | exp56_resnet_lr1e4_bs32 | DONE | 1.0786 | 1.3391 | 0.4709 | summary_train |
| 207 | 163 | exp163_pvitfoav2_lr5e5_w0.3 | DONE | 1.0832 | 1.3577 | 0.4426 | summary_train |
| 208 | 58 | exp58_resnet_lr5e4_bs32 | DONE | 1.0842 | 1.3557 | 0.4507 | summary_train |
| 209 | 175 | exp175_n3twin_lr1e3_lsh0.1 | DONE | 1.1201 | 1.4040 | 0.4576 | summary_train |
| 210 | 176 | exp176_n3twin_lr5e4_lsh0.1 | DONE | 1.1237 | 1.3995 | 0.4802 | summary_train |
| 211 | 09 | exp09_vit_lr5e4_bs32 | DONE | 1.1262 | 1.3828 | 0.5275 | summary_train |
| 212 | 183 | exp183_oracle_nc1_lr1e3_lsh0.3 | DONE | 1.1445 | 1.4256 | 0.4884 | summary_train |
| 213 | 182 | exp182_oracle_nc1_lr5e4_lsh0.1 | DONE | 1.1546 | 1.4427 | 0.4824 | summary_train |
| 214 | 181 | exp181_oracle_nc1_lr1e3_lsh0.1 | DONE | 1.1580 | 1.4557 | 0.4631 | summary_train |

### In-progress training runs

| Idx | Experiment | Status | Last epoch | Dir |
|---|---|---|---|---|
| 231 | exp231_n2_tenergy_ov3_lr1e3_lsh0.1 | IN_PROGRESS | 12/40 | n2_train |
| 232 | exp232_n2_tenergy_ov3_lr5e4_lsh0.1 | IN_PROGRESS | 12/40 | n2_train |
| 233 | exp233_n2_tenergy_ov3_lr1e3_lsh0.3 | IN_PROGRESS | 12/40 | n2_train |
| 234 | exp234_n2_temap_ov3_lr1e3_lsh0.1 | IN_PROGRESS | 12/40 | n2_train |
| 244 | exp244_n1_pvit_trms_film_lr1e4 | IN_PROGRESS | 19/40 | n1_train |
| 245 | exp245_n1_pvit_eattn_lr1e4 | IN_PROGRESS | 19/40 | n1_train |
| 246 | exp246_n1_pvit_mssh_lr1e4 | IN_PROGRESS | 19/40 | n1_train |
| 247 | exp247_emap_unet_repeat_lr1e3 | IN_PROGRESS | 6/60 | n3_train |
| 248 | exp248_emap_unet_conv_lr1e3 | IN_PROGRESS | 6/60 | n3_train |
| 249 | exp249_emap_unet_edge_lr1e3 | IN_PROGRESS | 6/60 | n3_train |
| 250 | exp250_emap_vit_repeat_lr1e4 | IN_PROGRESS | 6/60 | n3_train |

# Findings Report — 04/25 Test-Log Analysis

Source directories
- `baseline/logs/n2_revisit_test/` — 23 baseline architecture runs (unet / vit / echodiffusion / pretrained_resnet), ERP only
- `baseline/logs/renew_test/`     — 20 renew_* + n9_0424 runs, ambisonic=ON
- `baseline/logs/n4_test/`        — 3 n4_0425 runs, ambisonic=ON

All runs evaluate on the same test split (3192 samples, 9 scenes). Test caches differ between groups: `e2314b68a4f5` (n2_revisit, no ambisonic) vs. `7027059baf06` (renew + n4, ambisonic=ON), so absolute RMSE is **not directly comparable across the two cache families** (see Finding 5).

---

## 1. Result tables

### 1.1 `n2_revisit_test` — baseline architecture sweep (no ambisonic)

| exp | model | lr | bs | best_ep | ABS_REL ↓ | RMSE ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Log10 ↓ | MAE ↓ |
|-----|-------|----|----|---------|-----------|--------|------|------|------|---------|-------|
| 350 | unet_baseline    | 1e-3 | 32  | 12 | 0.4486 | 1.2337 | 0.4952 | 0.7116 | 0.8329 | 0.1554 | 0.7857 |
| 351 | unet_baseline    | 5e-4 | 32  | 16 | 0.4573 | 1.2386 | 0.4992 | 0.7108 | 0.8310 | 0.1577 | 0.7940 |
| 352 | unet_baseline    | 1e-4 | 32  | 12 | 0.4703 | 1.2326 | 0.4992 | 0.7099 | 0.8276 | 0.1577 | 0.7894 |
| 353 | unet_baseline    | 1e-3 | 16  | 16 | 0.4666 | 1.2319 | 0.4972 | 0.7103 | 0.8301 | 0.1563 | 0.7873 |
| 354 | unet_baseline    | 5e-4 | 16  | 12 | 0.4676 | **1.2117** | 0.4988 | 0.7125 | **0.8329** | **0.1547** | **0.7808** |
| 355 | vit_baseline     | 1e-4 | 8   | 40 | 0.4989 | 1.2621 | 0.4777 | 0.6869 | 0.8144 | 0.1647 | 0.8213 |
| 356 | vit_baseline     | 5e-5 | 8   | 40 | 0.5005 | 1.2631 | 0.4836 | 0.6976 | 0.8199 | 0.1633 | 0.8170 |
| 357 | vit_baseline     | 1e-4 | 4   | 20 | 0.4703 | 1.3107 | 0.4574 | 0.6762 | 0.8077 | 0.1691 | 0.8403 |
| 358 | vit_baseline     | 5e-4 | 8   | 36 | 0.5664 | 1.3783 | 0.4373 | 0.6512 | 0.7826 | 0.1839 | 0.9007 |
| 359 | vit_baseline     | 1e-5 | 8   | 12 | 0.5130 | 1.2593 | 0.4739 | 0.6830 | 0.8111 | 0.1660 | 0.8254 |
| 360 | echodiffusion    | 1e-4 | 16  | 8  | 0.4909 | 1.2204 | 0.4945 | 0.7026 | 0.8247 | 0.1578 | 0.7944 |
| 360 | echodiffusion    | 1e-4 | 48  | 20 | 0.4774 | 1.2473 | 0.4972 | 0.7081 | 0.8250 | 0.1591 | 0.7992 |
| 361 | echodiffusion    | 5e-5 | 16  | 8  | 0.4793 | 1.2419 | 0.4926 | 0.7033 | 0.8258 | 0.1590 | 0.8010 |
| 361 | echodiffusion    | 5e-5 | 48  | 8  | 0.5748 | 1.2396 | 0.4713 | 0.6860 | 0.8110 | 0.1720 | 0.8511 |
| 362 | echodiffusion    | 1e-4 | 32  | 8  | 0.4557 | 1.2292 | 0.4923 | 0.7098 | 0.8314 | 0.1569 | 0.7907 |
| 363 | echodiffusion    | 5e-4 | 16  | 20 | 0.5035 | 1.2307 | 0.4861 | 0.6982 | 0.8218 | 0.1627 | 0.8118 |
| **363** | **echodiffusion** | **5e-4** | **48** | **28** | **0.4482** | 1.2198 | 0.4936 | 0.7097 | 0.8328 | 0.1546 | **0.7772** |
| 365 | pretrained_resnet | 1e-4 | 32  | 8  | 0.5554 | 1.3154 | 0.4688 | 0.6761 | 0.8044 | 0.1732 | 0.8663 |
| 366 | pretrained_resnet | 5e-5 | 32  | 4  | 0.5022 | 1.3332 | 0.4482 | 0.6641 | 0.7982 | 0.1751 | 0.8658 |
| 367 | pretrained_resnet | 5e-4 | 160 | 16 | 0.5171 | 1.3413 | 0.4636 | 0.6690 | 0.7982 | 0.1750 | 0.8685 |
| 367 | pretrained_resnet | 5e-4 | 32  | 8  | 0.5373 | 1.2906 | 0.4638 | 0.6784 | 0.8089 | 0.1708 | 0.8504 |
| 368 | pretrained_resnet | 1e-4 | 16  | 4  | 0.5274 | 1.2748 | 0.4722 | 0.6849 | 0.8108 | 0.1667 | 0.8298 |
| 368 | pretrained_resnet | 1e-4 | 96  | 12 | 0.5183 | 1.3213 | 0.4668 | 0.6747 | 0.8020 | 0.1716 | 0.8549 |

### 1.2 `renew_test` — renew + n9 series (ambisonic=ON)

| exp | model | lr | bs | best_ep | ABS_REL ↓ | RMSE ↓ | δ1 ↑ | Log10 ↓ | MAE ↓ | FOA_L1 ↓ | FOA_COS ↑ | FOA_DIR ↑ |
|-----|-------|----|----|---------|-----------|--------|------|---------|-------|----------|-----------|-----------|
| 301 | renew_single        | 1e-4 | 16 | 12 | 0.4492 | 1.0994 | 0.4859 | 0.1580 | 0.6847 | 0.3331 | 0.7346  | -0.1203 |
| **302** | **renew_single (v2)** | **1e-4** | **32** | **16** | 0.4615 | **1.0921** | 0.5023 | 0.1575 | 0.6798 | 0.0231 | **0.9985** | **0.9963** |
| 303 | renew_dpt_only      | 1e-4 | 32 | 12 | 0.4685 | 1.0999 | 0.4928 | 0.1589 | 0.6882 | 0.0240 | 0.9986  | 0.9964  |
| **304** | **renew_dpt_only (no-KL)** | 1e-4 | 32 | 8  | 0.5011 | **1.0696** | 0.4916 | 0.1586 | **0.6845** | 0.0248 | 0.9985  | 0.9962  |
| 305 | renew_single (radial) | 1e-4 | 32 | 28 | 0.4575 | 1.2510 | 0.5045 | 0.1580 | 0.7936 | 0.2923 | 0.8347  | 0.6250  |
| 306 | renew_single (radial v2) | 1e-4 | 32 | 18 | 0.4742 | 1.2233 | 0.5033 | 0.1554 | 0.7835 | 0.0238 | 0.9986  | 0.9965  |
| 307 | renew_dpt_only (radial) | 1e-4 | 32 | 6  | 0.4414 | 1.2540 | 0.4761 | 0.1600 | 0.8010 | 0.0275 | 0.9981  | 0.9956  |
| 308 | renew_dpt_only (radial no-KL) | 1e-4 | 32 | 8 | 0.4501 | 1.2353 | 0.4873 | 0.1568 | 0.7895 | 0.0279 | 0.9984  | 0.9960  |
| 390 | n9_0424 (A)         | 1e-4 | 8  | 12 | 0.4739 | 1.2274 | 0.4795 | 0.1597 | 0.7939 | 0.0644 | -0.0755 | 0.0130  |
| 391 | n9_0424 (A)         | 5e-5 | 32 | 8  | 0.4474 | 1.2349 | 0.4835 | 0.1569 | 0.7872 | 0.1521 | -0.3093 | -0.0165 |
| 392 | n9_0424 (B, lsh0.05) | 1e-4 | 64 | 16 | 0.4685 | 1.2198 | 0.4976 | **0.1554** | 0.7833 | 0.0110 | 0.2824  | 0.1023  |
| 396 | n9_0424 (C, lsh0.05) | 1e-4 | 64 | 20 | 0.4646 | 1.2379 | 0.4972 | 0.1575 | 0.7937 | 0.0123 | 0.4817  | 0.1158  |
| 397 | n9_0424 (C, lsh0.1)  | 1e-4 | 64 | 12 | 0.4859 | 1.2489 | 0.4765 | 0.1629 | 0.8102 | 0.0204 | 0.2982  | -0.0081 |
| 398 | n9_0424 (C, lsh0.05) | 5e-5 | 64 | 12 | 0.5135 | 1.2228 | 0.4861 | 0.1618 | 0.8058 | 0.0171 | 0.2313  | -0.0063 |
| 399 | n9_0424 (C, lsh0.05) | 3e-4 | 64 | 20 | 0.5075 | 1.2331 | 0.4822 | 0.1614 | 0.8074 | 0.0085 | 0.4705  | 0.1256  |
| 400 | n9_0424 (C, lsh0.05) | 1e-4 | 128 | 20 | 0.4711 | 1.2413 | 0.4831 | 0.1601 | 0.7994 | 0.0104 | 0.2033  | -0.0007 |
| **401** | **n9_0424 (C, lsh0.05)** | 1e-4 | 32 | 20 | **0.4350** | 1.2606 | 0.4984 | 0.1591 | 0.8000 | 0.0101 | 0.3621 | 0.0187 |

(Three exp396 reruns — `_earlypeek`, `_earlypeek_ep20`, `_fixed` — are diagnostic, see Finding 6.)

### 1.3 `n4_test` — n4_0425 ablation (ambisonic=ON)

| exp | variant | best_ep | ABS_REL ↓ | RMSE ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | MAE ↓ | FOA_L1 ↓ | FOA_COS ↑ | FOA_DIR ↑ |
|-----|---------|---------|-----------|--------|------|------|------|-------|----------|-----------|-----------|
| 400 | lam=0.0 | 12 | 0.4883 | **1.2171** | 0.4869 | 0.7032 | 0.8282 | 0.7919 | **0.0009** | **0.9997** | **0.9937** |
| **403** | **lam=0.1** | 20 | 0.4793 | 1.2301 | 0.5035 | **0.7149** | 0.8311 | 0.7900 | 0.0014 | **0.9997** | **0.9937** |
| 410 | drop=0  | 20 | **0.4605** | 1.2527 | **0.5039** | 0.7149 | 0.8291 | 0.7955 | 0.0085 | 0.4736   | 0.4444   |

---

## 2. Findings

### Finding 1 — `echodiffusion` is the best baseline architecture, not U-Net
Within `n2_revisit_test`, **exp363 (echodiff, lr 5e-4, bs 48, ep 28)** is the winner on ABS_REL (0.4482), MAE (0.7772) and Log10 (0.1546), and is statistically tied with the best U-Net on RMSE/δ1. U-Net's only standalone win is RMSE 1.2117 at exp354. ViT and pretrained_resnet are uniformly worse (ViT ABS_REL ≥ 0.47, ResNet ≥ 0.50). **Implication:** if a single baseline is needed for downstream comparisons, use echodiff lr5e-4/bs48, not U-Net.

### Finding 2 — `pretrained_resnet` consistently underperforms; do not use as baseline
All six pretrained_resnet runs land in the worst tier on every metric (ABS_REL 0.50–0.56, RMSE 1.27–1.34, δ1 ≤ 0.47). No LR or batch-size choice rescues it. Pre-training transfer appears to hurt rather than help on this audio-conditioned depth task — likely because the conditioning channel breaks ImageNet input statistics. Drop it from the architecture pool.

### Finding 3 — ViT is LR-fragile; lr 5e-4 catastrophically diverges (exp358)
ViT runs cluster at ABS_REL ~ 0.49–0.51, but **exp358 at lr 5e-4** jumps to ABS_REL 0.5664 / RMSE 1.3783, a clear divergence relative to its lr 1e-4 sibling (0.4989 / 1.2621). U-Net and echodiff at the same 5e-4 are stable. Cap ViT LR at 1e-4 for the next sweep.

### Finding 4 — n9_0424 has a FOA-learning regression vs. the original `renew` family
The non-radial renew_v2 / dpt_only runs (exp302–304, 306–308) achieve **FOA_COS 0.9981–0.9986 and FOA_DIR 0.9956–0.9965** — essentially solved. Every n9_0424 run (exp390–401) collapses to **FOA_COS ≤ 0.48** and FOA_DIR near zero, with two of them (exp390, exp391) actually negative. exp391's negative FOA_COS (-0.31) suggests an inverted prediction, not just noise. Depth quality on n9 is comparable to the renew radial variants, so the regression is in the FOA head / loss path specifically. **Investigate the n9 FOA loss vs. the renew_single v2 head before further n9 sweeps.**

### Finding 5 — "radial" variants of `renew_*` worsen RMSE by ~12% with no FOA gain
exp302/303/304 (non-radial) RMSE 1.0696–1.0999. exp305/306/307/308 (same models, "radial") RMSE 1.2233–1.2540 — a step jump matching the n9_0424 RMSE band. FOA quality is **already saturated** in the non-radial set (≥0.998), so the radial change cannot be justified by FOA; it appears to be an unambiguous depth regression. Either revert the radial change or identify the bug it introduced. (Note: the RMSE delta could also reflect a depth rescaling tied to the radial parametrisation — worth confirming the metric is computed consistently.)

### Finding 6 — exp396 reruns confirm a now-fixed FOA evaluation bug
Three diagnostic reruns of exp396 share identical depth metrics but diverge wildly on FOA:

| run | FOA_L1 | FOA_COS | FOA_DIR |
|-----|--------|---------|---------|
| `_earlypeek` (ep 8)  | 0.5615 | 0.4124 | -0.2183 |
| `_earlypeek_ep20`    | 0.5636 | 0.3814 | -0.6854 |
| `_fixed`  (ep 20)    | 0.0123 | 0.4817 |  0.1158 |
| canonical `*_test.log` (ep 20) | 0.0123 | 0.4817 | 0.1158 |

The `_earlypeek*` runs were taken **before the FOA-eval fix** and report FOA_L1 ~45× larger and a meaningless FOA_DIR. The canonical and `_fixed` numbers agree exactly. **Action:** any older n9 / renew result that quotes FOA_L1 in the 0.3–0.6 range may pre-date the fix and should be rerun before publishing.

### Finding 7 — n4_0425: dropout removal trades FOA for depth (not worth it)
- exp400 (lam=0) and exp403 (lam=0.1) both nail FOA (`COS 0.9997`, `DIR 0.9937`) — among the best FOA numbers in the entire corpus.
- exp410 (drop=0) gives the **best ABS_REL of the n4 set (0.4605)** and best δ1, but FOA_COS collapses to 0.47 and FOA_DIR to 0.44.

The drop=0 setting evidently lets the depth head over-fit at the expense of the shared FOA representation. **exp403 (lam=0.1) is the recommended n4 default**: equal-best depth among regularised variants and FOA on par with exp400.

### Finding 8 — n4_0425 lam=0.1 is currently the best-balanced model overall
exp403 reaches FOA quality on par with the best renew runs (FOA_COS 0.9997 vs. renew's 0.9986) **and** depth (ABS_REL 0.4793, δ1 0.5035) competitive with the n2 echodiff baseline — without the renew radial RMSE penalty. If a single model has to carry both depth and FOA reconstruction forward, exp403 is the pick.

### Finding 9 — exp390 evaluation crashed during visualization but metrics are valid
`exp390_n9A_lr1e4_test.log` ends with a `KeyboardInterrupt` inside `matplotlib.savefig` after writing 4 of 9 scenes' viz images. The metric block (line 333-345) and `stats_*.pt` were written before the crash, so the FOA_COS = -0.0755 is a real result, not a side-effect of the crash. Visualization for that run is incomplete on disk.

### Finding 10 — Two cache families; cross-group RMSE is not directly comparable
- `n2_revisit_test` uses cache `samples_test_erp_e2314b68a4f5.json` (no ambisonic) → RMSE band 1.21–1.38.
- `renew_test` and `n4_test` use cache `samples_test_erp_7027059baf06.json` (ambisonic=ON). Within this cache the non-radial renew runs sit at RMSE ~1.07 while everything else sits at ~1.22.

Same nominal split (3192 samples, same 9 scenes) but different preprocessing → **don't compare absolute RMSE between the two groups.** ABS_REL, δ-thresholds and Log10 are scale-invariant and remain comparable; prefer those for cross-group statements.

---

## 3. Recommended next actions

1. **Drop `pretrained_resnet`** from the baseline pool (Finding 2).
2. **Lock ViT LR at ≤ 1e-4** in the next sweep (Finding 3).
3. **Investigate the FOA head/loss in n9_0424** vs. renew_single v2; the regression from FOA_COS 0.998 → ≤ 0.48 is the largest delta in this corpus and is not explained by depth-side hyper-parameters (Finding 4).
4. **Decide whether to retain "radial" parametrisation** in renew_*; current evidence is a clear RMSE loss with no FOA gain (Finding 5).
5. **Audit older n9/renew FOA numbers** for pre-fix evaluation (Finding 6) — specifically anything with FOA_L1 > 0.3 should be rerun.
6. **Promote `exp403` (n4 lam=0.1) as the joint depth+FOA reference** going forward (Findings 7–8).
7. **Re-run exp390 visualisation** so the artefact set matches the metrics (Finding 9).

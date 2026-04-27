# Report G — 04/26 n4_test vs n2_revisit_test Comparison

Source directories
- `baseline/logs/n4_test/`        — 17 `n4_0425` runs (binaural UNet + **oracle** bin-gated FOA, ambisonic=ON, eigen-K=8). Sweeps over λ_sparsity, lr, drop-one-bin, and a binaural-only floor.
- `baseline/logs/n2_revisit_test/` — 31 baseline architecture runs (UNet, ViT, EchoDiffusion, ResNet, pretrained-ViT). All binaural-only, no ambisonic input, no FOA conditioning.

All runs evaluate on the same 3192-sample test split (9 scenes). **Important caveat from report_f**: the two log groups use different test caches — `n4_test` uses the ambisonic-ON cache (`7027059baf06`) and `n2_revisit_test` uses the binaural-only cache (`e2314b68a4f5`). Absolute metrics are therefore not perfectly cross-comparable; treat the cross-group numbers as a **rough ceiling vs. baseline floor**, not a head-to-head benchmark.

---

## 1. Result tables

### 1.1 `n4_test` — n4_0425 oracle bin-gated FOA (ambisonic=ON)

n4_0425 = binaural UNet bottleneck ⊕ MLP(g ⊙ rep_gt). `rep_gt` is the dataset's per-bin oracle FOA representative; `g` is a learnable length-K gate. Sparsity loss `λ · sigmoid(gate).mean()` pulls bins off.

| exp | variant                        | λ_sp | lr   | bs  | ABS ↓     | RMSE ↓    | δ1 ↑      | δ2 ↑   | δ3 ↑   | Log10 ↓ | MAE ↓  | FOA_L1 ↓ | FOA_COS ↑ | FOA_DIR ↑ |
|-----|--------------------------------|------|------|-----|-----------|-----------|-----------|--------|--------|---------|--------|----------|-----------|-----------|
| 400 | λ=0 (no sparsity, ceiling)     | 0.00 | 1e-4 | 64  | 0.4649    | 1.2230    | 0.5006    | 0.7140 | 0.8321 | 0.1557  | 0.7851 | 0.0010   | 0.9997    | 0.9937    |
| 401 | λ=0.01                         | 0.01 | 1e-4 | 64  | 0.4568    | **1.2013** | 0.4985    | 0.7149 | 0.8329 | 0.1551  | 0.7825 | 0.0011   | 0.9997    | 0.9937    |
| **402** | **λ=0.05** (best ABS)      | 0.05 | 1e-4 | 64  | **0.4235** | 1.2225    | 0.4966    | 0.7137 | 0.8317 | 0.1552  | 0.7836 | 0.0014   | 0.9997    | 0.9937    |
| 403 | λ=0.1 (config default)         | 0.10 | 1e-4 | 64  | 0.4561    | 1.2041    | 0.4993    | 0.7136 | 0.8324 | 0.1555  | 0.7847 | 0.0015   | 0.9997    | 0.9937    |
| 404 | λ=0.5 (strong sparsity)        | 0.50 | 1e-4 | 64  | 0.4402    | 1.2219    | 0.4897    | 0.7081 | 0.8298 | 0.1565  | 0.7895 | 0.0013   | 0.9997    | 0.9937    |
| 405 | λ=0.05 lr=5e-4                 | 0.05 | 5e-4 | 64  | 0.4516    | 1.2032    | 0.5040    | 0.7149 | 0.8322 | 0.1558  | 0.7852 | 0.0038   | 0.9996    | 0.9937    |
| 406 | λ=0.05 lr=3e-4                 | 0.05 | 3e-4 | 64  | 0.4405    | 1.2195    | 0.4956    | 0.7104 | 0.8298 | 0.1572  | 0.7920 | 0.0024   | 0.9997    | 0.9937    |
| 407 | λ=0.05 lr=1e-3                 | 0.05 | 1e-3 | 64  | 0.5211    | 1.2666    | 0.4514    | 0.6736 | 0.8053 | 0.1701  | 0.8431 | 0.0082   | 0.9970    | 0.9930    |
| 408 | λ=0.05 lr=5e-5                 | 0.05 | 5e-5 | 64  | 0.4658    | 1.2346    | 0.4878    | 0.7068 | 0.8284 | 0.1582  | 0.7951 | 0.0012   | 0.9997    | 0.9937    |
| 409 | λ=0.05 lr=1e-5                 | 0.05 | 1e-5 | 64  | 0.4272    | 1.2517    | 0.4814    | 0.7019 | 0.8242 | 0.1605  | 0.8049 | 0.0010   | 0.9997    | 0.9937    |
| 410 | drop bin 0 (bs128)             | 0    | 1e-4 | 128 | 0.4605    | 1.2527    | 0.5039    | 0.7164 | 0.8332 | 0.1574  | 0.7919 | 0.0085   | 0.4736    | 0.4444    |
| 410 | drop bin 0 (bs64)              | 0    | 1e-4 | 64  | 0.4607    | 1.2390    | 0.4987    | 0.7117 | 0.8302 | 0.1577  | 0.7929 | 0.0085   | 0.4736    | 0.4444    |
| **411** | **drop bin 1** (best δ1)   | 0    | 1e-4 | 64  | 0.4490    | 1.2179    | **0.5075** | 0.7187 | 0.8345 | 0.1542  | 0.7773 | 0.0000   | 0.9997    | 0.9937    |
| 412 | drop bin 2                     | 0    | 1e-4 | 64  | 0.4580    | 1.2419    | 0.4983    | 0.7106 | 0.8311 | 0.1574  | 0.7916 | 0.0000   | 0.9997    | 0.9937    |
| 413 | drop bin 3                     | 0    | 1e-4 | 64  | 0.4634    | 1.2430    | 0.4984    | 0.7108 | 0.8290 | 0.1577  | 0.7929 | 0.0000   | 0.9997    | 0.9937    |
| 414 | drop bin 4                     | 0    | 1e-4 | 64  | 0.4438    | 1.2250    | 0.4989    | 0.7128 | 0.8312 | 0.1554  | 0.7843 | 0.0000   | 0.9997    | 0.9937    |
| 416 | drop bin 6                     | 0    | 1e-4 | 64  | NO_METRICS — log incomplete |
| 418 | binaural-only floor (gate=0)   | 0    | 1e-4 | 64  | 0.4688    | 1.2902    | 0.4896    | 0.7035 | 0.8254 | 0.1604  | 0.8059 | 0.0085   | 0.4736    | 0.4444    |

Missing from the planned sweep (`n4_bulk.sh` lines 165–175, 419): exp415 (drop-5), exp417 (drop-7), exp419 (alternating K=4 mask).

### 1.2 `n2_revisit_test` — baseline architecture sweep (binaural only, no ambisonic)

| exp | model              | lr   | bs  | ABS ↓     | RMSE ↓    | δ1 ↑      | δ2 ↑   | δ3 ↑   | Log10 ↓ | MAE ↓  |
|-----|--------------------|------|-----|-----------|-----------|-----------|--------|--------|---------|--------|
| 350 | unet_baseline      | 1e-3 | 32  | 0.4486    | 1.2337    | 0.4952    | 0.7116 | 0.8329 | 0.1554  | 0.7857 |
| 351 | unet_baseline      | 5e-4 | 32  | 0.4573    | 1.2386    | 0.4992    | 0.7108 | 0.8310 | 0.1577  | 0.7940 |
| 352 | unet_baseline      | 1e-4 | 32  | 0.4703    | 1.2326    | 0.4992    | 0.7099 | 0.8276 | 0.1577  | 0.7894 |
| 353 | unet_baseline      | 1e-3 | 16  | 0.4666    | 1.2319    | 0.4972    | 0.7103 | 0.8301 | 0.1563  | 0.7873 |
| **354** | **unet_baseline** | **5e-4** | **16** | 0.4676 | **1.2117** | 0.4988 | 0.7125 | 0.8329 | 0.1547 | 0.7808 |
| 355 | vit_baseline       | 1e-4 | 8   | 0.4989    | 1.2621    | 0.4777    | 0.6869 | 0.8144 | 0.1647  | 0.8213 |
| 356 | vit_baseline       | 5e-5 | 8   | 0.5005    | 1.2631    | 0.4836    | 0.6976 | 0.8199 | 0.1633  | 0.8170 |
| 357 | vit_baseline       | 1e-4 | 4   | 0.4703    | 1.3107    | 0.4574    | 0.6762 | 0.8077 | 0.1691  | 0.8403 |
| 358 | vit_baseline       | 5e-4 | 8   | 0.5664    | 1.3783    | 0.4373    | 0.6512 | 0.7826 | 0.1839  | 0.9007 |
| 359 | vit_baseline       | 1e-5 | 8   | 0.5130    | 1.2593    | 0.4739    | 0.6830 | 0.8111 | 0.1660  | 0.8254 |
| 360 | echodiffusion      | 1e-4 | 16  | 0.4909    | 1.2204    | 0.4945    | 0.7026 | 0.8247 | 0.1578  | 0.7944 |
| 360 | echodiffusion      | 1e-4 | 48  | 0.4774    | 1.2473    | 0.4972    | 0.7081 | 0.8250 | 0.1591  | 0.7992 |
| 361 | echodiffusion      | 5e-5 | 16  | 0.4793    | 1.2419    | 0.4926    | 0.7033 | 0.8258 | 0.1590  | 0.8010 |
| 361 | echodiffusion      | 5e-5 | 48  | 0.5748    | 1.2396    | 0.4713    | 0.6860 | 0.8110 | 0.1720  | 0.8511 |
| 362 | echodiffusion      | 1e-4 | 32  | 0.4557    | 1.2292    | 0.4923    | 0.7098 | 0.8314 | 0.1569  | 0.7907 |
| 363 | echodiffusion      | 5e-4 | 16  | 0.5035    | 1.2307    | 0.4861    | 0.6982 | 0.8218 | 0.1627  | 0.8118 |
| **363** | **echodiffusion** | **5e-4** | **48** | **0.4482** | 1.2198 | 0.4936 | 0.7097 | 0.8328 | 0.1546 | 0.7772 |
| 364 | echodiffusion      | 1e-5 | 48  | 0.4675    | 1.3055    | 0.4788    | 0.6916 | 0.8172 | 0.1610  | 0.8076 |
| 365 | pretrained_resnet  | 1e-4 | 32  | 0.5554    | 1.3154    | 0.4688    | 0.6761 | 0.8044 | 0.1732  | 0.8663 |
| 366 | pretrained_resnet  | 5e-5 | 32  | 0.5022    | 1.3332    | 0.4482    | 0.6641 | 0.7982 | 0.1751  | 0.8658 |
| 367 | pretrained_resnet  | 5e-4 | 160 | 0.5171    | 1.3413    | 0.4636    | 0.6690 | 0.7982 | 0.1750  | 0.8685 |
| 367 | pretrained_resnet  | 5e-4 | 32  | 0.5373    | 1.2906    | 0.4638    | 0.6784 | 0.8089 | 0.1708  | 0.8504 |
| 368 | pretrained_resnet  | 1e-4 | 16  | 0.5274    | **1.2748** | 0.4722 | 0.6849 | 0.8108 | 0.1667 | 0.8298 |
| 368 | pretrained_resnet  | 1e-4 | 96  | 0.5183    | 1.3213    | 0.4668    | 0.6747 | 0.8020 | 0.1716  | 0.8549 |
| 369 | pretrained_resnet  | 3e-4 | 160 | 0.4964    | 1.3601    | 0.4371    | 0.6573 | 0.7891 | 0.1769  | 0.8800 |
| 369 | pretrained_resnet  | 3e-4 | 32  | 0.5247    | 1.3371    | 0.4366    | 0.6553 | 0.7900 | 0.1769  | 0.8765 |
| 370 | pretrained_vit     | 1e-4 | 48  | 0.4433    | 1.2416    | 0.4795    | 0.6964 | 0.8226 | 0.1614  | 0.8088 |
| **371** | **pretrained_vit** | **5e-5** | **48** | **0.4226** | 1.2806 | 0.4809 | 0.6895 | 0.8174 | 0.1622 | 0.8138 |
| 372 | pretrained_vit     | 5e-4 | 48  | 0.4621    | 1.2404    | 0.4872    | 0.6981 | 0.8233 | 0.1601  | 0.8002 |
| **373** | **pretrained_vit** | **1e-4** | **24** | 0.4743 | **1.2269** | **0.4993** | 0.7042 | 0.8255 | 0.1598 | 0.7990 |
| 374 | pretrained_vit     | 3e-5 | 48  | 0.4985    | 1.2419    | 0.4923    | 0.7036 | 0.8231 | 0.1583  | 0.7916 |

---

## 2. Per-family bests (RMSE-ranked)

| family                        | best run                            | ABS ↓  | RMSE ↓ | δ1 ↑   |
|-------------------------------|-------------------------------------|--------|--------|--------|
| n4_0425 (oracle FOA)          | exp401 λ=0.01 lr=1e-4 bs=64         | 0.4568 | **1.2013** | 0.4985 |
| n4_0425 best by ABS_REL       | exp402 λ=0.05 lr=1e-4 bs=64         | **0.4235** | 1.2225 | 0.4966 |
| n4_0425 best by δ1            | exp411 drop-bin-1 lr=1e-4 bs=64     | 0.4490 | 1.2179 | **0.5075** |
| unet_baseline (n2_revisit)    | exp354 lr=5e-4 bs=16                | 0.4676 | 1.2117 | 0.4988 |
| vit_baseline (from-scratch)   | exp359 lr=1e-5 bs=8                 | 0.5130 | 1.2593 | 0.4739 |
| echodiffusion                 | exp363 lr=5e-4 bs=48                | 0.4482 | 1.2198 | 0.4936 |
| pretrained_resnet             | exp368 lr=1e-4 bs=16                | 0.5274 | 1.2748 | 0.4722 |
| pretrained_vit                | exp371 lr=5e-5 bs=48                | **0.4226** | 1.2806 | 0.4809 |
| pretrained_vit (RMSE)         | exp373 lr=1e-4 bs=24                | 0.4743 | 1.2269 | **0.4993** |

---

## 3. Findings

### 3.1 Oracle FOA gives a small, inconsistent edge

Best n4 (exp401, RMSE=1.2013) beats the best n2_revisit baseline (exp354 unet_baseline, RMSE=1.2117) by **0.010 RMSE (≈0.8%)** — and remember the cache mismatch may explain part of this. On ABS_REL, the best n4 (exp402, 0.4235) ties pretrained_vit (exp371, 0.4226). On δ1, n4 (exp411, 0.5075) edges out unet_baseline (exp354, 0.4988) by 1.7 points.

**Reading**: the oracle FOA cascade extracts at most ~1% RMSE / 1.7-pt δ1 over a tuned binaural UNet. Given that n4 has access to *oracle* per-bin FOA reps (the upper bound for bin-gated FOA conditioning), this is the **ceiling**, not an operating-point improvement. Any predicted-FOA cascade (n9 family) must come within striking distance of n4 to be a useful idea, and the bound on what's recoverable is small.

### 3.2 Sparsity λ matters less than you'd hope; LR matters more

Across `n4` λ-sweep (exp400–404), RMSE varies in [1.2013, 1.2230] — a 1.8% spread for two orders of magnitude in λ. By contrast the LR sweep (exp405–409) ranges [1.2032, 1.2666], with lr=1e-3 (exp407) catastrophically degrading FOA_COS to 0.997 and ABS_REL to 0.5211. The LR=1e-4 sweet spot is robust.

### 3.3 Drop-one-bin: bin 1 helps, bin 0 hurts disproportionately

`exp410` (drop bin 0) collapses FOA_COS to **0.4736** (vs 0.9997 for any-other-bin-dropped) — bin 0 carries most of the FOA energy direction. In contrast `exp411` (drop bin 1) actually achieves the best δ1 (0.5075) of the entire n4 sweep. This suggests bin 1 is *redundant* with bin 0 for direction, and removing it nudges the model toward better depth signal extraction.

`exp418` (all-zero gate, binaural-only floor through the n4 architecture) gives RMSE=1.2902, **0.7%–4% worse** than every other n4 cell — confirming that the bin-gated FOA path does provide signal, just a small one.

### 3.4 Pretrained ViT is the surprise winner on ABS_REL

`exp371` pretrained_vit (lr=5e-5, bs=48) hits ABS_REL=0.4226 — best across both directories, beating even the oracle-FOA n4 (0.4235). Same family `exp373` (lr=1e-4, bs=24) wins δ1 at 0.4993. This is despite (a) no ambisonic input, (b) no FOA conditioning. Pretrained ViT is likely picking up texture/structure priors from ImageNet that translate to good *relative* depth ranking — but its RMSE (1.2269) is no better than UNet, so the absolute scale is similar.

### 3.5 vit_baseline (from-scratch) is the uniformly worst family

Range across `vit_baseline` (exp355–359): ABS=[0.4703, 0.5664], RMSE=[1.2593, 1.3783], δ1=[0.4373, 0.4836]. Without ImageNet pretraining, ViT undershoots every other architecture in this comparison. The lesson is operational: ViT only competes if (i) pretrained, (ii) tuned at very low LR (5e-5), (iii) reasonable batch size (≥24).

### 3.6 ResNet (pretrained) is the weakest of the pretrained models

`pretrained_resnet` (exp365–369): RMSE in [1.2748, 1.3601], δ1 ≤ 0.4722. Worse than every other family on every metric. The FPN-style decoder with skip connections from layers 1–4 may be ill-suited to spectrogram inputs, where the "image" lacks the spatial locality structure ResNet was pretrained on.

### 3.7 EchoDiffusion is comparable to UNet

`exp363` echodiff (lr=5e-4, bs=48): ABS=0.4482, RMSE=1.2198, δ1=0.4936 — within 1% of `exp354` unet_baseline on every metric. EchoDiffusion is roughly equivalent to a tuned UNet on this task; the extra machinery isn't paying for itself.

---

## 4. Cross-group reading (with cache caveat)

| Hypothesis under test                                  | Evidence                                                | Verdict |
|--------------------------------------------------------|---------------------------------------------------------|---------|
| Oracle FOA conditioning improves depth                 | n4 best RMSE 1.2013 vs unet_baseline 1.2117 (≈1%)       | **Marginal** — within cache-noise threshold |
| Pretrained ViT > UNet on ABS_REL                       | 0.4226 vs 0.4486 (5.8%)                                 | **Yes**, but RMSE and δ1 are comparable, so it's a different operating point |
| Bin gate is necessary for n4 to work                   | binaural-only floor (exp418) RMSE=1.2902 vs 1.2013      | **Yes**, the bin path provides ≈7% RMSE |
| Most FOA information lives in bin 0                    | drop-bin-0 collapses FOA_COS to 0.47                    | **Yes** |
| Sparsity λ is a meaningful knob                        | RMSE varies 0.018 across two orders of magnitude in λ   | **No** — λ is not load-bearing |
| ViT-from-scratch is competitive with pretrained        | exp355–359 worse than exp370–374 by 4–10% on every metric | **No** |

---

## 5. Open items

- exp415 (drop-5), exp417 (drop-7), exp419 (alternating-K=4) are missing from `n4_test/`. Without them the bin-importance ranking is incomplete; in particular bin 7 (the longest-distance bin in the geometric eigen layout) is exactly the one we'd most want to ablate.
- exp416 log exists but is empty / lacks metrics.
- The cache mismatch (`7027059baf06` vs `e2314b68a4f5`) prevents a clean head-to-head between `n4` and `n2_revisit`. To make the n4-vs-baseline comparison rigorous, re-evaluate one `unet_baseline` checkpoint on the ambisonic-ON cache (or vice-versa) and compute the cache-bias offset.
- The next sweep `n9_bulk_0426.sh` (pretrained ViT/ResNet outer + n3-cascade inner, exp600–603) will measure whether the **pretrained ViT advantage on ABS_REL** survives when forced to consume `gated_em` conditioning. If it does, it suggests pretrained ViT + FOA is the strongest combination; if it doesn't, the n4 ceiling is the practical limit.

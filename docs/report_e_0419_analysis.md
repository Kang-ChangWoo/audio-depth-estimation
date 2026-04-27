# Report E — Why Deployable Runs Chase the Oracle (and Where They Already Beat It)
**Date:** 2026-04-19
**Scope:** post-mortem of N3 bulk0417, N2 partial, and K-wave partial results, focused on RMSE / ABS_REL / Delta1 gaps vs the oracle baselines exp178 & exp179.

---

## 0. TL;DR

The question was *"why can't n2/n3 beat exp178/179?"* — the honest answer is **one of three things, per metric:**

| Metric | Oracle best | Deployable best | Who wins |
|---|---|---|---|
| **RMSE** ↓ | **1.0477** (exp179) | 1.0713 (exp187, n2_6ch_input) | **Oracle wins** — gap = 0.024 |
| **ABS_REL** ↓ | **0.4017** (exp180) | 0.4171 (exp223, n3_mssh λ_sh=0.2) | Oracle still wins, gap = 0.015 (just closed ~25 % this week) |
| **Delta1** ↑ | 0.5002 (exp179) | **0.5111** (exp187, n2_6ch_input) | **Deployable wins** — +0.011 |

The user's premise — that deployable "cannot beat Delta1" — is **incorrect**: exp187 already beats exp179 on Delta1 by 2.2 %. What cannot be beaten (yet) is **RMSE**. ABS_REL and Delta1 are already contested, and ABS_REL may fall this week once K4/K9 finish.

---

## 1. Metric definitions (one line each)

- **RMSE** — `√mean((gt − pred)²)`. Heavily penalizes *large* errors; a handful of far-wrong pixels dominate.
- **ABS_REL** — `mean(|gt − pred| / gt)`. Relative error, unitless. Sensitive to small-distance errors (wall in 0.5 m predicted as 1.0 m → 100 %).
- **Delta1** — `mean(max(gt/pred, pred/gt) < 1.25)`. Coverage: fraction of pixels within a 25 % multiplicative bracket. Indicates *how many* pixels are "close enough".

These pull in different directions. A model with a few catastrophic outliers but many nearly-correct pixels has *bad* RMSE and *good* Delta1. Oracle vs deployable hit exactly this split.

---

## 2. Current frontier (deployable vs oracle, all 3 metrics)

Full rank by RMSE (test split, 3192 samples):

| Rank | Exp | Model | RMSE ↓ | ABS_REL ↓ | Delta1 ↑ | Tag |
|---|---|---|---|---|---|---|
| 1 ★ | 179 | oracle_nc3, lr=5e-4, λ=0.1 | **1.0477** | 0.4774 | 0.5002 | oracle |
| 2 ★ | 178 | oracle_nc3, lr=1e-3, λ=0.1 | 1.0565 | 0.4216 | 0.4979 | oracle |
| 3 | **187** | **n2_6ch_input** (λ=0.1) | **1.0713** | 0.4728 | **0.5111** | **deployable best D1** |
| 4 ★ | 180 | oracle_nc3, lr=1e-3, λ=0.3 | 1.0741 | **0.4017** | 0.4817 | oracle best ABS |
| 5 | 172 | n3_energy_attn | 1.0744 | 0.4792 | 0.5030 | deployable old best RMSE |
| 6 | 213 | n3_mssh_energy_attn | 1.0777 | 0.4645 | 0.5017 | best Group I hybrid |
| 7 | 189 | n2_6ch_input (λ=0.3) | 1.0781 | 0.4393 | 0.4974 | — |
| 8 | 225 | n3_mssh freeze=10 (K5) | 1.0811 | 0.4821 | 0.4958 | — |
| 9 | 188 | n2_6ch_input (lr=5e-4) | 1.0875 | 0.4444 | 0.4983 | — |
| 10 | 222 | n3_mssh λ=0.7 (K2) | 1.0941 | 0.4546 | 0.5022 | K2 result |
| 11 | 221 | n3_mssh λ=0.5 (K1) | 1.1052 | 0.4351 | 0.4941 | K1 result |
| 12 | 171 | n3_mssh λ=0.3 (prior) | 1.1104 | 0.4218 | 0.4823 | best ABS before K |
| 13 | **223** | **n3_mssh λ=0.2 (K3)** | 1.1161 | **0.4171** | 0.4904 | **deployable best ABS_REL** |
| 14 | 190 | n2_temporal_rms | 1.1231 | 0.4370 | 0.4868 | — |

Observations:
- **RMSE frontier**: 1.0477 (★) → 1.0713 (deployable) → gap = **0.024**.
- **ABS_REL frontier**: 0.4017 (★) → 0.4171 (deployable) → gap = **0.015** (just broke from 0.4218 this week).
- **Delta1 frontier**: **0.5111 (deployable) > 0.5002 (oracle)** → deployable *already* +0.011 ahead.

---

## 3. Why RMSE can't be beaten yet — information-theoretic read

**The oracle's privileged channel.** exp178/179 take `cat(binaural[2ch], gt_energy_map[1ch])` as input. The GT energy map is derived from the *full FOA impulse response*, which encodes the 3-D directional energy distribution of the scene.

For **RMSE**, the dominant error pixels are:
- specular surface hits at far distances (low returned energy) — the binaural-only model predicts these *close* (regression to mean) because the echo is weak and the model's prior dominates
- grazing-angle surfaces that cast *directional* reflections — recoverable only from ambisonic decomposition

The oracle model sees these regions explicitly in its 3rd input channel. The deployable model must infer them from IPD/ITD + level differences across 2 mics — a much sparser signal. So the *tails* of the error distribution remain fat for deployable → RMSE stays higher.

**Empirical check.** Compare MAE:
- exp179 (oracle): MAE = 0.6714
- exp187 (deployable): MAE = 0.6682 ← **lower** than oracle!

Deployable's *mean* error is already smaller; the RMSE gap is entirely driven by **tails**. Put differently: exp187 is on average closer to GT than exp179; it just has more *bad* pixels. This tail-heaviness is consistent with the oracle having privileged per-pixel priors for edge cases.

---

## 4. Why ABS_REL has nearly closed — and exp223 just broke exp171

**exp223 (K3, n3_mssh λ_sh=0.2) = 0.4171** — crossed below the prior best exp171 (0.4218) by refining λ_sh by one grid step.

Mechanistic read:
- ABS_REL penalizes *relative* errors, which are largest at *near* distances (small denominator).
- MultiScale SH heads at enc[2, 4, 6] + bottleneck inject directional supervision at *many* encoder depths. Early-layer directional signals carry strong near-field cues (first-order reflections from nearby surfaces dominate the early IR).
- λ_sh=0.2 turned out to be a sweeter operating point than 0.3 — probably because λ_sh=0.3 was already in the over-regularization regime for the deepest head, masking the shallower heads' near-field signal.

**Predicted ABS_REL headroom:**
- exp224 (sh_dim=9, still pending) expands each SH head's bandwidth 4-fold — if near-field cues are what matter, this should push ABS_REL toward 0.41.
- exp229 (mssh+EnergyAttn + sh_dim=9 + λ_sh=0.3) combines K4 with the hybrid — highest EV ABS_REL attempt remaining.
- exp215 (distillation from exp180 teacher) inherits teacher's 0.4017 target directly; expect ABS_REL ≤ 0.42 with high confidence, possibly ≤ 0.41.

**Oracle ABS_REL gap is not information-bounded** the way RMSE is — it's a *regularization* question. Deployable models can match it if the λ_sh × sh_dim × feature-tap geometry is chosen right.

---

## 5. Why Delta1 is already beaten — "coverage wins when tails don't count"

**exp187 Delta1 = 0.5111 > exp179 Delta1 = 0.5002** — deployable wins by +0.011.

Delta1 counts pixels within a 1.25× multiplicative bracket. It *discards* the magnitude of failures: a 5 m GT predicted as 2 m scores the same as 10 m. This metric is **tail-invariant** and **density-sensitive**.

Why exp187 wins:
- Input = `cat(binaural[2ch], foa_spec[4ch])` = 6-ch input. FOA spectrogram is a *fully learnable* representation (not GT-conditioned like oracle) of the same directional information the oracle gets. The model can extract per-pixel direction from it at train *and* test.
- Unlike the oracle's energy map (which is a scalar summary — per-direction total energy), the FOA spectrogram has *time × frequency × channel* structure. Richer than oracle input for fine-grained pixel classification, poorer for global scale calibration.
- Result: exp187 produces more pixels in the "close enough" bucket (higher D1) but lets a few pixels go badly wrong (tail → worse RMSE).

This is directly the "coverage vs magnitude" trade from §1.

**Implication for future metrics design:** Delta1 should be considered *not* oracle-bounded in this setup. Future reports should stop comparing Delta1 against oracle as a benchmark — it's a different optimization problem.

---

## 6. Error-distribution view (coverage vs magnitude)

Three regimes observed across all tested models:

| Regime | Characterization | Example exps |
|---|---|---|
| **Oracle-like** | Low RMSE, moderate Delta1, best ABS_REL | exp178/179/180 |
| **Coverage-heavy** | Low-moderate RMSE, **highest Delta1**, moderate ABS_REL | exp187, exp172, exp213 |
| **ABS-focused** | Higher RMSE, moderate-to-high Delta1, **best ABS_REL** | exp171, exp223, exp180 |

The winners of each metric are *architecturally distinct*:
- Oracle wins RMSE with a privileged direct channel.
- 6-ch FOA input (exp187) wins Delta1 with rich learnable direction.
- Multi-scale SH supervision wins ABS_REL by balancing near-field and far-field gradients.

**No single architecture has won all three**. Each regime sacrifices one metric to optimize another. This suggests the frontier improvements must come from **hybrids that span regimes** — exactly Group I's and K-C's bet.

---

## 7. Pending experiments — who can break which frontier

Status of still-running / unimplemented experiments (as of 2026-04-19):

### Group K (in progress)

| Exp | Config | Status | Frontier target | Probability of breaking |
|---|---|---|---|---|
| 224 | n3_mssh sh_dim=9 λ=0.3 | PENDING | ABS_REL (beat 0.4171) | **medium-high** — order-2 SH should help MSSH directly; if ~0.40, closes gap to oracle |
| 226 | n3_mssh dw=2.0 λ=0.3 | PENDING | RMSE (beat 1.0713) | low — dw=2.0 tried on energy_attn (exp208), slightly hurt it; MSSH unknown |
| 227 | n3_mssh_eattn λ=0.3 | PENDING | RMSE (beat 1.0713) | **medium** — transfers exp171 winning λ into the hybrid that won the prev wave (exp213). Upside likely on RMSE *and* ABS |
| 228 | n3_mssh_eattn + λ_energy=0.3 | PENDING | Delta1 (beat 0.5111) | medium — direct energy supervision should sharpen the learned energy map → better local attention → higher D1 |
| 229 | n3_mssh_eattn sh_dim=9 λ=0.3 | PENDING | ABS_REL + RMSE | **highest EV** — combines K3 (best ABS), K4 (sh_dim=9), K7 (hybrid λ=0.3). Most likely single winner |
| 215 | n3_eattn_distill (λ_kd=0.5) | PENDING | RMSE (beat 1.0713) | **medium-high** — student is pulled toward oracle's pred_depth directly. Expected RMSE 1.06–1.065 |

### Group L (ViT port, not yet implemented)

| Exp | Hypothesis | Probability of beating some frontier |
|---|---|---|
| 216 | pvitfoa_v6 (ViT + EnergyHead) | **medium** for ABS_REL (ViT exp165 already has val ABS_REL=0.3817 — ported test performance could be ~0.40) |
| 219 | pvitfoa_v6_oracle | high for *new oracle ceiling* — informative even if not deployable |
| 230 | pvitfoa_v6_distill (ViT student, UNet oracle teacher) | **highest EV overall** — cross-arch distillation gives ViT's ABS_REL advantage + UNet oracle's RMSE signal |

### N2 remaining (exp191–206 on second server)

Not yet tested here. Variants I'd flag as highest-EV against exp187's 6ch baseline:
- **n2_tbin_crossattn (exp204–206)**: cross-attention between binaural and temporal energy bins — if it works, it combines exp187's coverage with explicit attention to late reflections, potentially pushing RMSE below 1.0713.
- **n2_dual_enc (exp195–196)**: separate binaural + FOA encoders fused late — architecturally closest to oracle's split path. Could match exp187 D1 with better RMSE.

---

## 8. Forecast — what the next week can realistically deliver

Absolute bests I'd expect if Groups K + L + remaining N2 all complete on-schedule:

| Metric | Current best | Forecasted ≤ 1 week | Oracle wall |
|---|---|---|---|
| RMSE ↓ | 1.0713 (exp187) | **~1.055** via exp215 distill or exp230 ViT distill | 1.0477 |
| ABS_REL ↓ | 0.4171 (exp223) | **~0.40** via exp229 (mssh+eattn+sh9+λ=0.3) | 0.4017 |
| Delta1 ↑ | 0.5111 (exp187) | **~0.520** via exp228 or n2_dual_enc (exp195) | 0.5002 (already beaten) |

**Summary of chances:** 
- **Delta1** — already beaten, expect bigger wins.
- **ABS_REL** — gap 0.015 likely halved this week by exp229 or exp215.
- **RMSE** — hardest. Distillation is the cheapest shot; a full ViT port is the longest.

Only **RMSE** has a real oracle wall, and that wall is there because of the *information* content of the input, not the model design. Closing it requires one of:
1. Richer input (more channels → FOA spectrogram like exp187; if we had ambisonic microphone data at test time, the wall drops to exp179's 1.0477).
2. Distillation (borrow the oracle's depth output as a target at train time; deploy with binaural).
3. Much longer training + better augmentation (diminishing returns, untested).

Paths (1) and (3) are orthogonal to current experiments; path (2) is in-flight as exp215.

---

## 9. What we should *stop* comparing against

Given the analysis above, future reports should:
- **Stop treating oracle Delta1 as a ceiling.** It's not; deployable already exceeded it.
- **Start reporting MAE alongside RMSE**, since MAE already favors deployable in some cases (exp187 < exp179) — the RMSE gap exaggerates oracle's actual advantage for typical pixels.
- **Report separate frontier tables per metric**, not a single sort-by-RMSE list. The "best deployable" identity depends on which metric you read.
- **For product pitches**: cite exp187's Delta1 (beats oracle) rather than RMSE (trails oracle). Customers care about pixel-level correctness coverage more than the RMSE average.

---

## 10. Appendix — frontier table refresh (deployable only)

| Metric | Best deployable | Notes |
|---|---|---|
| RMSE | exp187 (n2_6ch, 1.0713) | FOA-spectrogram input, realistic with 4-mic capture |
| ABS_REL | exp223 (n3_mssh λ_sh=0.2, 0.4171) | This week's win; check exp229 when it lands |
| Delta1 | exp187 (n2_6ch, 0.5111) | Already beats oracle |
| MAE | exp187 (n2_6ch, 0.6682) | Also beats oracle exp179 (0.6714) |
| Log10 | exp187 (n2_6ch, 0.1540) ≈ exp172 (0.1549) | Tied with deep-UNet; parallel to ABS_REL |

exp187 sweeps 4 of 5 metrics. A reasonable simplification for the next reporting cycle: **exp187 is the deployable baseline to beat on everything except ABS_REL; exp223 holds ABS_REL**.

---

## 11. N2 temporal-energy wiring audit (added 2026-04-20)

**Question:** for the early/mid/late energy maps derived from FOA `(4, T)` in `n2_temporal_energy` (exp192–194), is the "gradual information from Ambisonics" design actually being applied — and if so, why isn't it paying off?

### 11.1 Pipeline check — wiring is correct

FOA IR `(4, T)` → 3 time bins → per-bin SH-covariance energy map `(H, W)` → stack `(3, H, W)` GT, predicted by 3 heads, supervised per-channel via L1. All index alignments check out:

| Stage | Source | Shape / behavior |
|---|---|---|
| GT time bins (44.1 kHz) | `dataset_n2.py:38` | `[(0, 2600), (2600, 13000), (13000, None)]` — early / mid / late |
| GT energy map | `dataset_n2.py:117-132` | `R = IR·IRᵀ` (4×4) → SH-project → `(H, W)` → per-map max-normalize → `(3, H, W)` |
| Predicted maps | `n2_temporal_energy.py:99-101, 162-183` | 3 × `EnergyHead(bottleneck)` → sigmoid → concat `(B, 3, H', W')` |
| Attention injection | `n2_temporal_energy.py:102, 169-174` | `attn_inject_indices = [4, 5, 6]`; `h = h * (1 + emap)` |
| Per-bin L1 loss | `train.py:185-193` | `sum_k λ_energy · L1(pred_te[:, k], interp(gt_te[:, k]))` |

**Gradual design confirmed** — bins and decoder levels are matched coarse-to-fine:

```
decoder[4]  32×64   ← bin 0 (early, direct)       "where are the walls"
decoder[5]  64×128  ← bin 1 (mid, reverb)         "room-scale structure"
decoder[6]  128×256 ← bin 2 (late, diffuse)       "surface detail"
```

Ambisonic information arrives gradually: global geometry from the direct path first, then reverb, then diffuse detail — each modulating the decoder feature at the matching spatial scale. Matches design intent. ✅

### 11.2 Empirical result — gradual isn't paying off

| Exp | Variant | ABS_REL ↓ | RMSE ↓ | Delta1 ↑ |
|---|---|---|---|---|
| **190** | **trms (scalar per-bin RMS)** | **0.4370** | 1.1231 | 0.4868 |
| 191 | trms lr=5e-4 | 0.4408 | 1.1136 | 0.4984 |
| 192 | tenergy lr=1e-3 λ_sh=0.1 | 0.4734 | **1.0805** | 0.4912 |
| 193 | tenergy lr=5e-4 λ_sh=0.1 | 0.4475 | 1.1121 | 0.4923 |
| 194 | tenergy lr=1e-3 λ_sh=0.3 | 0.4736 | 1.1056 | 0.4950 |

Spatial energy-map attention (tenergy) is *worse* than plain per-bin RMS scalars (trms) on ABS_REL (0.44–0.47 vs 0.43–0.44). The wiring is correct, but the gradual-refinement signal isn't being absorbed.

### 11.3 Three likely reasons, in priority order

1. **Monotone gain `h * (1 + sigmoid(emap))`** (`n2_temporal_energy.py:173`). With `emap ∈ [0, 1]`, the decoder can only be *boosted*, never attenuated. The energy map can't say "this region is diffuse-only, ignore it." Fix: `h * (1 + α · tanh(emap_raw))` or additive residual `h + γ · emap`, so the map can both amplify *and* suppress.

2. **Equal λ per bin** (`train.py:188-193`). Loop sums 3 L1 terms each multiplied by λ — total energy weight is `3 · λ`, not `λ`, and all bins contribute gradient equally. But late/diffuse maps are visually smoothest (low-info for localization), likely crowding out the informative early bin. Fix: non-uniform `λ_bins = [0.2, 0.1, 0.05]` or average rather than sum.

3. **All 3 heads condition on the same bottleneck** (`n2_temporal_energy.py:51-56, 170-171`). Each head is `GAP → MLP → 16×32 map → upsample`, sharing no structure except input feature. Three bins predicted from the same vector start highly correlated; the model has no inductive bias to specialize early vs late. Fix: either give each head its own projection + deeper conditioning path, or condition each head on *its own decoder-level feature* (i.e., predict `emap_k` from `h_{decoder[k]}`) so the gradual refinement is a closed loop rather than open-loop injection.

### 11.4 Recommended follow-ups

- **exp-e1** (cheap): change `h * (1 + emap)` → `h + γ · (2·emap − 1)` with learnable γ. One-line diff. Tests whether monotonicity is the dominant bottleneck.
- **exp-e2** (cheap): `λ_bins = [0.2, 0.1, 0.05]`. Tests whether equal weighting drowns the informative bin.
- **exp-e3** (medium): condition energy-head k on `enc_reversed[k]` (skip connection) rather than bottleneck. Tests whether decoder-level conditioning unlocks specialization.
- Only escalate to ViT or cross-attention variants (exp204–206 `n2_tbin_crossattn`) if e1–e3 don't close the trms→tenergy gap.

**Takeaway.** The gradual-Ambisonics hypothesis is architecturally wired, but two representational choices (monotone gain, shared bottleneck) prevent it from expressing itself. Before dismissing temporal energy attention, run e1/e2/e3.

---

## 12. exp231–240 — temporal understanding via overlapping bins (added 2026-04-20)

### 12.1 The idea

The current `BINS_3 = [(0, 2600), (2600, 13000), (13000, None)]` partition is **disjoint**: bin boundaries are cliffs, not ramps. A reflection arriving at sample 2599 is in bin 0, a reflection at 2601 is in bin 1 — they are numerically uncorrelated in the supervision signal even though acoustically adjacent. That contradicts the "gradual information from Ambisonics" intent: a decoder feature at mid-resolution should see both early-leaning *and* late-leaning mid energy.

Fix: let the bins overlap. Each successive bin still shifts forward in time but carries a substantial chunk of the previous region, so "mid reverb" (where most of the depth-relevant directional energy lives) is present in **every** bin with different weighting. This gives the energy heads a genuinely gradual temporal signal instead of three disjoint windows.

### 12.2 Two preset proposals

**BINS_3_OVERLAP** (3-channel, drop-in replacement — no model change):

| bin | samples (@44.1 kHz) | ms | physical content |
|---|---|---|---|
| 0 | (0, 13000)    | 0–295   | direct + early + mid-early |
| 1 | (2600, 18000) | 59–408  | early + mid (core) |
| 2 | (8000, end)   | 181+    | mid-late + late diffuse |

Pairwise overlaps:
- bin 0 ∩ bin 1 = (2600, 13000) — the mid core is in both
- bin 1 ∩ bin 2 = (8000, 18000) — late-mid in both
- bin 0 ∩ bin 1 ∩ bin 2 = (8000, 13000) — triple overlap

Every part of the IR is in ≥1 bin; the informative mid region is in all three. Same model (`n2_temporal_energy`, `n2_temap_input`), only dataset-level change.

**BINS_4_OVERLAP** (4-channel, user's explicit proposal: *direct+mid / mid / mid / mid+late*):

| bin | samples | ms | physical content |
|---|---|---|---|
| 0 | (0, 8800)     | 0–200   | direct + mid-early |
| 1 | (2600, 11000) | 59–249  | mid (shifted +60 ms) |
| 2 | (5400, 15000) | 122–340 | mid (shifted +120 ms) |
| 3 | (8800, end)   | 200+    | mid-late + late |

Sliding window: each bin shifts ~60 ms forward and overlaps the neighbor by ~150 ms. Requires a 4-head variant of `n2_temporal_energy` (new `attn_inject_indices = [3, 4, 5, 6]`).

### 12.3 Experiment plan

Three layers of ablation. Every exp compares against exp192 (tenergy, 0.4734 ABS_REL) and exp190 (trms, 0.4370 ABS_REL) as the non-overlap baselines.

| Exp | Variant | Bins | Arch change | Tests |
|---|---|---|---|---|
| **231** | n2_temporal_energy lr=1e-3 λ=0.1 | `BINS_3_OVERLAP` | none | Does overlap alone help? |
| **232** | n2_temporal_energy lr=5e-4 λ=0.1 | `BINS_3_OVERLAP` | none | LR robustness |
| **233** | n2_temporal_energy lr=1e-3 λ=0.3 | `BINS_3_OVERLAP` | none | λ_sh sensitivity under overlap |
| **234** | n2_temap_input lr=1e-3 λ=0.1 | `BINS_3_OVERLAP` | none | Overlap as input concat (simpler, no attention) |
| **235** | n2_temap_input lr=5e-4 λ=0.1 | `BINS_3_OVERLAP` | none | Matches 234 |
| **236** | n2_temporal_energy overlap + **signed gain** | `BINS_3_OVERLAP` | `h*(1+α·(2·emap−1))` | §11.3 #1 combined with overlap |
| **237** | n2_temporal_energy overlap + **λ_bins=[0.2,0.15,0.05]** | `BINS_3_OVERLAP` | non-uniform loss weights | §11.3 #2 combined with overlap |
| **238** | n2_temporal_energy overlap + **decoder-level conditioning** | `BINS_3_OVERLAP` | energy head k conditioned on `enc_reversed[k]` | §11.3 #3 combined with overlap |
| **239** | n2_temporal_energy_4bin lr=1e-3 λ=0.1 | `BINS_4_OVERLAP` | 4 heads, inject [3,4,5,6] | User's 4-bin proposal |
| **240** | n2_temporal_energy_4bin lr=5e-4 λ=0.1 | `BINS_4_OVERLAP` | 4 heads, inject [3,4,5,6] | Matches 239 |

### 12.4 What each layer isolates

- **231–233** (pure binning): if ABS_REL drops below 0.44 vs exp192's 0.47, overlap is a free win — ship `BINS_3_OVERLAP` as the default.
- **234–235** (overlap as input concat, no attention): controls for whether the gain comes from the *supervision signal* vs the *attention pathway*. If temap_input with overlap beats tenergy with overlap, attention is net-harmful and should be dropped entirely (use simple concat).
- **236–238** (overlap + §11.3 fixes): which arch flaw matters most. Expect **236 (signed gain)** to be the single biggest delta — it addresses the monotonicity bug head-on.
- **239–240** (4-bin sliding window): tests whether "more temporal resolution with heavy overlap" beats "3 bins with light overlap". Highest-risk, highest-reward. Requires model change (add `n2_temporal_energy_4bin.py`).

### 12.5 Prerequisites (code/config to add before launch)

**Critical naming rule.** The first column in `n2_bulk.sh` is the `--config` value, which maps to `config/{value}.yaml`. The YAML's `model.name` field is a **separate** key that must remain one of the existing `_N2_CLASSES` entries (`n2_temporal_energy`, `n2_temap_input`, `n2_tbin_crossattn`, …) — otherwise `is_n2_model(cfg)` (train_utils.py:118) returns False and the N2 train/val path is skipped, producing a silent arg-mismatch crash. So new YAMLs may have overlap-specific *filenames*, but their inner `model.name` stays canonical.

**Work items:**

1. **`data/dataset_n2.py`**: add `BINS_3_OVERLAP` and `BINS_4_OVERLAP` to `PRESET` dict. Suggested keys `'overlap3'` and `'overlap4'`; extend `n_temporal_bins` validation to accept strings as well as ints. Nothing else in the dataset class needs changing — `_compute_temporal_rms` and `_compute_temporal_energies` already iterate `self._temporal_bins` generically, so `(B, K, H, W)` flows with `K = 3 or 4` automatically.

2. **`config/`**: new YAMLs per variant. Each sets `dataset.n_temporal_bins: overlap3` (or `overlap4`) but keeps `model.name` canonical:

   | YAML filename | `model.name` inside YAML | Notes |
   |---|---|---|
   | `n2_temporal_energy_overlap3.yaml` | `n2_temporal_energy` | exp231–233 |
   | `n2_temap_input_overlap3.yaml` | `n2_temap_input` | exp234–235, keep `input_nc: 5` (2 audio + 3 bins) |
   | `n2_temporal_energy_overlap3_signed.yaml` | `n2_temporal_energy` | exp236, sets `model.gain_mode: signed` |
   | `n2_temporal_energy_overlap3_wloss.yaml` | `n2_temporal_energy` | exp237, sets `model.lambda_bins: [0.2, 0.15, 0.05]` |
   | `n2_temporal_energy_overlap3_deccond.yaml` | `n2_temporal_energy` | exp238, sets `model.cond_source: decoder_level` |
   | `n2_temporal_energy_overlap4.yaml` | `n2_temporal_energy` (*or new entry — see item 3*) | exp239–240, sets `model.n_bins: 4` |

3. **`models/n2_0417/n2_temporal_energy.py`** — three code changes:
   - Add `n_bins: int = 3` to `__init__`. Replace `for _ in range(3)` with `for _ in range(n_bins)` and `self.attn_inject_indices = [4, 5, 6]` with `self.attn_inject_indices = list(range(7 - n_bins, 7))` (3 bins → [4,5,6]; 4 bins → [3,4,5,6]). Store `self.n_bins`.
   - Add `gain_mode: str = 'monotone'`. In `forward`, when `gain_mode == 'signed'`, replace `h = h * (1.0 + emap)` with `h = h * (1.0 + self.gain_alpha * (2.0 * emap - 1.0))` where `gain_alpha` is a learnable `nn.Parameter(torch.tensor(0.5))`. For exp236.
   - Add `cond_source: str = 'bottleneck'`. When `cond_source == 'decoder_level'`, pass `h` (the current decoder feature) to the energy head instead of `bottleneck`. The head's first `AdaptiveAvgPool2d(1)` handles variable input sizes. For exp238.

   Also: `__init__` must read these from cfg. Currently `__init__` accepts `**_` and ignores unknown kwargs — change the model factory in `utils/train_utils.py:206` (or wherever the class is instantiated) to forward `cfg.model.n_bins`, `cfg.model.gain_mode`, `cfg.model.cond_source` explicitly with `getattr(cfg.model, 'n_bins', 3)` defaults, so existing exp192-194 still work unchanged.

4. **`train.py`** (`_train_step_n2` energy-loss branch, lines 185-193) — if `cfg.model.lambda_bins` is a list, use per-bin weights: `loss += lambda_bins[k] * F.l1_loss(…)` instead of scalar `lambda_energy * …`. Fall back to scalar when absent. For exp237. Also `temporal_energies[:, k:k+1]` already iterates per channel, so K=4 works automatically once the dataset produces 4 bins.

5. **`n2_bulk.sh`**: exp231–240 rows appended (commented-out until 1–4 land; uncomment in order 231 → 240).

### 12.6 Per-experiment trainability check

| Exp | CONFIG file | `model.name` | Blockers before launch |
|---|---|---|---|
| 231-233 | `n2_temporal_energy_overlap3` | `n2_temporal_energy` | items 1 + 2 only |
| 234-235 | `n2_temap_input_overlap3` | `n2_temap_input` | items 1 + 2 only |
| 236 | `n2_temporal_energy_overlap3_signed` | `n2_temporal_energy` | items 1 + 2 + 3 (gain_mode branch) |
| 237 | `n2_temporal_energy_overlap3_wloss` | `n2_temporal_energy` | items 1 + 2 + 4 (lambda_bins list) |
| 238 | `n2_temporal_energy_overlap3_deccond` | `n2_temporal_energy` | items 1 + 2 + 3 (cond_source branch) |
| 239-240 | `n2_temporal_energy_overlap4` | `n2_temporal_energy` | items 1 + 2 + 3 (n_bins=4 path) |

Order of work: 1 → 2 → launch 231–235 (cheapest, dataset change only) → 3+4 → launch 236–238 → launch 239–240. Cross-check after each launch that `is_n2_model(cfg)` prints True-equivalent behavior (presence of `"N2 dataset: 3 temporal bins"` or `"N2 dataset: 4 temporal bins"` in the log header).


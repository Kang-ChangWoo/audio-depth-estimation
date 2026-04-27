# EXPERIMENT_NARRATIVE — Audio→Depth Estimation

Compressed timeline + per-family why / context / findings, consolidated from `report_a..g_*.md`, `answer.md`, `answer2.md`, `experiment_plan_*.md`, `report_1_cleaning_exp.md`, KB §3 + §6, on 2026-04-27. All exact metrics live in `docs/results/ledger_*.csv`. Original sources are deleted after extraction; preserved facts are tagged here.

## 0. Project objective

Audio-only omnidirectional depth estimation. **Test-time input must be binaural audio only.** Output is an ERP depth map, normalized by `max_depth=10.0`, evaluated on the standard 9-scene / 3192-sample split. Ambisonics / FOA / SH / RGB / GT-energy / oracle-geometry are admissible **only as training-time privileged signals, auxiliary supervision, distillation targets, or intermediate representations** — never at test time.

Strict success criteria the project agreed on (KB §0):

- Same split / cache / protocol as the comparison baseline.
- ≥1 % RMSE or AbsRel gain over a strong baseline, with paired per-sample or repeated-seed evidence.
- Controls: shuffled FOA, wrong-scene FOA, random-rotated FOA.
- If Ambisonics is a main claim: RGB+Ambisonics > RGB-only must hold.

## 1. Timeline

| Date (2026) | Phase | Focus | Volume |
|---|---|---|---|
| Pre-04/07 | bulk0407 | Initial 60 baseline + FOA-aux runs | exp01-60 |
| 04/08-04/10 | bulk0408 / bulk0410 | Fill-out sweep + 78-experiment full-test | exp56r-125 train; 78 test runs (53 PASSED, 25 FAILED-eval-bug) |
| 04/15 | bulk0415 | FOA0415 v1-v5 family (25 train logs) | foa_0415 variants — no per-row test metrics in KB §2A |
| 04/16 | bulk0416 + report_a + report_1_cleaning_exp | ViT+FOA + repo cleanup; first frontier read | exp160-165 + checkpoint pruning |
| 04/17 | report_b (N3 design) + report_c (N2 design) | Architectural sweep design | exp166-186 (N3); exp187-206 (N2) |
| 04/18 | exp_report_test/_train update | 147 tested; 124 PASSED, 25 FAILED, 4 missing-ckpt, ~54 awaiting | — |
| 04/19 | report_d (Group H/I/J) + report_e (post-mortem) | Re-tune training recipe; hybrids; ViT port | exp207-230 |
| 04/20 | report_e §11 + §12 addenda | n2_temporal_energy wiring audit; overlap-bin design | exp231-240 (planned) |
| 04/24 | n9_0424 first wave | Predicted sound-field projection fusion | exp390-401 |
| 04/25 | report_f (analysis) + n4_0425 | Oracle bin-gated FOA; renew v2 fix; FOA eval bug surfaced | exp400-418 (n4); exp301-308 (renew); exp390-401 (n9_0424) |
| 04/26 | report_g + answer + answer2 + table.md | Cross-family ranking; HP-noise floor; bin-gate equilibrium failure | exp350-374 baselines; cross-comparison done |
| 04/27 | This cleanup | Repository token reduction for LLM ingestion | — |

The frontier RMSE moved from 1.0817 (baseline UNet exp01) → 1.0744 (n3 energy_attn exp172) → 1.0713 (n2_6ch_input exp187, **but uses FOA at test → ORACLE**) on the old protocol, and from baseline UNet 1.2117 (exp354 n2_revisit) → n4_0425 oracle 1.2013 (exp401, **uses oracle bin-gated FOA → ORACLE**) on the new protocol. **No deployable result has cleared the noise floor (≥1 % over a strong same-bucket baseline) with paired/seed evidence as of 2026-04-27.**

## 2. Family by family

### 2.1 Baseline binaural→depth (comparison methods)

**Why pursued.** Establish strong non-privileged references against which any FOA-using improvement is measured. Without these, there is no honest definition of "FOA helps".
**Context.** Comparison set adopted from prior depth-from-echoes literature: UNet (pix2pix), BatVision (Brunetto 2023), ViT (from-scratch + ImageNet-pretrained), Echo-Net (Parida 2021), EchoDiffusion, ImageNet-pretrained ResNet-50 with FPN.
**Best deployable rows.**
- Old protocol (`source=summary_test`, master_ranking): exp01 unet_baseline RMSE 1.0817; exp65 vit RMSE 1.0818; exp71 batvision RMSE 1.0835; exp123 echodiffusion RMSE 1.1062.
- New protocol (`source=n2_revisit_test`): exp354 unet_baseline RMSE **1.2117** (best HP cell); exp363 echodiffusion RMSE **1.2198** (best in family by ABS_REL=0.4482); exp371 pretrained_vit ABS_REL=**0.4226**.

**Findings.**
- Baseline UNet is by far the **most HP-robust family** (RMSE std=0.0104 over 5 cells; δ1 std=0.0017). Use it as the *single-run reliable* reference.
- EchoDiffusion exp363 wins ABS_REL on n2_revisit (0.4482) but the gap to UNet is **0.29σ within echodiff's 0.0336 RMSE std → not statistically distinguishable**.
- Pretrained ResNet uniformly worst in n2_revisit (RMSE 1.27-1.36, δ1 ≤ 0.47). FPN decoder + ImageNet priors are mismatched to spectrogram inputs.
- ViT (from-scratch) is uniformly worst by RMSE/δ1; **lr=5e-4 catastrophically diverges** (exp358 RMSE 1.3783). Cap ViT LR at ≤1e-4.
- ImageNet-pretrained ViT exp371 hits ABS_REL 0.4226 — best across all families (incl. oracle FOA), but RMSE 1.2806 is no better than UNet → different operating point.
- **Wav2Vec2 / CIDE (used inside EchoDiffusion) provides no statistically meaningful depth value.** Speech-pretrained features (LibriSpeech 960h) are off-distribution for room impulse responses; 132.6M-param echodiff doesn't beat 55M-param UNet baseline (`answer2.md`).

**Caveats.** Per `docs/table.md`, **HP-induced std (RMSE) is 0.0104-0.0514** across families → any single-run gap below ~0.02 RMSE is below 1× HP-σ, undisclosable as a real improvement without seed-variance.

**Status.** All 7 comparison methods stay active in `models/` for reproducibility. See `docs/REPRODUCIBILITY.md`.
**Source rows.** `ledger_master.csv` (exp01-05, exp61-75, exp11-14 + exp121-125), `ledger_supplemental.csv` (exp350-374). HP variance: `hp_variance.csv`.

---

### 2.2 FOA-as-input (ORACLE — non-deployable)

**Why pursued.** Quantify the **upper bound** on what richer 4-mic capture would deliver if available at test time. Acts as a ceiling for any deployable claim.
**Context.** Several routes: 6-channel input (binaural ⊕ 4-ch FOA spectrogram), dual encoder, temporal energy maps as input, GT energy map oracle (foa_oracle_nc1/nc3), per-bin oracle reps (n4_0425).
**Best rows (ORACLE — never deploy).**
- exp241 n1_pvit_temap RMSE **1.0454** (top of master ranking; uses temporal FOA energy maps at test).
- exp201 n2_temap RMSE 1.0464.
- exp179 foa_oracle_nc3 RMSE 1.0477 / exp180 foa_oracle_nc3 ABS_REL=**0.4017**.
- exp187 n2_6ch_input RMSE 1.0713 / **δ1 0.5111** (only oracle row that beats deployable on δ1 — see §2.6).
- exp401 n4_0425 oracle bin-gated FOA RMSE 1.2013 (n4_test bucket).

**Findings.**
- Oracle ceiling is **at most ~3 % RMSE** above the best deployable old-protocol baseline (exp01 1.0817 vs exp179 1.0477) → "FOA contains usable depth signal" hypothesis weakly confirmed.
- New-protocol oracle ceiling is **~1 % RMSE** above same-bucket baseline (exp401 1.2013 vs exp354 1.2117). At noise floor.
- The `4-D FOA RMS` representation is over-compressed (loses timing / cross-channel covariance / spherical structure) — should be retired as a primary FOA target.
- **Energy map (covariance E(Ω)=yᵀRy from data/sh_basis.py) preserves more signal** than 4-D RMS. Oracle nc3 ABS_REL improvements (~0.06 vs deployable) come from the energy channel, not the architecture.
- δ1 frontier inverted: exp187 n2_6ch (deployable-architecture but FOA-at-test) = **0.5111** > exp179 oracle nc3 = 0.5002. Multi-channel FOA spectrogram input is better at "coverage" (within-1.25× pixels) than the GT energy oracle. Tail-vs-mean trade-off: exp187 has lower MAE (0.6682) than exp179 (0.6714), oracle's RMSE win is from suppressed tails.

**Caveats.** **Top of the entire ledger uses FOA at test → not deployable.** Whenever ranking by RMSE, separate ORACLE rows from DEPLOYABLE rows.

**Status.** Active oracle ceilings kept in `models/` per ≥1 % rule: `n4_0425/`, `pretrained_vit_foa_v6_oracle_nc3.py`, foa_oracle (config/foa_oracle*.yaml; class likely lives in `models/unet_foa.py`). `n1_4020/` archived per user override 2026-04-27. n2_0417 ambisonics-input subset archived.
**Source rows.** `ledger_master.csv` (exp178-189, exp201-206, exp241-243, exp219); `ledger_supplemental.csv` (exp400-418).

---

### 2.3 FOA-as-aux supervision (deployable, binaural at test)

**Why pursued.** Use FOA only as a *training-time auxiliary loss*; predict FOA RMS or SH coefficients from binaural features so the encoder learns spatial structure without needing FOA at test. This was the original FOA UNet design (exp36-55, foa.yaml).
**Context.** Original `audio_depth_foa` (UNet + SH branch, `foa_weight*L1+cosine + hist_weight*SHHistogramAlignmentLoss`). Later attention-based bridges (CrossAttn / FeatBank / MSAttn / ChannelAttn). foa_v2 / foa_v2_js variants. FOA0415 v1-v5 (rotate_canonical=True, λ_sh).
**Best rows.** exp49 original FOA UNet RMSE 1.0803 / exp111 freeze=15 RMSE 1.0781 / exp16 crossattn RMSE 1.0880 / exp23 featbank RMSE 1.0869 / exp28 msattn RMSE 1.0816 / exp91 channelattn RMSE 1.0907.

**Findings.**
- **All FOA-aux variants are within HP-noise of binaural baseline.** exp49 vs exp01 = 0.0014 RMSE = 0.13σ → **not a real improvement**.
- The auxiliary head is "fire and forget": predicted SH never feeds back into depth decoding (report_b / report_d Diagnosis 1). Architecturally guaranteed to be weak.
- 25 of the original 83 FOA experiments **failed during evaluation due to checkpoint loading / dimension mismatch**, not architecture failure (KB §2E exp16-20, 21-25, 26-30, 31-35, 56-60 + report_a §1.2). Standalone canonical reruns later confirmed the architectures themselves train fine.
- pvitfoa v1/v2/v3 (FOA-aux on ImageNet ViT): best exp164 v3 RMSE 1.0929 — **worse than binaural ViT baseline 1.0818**. Pretrained backbone doesn't rescue FOA-aux.

**Caveats.** "FOA helps" was the project's main claim entering this cleanup; the data does not support it as a deployable/main claim. KB §0 explicitly demotes Ambisonics unless it beats strong binaural/RGB baselines under strict controls.

**Status.** **All archived to `trash/` 2026-04-27** except `pretrained_vit_foa_v6_eattn` and `pretrained_vit_foa_v6_mssh` which join the n3 line (§2.4). Original `unet_foa.py` archived; if `foa_oracle` class lives inside it, extract to `models/oracle/foa_oracle.py` first.
**Source rows.** `ledger_master.csv` exp16-35 (failed-eval rows), exp36-55, exp56-60 (foa_v2), exp80-95 (channelattn), exp96-120 (FOA UNet HP sweep), exp160-165 (pvitfoa).

---

### 2.4 N3 — energy / SH-aware deployable variants (active)

**Why pursued.** Move beyond pure auxiliary supervision: inject directional cues *back into* the depth decoder. report_b §0 hypothesis: "the predicted SH never feeds back into depth decoding" → fix by FiLM conditioning, multi-scale SH heads, predicted energy attention, or temporal windowing.
**Context.** 4 architectural cells (C1-C5) in report_b plus oracle ceilings (D1, D5):
- **C1 MultiScale SH** (exp169-171): tap SH heads at enc[2,4,6] + bottleneck — captures directional cues at multiple scales.
- **C2 FiLM** (exp166-168): predicted SH → (γ, β) → modulate decoder[0] features. Tests channel-wise modulation.
- **C3 Energy Attention** (exp172-174, exp186): predict energy map from binaural features, use as residual multiplicative attention (`h * (1 + emap)`).
- **C5 Temporal Window** (exp175-177): split IR into early/mid/late windows, shared encoder, fused bottleneck.
- **D1 Oracle nc3** (exp178-180): ground-truth energy map concatenated as 3rd input channel.
- **D5 Oracle nc1** (exp181-183): GT energy map only (no binaural). Diagnostic.

**Best rows.** exp172 n3_energy_attn RMSE **1.0744** (best deployable old-protocol) / exp213 n3_mssh_eattn hybrid 1.0777 / exp223 n3_mssh λ_sh=0.2 ABS_REL **0.4171** (best deployable ABS_REL).

**Findings (report_d / report_e).**
- N3 **moved RMSE by 0.4 %** vs prior best (exp111 1.0781 → exp172 1.0744). Below 1 % threshold → not significant.
- **Oracle nc3 (exp179) RMSE 1.0477 = ceiling**; deployable best 1.0744 leaves a 2.5 % headroom that no architecture has closed.
- **MSSH likes stronger SH supervision** (λ_sh=0.3 → exp171 ABS_REL 0.4218; λ_sh=0.2 → exp223 0.4171); FiLM degrades with stronger SH. The two mechanisms have **incompatible operating points** — cannot be casually combined.
- **Temporal window family is dead** (exp175-177): 3× encoder cost with worse metrics across the board. Drop.
- **Group H re-tune (exp207-211)**: BS=32, dw=2.0, freeze=15, λ_energy=0.3 all *hurt* the energy_attn model relative to BS=128 default. Refuted "training recipe is the bottleneck" diagnosis.
- **Hybrids (Group I, exp212-215)** moved frontier marginally: best exp213 n3_mssh+eattn RMSE 1.0777, +0.003 vs exp172 — within noise.
- **Distillation (exp215, exp230)** mildly competitive: exp230 pvit_distill RMSE 1.0735, exp215 n3eattn_distill 1.0853. Promising line but not yet decisive.
- exp172's EnergyHead (`n3_0417/n3_energy_attn.py`) was **trained with no direct supervision against GT energy** — only via downstream depth gradient. Adding λ_energy*L1(pred_energy, gt_energy) (planned exp210) hurt it (Group H result).

**Caveats.** **n3 family was bound by `--batch-size 128` in `scripts/n3_bulk.sh` line 66** regardless of config defaults (32). May affect comparisons against any non-N3 family using config defaults.

**Status.** `models/n3_0425/` active (latest gen). `models/n3_0417/`, `models/n3_0419/` archived as superseded. `pretrained_vit_foa_v6_eattn/_mssh` (config: `pvitfoa_v3_eattn.yaml`, `pvitfoa_v3_mssh.yaml`) stay active as ViT ports of the C3/C1 mechanisms.
**Source rows.** `ledger_master.csv` exp166-186, exp207-229; `ledger_planned.csv` exp244-247.

---

### 2.5 N2 — temporal FOA decomposition (mostly archived)

**Why pursued.** report_c §0: FOA IR is time-resolved; **early (≤59 ms) carries first-reflection geometry, mid (59-295 ms) carries room shape, late (≥295 ms) is diffuse**. Averaging over the full IR loses this. Try (a) raw FOA spectrogram input (E1 6ch), (b) explicit temporal-bin features (E2-E3, E6-E7), (c) temporal bin cross-attention (E8).
**Context.** 7-tuple dataset (`data/dataset_n2.py`): audio + depth + FOA + energy + FOA_spec + temporal_RMS + temporal_energy_maps. Two cells per family at lr ∈ {1e-3, 5e-4} × λ_sh ∈ {0.1, 0.3}.
**Best rows.**
- ORACLE (FOA at test): exp201 n2_temap RMSE 1.0464; exp187 n2_6ch RMSE 1.0713 / **δ1 0.5111**; exp196 n2_dual RMSE 1.0614.
- DEPLOYABLE (binaural at test, FOA aux only): exp192 n2_tenergy RMSE 1.0805; exp199 n2_trms_film 1.0937; exp197 n2_stft 1.0941; exp204 n2_xattn 1.0931.

**Findings.**
- **Deployable N2 variants do not beat baseline** (best exp192 RMSE 1.0805 vs exp01 1.0817 = 0.11 % — noise).
- ORACLE-input variants are top of the entire ledger but **non-deployable**.
- report_e §11 wiring audit of `n2_temporal_energy`: pipeline is correct (early/mid/late bins, decoder[4/5/6] coarse-to-fine injection). **But three representational choices defeat the design**:
  1. Monotone gain `h * (1 + sigmoid(emap))` — can only boost, never attenuate diffuse regions.
  2. Equal λ per bin sums to 3λ — late/diffuse bin (low-info) crowds out early bin (high-info).
  3. All 3 heads condition on the same bottleneck → highly correlated per-bin predictions.
- Recommended fixes (exp-e1/e2/e3) were drafted but not executed.
- exp231-238 (overlap-bin variants from n2_temporal_energy_overlap{3,4}.yaml) **planned not reported** — see `ledger_planned.csv`.

**Status.** `models/n2_0417/` archived (deployable subset to `trash/n2_0417/binaural/`, ORACLE subset to `trash/n2_0417/ambisonics/`). `models/n2_0427/` archived (newest dir, no metrics yet — per user override 2026-04-27).
**Source rows.** `ledger_master.csv` exp187-206; `ledger_planned.csv` exp231-238 + 247.

---

### 2.6 N4 — oracle bin-gated FOA conditioning (active oracle ceiling)

**Why pursued.** Cleanly probe whether **per-distance-bin FOA reps** (K=8 bins of `rep_kind=eigen` from `data/dataset.py`) carry extra depth signal beyond the global energy map. Architecturally: binaural UNet bottleneck ⊕ MLP(g ⊙ rep_gt), where `g` is a learnable per-bin gate; `λ_sparsity * sigmoid(gate).mean()` pulls bins off; drop-one-bin ablations isolate contribution.
**Context.** Driven by `experiment_plan_bin_selection.md`: which temporal bins matter, are they localized vs distributed, and do they vary across depth ranges. n4_0425.yaml + `models/n4_0425/`.
**Best rows.** exp401 λ=0.01 RMSE **1.2013** / exp402 λ=0.05 ABS_REL **0.4235** / exp411 drop-bin-1 δ1 **0.5075**. Same-cache baseline = unet_baseline exp350 RMSE 1.2337.

**Findings.**
- Oracle FOA gives a **0.8 % RMSE / 1.7-pt δ1** edge over tuned binaural UNet on n4_test. Below 1 % threshold but tracked as ceiling.
- **Sparsity λ varies RMSE by only 1.8 % across two orders of magnitude** (exp400-404). λ is not load-bearing; the LR sweet spot 1e-4 is the real lever.
- **Bin 0 carries most FOA energy direction**: dropping bin 0 collapses FOA_COS to **0.4736** (vs 0.9997 elsewhere). Bin 1 is **redundant** with bin 0; dropping it gives the **best δ1 of the whole sweep** (exp411 0.5075).
- exp418 binaural-only floor (gate=0, bin path zeroed): RMSE **1.2902** — *worse* than every gated variant by 0.7-4 %, confirming the bin path provides *some* signal.
- **Bin-gate equilibrium failure** (`EDA/_9_comparison_0426/` summary): gate values converge slowly; gauge equivalence between (g, MLP) means similar performance under different gate solutions; making per-bin importance not uniquely identifiable from `g` alone.
- exp415 (drop-5), exp417 (drop-7), exp419 (alt-K=4) **missing**; exp416 has empty log.

**Status.** Active in `models/n4_0425/` with `NON_DEPLOYABLE.md` marker (≥1 % over same-bucket baseline; passes oracle keep-rule).
**Source rows.** `ledger_supplemental.csv` exp400-418. Planning: `experiment_plan_bin_selection.md` (consolidated here, original deleted).

---

### 2.7 Renew — dual-ViT SH36 sound-field bottleneck (active flagship)

**Why pursued.** KB §0: strongest paper framing as **Spatial Geometry Distillation for Binaural Echo Depth**. Use a dual-ViT (`spec ViT → SH36 → sound-field ViT → DPT decoder`) where the SH36 sound-field bottleneck is supervised against GT FOA energy / SH; depth path remains binaural-only at test.
**Context.** `models/renew/`, `config/renew_*.yaml`. Two generations:
- **v1**: λ_sh=0.1, λ_energy=0.1, λ_kl_energy=0.1, no freeze.
- **v2**: λ_sh=0.3, λ_energy=0.1, λ_kl_energy=0.05, `renew_freeze_epochs=3`.

**Best rows.**
- exp302 renew_single v2 RMSE **1.0921** (renew_test bucket), FOA_L1=0.0231, FOA_COS=**0.9985**, FOA_DIR=**0.9963**.
- exp304 renew_dpt_only no-KL RMSE **1.0696** (best in renew_test).
- Radial variants exp305-308: RMSE 1.2233-1.2540 — **12 % worse than non-radial** for no FOA gain (FOA already saturated in non-radial).

**Findings (report_f).**
- **Renew v2 fixed FOA quality** (FOA_L1: 0.33 v1 → 0.023 v2 → ~45× improvement on the auxiliary). Critical sanity check that the SH bottleneck actually learns.
- **No-KL DPT-only (exp304) gives best RMSE in the bucket** but with same-tier FOA quality. KL energy term weakly hurts depth on this protocol.
- **Radial parametrisation is a regression**: depth RMSE worsens by 12 % across exp305-308 with no FOA benefit. Either revert or investigate the metric inconsistency it introduces.
- exp390 visualization KeyboardInterrupt during `matplotlib.savefig` after 4/9 scenes; **metrics block written before crash, valid**.

**Caveats.** renew_test cache `7027059baf06`; not directly comparable to n2_revisit cache `e2314b68a4f5`.

**Status.** Active flagship line in `models/renew/`. Both v1 and v2 preserved (v1 kept for the FOA-quality-fix story).
**Source rows.** `ledger_supplemental.csv` exp301-308.

---

### 2.8 N9 — predicted sound-field projection fusion / cascade (archived per user 2026-04-27)

**Why pursued.** Generalize renew's SH36 bottleneck to a **predicted (not GT-supervised)** sound-field cascade: outer ViT/UNet backbone consumes a `gated_em` feature derived from an inner n3-style sound-field predictor.
**Context.** Three generations: `n9_0424/` (projection fusion), `n9_0425/` (n3→depth cascade with `freeze_n3=true`), `n9_0426/` (latest, no metrics yet).
**Best rows.** exp392 n9_0424 (B, lsh=0.05) RMSE 1.2198; exp401 n9_0424 (C) ABS_REL 0.4350. n9_0425 / n9_0426 — **no per-row metrics in ledger**.

**Findings (report_f).**
- **n9_0424 has a regression in FOA learning** vs renew v2: every n9_0424 run collapses to FOA_COS ≤ 0.48 (vs ≥0.998 for renew). Two runs (exp390/391) actually negative, indicating inverted prediction. **The regression is in the n9 FOA loss / head path, not depth.**
- Depth quality on n9_0424 is on par with renew radial variants (RMSE 1.22-1.26) — i.e. weakened by whatever broke the FOA head.
- Bug audit recommended before further n9 sweeps.

**Status.** `n9_0424/` and `n9_0425/`, `n9_0426/` archived to `trash/` per user 2026-04-27 (line of work paused).
**Source rows.** `ledger_supplemental.csv` exp390-401.

---

### 2.9 EchoDiffusion + Ambi (700-series, archived per user 2026-04-27)

**Why pursued.** Combine echodiffusion's diffusion-UNet + Wav2Vec2/CIDE conditioning with ambisonic-side input or SH-side features.
**Context.** Variants:
- **echodiffusion_ambi** (config foa-mode=`input` vs `condition`, `rep_kind=eigen`, `rep_K=8`).
- **echodiffusion_ambi_cide** (adds Wav2Vec2 conditioning from `use_waveform=True`).
- **echodiffusion_ambi_sh** (SH-side+).
- **echodiff_sh_side_plus** (sideplus baseline + oracle UB gate-ones).
**Best rows.** **None reported** in `ledger_master.csv` or `ledger_supplemental.csv`. exp700-731 sit in `ledger_planned.csv` as PLANNED / NOT_REPORTED. `results/<run_dir>` exists for many but not surfaced as canonical metrics.

**Findings.** Predictions from `answer2.md` §5 (using exp363 echodiff baseline patterns):
- Best LR likely 5e-4 (consistent with exp363).
- Val/test RMSE disparity expected (~1.40 val vs ~1.22 test).
- Combined oracle-FOA ceiling: expect ≤1.18 RMSE = meaningful gain (~1.5σ); <1.16 RMSE = real win. 1.18-1.22 = within noise.

**Caveats.** 700-series is the **largest unfinished EDA-Ambi sweep**. answer2.md predictions are explicit hypotheses, not results.

**Status.** Code and configs archived to `trash/echodiffusion_ambi/` 2026-04-27 (line marked done by user). Base `echodiffusion` stays active as comparison method.
**Source rows.** `ledger_planned.csv` exp700-731.

---

### 2.10 Distillation (open hypothesis, partially tested)

**Why pursued.** Borrow oracle / RGB / Ambisonic teacher knowledge at train time only; deploy with binaural input. Bypasses the oracle ceiling without the oracle constraint.
**Context.** exp215 n3_eattn distill (teacher=oracle nc3 exp180, λ_kd=0.5); exp230 pvit_distill (similar). RGB teacher direction is in `data/dataset_rgb.py` but unused.
**Best rows.** exp230 pvit_distill RMSE **1.0735** (n3_test); exp215 RMSE 1.0853.

**Findings.** Mildly competitive — exp230 is one of the better deployable rows old-protocol — but not yet decisively better than non-distill exp172 (1.0744). RGB-teacher comparison **never run**.

**Status.** Distillation code stays active under `models/n3_0425/` and `models/pretrain/pretrained_vit_foa_v6_*`. RGB-teacher path remains open (KB §6 Table B Hypothesis 2 + 5).
**Source rows.** `ledger_master.csv` exp215, exp230.

---

### 2.11 HOA / higher-order ambisonics (planned, not run)

**Why pursued.** `answer.md` Option 1 — `ambi3_npy` (16-ch order-3) and `ambi5_npy` (36-ch order-5) sit on disk, unused. Going from FOA (4-ch, ~120° main lobe) to 5th-order (~30° main lobe) is a real **information-theoretic increase**, not an architectural tweak.
**Context.** Single recommended cell: swap `ambi1_npy → ambi3_npy` in `n4_0425.yaml`, keep all else; head-to-head vs current best (exp401/exp402). ≥0.02 RMSE win → real (information saturates beyond FOA); else negative ⇒ FOA already saturates the task.
**Best rows.** None — never executed.
**Findings.** None. Open hypothesis; cleanest single-figure win or negative result available.
**Caveats.** Recommended **after** seed-variance run on exp402 (3 seeds same HP) so HOA result becomes interpretable as real-or-noise.

**Status.** Future experiment. No code path yet.
**Source rows.** None.

---

## 3. Cross-cutting findings (KB §6 Table A — high-confidence)

| Finding | Confidence |
|---|---|
| Current Ambisonics main claim is weak; FOA gains are <1 % and inconsistent. | High |
| 4-D FOA RMS is over-compressed for a main claim. | High |
| Temporal / covariance representations are more physically meaningful than 4-D RMS. | Medium |
| Canonical FOA rotation is essential to preserve frame alignment (raw IR, sample_idx % 4, ACN [W Y Z X]). | High |
| Oracle energy / FOA improves ceilings but is not deployable. | High |
| Direct FOA / energy input dominates RMSE rankings but violates final binaural-only constraint. | High |
| Baseline UNet is the most HP-robust family on n2_revisit. | High |
| Wav2Vec2 / CIDE shows no clear depth value. | Medium-high |
| n4 bin-gate / drop analysis exhibits gauge / equilibrium failure signs. | Medium |
| Some reported FOA metrics pre-date the FOA-eval bug fix; high-FOA_L1 rows must be rerun. | High |
| Cache / protocol comparison has documented CONFLICT (preserved in `conflicts.csv` C1). | High |

## 4. Open hypotheses (KB §6 Table B)

| Hypothesis | Test |
|---|---|
| Temporal pressure-direction covariance improves deployable depth | Binaural-only student + aux covariance descriptor; vs RMS/4D and shuffled controls |
| RGB geometry teacher outperforms Ambisonics-only teacher | Binaural + RGB teacher vs +Ambi teacher vs +RGB+Ambi, same split/cache |
| HOA-3 / HOA-5 improves over FOA | Swap `ambi1_npy → ambi3_npy` in n4_0425, same HP, 3 seeds. **First open recommendation per `answer.md`** |
| Binaural spatial features (IPD/ILD/GCC) beat magnitude-only spec | Strong binaural feature baseline: mag+phase+IPD/ILD/GCC vs UNet/ViT |
| Distillation from oracle / RGB reduces RMSE while remaining deployable | Teacher depth + gradient + feature distillation; no test-time teacher |
| Late-reverb downweighting improves auxiliary supervision | Full vs early/mid/late/early_mid covariance descriptor; per-depth-range analysis |

## 5. Critical caveats (KB §5; preserved verbatim into `conflicts.csv` and the active codebase)

1. **Cache mismatch CONFLICT** (`conflicts.csv` C1): `e2314b68a4f5` (n2_revisit) vs `7027059baf06` (renew/n4) — `report_f/g` says NOT comparable; `docs/table.md` says identical samples verified 2026-04-26. Preserve both.
2. **Old-protocol vs new-protocol RMSE not directly comparable** (RMSE 1.04-1.10 vs 1.20-1.31). Always rank within bucket.
3. **FOA evaluation bug** (pre-2026-04-25 evaluator): pre-fix high-FOA_L1 rows are invalid for FOA metrics; depth metrics still valid. Audit anything with FOA_L1 ∈ [0.3, 0.6].
4. **Cache hash gap**: the cache filename hash does **not** encode `depth_dir`, `rep_kind`, `rep_K`, `use_waveform`, `use_rgb`. Same hash can refer to subtly different sample tensors.
5. **Notation conflict** (`conflicts.csv` C2): code uses ACN `[W Y Z X]`; some comments write `[W X Y Z]`. Code is authoritative.
6. **Best checkpoint** is selected by `0.7*RMSE + 0.3*AbsRel` (program.md §11), not AbsRel alone. Affects whether a "best" checkpoint is a fair RMSE or AbsRel reference.
7. **`scripts/n3_bulk.sh` line 66 forces `--batch-size 128`** for all N3, overriding config defaults of 32. May explain the under-tuning observed in report_d Diagnosis 2.
8. **Pin memory fix** (`data/dataloader.py`): val/test set `pin_memory=False`, `persistent_workers=False`, `timeout=120` to avoid validation hangs.

## 6. Preservation rules (user-imposed 2026-04-27)

These three rules override the default "archive variants with bad metrics" cleanup heuristic. Each rule preserves a class of artifacts in the active tree regardless of metric performance.

| Rule | Scope | What stays active |
|---|---|---|
| **exp700-707 reproducible** | echodiffusion_ambi (foa-mode=input/condition) | `models/echodiffusion/echodiffusion_ambi*.py`, `config/echodiffusion_ambi.yaml`, `scripts/echodiff_ambi_bulk.sh`, plus the corresponding `checkpoints/<run_dir>/` and `results/<run_dir>/`. Conservatively also keep 711-713 (cide), 720-723 (sh), 730-731 (sideplus) since the line is being actively explored. |
| **Comparison methods' radial runs** | Any radial-protocol (`depth_dir=erp_depth_radial`) experiment touching a comparison method | All `*_radial.yaml` configs stay; all radial sweeps (renew exp305-308; n4_0425 entire family; echodiffusion_ambi*; echodiff_sh_side_plus exp730-731) preserved with their checkpoints. |
| **Pre-exp140 code preserved** | Any `models/*.py` that produced an experiment with id ≤ 140 | `unet_foa.py` (exp36-55, 96-120); `foa_crossattn.py` (exp16-20, 76-79); `foa_featbank.py` (exp21-25, 80-86); `foa_msattn.py` (exp26-30, 87-90); `foa_channelattn.py` (exp31-35, 91-95); `foa_v2.py` (exp56-60); `foa_v2_js*.py`, `foa_js_swin.py` (foa_v2_js family ckpts existed pre-cleanup); `foa_0415_v1..v5.py` (exp130-150 mixed). All stay active. |

The result: the original archive list shrinks materially. Only `pretrained_vit_foa v1/v2/v3-base/v4/v5` (exp160-165), `n1_4020/` (per user "5. oracle"), `n2_0417/`, `n2_0427/`, `n3_0417/`, `n3_0419/`, `n9_0424/0425/0426/` move to trash — and even there only the runs with **≤10-trained-epoch checkpoints** are eligible for ckpt removal. Most ckpts stay in `checkpoints/`.

## 7. Cleanup history (from `report_1_cleaning_exp.md`, 2026-04-16)

For provenance: an earlier cleanup pass on 2026-04-16 removed 9 stalled / legacy checkpoints (e.g. `exp78_crossattn`, old foa0415 v1-v5 placeholders, legacy pvitfoa naming) and *accidentally removed* 6 JS-related checkpoints that had `best_model.pth`. Affected: `feat_attn_foa_v2_js`, `foa_basic_js_foa_v2_js`, `foa_feat_attn_v2_foa_v2_js`, `full_run_foa_v2_js`, `sh_coeff_hierarch_foa_v2_js`, `full_run_foa`. **These need retraining** if the corresponding rows in `ledger_master.csv` (foa_v2_js family) are ever to be reproduced. Recorded in `known_issues.csv:js_variants`.

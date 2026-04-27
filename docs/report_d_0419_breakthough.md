# Report D — Pushing ABS_REL & RMSE Below the N3 Frontier
**Date:** 2026-04-19
**Baseline for this report:** exp166–186 (see `report_b_0417_n3_exp_design_166-186.md`)
**Scope:** design the next wave of experiments whose *sole* purpose is to drive ABS_REL and RMSE below the current best.

---

## 0. TL;DR

N3 produced a modest RMSE win (1.0744 from exp172, down from 1.0781 on exp111) but **nothing on ABS_REL for deployable models** — the best deployable ABS_REL in the whole N3 bulk is exp171 at **0.4218**, which is *worse* than the old bulk0410 best (exp02 at 0.4127) and the 0416 ViT best (exp160/165 at 0.3817–0.4211 on val). The oracle answered the gating question: GT energy helps a lot (exp180: ABS_REL=0.4017, RMSE=1.0741), confirming the spatial-direction signal is recoverable. But no deployable N3 variant captures that signal.

**Conclusion**: the depth encoder isn't information-starved — it's under-guided. The N3 sweep varied what the encoder *outputs*, not how the encoder is *trained* or *initialized*. Report D's proposal wave targets training-side levers (freeze/stage, LR schedule, loss balance, backbone transfer) that are orthogonal to N3 architecture changes.

Proposed wave: **14 experiments in 3 groups, ~320 GPU-hours total** (exp207–220). Expected gains:
- **ABS_REL**: current deployable best 0.4127 → projected **~0.38** (match ViT exp165, apply to UNet N3).
- **RMSE**: current deployable best 1.0744 → projected **~1.065** (close the gap to oracle 1.0477 by ~50%).

---

## 1. What N3 Actually Moved

Reading the metrics in `logs/summary_test/exp1[6-8]*_test.log`:

| Family | Best exp | ABS_REL | RMSE | Δ vs exp111 (1.0781) |
|---|---|---|---|---|
| N3 FiLM (exp166–168) | exp166 | 0.4271 | 1.1045 | +0.026 worse |
| N3 MultiScale SH (exp169–171) | exp170 | 0.4676 | **1.0878** | +0.010 worse |
| N3 Energy Attn (exp172–174, 186) | exp172 | 0.4792 | **1.0744** | **−0.004 better** |
| N3 Temporal Window (exp175–177) | exp176 | 0.5430 | 1.2022 | +0.124 far worse |
| Oracle nc3 (exp178–180) ★ | exp179 | 0.4774 | **1.0477** | oracle ceiling |
| foa_0415_v1 @ 60ep (exp184–185) | exp184 | 0.4836 | 1.0889 | +0.011 worse |

**Reading the gap between exp172 (deployable, 1.0744) and exp179 (oracle, 1.0477):** a 3 % RMSE headroom is *still on the table* without any architectural change — just better use of information the encoder already sees. That 3 % is the target of this report.

**Reading exp166–168 vs exp171 (both λ_sh=0.1 → 0.3):** FiLM degraded, MultiScale improved on ABS_REL. MultiScale SH tapping likes a stronger SH loss; FiLM doesn't. These two mechanisms are *not* compatible in their optimal operating points — a key negative result that the current bulk didn't surface explicitly.

**Reading temporal window (exp175–177) = disaster:** 3× encoder forward cost with worse metrics across the board. Drop this family; do not continue.

---

## 2. Why N3 Plateaued — Three Diagnoses

### Diagnosis 1: Information is there, supervision isn't

Oracle nc3 (3-ch input = binaural + GT energy map) beats the non-oracle N3 variants by ~3 % RMSE. That's the magnitude of improvement available if the network *reliably* predicted the energy map. But exp172's EnergyHead (`models/n3_0417/n3_energy_attn.py` lines 48–77) predicts an (H, W) energy map **with no loss term against GT energy** — it's trained only via the downstream depth gradient. That's a weak and indirect signal.

### Diagnosis 2: Hyperparams were copy-pasted from FOA, not re-tuned

- `depth_weight=1.0` hardcoded in every `config/n3_*.yaml`. FOA's exp115 (RMSE 1.2245) used `dw=2.0, fw=0.2`, and it's the #1 validation-score row of the *entire train report*. N3 never explored this.
- `--batch-size 128` was forced in `scripts/n3_bulk.sh` line 66 for *all* N3, overriding the config defaults of 32. This likely over-regularizes the gradient and makes λ_sh tuning less reliable (the optimizer sees 4× fewer updates per epoch).
- No freeze/stage schedule. The deployable baseline exp111 (RMSE 1.0781) used `freeze=15`. No N3 run used any freeze.

### Diagnosis 3: Backbone never changed

Every N3 is built on the `unet_256` UNet (`models/foa_0415_v1.py` via `N3*Generator` subclasses). The Bulk0416 ViT experiments (exp160–165) had the best *validation* ABS_REL in the repo (0.3817 on exp165). No one has plugged an N3 mechanism (FiLM conditioning, multi-scale SH tap, energy attention) into a ViT backbone. That's the biggest conspicuously-unexplored axis.

---

## 3. Proposal Wave — exp207–220

### Group H: Re-tune exp172 (cheapest, highest-probability wins) — 5 experiments

Target: push exp172's RMSE=1.0744 → ~1.06 by training-side fixes only. No new architecture.

| Exp | Config | Changes vs exp172 | Expected outcome |
|---|---|---|---|
| 207 | n3_energy_attn | BS=32 (back to config default) | Test whether BS=128 was over-smoothing. |
| 208 | n3_energy_attn | `depth_weight=2.0, λ_sh=0.1` | FOA exp115 pattern on the N3 encoder. |
| 209 | n3_energy_attn | `freeze=15` + lr=1e-3 then 1e-4 | Replicate the exp111 staged recipe on the strongest N3 family. |
| 210 | n3_energy_attn | Add `λ_energy=0.3 · L1(pred_energy, gt_energy)` | Supervise the EnergyHead directly. See §4.1 for implementation. |
| 211 | n3_energy_attn | 208 + 210 combined | Verify additivity of the two training-side wins. |

Budget: 5 × ~6 h = **30 GPU-hours** (single GPU each, 60 epochs, BS=32 or 128).

### Group I: Hybrid architectures — 4 experiments

Target: break the exp172 frontier by combining independent-axis wins.

| Exp | Config (new) | Mechanism | Hypothesis |
|---|---|---|---|
| 212 | `n3_film_energy_attn` | FiLM on decoder[0] **+** EnergyAttn on decoder[3] | FiLM improved ABS_REL (exp166), EnergyAttn improved RMSE (exp172) — stack them at non-overlapping decoder depths. |
| 213 | `n3_mssh_energy_attn` | Multi-scale SH tap **+** EnergyAttn | MSSH supervises SH from enc[2,4,6]; EnergyAttn uses a different (spatial) cue. Non-colliding. |
| 214 | `n3_energy_attn_highsh` | exp172 config with `sh_dim=9` (order-2 SH) | Higher SH bandwidth may stabilize the EnergyHead gradient, since both share the bottleneck. |
| 215 | `n3_oracle_distill` | Two-stage: train oracle nc3 (as exp180) → **distill** into n3_energy_attn via KL on `pred_energy` | Use the oracle as a teacher for the EnergyHead — deploys without oracle at test time. |

Budget: 4 × ~8 h = **32 GPU-hours**. exp215 is a two-stage run, ~12 h.

### Group J: ViT backbone port — 5 experiments (most ambitious)

Target: break the ABS_REL frontier (0.4127) by bringing N3 mechanisms to the ViT backbone that already has the best ABS_REL.

| Exp | Config (new) | Backbone | Mechanism | Hypothesis |
|---|---|---|---|---|
| 216 | `pvitfoa_v3_energy_attn` | PreViT-B/16 | FiLM on ViT block-9 + EnergyHead on patch tokens | exp165 + N3 energy attention. ViT already beats UNet on ABS_REL; energy attn should push RMSE. |
| 217 | `pvitfoa_v3_mssh` | PreViT-B/16 | MultiScale SH tap on blocks 3, 6, 9 | Transformer depth tap is cleaner than UNet skip tap — directional info layered by attention depth. |
| 218 | `pvitfoa_v3_film_dw2` | PreViT-B/16 | exp165 + `depth_weight=2.0` | Cheap hyperparam transfer from FOA exp115 to best ViT. |
| 219 | `pvitfoa_v3_oracle_nc3` | PreViT-B/16 | `input_nc=3` patch-embed; binaural + GT energy map | Oracle ceiling on ViT. Frames how much of the oracle gap is backbone-limited vs supervision-limited. |
| 220 | `pvitfoa_v3_freeze_stage` | PreViT-B/16 | exp165 + `freeze_encoder=20ep`, lr=5e-5 after | Longer freeze than exp164 (which used no freeze). |

Budget: 5 × ~50 h (ViT is slow) = **250 GPU-hours**. Run on 2 GPUs each (BS=16), ~25 h wall-clock.

### Total budget: ~320 GPU-hours ≈ 32 h wall-clock on 10 GPUs.

---

## 4. Implementation Notes

### 4.1 Energy-head supervision (needed for exp210, 211, 215)

Add to `train.py::_train_step_n3_eattn` (or a new branch keyed on `cfg.model.lambda_energy > 0`):

```python
if 'pred_energy' in out and cfg.model.lambda_energy > 0:
    # gt_energy_map: (B, 1, H, W) from dataset_n2 fourth return element
    e_loss = F.l1_loss(out['pred_energy'], gt_energy_map)
    loss = loss + cfg.model.lambda_energy * e_loss
```

The `gt_energy_map` is already computed in `data/dataset.py:275-281` when `use_ambisonic=True`. `n3_energy_attn.yaml` currently doesn't read it; add to dataloader by switching base class from `SoundSpacesDataset` to `SoundSpacesN2Dataset` (which exposes the 7-tuple), or simpler: extend `SoundSpacesDataset` to optionally return the energy map.

### 4.2 Hybrid model (exp212–215)

`models/n3_0417/n3_film_energy_attn.py`: subclass `N3FiLMGenerator` and additionally instantiate `EnergyHead` from `n3_energy_attn.py`. Apply FiLM at decoder[0] (existing) and energy attention at decoder[3] (existing). Both mechanisms read from the same bottleneck — verify no gradient conflict via a mini 2-epoch dry run.

### 4.3 Oracle distillation (exp215)

Stage 1 = re-run exp180 as teacher (weights already exist: `checkpoints/unet_256_soundspaces_BS128_Lr0.001_AdamW_exp180_oracle_nc3_lr1e3_lsh0.3/best_model.pth`).
Stage 2 = exp172-style training with extra loss:
```python
with torch.no_grad():
    teacher_out = teacher(torch.cat([audio, gt_energy_map], dim=1))
    target_energy = teacher_out['pred_depth']  # or teacher['features'][k]
loss += λ_kd * F.kl_div(student['pred_energy'].log_softmax(-1),
                         teacher_target.softmax(-1))
```

Test-time: student uses binaural only (teacher not called).

### 4.4 ViT + N3 port (exp216–220)

`models/pretrain/pretrained_vit_foa_v3.py` already has the FiLM mechanism — reuse. Add `EnergyHead` fed from the last transformer block's patch tokens, reshaped to (B, D, H/P, W/P). `input_nc` change for exp219 requires adapting `patch_embed.proj` conv from 3→13 (binaural 12 + energy 1) or writing an adapter layer.

---

## 5. Priority and Kill Criteria

**Run order (information gain per GPU-hour):**

1. **Group H first** (exp207–211) — cheapest, diagnoses whether N3 plateaued because of training recipe vs architecture. If Group H gives RMSE < 1.068, then training recipe was the bottleneck and Group J becomes high-priority.
2. **Group I in parallel with H** — hybrid models are orthogonal tests. If exp212 (FiLM+EnergyAttn) beats exp172, the mechanisms compose, and we have a 2-variable exploration axis for future work.
3. **Group J last** — biggest compute cost, biggest potential ABS_REL win, but also the most fragile. Run after Group H confirms the training recipe so ViT ports start from the right hyperparams.

**Kill criteria (abort family):**
- exp207 (BS=32) within ±0.001 RMSE of exp172 → batch size wasn't the lever. Drop from Group H.
- exp210 (explicit energy supervision) within ±0.002 RMSE of exp172 → EnergyHead wasn't the weak link. Drop exp211 and exp215.
- exp216 (ViT+EnergyAttn) within ±0.01 RMSE of exp165 → ViT+N3 port didn't help. Drop exp217–218; keep exp219 (oracle ceiling, still informative).

**Success definition for this report:**
- *Must have* (baseline expectation): one deployable experiment with **RMSE ≤ 1.068 and ABS_REL ≤ 0.40**.
- *Stretch* (report writes itself): one deployable experiment with **RMSE ≤ 1.055 and ABS_REL ≤ 0.385**, closing the oracle gap by 50 %.

---

## 6. What This Report Does NOT Propose

For clarity about scope:

- **No new architecture families.** FiLM, multi-scale SH, energy attention, temporal window were the N3 sweep. Temporal window is dead. The others are recombined here, not replaced.
- **No dataset-side changes.** `rotate_canonical` remains on. HOA/order-2 SH targets (from report_a §4.1) deferred to a later wave — need dataloader work.
- **No N2 evaluation.** exp191–206 are training on a second server; their results will open their own report.
- **No ensemble / checkpoint averaging.** These are post-hoc and should be measured once a single model exceeds the exp172 frontier.

---

## 7. Appendix — Current Frontier Reference

| Metric | Best experiment | Value | Source log |
|---|---|---|---|
| Test RMSE (deployable) | exp172 (n3_energy_attn, lr=1e-3, λ=0.1) | 1.0744 | `logs/summary_test/exp172_n3eattn_lr1e3_lsh0.1_test.log` |
| Test RMSE (oracle ★) | exp179 (foa_oracle_nc3, lr=5e-4, λ=0.1) | 1.0477 | `logs/summary_test/exp179_oracle_nc3_lr5e4_lsh0.1_test.log` |
| Test ABS_REL (deployable) | exp02 (baseline UNet, lr=5e-4) | 0.4127 | `logs/summary_test/exp02_baseline_*_test.log` |
| Test ABS_REL (N3 deployable) | exp171 (n3_mssh, lr=1e-3, λ=0.3) | 0.4218 | `logs/summary_test/exp171_n3mssh_lr1e3_lsh0.3_test.log` |
| Test ABS_REL (oracle ★) | exp180 (foa_oracle_nc3, lr=1e-3, λ=0.3) | 0.4017 | `logs/summary_test/exp180_oracle_nc3_lr1e3_lsh0.3_test.log` |
| Val ABS_REL (any) | exp165 (pvitfoa_v3, lr=5e-5, λ=0.3) | 0.3817 | train log at 40/40 |

Oracle gap = 0.4218 (deployable) − 0.4017 (oracle) = **0.0201 ABS_REL** and 1.0744 − 1.0477 = **0.0267 RMSE**. Closing half of each is the operational success definition above.

---

## 8. Addendum (2026-04-19 update) — Group K and Group L design

Group H (exp207–211) + I (exp212–214) completed with no frontier break.
Best new result: **exp213 n3_mssh_energy_attn** at RMSE=1.0777 (+0.003 vs exp172). Group H refuted the "training-side is the bottleneck" diagnosis — BS=32, dw=2.0, freeze=15, and λ_energy=0.3 all produced *worse* metrics than the BS=128 defaults. Re-scoping.

### Group K — HP sweep around exp171 (best ABS_REL) + exp213 (best hybrid)

Purpose: exp171 proved λ_sh=0.3 lowers ABS_REL on the MSSH family; exp213 ties that to EnergyAttn for a good RMSE. Both have narrow un-swept neighborhoods.

| Exp | Base | Axis | Expected |
|---|---|---|---|
| 221 | n3_mssh | λ_sh=0.5 | push ABS further (monotonic improvement hypothesis) |
| 222 | n3_mssh | λ_sh=0.7 | bracket: does ABS turn over? |
| 223 | n3_mssh | λ_sh=0.2 | fine granularity between 0.1 and 0.3 |
| 224 | n3_mssh | sh_dim=9 (order-2) | more SH bandwidth per head → better multi-scale fusion |
| 225 | n3_mssh | freeze=10 | staged on the model that actually won ABS |
| 226 | n3_mssh | dw=2.0 | depth-focus on MSSH (Group H tried this on EnergyAttn — hurt it) |
| 227 | n3_mssh_energy_attn | λ_sh=0.3 | transfer exp171's winning λ into the hybrid |
| 228 | n3_mssh_energy_attn | +λ_energy=0.3 | supervise EnergyHead in the hybrid |
| 229 | n3_mssh_energy_attn | sh_dim=9 + λ_sh=0.3 | order-2 MSSH inside hybrid |
| 215 | n3_energy_attn_distill | teacher=exp180, λ_kd=0.5 | oracle distillation (implemented, student deploys alone) |

Budget: 10 × ~6 h = **60 GPU-hours**. 2 parallel rounds on 10 GPUs.

Kill criteria (K only):
- K1/K2 worse than exp171 → drop the "higher λ_sh is better" hypothesis; MSSH is saturated.
- K4 (sh_dim=9) worse than K3 (λ_sh=0.2) → SH bandwidth is not the lever.
- K10 (distill) within 0.002 RMSE of exp172 → oracle features don't transfer through KD; only helpful as input (not as target).

### Group L — ViT backbone port (supersedes report_d §3.3 Group J)

User intent, plain-language: "replace the convnet backbone with a ViT and put the N3 mechanisms on top of it." Since the 0410 sweep already has ResNet (exp56r–60r, best RMSE **1.1444** — *worse* than UNet 1.0817) and PreViT (exp61–65, best RMSE 1.0818 — *on par* with UNet), the ResNet option can be killed. PreViT is the correct target.

**What needs to be built** (no-code placeholder in `scripts/n3_bulk.sh`, but requires ~200 LoC of new model code):

1. `models/pretrain/pretrained_vit_foa_v6.py` — extends the v3 FiLM architecture with an `EnergyHead` operating on patch tokens:
   - After the last transformer block, reshape patch tokens (B, N, D) → (B, D, H/P, W/P)
   - 3× ConvTranspose2d upsample + Sigmoid → (B, 1, H, W) energy map
   - Apply multiplicative attention on the final depth decoder feature map (matches n3_energy_attn's mechanism)
2. `config/pretrain_vit_foa_v6.yaml` — base of v3 + EnergyHead params
3. Oracle variant `pretrain_vit_foa_v6_oracle`:
   - PatchEmbed conv: 3 → 13 channels (binaural 12-spec + energy 1) adapter
   - Training step already supported via `_train_step_oracle` since `is_foa_oracle_model` keys on name prefix
4. Integrate with existing `_train_step_foa_0415` — v6 emits `pred_energy`, so λ_energy supervision works out of the box.

Proposed experiments once the above lands (exp216–220, exp230):

| Exp | Config | Hypothesis |
|---|---|---|
| 216 | pvitfoa_v6 | Baseline: ViT + EnergyAttn (deployable). |
| 217 | pvitfoa_v6 + sh_dim=9 | Order-2 SH on ViT patches. |
| 218 | pvitfoa_v6 + dw=2.0 | Transfer exp115 recipe to ViT (not tested yet for ViT). |
| 219 | pvitfoa_v6_oracle | New ViT oracle ceiling. Frames Group L's own gap. |
| 220 | pvitfoa_v6 + freeze=20 | Longer freeze on the deeper backbone. |
| 230 | pvitfoa_v6_distill | ViT student, UNet-oracle teacher (exp180). Cross-arch distillation. |

Budget: 5 × ~30 h on 2 GPUs (BS=16) = **~150 GPU-hours wall clock on 2 GPUs, ~15 h on 10**. Highest-EV deployable-metric play still on the table.

### Next action

1. Run Group K now — no new code needed beyond exp215, already implemented.
2. In parallel, draft `models/pretrain/pretrained_vit_foa_v6.py` (rough template: ~150 LoC, copy-edit from v3 + steal EnergyHead from `models/n3_0417/n3_energy_attn.py`).
3. Once K finishes, the best K experiment's hyperparams become the starting point for Group L — do not re-sweep inside Group L, copy over.

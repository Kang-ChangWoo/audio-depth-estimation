# Options to Boost Depth Performance — 04/26 Senior-Reviewer Answer

> Saved from chat 2026-04-26. Context: after diagnosing the bin-gate equilibrium failure (`docs/report_g_0426_report.md`, `EDA/_9_comparison_0426/`) and the noise-floor analysis (`docs/table.md`), the question was: "Are there other options to boost depth performance in similar directions?"

There are several plausible directions, but they need to be ranked honestly against the noise floor. `docs/table.md` showed that 1% RMSE is below HP-induced std for most architectures, so anything claimed at ≤2% is suspect without seed-variance data.

---

## High-leverage (concretely underexplored, real expected gain)

### 1. Higher-order ambisonics (HOA) — `ambi3_npy` and `ambi5_npy` are sitting unused

You have **16-channel 3rd-order** and **36-channel 5th-order** ambisonics already on disk. The codebase only loads `ambi1_npy` (4-channel, 1st-order). HOA gives:

| order        | channels | angular resolution | spatial side-lobes |
|--------------|----------|--------------------|--------------------|
| 1 (current)  | 4        | ~120° main lobe    | huge               |
| 3            | 16       | ~50° main lobe     | moderate           |
| 5            | 36       | ~30° main lobe     | small              |

**Why this should help (theory):** The `rep_gt` you currently feed is computed from an *acoustically blurred* directional representation. With FOA you cannot resolve two reflectors at <120° apart in the same time window. With 5th-order, you can resolve <30°. That's a real information increase, not a representation tweak.

**Why no one has tried it (guess):** the original SoundSpaces 2.0 release default was FOA. Going HOA requires a small change to `_load_foa_ir` and `_compute_rep_gt` in `data/dataset.py` (load 16-ch instead of 4-ch, project per-bin reps onto SH basis up to the corresponding order).

**Risk:** more channels means harder optimization. Need to balance K=8 bins × 16 SH coeffs (= 128-dim per sample) vs current K=8 × 4 (= 32-dim). Per-bin reps may be noisier with smaller IR samples.

**Recommended cell:** swap to `ambi3_npy` in `n4_0425.yaml`, keep all else identical. One head-to-head experiment vs your current best (exp401/exp402). If HOA-3 beats it by ≥0.02 RMSE → real.

### 2. Time-resolved FOA, not just distance-binned

The current `rep_gt` collapses time into K=8 distance bins, then averages each bin into a 4-vector. Each bin loses temporal structure. A more informative rep:

- **Per-frequency-band FOA** instead of per-distance-bin. Take the IR's spectrogram per channel → 4 SH coeffs per (time, freq) cell. The model gets a (4, T, F) tensor instead of (8, 4).
- This preserves the actual spectro-temporal cue your binaural spec already uses, but with directional info added.

This is conceptually closer to "use FOA as additional spectrogram channels" than to "select the right distance bin." Probably more aligned with how depth signals actually live in echoes.

---

## Medium-leverage (plausible but uncertain)

### 3. Multi-step diffusion at inference (use EchoDiffusion as a real diffusion model)

Currently `EchoDiffusion` runs the diffusion UNet at fixed `t=1` — it's a feature extractor, not a diffusion model. If you want the diffusion structure to actually pay off, run **multi-step denoising** at inference: start from depth-noise prior, iteratively refine over 5-10 steps. This is a real architectural change but is what made the original method "diffusion."

**Risk:** existing trained checkpoint can't be reused; needs full retrain with proper noise schedule.

### 4. Cross-dataset training

72 train scenes is small for any depth model. Augmenting with another sound-aware depth dataset (BatVision, Replica with simulated audio) would give you more pose variation. The codebase already has a `BatVisionUNet` class — the data loading path may already support BatVision.

**Risk:** dataset shift can hurt as much as help; needs careful scene-balancing.

### 5. Edge-aware / surface-normal supervision

Current loss is BerHu + SILog on depth. Adding a gradient/edge-aware loss that penalizes wrong depths near depth discontinuities (where depth-from-echoes typically fails most) is a free improvement on the loss side — no architectural change.

```python
loss = depth_loss + λ * gradient_loss(pred, gt)
```

---

## Low-leverage (will fall in noise — don't bother)

- More architectural tweaks on the n4/n9 cascade (gate variants, attention patterns, normalization changes) — your variance data shows the architecture isn't the bottleneck.
- More LR / BS sweeps on existing models — already swept.
- Adding more FoA-conditioning paths (more cross-attention layers, different projections of `rep_gt`) — same gauge-equivalence problem as the gate.
- Self-distillation with EMA teacher — well-known to give ~0.5% improvement in vision; almost certainly noise here.

---

## Honest ranking

Given the noise-floor reality, the **single most worthwhile experiment** is **HOA-3** (option 1). Here's why specifically:

- It tests an information-theoretic hypothesis (more directional channels → more depth signal) that has not been tested on this task.
- It uses data you already have — zero extra storage.
- The implementation is ~50 lines (load 16-ch instead of 4-ch, compute SH-basis projection up to order 3).
- If it works (≥3% RMSE improvement), it's a clean paper finding: "depth-from-echoes benefits from spatial resolution beyond FOA."
- If it doesn't work, it's an equally clean negative finding: "FOA already saturates the task's directional information."
- Either result is a publishable single-figure claim.

Other options either compete with the noise floor (architectural) or are large engineering investments with uncertain payoff (multi-step diffusion, cross-dataset).

**Specific recommendation right now:** before launching HOA, finish the seed-variance run on `n4_0425 exp402` (3 seeds at the same HP) so you have the noise floor. Then HOA becomes interpretable as "real or noise" by direct comparison. Without the variance baseline, you'll just generate another inconclusive sweep.

---

## Cross-reference

- Bin-gate failure diagnosis: `EDA/_9_comparison_0426/out/summary.txt`, `gate_per_bin.png`
- Noise-floor table: `docs/table.md` (HP-induced std per family)
- Cross-family ranking and gap analysis: `docs/report_g_0426_report.md`
- Dataset structure (where ambi3/ambi5 live): `/root/local1/changwoo/matterport3d_0303_renew/<scene>/ambi{1,3,5}_npy/`

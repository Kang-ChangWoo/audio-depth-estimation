# REPRODUCIBILITY — comparison methods and protocol

Replaces `program.md` (deleted 2026-04-27 per cleanup). For experiment-level details (per-row metrics, hypotheses, findings) see `docs/EXPERIMENT_NARRATIVE.md` and `docs/results/*.csv`.

## 1. Comparison methods (must remain reproducible)

| Method | Family | Config | Model entry point | Generator | Notes |
|---|---|---|---|---|---|
| UNet baseline | binaural-only | `config/baseline.yaml` | `models/unet.py` | `unet_256` (8-level, ngf=64) | Most HP-robust; RMSE std=0.0104 over 5 cells |
| ViT (from-scratch) | binaural-only | `config/vit.yaml` | `models/vit.py` | `vit` | LR-fragile — cap at ≤1e-4 |
| BatVision UNet | binaural-only | `config/batvision.yaml` | `models/batvision/` | `unet_256` | Brunetto 2023 reference |
| Echo-Net | binaural-only | `config/echonet.yaml` | `models/echonet/` | `echonet` | Parida 2021; full audio-visual with image branch zeroed |
| EchoDiffusion (base) | binaural + waveform | `config/echodiffusion.yaml` | `models/echodiffusion/` | `echodiffusion` | Wav2Vec2/CIDE conditioning; 132.6M params; off-distribution for RIR |
| Pretrained ResNet | binaural-only | `config/pretrain_resnet.yaml` | `models/pretrain/pretrained_resnet.py` | `pretrained_resnet` | ImageNet ResNet-50 + FPN; uniformly worst on n2_revisit |
| Pretrained ViT | binaural-only | `config/pretrain_vit.yaml` | `models/pretrain/pretrained_vit.py` | `pretrained_vit` | ImageNet ViT-B/16; best ABS_REL on n2_revisit (exp371 = 0.4226) |

### Reproduction commands

```bash
python train.py --config baseline           --experiment-name <name>
python train.py --config vit                --experiment-name <name>
python train.py --config batvision          --experiment-name <name>
python train.py --config echonet            --experiment-name <name>
python train.py --config echodiffusion      --experiment-name <name>
python train.py --config pretrain_resnet    --experiment-name <name>
python train.py --config pretrain_vit       --experiment-name <name>

python test.py  --config <same> --experiment-name <name> --checkpoints best
```

### Verified HP envelope (n2_revisit_test, 2026-04-25/26)

`docs/results/hp_variance.csv` documents per-family HP-induced std/range. Quick reference:

| Family | RMSE mean ± std | ABS_REL mean ± std |
|---|---|---|
| baseline (UNet) | 1.2297 ± 0.0104 | 0.4621 ± 0.0090 |
| vit_baseline | 1.2947 ± 0.0514 | 0.5098 ± 0.0353 |
| echodiffusion | 1.2483 ± 0.0336 | 0.4847 ± 0.0516 |
| pretrained_resnet | 1.3343 ± 0.0176 | 0.5179 ± 0.0230 |
| pretrained_vit | 1.2463 ± 0.0202 | 0.4602 ± 0.0290 |

**A claim of "method A beats method B" requires a gap exceeding ~2× the larger HP-std** (the variance bound), or paired per-sample / repeated-seed evidence at the same HP. Differences inside this envelope are not real.

## 2. Active research lines (binaural at test, deployable)

| Line | Config | Model | Purpose |
|---|---|---|---|
| n3_0425 | `config/n3_0425.yaml` + n3_emap_*, n3_film*, n3_mssh*, n3_energy_attn*, n3_temporal_window* | `models/n3_0425/` | FOA representation predictor (predicts RMS / Eigen reps from binaural; no oracle FOA at test) |
| renew | `config/renew_single*.yaml`, `config/renew_dpt_only*.yaml` | `models/renew/` | Dual-ViT SH36 sound-field bottleneck; KL energy aux; freeze curriculum |
| pvit n3 ports | `config/pvitfoa_v3_eattn.yaml`, `_mssh.yaml` | `models/pretrain/pretrained_vit_foa_v6_eattn.py`, `_mssh.py` | ViT-pretrained backbone with N3 mechanisms |
| echodiffusion_ambi (700-series) | `config/echodiffusion_ambi*.yaml`, `config/echodiff_sh_side_plus.yaml` | `models/echodiffusion/echodiffusion_ambi*.py` | Diffusion + ambisonic input/condition + Wav2Vec2 CIDE; **must remain reproducible (user rule)** |

## 3. Active oracle ceilings (NON-DEPLOYABLE)

Used as upper-bound references; NEVER ranked alongside deployable rows.

| Line | Config | Model | Why oracle |
|---|---|---|---|
| foa_oracle | `config/foa_oracle.yaml`, `_nc1.yaml`, `_nc3.yaml` | `models/unet_foa.py` (FOAOracleGenerator) | Concatenates GT energy map as 3rd input channel |
| n4_0425 | `config/n4_0425.yaml` | `models/n4_0425/` | Conditions on oracle per-bin FOA reps via gated MLP |
| pvitfoa_v6_oracle_nc3 | `config/pvitfoa_v3_oracle_nc3.yaml` | `models/pretrain/pretrained_vit_foa_v6_oracle_nc3.py` | ViT version of foa_oracle nc3 |

## 4. Pre-exp140 code (preserved by user rule, regardless of metric)

These are kept active for historical reproducibility of exp ≤ 140. They produced rows in `ledger_master.csv`.

| File | Configs | Approximate exp range |
|---|---|---|
| `models/unet_foa.py` | `foa.yaml` | exp36-55, 96-120 |
| `models/foa_crossattn.py` | `foa_crossattn.yaml` | exp16-20, 76-79 |
| `models/foa_featbank.py` | `foa_featbank.yaml` | exp21-25, 80-86 |
| `models/foa_msattn.py` | `foa_msattn.yaml` | exp26-30, 87-90 |
| `models/foa_channelattn.py` | `foa_channelattn.yaml` | exp31-35, 91-95 |
| `models/foa_v2.py`, `foa_v2_js.py`, `foa_v2_js_0415.py`, `foa_js_swin.py` | `foa_v2*.yaml` | exp56-60 + JS branch (some ckpts lost in 04-16 cleanup; need retraining if reproducing) |
| `models/foa_0415_v1..v5.py` | `foa_0415_v1..v5.yaml` | exp130-150 (mixed; some <=140) |

## 5. Dataset and evaluation invariants

| Item | Value | Source |
|---|---|---|
| Dataset | `soundspaces` (Matterport3D 0303renew) | dataset.py |
| Split | 9 scenes, 3192 samples; saved to `scene_split.json` (regenerated from `split_seed=42`, `split_ratio=[0.8,0.1,0.1]` if missing) | dataset.py + program.md |
| Output | ERP depth, resized to `[256, 512]`, normalized by `max_depth=10.0` if `depth_norm=true` | dataset.py |
| Audio path | `audio_wav/audio_{idx}.wav`; cut length `int((2*20/340)*sr)` | dataset.py |
| Spectrogram | `n_fft=512`, `win_length=400`, `hop_length=160`, `power=1.0`; resize to 256×512 | dataset.py |
| Ambisonics | `ambi1_npy/ambi1_{idx}.npy`; ACN `[W,Y,Z,X]` | dataset_rotated.py |
| Canonical rotation | Raw FOA rotated by `raw_sample_idx % 4`; W/Z invariant, Y/X sign-swap | dataset_rotated.py |
| FOA RMS target | Per-channel RMS / max-abs (4-D; over-compressed for main claim) | dataset.py |
| Covariance energy map | `R = (IR · IRᵀ) / T`; `E(Ω) = y(Ω)ᵀ R y(Ω)`; max-abs normalized | data/sh_basis.py |
| N2 temporal bins | Default 3 bins at 44.1 kHz: `[0, 2600)`, `[2600, 13000)`, `[13000, end)`; overlap3/4 also defined | data/dataset_n2.py |
| Distance bins | `use_distance_bins=True`; default `rep_kind=eigen`, `rep_K=8`; geometric edges 0.2-10 m for K=8, equal-time edges otherwise | dataset.py |
| Default depth loss | BerHu + SILog, `w_berhu=1.0`, `w_silog=0.5` (some FOA0415/N2 use `w_silog=1.0`) | losses.py + program.md |
| Best-checkpoint selection | `score = 0.7 * RMSE + 0.3 * AbsRel` (lower better) | program.md |
| Pin memory fix | Validation/test: `pin_memory=False`, `persistent_workers=False`, `timeout=120` (avoids val hangs) | data/dataloader.py |
| Checkpoint loading fallback | `test.py` uses `strict=False` and strips `module.` if needed | test.py |

## 6. Cache families (CONFLICT — see `docs/results/conflicts.csv` C1)

Two test caches in active use; **absolute RMSE is not directly comparable across families** per `report_f` Finding 10, although `docs/table.md` (2026-04-26) states identical samples were verified. Preserve both claims.

| Cache hash | Used by | Ambisonic | RMSE band (test) |
|---|---|---|---|
| `e2314b68a4f5` | n2_revisit_test (exp350-374) | OFF | 1.21-1.38 |
| `7027059baf06` | renew_test, n4_test (exp301-308, 390-418) | ON | non-radial 1.07; radial 1.22 |

The cache filename hash does **not** encode `depth_dir`, `rep_kind`, `rep_K`, `use_waveform`, `use_rgb`. Same hash can refer to subtly different sample tensors when those flags differ.

## 7. Modality contract (final claim)

Test-time input must be **binaural audio only**. Anything that uses FOA / energy map / GT depth / oracle geometry at test is labeled NON_DEPLOYABLE in the ledger and may not be ranked alongside deployable rows. The taxonomy: DEPLOYABLE | ORACLE_FOA | ORACLE_ENERGY | ORACLE_GT | RGB_TEACHER_TRAIN_ONLY (none yet).

## 8. Reproduction smoke test (run before any merge that touches these files)

```bash
# For each comparison method <m> in {baseline, vit, batvision, echonet, echodiffusion, pretrain_resnet, pretrain_vit}:
python test.py --config <m> --experiment-name <recent-ckpt-name> --checkpoints best 2>&1 | tail -20
# Verify: RMSE / ABS_REL / δ1 lines appear and match the row in docs/results/ledger_master.csv (or _supplemental.csv for n2_revisit_test entries).
```

If any comparison method fails this check, do not proceed with the cleanup move.

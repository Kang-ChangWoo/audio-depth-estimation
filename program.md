# Program

This document is the single source of truth for planning, adding, running, tracking, and maintaining experiments in this repository for audio-to-depth estimation.

The project may involve:
- replacing or extending model architectures,
- switching or extending datasets,
- adding FOA visualizations,
- adding intermediate analysis and debugging visualizations,
- optionally analyzing selected model parameters,
- and managing experiments in a consistent and reproducible way.

This file should be updated whenever a new experiment, model, dataset, or experiment rule is introduced.

---

## 1. Purpose

The purpose of this repository is to support structured and reproducible experimentation for audio-to-depth estimation.

The main goals are:
1. Compare different models under controlled settings
2. Compare different datasets under controlled settings
3. Add FOA and intermediate analysis utilities for debugging and understanding
4. Track experiments consistently with W&B
5. Keep outputs, checkpoints, and experiment definitions organized

---

## 2. General Principles

Always follow these principles when working on this repository:

- Reuse the existing codebase as much as possible.
- Avoid creating a new training script unless it is absolutely necessary.
- Prefer extending existing config, registry, dataloader, model, loss, evaluation, or utility modules.
- Each experiment should change one major factor at a time unless the goal is explicitly a combined setting.
- Every experiment must have a unique experiment name.
- Every experiment must be reproducible from a single command.
- Every experiment must save outputs to predictable locations.
- Heavy visual outputs must be stored locally, not uploaded to W&B by default.
- A new experiment is not considered complete until it is documented in this file.

---

## 3. Repository Structure

```
baseline/
├── config/                      # Per-model YAML configs
│   ├── baseline.yaml            # UNet baseline config
│   └── foa.yaml                 # FOA (UNet + SH branch) config
├── train.py                     # Training entry point (with W&B)
├── test.py                      # Testing/evaluation entry point
├── data/                        # Dataset & dataloader
│   ├── dataset.py               # SoundSpacesDataset, get_scene_split
│   ├── dataloader.py            # make_dataloader factory
│   └── sh_basis.py              # SH basis (ACN/SN3D), covariance energy
├── models/                      # Network architectures & losses
│   ├── unet.py                  # UNet generator (pix2pix-based)
│   ├── unet_foa.py              # AudioDepthFOAGenerator, DeepScaleShift
│   └── losses.py                # DepthLoss, FOA losses, SH alignment
├── utils/                       # Helpers
│   ├── config.py                # load_config(config_name, mode, exp)
│   ├── metrics.py               # compute_errors, compute_foa_errors
│   ├── visualization.py         # save_batch_visualization, load_gt_rgb
│   ├── train_utils.py           # build_model, build_criterion, helpers
│   └── test_utils.py            # evaluate
├── scripts/                     # Shell scripts
│   ├── train.sh
│   └── test.sh
├── program.md                   # This file
├── README.md
├── .gitignore
├── checkpoints/                 # Saved weights (gitignored)
├── results/                     # Visualizations (gitignored)
└── eval/                        # Evaluation stats (gitignored)
```

### Module responsibilities

| Module | Role | Modification policy |
|--------|------|---------------------|
| `config/*.yaml` | Per-model hyperparameters | Edit freely per experiment |
| `train.py` | Training loop orchestration | Edit for training flow changes |
| `test.py` | Evaluation loop orchestration | Edit for evaluation flow changes |
| `data/dataset.py` | Dataset class, scene splitting | Edit to add new datasets |
| `data/dataloader.py` | DataLoader factory | Rarely needs changes |
| `data/sh_basis.py` | SH math + covariance energy | Extend only |
| `models/unet.py` | UNet baseline architecture | Edit or add new model files |
| `models/unet_foa.py` | AudioDepthFOAGenerator (UNet+SH) | Edit for FOA model changes |
| `models/losses.py` | Loss functions (depth, FOA, hist) | Edit to add new losses |
| `utils/config.py` | Config loading | Rarely needs changes |
| `utils/metrics.py` | Evaluation metrics | Do not modify |
| `utils/visualization.py` | Plotting utilities | Edit to add new visualizations |
| `utils/train_utils.py` | Model/criterion builders | Edit when adding models/losses |
| `utils/test_utils.py` | Evaluation loop | Edit for evaluation changes |

### How to add new components

- **New model**: Add `models/new_model.py`, register in `models/__init__.py`, update `utils/train_utils.py:build_model`, create `config/new_model.yaml`
- **New dataset**: Add to `data/dataset.py` or create `data/new_dataset.py`, update `data/dataloader.py`
- **New loss**: Add to `models/losses.py`, update `utils/train_utils.py:build_criterion`
- **New metric**: Add to `utils/metrics.py`
- **New visualization**: Add to `utils/visualization.py`

### Config system

Configs live in `config/` as separate YAML files per model variant:
- `config/baseline.yaml` -- UNet baseline (no ambisonic)
- `config/foa.yaml` -- FOA model (UNet + SH branch, with ambisonic)

Select config via `--config` flag:
```bash
python train.py --config baseline --experiment-name exp001
python train.py --config foa --experiment-name exp002
```

### Scene split

The train/val/test scene split is saved as an explicit `scene_split.json` dictionary inside the dataset directory. On first run it is generated from `split_ratio` and `split_seed` in config, then loaded from file on all subsequent runs. To regenerate, delete `scene_split.json`.

---

## 4. W&B Policy

W&B is used for lightweight experiment tracking only.
All experiments log to the W&B project **`neurips_audio_depth`**.

### 4.1 Setup

Set the W&B API key via environment variable before training:
```bash
export WANDB_API_KEY=<your-token>
```
Do NOT hardcode the token in any source file.

### 4.2 What must be logged to W&B

Per epoch:
- `train/loss`, `train/depth`, `train/foa`, `train/hist`
- `val/val_loss`, `val/abs_rel`, `val/rmse`, `val/delta1-3`, `val/log10`, `val/mae`
- `best/rmse`, `best/abs_rel`, `best/epoch` (on best-model update)

Run config:
- experiment_name, model, dataset, optimizer, lr, batch_size, epochs, params_M
- (FOA only) depth_weight, foa_weight, hist_weight, sh_order, proj_dim, foa_freeze_epochs

### 4.3 What must NOT be logged to W&B by default

To avoid storage problems, do not upload images/media to W&B. Save all visual outputs locally under `results/`.

---

## 5. Default Loss Setup

Both baseline and FOA models use the same default depth loss: **BerHu + SILog**.

```yaml
# In config/*.yaml
train:
  use_berhu: true
  use_silog: true
  w_berhu: 1.0
  w_silog: 0.5
```

- **BerHuLoss**: Reverse Huber -- L1 for small errors, L2 for large errors, with adaptive threshold `c = 0.2 * max(|diff|)`
- **SILogLoss**: Scale-invariant log loss with variance weight 0.5

For the FOA model, this depth loss is wrapped in `AudioDepthFOALoss` which adds:
- `foa_weight * FOAGuidedLoss` (L1 + cosine on FOA coefficients)
- `hist_weight * SHHistogramAlignmentLoss` (SH energy-depth alignment)

---

## 6. Models

### 6.1 Baseline UNet (`config/baseline.yaml`)

Standard pix2pix UNet (8-level, ngf=64). Input: `(B, 2, H, W)` binaural spectrogram, output: `(B, 1, H, W)` depth. No ambisonic data used.

### 6.2 AudioDepthFOAGenerator (`config/foa.yaml`)

UNet encoder-decoder with SH auxiliary branch. Located in `models/unet_foa.py`.

```
Input: (B, 2, H, W) binaural spectrogram
  ├── UNet Encoder (8-level, ngf=64) -> bottleneck (512-dim)
  ├── SH Branch:
  │     pool -> audio_proj(512->128) -> foa_head(128->4) + hoa_head(128->32)
  │     -> pred_sh (36 SH5 coefficients)
  │     -> DeepScaleShift(36) for histogram alignment
  ├── UNet Decoder (with skip connections) -> pred_depth (B, 1, H, W)
  Output: dict { pred_depth, pred_foa, pred_sh, foa_latent, ... }
```

**Hyperparameters** (from `config/foa.yaml`):

| Parameter | Value |
|-----------|-------|
| generator | unet_256 (num_downs=8, ngf=64) |
| proj_dim | 128 |
| sh_order | 5 (36 SH coefficients) |
| depth_weight | 1.0 |
| foa_weight | 0.1 |
| hist_weight | 0.1 |
| foa_cosine_weight | 0.1 |
| optimizer | AdamW, lr=0.001 |
| batch_size | 32 |
| epochs | 40 |
| depth loss | BerHu (w=1.0) + SILog (w=0.5) |
| total params | ~55M |

### 6.3 FOA Freeze Warmup

`foa_freeze_epochs` controls a warmup period where only the depth branch trains.
During freeze: SH branch (audio_proj, foa_head, hoa_head, scale_shift) has requires_grad=False.

---

## 7. Ambisonic Energy Map: Covariance Correction

### 7.1 Problem

The old method computed energy maps via RMS-per-channel then linear SH projection. This is physically incorrect -- it ignores cross-channel correlations.

### 7.2 Correct Method

```
E(Omega) = y(Omega)^T R y(Omega)
where R = (1/T) sum_t b(t) b(t)^T  (inter-channel covariance)
```

Implementation in `data/sh_basis.py`: `compute_covariance()`, `energy_map_from_cov()`.

### 7.3 Dataset Output Format

With `use_ambisonic=True` (FOA model), the dataset returns a **4-tuple**:
```
(audio, gt_depth, foa_target, energy_map)
```
- `foa_target`: (4,) channel RMS from IR, normalized
- `energy_map`: (1, H, W) covariance-based directional energy, normalized

Without ambisonic (baseline), returns a **2-tuple**: `(audio, gt_depth)`.

---

## 8. Standard Workflow for Adding a New Experiment

1. Inspect existing code before writing new files
2. Define: what changes, what stays fixed, comparison baseline, target metric
3. Implement with minimal changes (config, model, loss, builder)
4. Sanity checks: forward pass, loss, validation, checkpoint save, W&B logging
5. Document the experiment in section 13

---

## 9. Best Model Selection Policy

Use RMSE as primary selection metric. Always monitor both abs_rel and RMSE.

Required rule:
- Never save best checkpoint based on abs_rel alone
- Log all 7 metrics for every best-model update

---

## 10. Rules for Adding a New Model

- Place in `models/` as a new file
- Export in `models/__init__.py`
- Update `utils/train_utils.py:build_model`
- Create `config/{model_name}.yaml`
- Interface:
  - Baseline-style: input `(B, 2, H, W)` -> output `(B, 1, H, W)` tensor
  - FOA-style: input `(B, 2, H, W)` -> output dict with `pred_depth`, `pred_foa`, etc.

---

## 11. Naming Convention

Format: `{exp_id}_{model}_{input}_{dataset}_{setting}`

Examples:
- `exp001_unet_foa_soundspaces_baseline`
- `exp002_unet_binaural_soundspaces_no_ambisonic`

---

## 12. Required Output Structure

```text
results/{experiment_name}/
checkpoints/{experiment_name}/best_model.pth, checkpoint_{epoch}.pth
eval/{dataset_name}/{split}/stats_{experiment_name}.pt
```

---

## 13. Experiment Registry

_(Add experiments below as they are run.)_

---

## 14. Maintenance Rule

This file must be updated whenever:
- a new experiment, dataset, or model family is added,
- the logging/loss/output policy changes,
- or the experiment naming convention changes.

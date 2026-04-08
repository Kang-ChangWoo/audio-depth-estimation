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
│   ├── echodiffusion.yaml       # EchoDiffusion config
│   ├── foa.yaml                 # FOA (UNet + SH branch) config
│   ├── foa_crossattn.yaml       # FOA + Cross-Attention Bridge
│   ├── foa_featbank.yaml        # FOA + Feature Bank
│   ├── foa_msattn.yaml          # FOA + Multi-Scale Attention
│   ├── foa_channelattn.yaml     # FOA + Channel Attention (SE)
│   └── vit.yaml                 # ViT baseline config
├── train.py                     # Training entry point (with W&B)
├── test.py                      # Testing/evaluation entry point
├── data/                        # Dataset & dataloader
│   ├── dataset.py               # SoundSpacesDataset, get_scene_split
│   ├── dataloader.py            # make_dataloader factory
│   └── sh_basis.py              # SH basis (ACN/SN3D), covariance energy
├── models/                      # Network architectures & losses
│   ├── unet.py                  # UNet generator (pix2pix-based)
│   ├── unet_foa.py              # AudioDepthFOAGenerator, DeepScaleShift
│   ├── foa_crossattn.py         # FOACrossAttnGenerator
│   ├── foa_featbank.py          # FOAFeatBankGenerator
│   ├── foa_msattn.py            # FOAMultiScaleAttnGenerator
│   ├── foa_channelattn.py       # FOAChannelAttnGenerator
│   ├── vit.py                   # AudioDepthViT (ViT encoder + conv decoder)
│   ├── echonet/                 # Echo-Net (Parida et al., CVPR 2021)
│   │   ├── __init__.py
│   │   └── echonet.py           # EchoNet: full audio-visual (image=zeros)
│   ├── batvision/               # BatVision UNet (Brunetto et al., IROS 2023)
│   │   ├── __init__.py
│   │   └── batvision.py         # BatVisionUNet: 8-block pix2pix UNet
│   ├── pretrain/                # Pretrained ImageNet backbones
│   │   ├── __init__.py
│   │   ├── pretrained_vit.py    # PretrainedViT: ViT-B/16 + decoder
│   │   └── pretrained_resnet.py # PretrainedResNet: ResNet-50 + FPN decoder
│   └── losses.py                # DepthLoss, FOA losses, SH alignment, KL
├── utils/                       # Helpers
│   ├── config.py                # load_config(config_name, mode, exp)
│   ├── metrics.py               # compute_errors, compute_foa_errors
│   ├── visualization.py         # save_batch_visualization, load_gt_rgb
│   ├── train_utils.py           # build_model, build_criterion, helpers
│   └── test_utils.py            # evaluate
├── scripts/                     # Shell scripts
│   ├── train.sh
│   ├── test.sh
│   ├── bulk_train.sh            # Parallel training: 4 models × 2 GPUs
│   ├── bulk_test.sh             # Parallel testing: 4 models × 2 GPUs
│   ├── bulk0407_55exps.sh       # 55-experiment sweep (5 slots × 2 GPUs, GPUs 0-9)
│   └── bulk0408_65exps.sh       # 65-experiment sweep (4 slots × 2 GPUs, GPUs 0-7)
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
- `config/echodiffusion.yaml` -- EchoDiffusion (diffusion UNet + CIDE conditioning)
- `config/echonet.yaml` -- Echo-Net (full audio-visual, image branch deactivated)
- `config/batvision.yaml` -- BatVision UNet (8-block pix2pix UNet, audio-only)
- `config/pretrain_vit.yaml` -- Pretrained ViT-B/16 (ImageNet) + decoder
- `config/pretrain_resnet.yaml` -- Pretrained ResNet-50 (ImageNet) + FPN decoder
- `config/foa.yaml` -- FOA model (UNet + SH branch, with ambisonic)
- `config/vit.yaml` -- ViT baseline (Vision Transformer encoder + conv decoder)

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

### 6.3 EchoDiffusion (`config/echodiffusion.yaml`)

Diffusion UNet backbone repurposed as a feature extractor. Located in `models/echodiffusion/`.

```
Input: (B, 2, H, W) binaural spectrogram + (B, 2, T) raw waveform
  ├── ASPP+ASFF UNet -> 128ch latent (32x32)
  ├── CIDE: Wav2Vec2 -> scene embeddings -> cross-attention context
  ├── Diffusion UNet (t=1) -> hierarchical features
  ├── Feature aggregation + Decoder -> depth (B, 1, H, W)
  Output: depth map resized to original input dimensions (nearest interpolation)
```

| Parameter | Value |
|-----------|-------|
| embed_dim | 192 |
| emb_dim (CIDE) | 768 |
| optimizer | AdamW, lr=0.0001 |
| batch_size | 32 |
| depth loss | BerHu (w=1.0) + SILog (w=0.5) |

### 6.4 ViT Baseline (`config/vit.yaml`)

Vision Transformer encoder with convolutional decoder. Located in `models/vit.py`.

```
Input: (B, 2, H, W) binaural spectrogram
  ├── Patch embedding (16x16 patches) -> sequence of tokens
  ├── CLS token + learnable positional embedding
  ├── Transformer encoder (12 layers, 12 heads, dim=768)
  ├── ConvDecoder: progressive upsampling (4 stages) -> depth
  Output: (B, 1, H, W) depth map (nearest interpolation if size mismatch)
```

| Parameter | Value |
|-----------|-------|
| patch_size | 16 |
| embed_dim | 768 |
| depth | 12 layers |
| num_heads | 12 |
| mlp_ratio | 4.0 |
| optimizer | AdamW, lr=0.0001 |
| batch_size | 32 |
| depth loss | BerHu (w=1.0) + SILog (w=0.5) |
| total params | ~87M |

### 6.5 FOA Variants (4 architectures)

All FOA variants share the base UNet encoder-decoder + SH branch from AudioDepthFOAGenerator, but add different bridge modules and KL divergence regularization.

| Variant | Config | Bridge Module | KL Loss Target |
|---------|--------|--------------|----------------|
| `foa_crossattn` | `foa_crossattn.yaml` | Cross-attention (SH queries, encoder KV) | VAE latent (mu, logvar) |
| `foa_featbank` | `foa_featbank.yaml` | Learnable feature bank (K=64 prototypes) | Bank attention uniformity |
| `foa_msattn` | `foa_msattn.yaml` | Multi-scale attention across encoder levels | Scale attention uniformity |
| `foa_channelattn` | `foa_channelattn.yaml` | SE channel attention on bottleneck + skips | Channel attention uniformity |

All variants output a `kl_loss` key in their forward dict, weighted by `kl_weight` (default 0.01) in the loss.

### 6.6 Echo-Net (`config/echonet.yaml`)

Full architecture from "Beyond Image to Depth" (Parida et al., CVPR 2021). Located in `models/echonet/`. Image input is **deactivated** (fed as zeros), but the image branch is preserved and trainable.

```
Input: (B, 2, H, W) binaural spectrogram
  ├── (1) Echo SubNet   – 3-conv encoder → 512-d bottleneck → 7 upconv decoder
  │       echo → D_echo (B,1,H,W) + feat (B,512,1,1)
  ├── (2) Visual SubNet – 5-level UNet with skip connections (input: zeros)
  │       zeros(B,3,H,W) → D_image (B,1,H,W) + feat (B,512,h,w)
  ├── (3) Material SubNet – ResNet-18 backbone (input: zeros)
  │       zeros(B,3,H,W) → feat (B,512,h,w)
  ├── (4) Multimodal Fusion – bilinear(img,echo) + bilinear(mat,echo) → concat
  │       → fused (B,1024,h,w)
  ├── (5) Attention Net – 5 upconv layers → α ∈ [0,1] per pixel
  └── Final: D = α ⊙ D_echo + (1 − α) ⊙ D_image
  Output: (B, 1, H, W) depth map
```

| Parameter | Value |
|-----------|-------|
| conv1x1_dim | 8 |
| bottleneck_dim | 512 |
| optimizer | AdamW, lr=0.001 |
| batch_size | 32 |
| depth loss | BerHu (w=1.0) + SILog (w=0.5) |
| total params | ~321M (fusion bilinear layers dominate) |

Run: `python train.py --config echonet --experiment-name echonet_baseline`

### 6.7 BatVision UNet (`config/batvision.yaml`)

Audio-only depth prediction from "The Audio-Visual BatVision Dataset for Research on Sight and Sound" (Brunetto, Hornauer, Yu, Moutarde — IROS 2023). Located in `models/batvision/`.

```
Input: (B, 2, H, W) binaural spectrogram
  ├── 8-block pix2pix-style recursive UNet with skip connections
  │   Encoder: Conv2d(k=4,s=2,p=1) + BatchNorm + LeakyReLU(0.2)
  │   Decoder: ConvTranspose2d(k=4,s=2,p=1) + BatchNorm + ReLU
  │   Skip connections via channel concatenation
  │   Filter progression: 64→128→256→512→512→512→512→512 (bottleneck)
  │   Output: Sigmoid (depth_norm=True)
  Output: (B, 1, H, W) depth map
```

| Parameter | Value |
|-----------|-------|
| num_downs | 8 (unet_256) |
| ngf | 64 |
| optimizer | AdamW, lr=0.001 |
| batch_size | 32 |
| depth loss | BerHu (w=1.0) + SILog (w=0.5) |
| total params | ~54M |

Run: `python train.py --config batvision --experiment-name batvision_baseline`

### 6.8 Pretrained ViT-B/16 (`config/pretrain_vit.yaml`)

ImageNet-pretrained ViT-B/16 encoder adapted for audio-to-depth. Located in `models/pretrain/pretrained_vit.py`.

The 2-channel spectrogram is projected to 3-channel pseudo-RGB via a learnable `Conv2d(2, 3, 1)` input adapter, enabling direct use of ImageNet-pretrained weights. Positional embeddings are bicubically interpolated from the original 14×14 grid to 16×32 at init.

```
Input: (B, 2, H, W) binaural spectrogram
  ├── Input adapter: Conv2d(2→3, 1×1) — spectrogram → pseudo-RGB
  ├── Patch embedding: Conv2d(3→768, k=16, s=16) → (B, 512, 768) tokens
  ├── Prepend CLS token + interpolated positional embedding
  ├── ViT-B/16 Transformer encoder (12 layers, 12 heads, dim=768)
  ├── Drop CLS, reshape patch tokens to (768, 16, 32) spatial grid
  ├── Progressive ConvTranspose decoder: 768→256→128→64→32→16→1
  Output: (B, 1, H, W) depth map (bilinear resize if needed)
```

| Parameter | Value |
|-----------|-------|
| backbone | ViT-B/16 (ImageNet pretrained) |
| pretrained | true |
| freeze_encoder | false (fine-tune all) |
| optimizer | AdamW, lr=0.0001 |
| batch_size | 16 |
| total params | ~90M |

Run: `python train.py --config pretrain_vit --experiment-name pretrain_vit`

### 6.9 Pretrained ResNet-50 (`config/pretrain_resnet.yaml`)

ImageNet-pretrained ResNet-50 encoder with FPN-style decoder for audio-to-depth. Located in `models/pretrain/pretrained_resnet.py`.

Same `Conv2d(2, 3, 1)` input adapter as ViT. Fully convolutional — handles (256, 512) natively without resizing. Multi-scale features from layers 1–4 are progressively decoded with skip connections.

```
Input: (B, 2, H, W) binaural spectrogram
  ├── Input adapter: Conv2d(2→3, 1×1) — spectrogram → pseudo-RGB
  ├── ResNet-50 encoder (multi-scale features):
  │     stem  (64,  H/4, W/4)
  │     layer1 (256, H/4, W/4)
  │     layer2 (512, H/8, W/8)
  │     layer3 (1024, H/16, W/16)
  │     layer4 (2048, H/32, W/32)
  ├── FPN decoder with skip connections:
  │     reduce4(2048→512) → +layer3 → +layer2 → +layer1 → +stem → head(→1)
  Output: (B, 1, H, W) depth map
```

| Parameter | Value |
|-----------|-------|
| backbone | ResNet-50 (ImageNet V2 pretrained) |
| pretrained | true |
| freeze_encoder | false (fine-tune all) |
| optimizer | AdamW, lr=0.0001 |
| batch_size | 32 |
| total params | ~30M |

Run: `python train.py --config pretrain_resnet --experiment-name pretrain_resnet`

### 6.10 FOA Freeze Warmup

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

Best model is selected by **weighted score**: `0.7 * RMSE + 0.3 * abs_rel` (lower is better).

This prevents early-epoch models with good abs_rel but poor qualitative (RMSE) results from being selected.

Required rule:
- Never save best checkpoint based on abs_rel alone
- Log all 7 metrics for every best-model update
- Track and log the weighted score alongside RMSE and abs_rel

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

### Sanity check: 5-epoch test (2026-04-07)

All 4 models trained for 5 epochs on 2 GPUs each (8 GPUs total). No errors.

| Model | GPUs | Train Loss | Val RMSE | Val ABS_REL | Val Delta1 |
|-------|------|-----------|----------|-------------|------------|
| baseline | 0,1 | 0.1905 | 1.3456 | 0.4054 | 0.4536 |
| echodiffusion | 2,3 | 0.1918 | 1.3088 | 0.4028 | 0.4913 |
| foa | 4,5 | 0.2426 | 1.2958 | 0.4597 | 0.4936 |
| vit | 6,7 | 0.2021 | 1.3297 | 0.4287 | 0.4610 |

### 55-Experiment Sweep (2026-04-07) — `scripts/bulk0407_55exps.sh`

Run: `bash scripts/bulk0407_55exps.sh`
- 5 concurrent slots × 2 GPUs each = 10 GPUs
- 40 epochs per experiment, validation every 4 epochs
- Logs in `logs/bulk0407/`

| Exp | Model | What varies |
|-----|-------|------------|
| 01-05 | baseline | lr ∈ {1e-3, 5e-4, 1e-4}, bs ∈ {32, 16} |
| 06-10 | vit | lr ∈ {1e-4, 5e-5, 5e-4, 1e-5}, bs ∈ {32, 16} |
| 11-15 | echodiffusion | lr ∈ {1e-4, 5e-5, 5e-4, 1e-5}, bs ∈ {32, 16} |
| 16-20 | foa_crossattn | lr ∈ {1e-3, 5e-4, 1e-4}, foa_weight ∈ {0.1, 0.2} |
| 21-25 | foa_featbank | lr ∈ {1e-3, 5e-4, 1e-4}, foa_weight ∈ {0.1, 0.2} |
| 26-30 | foa_msattn | lr ∈ {1e-3, 5e-4, 1e-4}, foa_weight ∈ {0.1, 0.2} |
| 31-35 | foa_channelattn | lr ∈ {1e-3, 5e-4, 1e-4}, foa_weight ∈ {0.1, 0.2} |
| 36-55 | foa (original) | lr, bs, depth_weight, foa_weight, hist_weight, foa_freeze_epochs |

### 65-Experiment Sweep (2026-04-08) — `scripts/bulk0408_65exps.sh`

Run: `bash scripts/bulk0408_65exps.sh`
- 4 concurrent slots × 2 GPUs each = 8 GPUs (0-7)
- 40 epochs per experiment, validation every 4 epochs
- Logs in `logs/bulk0408/`
- Index 56–120 (no conflict with bulk0407 exps 01–55)

**Group A — New model baselines (20 exps: 56–75)**

| Exp | Model | What varies |
|-----|-------|------------|
| 56-60 | pretrain_resnet | lr ∈ {1e-4, 5e-5, 5e-4, 3e-4}, bs ∈ {16, 32} |
| 61-65 | pretrain_vit | lr ∈ {1e-4, 5e-5, 5e-4, 3e-5}, bs ∈ {8, 16} |
| 66-70 | echonet | lr ∈ {1e-3, 5e-4, 1e-4, 2e-3}, bs ∈ {8, 16} |
| 71-75 | batvision | lr ∈ {1e-3, 5e-4, 1e-4, 2e-3}, bs ∈ {16, 32} |

**Group B — FOA variants, new combos (20 exps: 76–95)**

| Exp | Model | What varies (vs 0407: new kl_weight, fw, hw combos) |
|-----|-------|------------|
| 76-80 | foa_crossattn | fw ∈ {0.05, 0.1, 0.2, 0.3}, kl ∈ {0.005, 0.01, 0.02}, hw=0.2 |
| 81-85 | foa_featbank | fw ∈ {0.05, 0.1, 0.2, 0.3}, kl ∈ {0.005, 0.01, 0.02}, hw=0.2 |
| 86-90 | foa_msattn | fw ∈ {0.05, 0.1, 0.2, 0.3}, kl ∈ {0.005, 0.01, 0.02}, hw=0.2 |
| 91-95 | foa_channelattn | fw ∈ {0.05, 0.1, 0.2, 0.3}, kl ∈ {0.005, 0.01, 0.02}, hw=0.2 |

**Group C — FOA main, wider search (25 exps: 96–120)**

| Exp | What varies |
|-----|------------|
| 96-98 | lr ∈ {2e-4, 3e-4, 7e-4} (new LR points) |
| 99-102 | dw ∈ {1.5}, fw ∈ {0.15}, hw ∈ {0.15} (intermediate values) |
| 103-107 | fw ∈ {0.15, 0.3}, hw ∈ {0.15, 0.3} (wider range) |
| 108 | dw=2.0 at lr=5e-4 |
| 109-112 | freeze ∈ {3, 5, 10, 15} (freeze schedule sweep) |
| 113-115 | extreme combos: fw=0.05/hw=0.05, dw=2.0/fw=0.2, dw=0.5/hw=0.2 |
| 116-120 | lr ∈ {3e-4, 2e-4, 7e-4}, bs=16, foa_freeze combos |

---

## 14. Maintenance Rule

This file must be updated whenever:
- a new experiment, dataset, or model family is added,
- the logging/loss/output policy changes,
- or the experiment naming convention changes.

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
UNetSoundOnly/
├── config.yaml              # All settings: dataset, model, train, test
├── train.py                 # Training entry point
├── test.py                  # Testing/evaluation entry point
├── data/                    # Dataset & dataloader
│   ├── dataset.py           # SoundSpacesDataset, get_scene_split
│   ├── dataloader.py        # make_dataloader factory
│   └── sh_basis.py          # Spherical Harmonics basis (ACN/SN3D)
├── models/                  # Network architecture & losses
│   ├── unet.py              # UNet generator (pix2pix-based), define_G
│   └── losses.py            # SIlogLoss
├── utils/                   # Helpers
│   ├── config.py            # load_config (reads config.yaml)
│   ├── metrics.py           # compute_errors, compute_foa_errors
│   ├── visualization.py     # save_batch_visualization, load_gt_rgb
│   ├── train_utils.py       # build_model, build_criterion, compute_loss
│   └── test_utils.py        # evaluate
├── scripts/                 # Shell scripts
│   ├── train.sh
│   └── test.sh
├── program.md               # This file — experiment program document
├── README.md                # Quick start guide
├── .gitignore
├── checkpoints/             # Saved model weights (gitignored)
├── results/                 # Visualizations & logs (gitignored)
└── eval/                    # Evaluation statistics (gitignored)
```

### Module responsibilities

| Module | Role | Modification policy |
|--------|------|---------------------|
| `config.yaml` | All hyperparameters, paths, settings | Edit freely per experiment |
| `train.py` | Training loop orchestration | Edit for training flow changes |
| `test.py` | Evaluation loop orchestration | Edit for evaluation flow changes |
| `data/dataset.py` | Dataset class, scene splitting | Edit to add new datasets |
| `data/dataloader.py` | DataLoader factory | Rarely needs changes |
| `data/sh_basis.py` | SH math utilities | Do not modify |
| `models/unet.py` | UNet architecture | Edit or add new model files |
| `models/losses.py` | Loss functions | Edit to add new losses |
| `utils/config.py` | Config loading | Rarely needs changes |
| `utils/metrics.py` | Evaluation metrics | Do not modify |
| `utils/visualization.py` | Plotting utilities | Edit to add new visualizations |
| `utils/train_utils.py` | Model/criterion builders | Edit when adding models/losses |
| `utils/test_utils.py` | Evaluation loop | Edit for evaluation changes |

### How to add new components

- **New model**: Add `models/new_model.py`, register in `models/__init__.py`, update `utils/train_utils.py:build_model`
- **New dataset**: Add to `data/dataset.py` or create `data/new_dataset.py`, update `data/dataloader.py`
- **New loss**: Add to `models/losses.py`, update `utils/train_utils.py:build_criterion`
- **New metric**: Add to `utils/metrics.py`
- **New visualization**: Add to `utils/visualization.py`

### Scene split

The train/val/test scene split is saved as an explicit `scene_split.json` dictionary inside the dataset directory. On first run it is generated from `split_ratio` and `split_seed` in config, then loaded from file on all subsequent runs. To regenerate, delete `scene_split.json`.

---

## 4. W&B Policy

W&B is used for lightweight experiment tracking only.

### 4.1 What must be logged to W&B

Always log the following:
- experiment name
- model name
- dataset name
- input type
- optimizer
- scheduler if used
- learning rate
- batch size
- number of epochs
- seed
- loss type
- training loss
- validation loss
- validation metrics
- best metric
- best epoch
- checkpoint path
- result directory path
- tags or group information if used

Required validation metrics:
- abs_rel
- RMSE
- Delta1
- Delta2
- Delta3
- Log10
- MAE

### 4.2 What must NOT be logged to W&B by default

To avoid storage problems, do not upload the following by default:
- FOA visualization images
- predicted depth image grids
- ground-truth vs prediction comparison images
- error map images
- intermediate feature maps
- activation heatmaps
- parameter histogram images
- large debug media tables
- per-sample visualization galleries
- heavy plots generated every epoch

These must be saved locally instead.

### 4.3 Optional exception

Only upload visual media to W&B if all of the following are true:
- it is explicitly requested,
- it is limited to a very small subset of runs,
- and it is limited to a very small subset of samples or epochs.

By default, keep W&B scalar-only.

---

## 4.4 Metric Policy

The following validation metrics must always be tracked together:

- abs_rel
- RMSE
- Delta1
- Delta2
- Delta3
- Log10
- MAE

These metrics should be treated as a set, not as isolated numbers.

Important rule:
- Do not judge model quality using only abs_rel.
- A model that improves abs_rel but severely degrades RMSE should not automatically be considered better.
- A model that improves one metric while clearly breaking overall depth quality should be treated as unstable or unbalanced.

Recommended interpretation:
- abs_rel reflects relative error behavior.
- RMSE reflects large-error sensitivity and failure on badly predicted regions.
- Delta metrics reflect threshold accuracy.
- Log10 reflects scale-consistent error.
- MAE reflects average absolute error magnitude.

Therefore, abs_rel and RMSE must always be checked together.

---

## 5. Local Visualization Policy

Visualizations are important for debugging and analysis, but they must be stored locally.

Recommended local outputs include:
- FOA channel visualizations
- predicted depth vs ground truth
- absolute error maps
- selected intermediate feature visualizations
- debug sample dumps
- parameter analysis plots for important models

Recommended output locations:
- `results/{experiment_name}/visualizations/`
- `results/{experiment_name}/analysis/`

Recommended naming rule:
- include split, epoch, and sample identifier when applicable

Examples:
- `val_epoch_010_sample_0003_pred.png`
- `val_epoch_010_sample_0003_gt.png`
- `val_epoch_010_sample_0003_error.png`
- `val_epoch_010_sample_0003_foa.png`
- `analysis_epoch_010_layer3_feature.png`

Visualization frequency should be controlled. Avoid saving outputs for every batch.

Recommended strategy:
- save only for fixed sample indices,
- save only every few epochs,
- and keep the saved set consistent across runs for fair comparison.

---

## 6. Standard Workflow for Adding a New Experiment

Whenever a new experiment is added, follow this process.

### Step 1. Inspect the existing code
Before writing new code:
- find similar models, datasets, or utilities already implemented,
- identify reusable training logic,
- and avoid duplicating scripts.

### Step 2. Define the experiment clearly
Before implementation, write down:
- what changes,
- what remains fixed,
- which baseline it should be compared against,
- and which metric is the main target.

### Step 3. Implement with minimal changes
Typical modification targets:
- `config.yaml`
- `models/` (new or modified model file)
- `models/losses.py` (new loss)
- `data/dataset.py` (new dataset)
- `utils/train_utils.py` (builder updates)
- `utils/visualization.py` (new visualization)

### Step 4. Run sanity checks
Before launching full training:
- one-batch forward pass
- input/output shape check
- loss computation check
- validation step check
- checkpoint save check
- local visualization save check
- W&B scalar logging check

### Step 5. Document the experiment
Add the new experiment to the registry section in this file before or immediately after running it.

---

## 6.1 Best Model Selection Policy

Best model selection must not rely on a single metric alone.

### Primary concern
In this project, abs_rel may decrease while RMSE becomes unstable or worse.
This usually means the model is improving relative error on many samples while still producing large failures on some regions or examples.

Therefore, the best model policy must explicitly guard against this failure mode.

### Required rule
When selecting the best model:
- always monitor both abs_rel and RMSE,
- and never save the best checkpoint based on abs_rel alone.

### Recommended selection strategies

#### Option A. RMSE-gated abs_rel selection
Use abs_rel as the primary metric, but only accept a new best model if:
- abs_rel improves,
- and RMSE does not degrade beyond a small tolerance.

Example rule:
- update best model if `abs_rel < best_abs_rel` and `RMSE <= best_RMSE * (1 + tolerance)`

Recommended tolerance:
- 1% to 3% depending on validation noise

This is the recommended default when abs_rel is still the main target but RMSE collapse must be prevented.

#### Option B. Composite score
Define a composite validation score using normalized metrics.

Example idea:
- minimize a weighted score built from abs_rel, RMSE, Log10, and MAE
- optionally add penalties if Delta1 drops

#### Option C. Pareto-style rule
Treat a checkpoint as better only if:
- it improves abs_rel and does not meaningfully worsen RMSE,
- or it improves RMSE and does not meaningfully worsen abs_rel.

This is simple and robust for ablation studies.

### Recommended default for this repository
Use RMSE-gated abs_rel selection as the default best-checkpoint rule.

### Tie-breaking recommendation
When two checkpoints have very similar abs_rel:
- prefer lower RMSE,
- then prefer higher Delta1,
- then prefer lower MAE.

### Logging requirement
For every best-model update, record:
- current abs_rel, RMSE, Delta1, Delta2, Delta3, Log10, MAE
- reason for update

---

## 7. Rules for Adding a New Model

When adding a new model:

- place it in `models/` as a new file (e.g., `models/resnet.py`),
- export it in `models/__init__.py`,
- update `utils/train_utils.py:build_model` to support the new model,
- keep the interface compatible: input `(B, 2, H, W)` audio, output `(B, 1, H, W)` depth,
- clearly document expected input/output format,
- verify compatibility with existing losses and evaluation metrics,
- and record the parameter count.

Minimum required checks:
- input shape is correct,
- output depth shape is correct,
- one training step works,
- one validation step works,
- checkpoint saving works,
- and result directories are created correctly.

---

## 8. Rules for Adding a New Dataset

When adding a new dataset:

- add a new class in `data/dataset.py` or create `data/new_dataset.py`,
- update `data/dataloader.py:make_dataloader` to support it,
- keep the returned sample format consistent: `(audio_tensor, depth_tensor)`,
- define train, validation, and test split logic clearly,
- save the split as an explicit JSON dictionary,
- document normalization and preprocessing,
- and verify audio/depth alignment.

Minimum required checks:
- dataset loading works,
- batch collation works,
- audio representation is correct,
- depth target scale is correct,
- sample visualization looks reasonable,
- and file paths are resolved correctly.

---

## 9. Rules for Visualization Features

Visualization code must not clutter the training loop.

Rules:
- keep visualization logic in `utils/visualization.py`,
- call visualization only at controlled intervals,
- save outputs locally under `results/{experiment_name}/`,
- avoid generating visualization for every batch,
- and use a small fixed subset of samples for consistency.

---

## 10. Rules for Parameter Analysis

Parameter analysis is optional and should only be used for selected important runs.

Possible analyses include:
- total parameter count
- parameter count by module
- weight norm by layer
- gradient norm by layer
- checkpoint-to-checkpoint parameter drift

Store these locally under:
- `results/{experiment_name}/analysis/`

Do not upload these plots to W&B unless explicitly requested.

---

## 11. Naming Convention

Every experiment must use a clear, structured, and unique name.

Recommended format:

`{exp_id}_{model}_{input}_{dataset}_{setting}`

Examples:
- `exp001_unet_foa_soundspaces_baseline`
- `exp002_resnet_foa_soundspaces_backbone_ablation`
- `exp003_unet_binaural_soundspaces_no_ambisonic`

This same name should be reused consistently for:
- output directory
- checkpoint directory
- W&B run name
- log file name
- shell script name

---

## 12. Required Output Structure

Each experiment should produce a predictable output structure.

```text
results/
  {experiment_name}/
    visualizations/
    analysis/

checkpoints/
  {experiment_name}/
    best_model.pth
    checkpoint_{epoch}.pth

eval/
  {dataset_name}/{split}/
    stats_{experiment_name}.pt
```

---

## 13. What Not to Do

Do not:
- create many nearly identical training scripts,
- upload heavy visualizations to W&B by default,
- change multiple major variables in one ablation unless explicitly intended,
- hardcode dataset-specific logic deep inside generic training code,
- put long visualization code directly inside the training loop,
- introduce inconsistent metric names across experiments,
- choose the best model using abs_rel alone,
- leave a run undocumented,
- or save outputs into ad hoc directories with inconsistent names.

---

## 14. Recommended Implementation Style for Assistants

When an assistant such as Claude helps with this repository, it should follow these rules:

- inspect existing code before proposing new files,
- prefer extension over duplication,
- clearly state what files will be changed before making changes,
- keep changes minimal and modular,
- preserve reproducibility,
- keep W&B logging lightweight,
- store visual outputs locally,
- and update this document whenever a new experiment is added.

For every new experiment, the assistant should clearly specify:
- goal,
- what changed,
- what stayed fixed,
- command,
- expected outputs,
- comparison baseline,
- best-model selection rule,
- and risks or failure points.

---

## 15. Pre-Run Checklist

Before starting a full training run, verify all of the following:

- [ ] experiment name is unique
- [ ] config is saved correctly
- [ ] one-batch forward pass works
- [ ] loss computation works
- [ ] validation step works
- [ ] checkpoint saving works
- [ ] W&B scalar logging works (if enabled)
- [ ] best-checkpoint rule uses both abs_rel and RMSE
- [ ] visual outputs are saved locally if enabled
- [ ] output directory is correct
- [ ] experiment entry is added to this file

---

## 16. Experiment Registry

Every experiment must be appended below using the same format.

### Template

#### EXP-XXX
**Name**: `expXXX_model_input_dataset_setting`
**Goal**: Short description of the purpose of the experiment.

**Changed**:
- item 1

**Fixed**:
- item 1

**Model**: model name
**Dataset**: dataset name
**Input**: input type

**Command**:
```bash
python train.py --experiment-name expXXX_model_input_dataset_setting
```

**Comparison Baseline**: baseline experiment name
**Notes**: important notes

---

## 17. Registered Experiments

_(Add experiments below as they are run.)_

---

## 18. Short Assistant Prompt

```text
Use this repository as a structured experiment framework for audio-to-depth estimation.

When adding a new experiment:
- reuse existing code first,
- avoid creating a new training script unless necessary,
- keep W&B logging lightweight,
- do not upload heavy visualizations to W&B,
- save all visual outputs locally,
- document every experiment in program.md,
- make experiments reproducible and comparable,
- and change one major factor at a time unless otherwise specified.

For every new experiment, clearly state:
- goal, what changed, what stayed fixed,
- command, expected outputs,
- comparison baseline,
- best-model selection rule,
- and risks or failure points.
```

---

## 19. Maintenance Rule

This file must be updated whenever any of the following changes:
- a new experiment is added,
- a new dataset is introduced,
- a new model family is introduced,
- the logging policy changes,
- the output structure changes,
- or the experiment naming convention changes.

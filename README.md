# Audio Depth Estimation

Depth estimation from binaural / ambisonics echoes on the SoundSpaces dataset.

Active line: **EchoRange** (bin-based radial depth, `models/bin_based/`).
Comparison baselines: **BatVision**, **EchoNet**, **EchoDiffusion**, **EchoDiffusion-Ambi (+SH)**, **EchoDiffusion-SH-Side+** — all under `comparison_methods/`.

## Structure

```
├── train.py, test.py, config.yaml
├── data/                  # dataset.py, dataloader.py, sh_basis.py
├── models/                # ours / shared code only
│   ├── unet_foa.py        # Ours v0 — simplest first attempt
│   ├── unet.py, vit.py    # Backbones for baseline.yaml / vit.yaml
│   ├── losses.py
│   ├── registry.py        # @register_builder model registry
│   ├── bin_based/         # Ours main: EchoRangeDepth, RangeDepthHead, spherical_loss
│   └── pretrain/          # Pretrained ResNet / ViT / ViT-FOA backbones
├── comparison_methods/    # External comparison baselines (kept separate from ours)
│   ├── batvision/
│   ├── echonet/
│   └── echodiffusion/     # EchoDiffusion + Ambi + Ambi-SH + SH-Side+
├── utils/                 # config, metrics, visualization, train_utils, test_utils
├── scripts/               # train.sh, test.sh, summary_*, paired_bootstrap.py
├── config/                # 9 active model configs
├── runs/                  # gitignored — per-run packages (train.py auto-writes)
└── logs/                  # training / test logs
```

Set aside in the sibling directory `../baseline_deprecated/` (not git-tracked):
- `models/` — 10 deprecated trial subpackages + one-off variants (n1/n2/n3/n4/n9/renew/pretrain v2–v6)
- `data/dataset_n2.py` — N2-feature dataset for the deprecated n2_0417 line
- `archive/` — historical experiment index (run/comparison/delete CSVs) + 41 packaged runs
- `docs/` — EXPERIMENT_NARRATIVE.md, REPRODUCIBILITY.md, ledger CSVs

For previous in-tree layouts, refer to git history before the relevant move commit.

## Quick Start

```bash
mkdir -p runs                       # first run only

# Train — writes runs/<experiment-name>/{checkpoints,results,config_resolved.yaml,...}
python train.py --config echorange --experiment-name 20260514_001_echorange_seed0

# Test — reads runs/<experiment-name>/checkpoints/ by default
python test.py --config echorange --experiment-name 20260514_001_echorange_seed0 \
    --eval-on test --checkpoints best
```

CLI arguments override `config/<name>.yaml`.

## Experiment workflow

The repo is config-driven, not copy-driven — keep code small, grow experiments via config + run dirs.

- **New model** → add a class + `@register_builder("name")` in its module; `models/registry.py` dispatches on `cfg.model.name`. No if/elif edits, no `unet_v1/v2/final` files.
- **Model variant** → pass CLI overrides (`--lambda-sh`, `--range-num-bins`, …) instead of forking a new YAML. `config/` holds one YAML per model family (9 total).
- **Every run** → `train.py` auto-writes `runs/<experiment-name>/` with `config_resolved.yaml`, `command.sh`, `git_commit.txt`, `git_diff.patch`, `meta.json`, plus `checkpoints/` and `results/`. That package is the provenance record.
- **Run naming** → recommended `YYYYMMDD_RUNID_METHOD_KEYCHANGE_SEED`, e.g. `20260514_004_echorange_radial_depth_seed0`.

## Active configs (9)

| config | model | role |
|---|---|---|
| `echorange` | EchoRangeDepth | **ours main** — radial bin-based |
| `foa` | AudioDepthFOAGenerator | ours v0 — simplest first attempt |
| `baseline` | UnetGenerator | plain UNet baseline |
| `vit` | AudioDepthViT | ViT baseline |
| `batvision` | BatVisionUNet | comparison baseline |
| `echonet` | EchoNet | comparison baseline |
| `pretrain_resnet` / `pretrain_vit` / `pretrain_vit_foa` | Pretrained* | pretrained backbones |

EchoDiffusion-family configs live only in `../baseline_deprecated/archive/runs/<id>/config.yaml` — copy one back into `config/` to re-run those baselines.

## Cleanup history (2026-05-13 → 2026-05-14)

The repo was reorganized from a flat trial-dump into a config-driven, run-packaged layout:

- **Archive pass** — indexed 433 runs, packaged 41 meaningful ones, removed dead scripts/configs/logs/qual, moved 10 trial model subpackages + the experiment ledger/KB out of the tree. Freed ~80 GB.
- **data/ consolidation** — merged 4 dataset files into `data/dataset.py`; moved `dataset_n2.py` out.
- **Stage 1 — model registry** — `utils/train_utils.py` if/elif → `@register_builder` functions.
- **Stage 2 — run packaging** — `train.py`/`test.py` now read/write `runs/<experiment-name>/` instead of flat `checkpoints/<exp_name>/`.
- **Stage 3 — comparison split** — comparison baselines moved from `models/` to `comparison_methods/`.
- **archive/ + docs/** — moved to `../baseline_deprecated/` (kept as reference, separate from active experiment code).

Comparison baselines and the ours-v0 file (`unet_foa.py`) were preserved as standalone code throughout.

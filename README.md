# Audio Depth Estimation

Depth estimation from binaural / ambisonics echoes on the SoundSpaces dataset.

Active line: **EchoRange** (bin-based radial depth, `models/bin_based/`).
Comparison baselines: **BatVision**, **EchoNet**, **EchoDiffusion**, **EchoDiffusion-Ambi (+SH)**, **EchoDiffusion-SH-Side(+)**.

## Structure

```
├── train.py, test.py, config.yaml
├── data/                      # SoundSpacesDataset, dataloader, SH basis (out of cleanup scope)
├── models/
│   ├── unet_foa.py            # Ours v0 — simplest first attempt (preserved separately)
│   ├── unet.py, vit.py        # Backbones for baseline.yaml / vit.yaml
│   ├── losses.py              # SILog, BerHu, depth, FOA-guided, SH-histogram, KL-reg
│   ├── bin_based/             # Ours main: EchoRangeDepth, RangeDepthHead, spherical_loss
│   ├── batvision/             # Comparison baseline
│   ├── echonet/               # Comparison baseline
│   ├── echodiffusion/         # Comparison baseline (+ Ambi, Ambi-SH)
│   ├── n2_0427/               # EchoDiffusion-SH-Side+
│   ├── pretrain/              # Pretrained ResNet/ViT/ViT-FOA v1–v6 backbones
│   └── deprecated/            # Older trial subpackages & one-off variants (kept for reference)
├── utils/                     # config, metrics, visualization, train_utils, test_utils
├── scripts/                   # Training/test/sweep shell scripts + paired_bootstrap.py
├── docs/                      # EXPERIMENT_NARRATIVE.md, REPRODUCIBILITY.md, results/*.csv
├── archive/                   # Cleanup artifacts — see below
├── checkpoints/, results/     # gitignored runtime outputs
└── logs/                      # Training/test logs (residuals after archive moves)
```

## Quick Start

```bash
# Train
python train.py --config config/echorange.yaml

# Test
python test.py --config config/echorange.yaml --eval-on test --checkpoints best

# Or shell scripts
bash scripts/train.sh
bash scripts/test.sh
```

CLI arguments override values in `config/<name>.yaml`.

## Archive (`archive/`)

Generated during the 2026-05-13 cleanup. Indexes meaningful historical runs without bloating active source.

```
archive/
├── README.md                      # cleanup methodology + column dictionary
├── runs.csv                       # 433 unique runs: role / era / status / paths / git_commit
├── comparison_baselines.csv       # 5 comparison methods → code/config/logs/ckpts
├── delete_candidates.csv          # paths flagged for review (safe_to_delete default = no)
├── artifacts.csv                  # orphan artifacts not matched to any ledger run
└── runs/<exp_id>/                 # 41 packaged runs
    ├── manifest.yaml              # run metadata + raw ledger metrics
    ├── config.yaml                # the config used at run time
    ├── logs/                      # train + test logs
    ├── checkpoints/, results/     # gitignored heavy artifacts (moved from top-level)
    └── code_ref/{git_commit.txt, source_files.txt}
                                   # the canonical way to reproduce: check out git_commit
```

To reproduce an archived run, check out the commit in `code_ref/git_commit.txt` and re-run with `config.yaml`. The active codebase intentionally does **not** carry per-run code snapshots.

## Cleanup History (2026-05-13)

Master was tidied across phases A–F:

| Phase | What changed | Result |
|---|---|---|
| A | Generated archive indexes (4 CSVs) | 0 mutation |
| B | Moved 54 meaningful run dirs into `archive/runs/<id>/` | logs/ckpts/results consolidated |
| C | Removed `__pycache__/` + empty checkpoint dirs | 73 paths |
| D | Moved one stray `foa_v2_js_0415.py` to `models/deprecated/` | 1 file |
| E1 | Slimmed `models/__init__.py` (57→32 lines), moved 10 trial subpackages + 12 top-level foa variants to `models/deprecated/` | active surface ~25 files |
| E2 | De-duplicated `runs.csv` (484 → 433 rows), removed 13 manifest-only orphan dirs | clean index |
| F1 | `n2_0427/echodiff_sh_side.py` → `deprecated/` (no archived runs) | 1 file |
| F2 | Removed 28 dead bulk/legacy scripts | scripts/ 40 → 11 |
| F3 | Deleted `qual/` entirely | ~788 MB freed |
| F5 | Removed top-level `audio_depth_experiment_ledger.csv`, `audio_depth_experiment_kb.md`, `make_qual.py` (digested into `archive/`) | 3 files |
| F6 | Cleared residual `results/`, `eval/` filesystem content | ~78 GB freed (.gitignored) |

Comparison baselines and the ours-v0 file (`unet_foa.py`) were preserved as standalone code throughout.

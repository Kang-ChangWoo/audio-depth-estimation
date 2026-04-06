# UNetSoundOnly — Depth Estimation from Binaural Echoes

Audio-based depth estimation using UNet architecture on the SoundSpaces dataset.

## Structure

```
├── config.yaml          # Dataset, model, train/test settings
├── train.py             # Training entry point
├── test.py              # Testing entry point
├── data/                # Dataset & dataloader
│   ├── dataset.py       # SoundSpacesDataset
│   ├── dataloader.py    # make_dataloader
│   └── sh_basis.py      # Spherical Harmonics (ACN/SN3D)
├── models/              # Network architecture & losses
│   ├── unet.py          # UNet generator (pix2pix-based)
│   └── losses.py        # SIlog loss
├── utils/               # Helpers
│   ├── config.py        # YAML config loader
│   ├── metrics.py       # Depth error metrics
│   ├── visualization.py # Prediction visualizations
│   ├── train_utils.py   # Model/criterion builders
│   └── test_utils.py    # Evaluation loop
└── scripts/             # Shell scripts
    ├── train.sh
    └── test.sh
```

## Quick Start

```bash
# Train
python train.py --lr 0.001 --batch-size 32 --epochs 40

# Test
python test.py --eval-on test --checkpoints best

# Or use shell scripts
bash scripts/train.sh
bash scripts/test.sh
```

## Config

All settings are in `config.yaml`. CLI arguments override config values.

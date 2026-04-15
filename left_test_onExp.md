# Left Experiments: Testing Guide for Claude Code

## Context

As of 2026-04-10, **78 out of 130 experiments** have been trained and have `best_model.pth` checkpoints. The remaining **52 experiments** (exp70-120 + exp125) from `bulk0408_65exps.sh` **never ran** — no logs, no checkpoints. They were defined in the script but the training was killed before reaching them (only exp56-69 completed from that batch).

Once these experiments finish training, use this guide to test them.

---

## What Never Ran (52 experiments)

### Group A — BatVision (exp71-75, config: `batvision`)
| Exp | Name | LR | BS |
|-----|------|----|----|
| 71 | exp71_batvision_lr1e3_bs32 | 0.001 | 32 |
| 72 | exp72_batvision_lr5e4_bs32 | 0.0005 | 32 |
| 73 | exp73_batvision_lr1e4_bs32 | 0.0001 | 32 |
| 74 | exp74_batvision_lr1e3_bs16 | 0.001 | 16 |
| 75 | exp75_batvision_lr2e3_bs32 | 0.002 | 32 |

### Group A — EchoNet (exp70, config: `echonet`)
| Exp | Name | LR | BS |
|-----|------|----|----|
| 70 | exp70_echonet_lr2e3_bs16 | 0.002 | 16 |

### Group B — FOA CrossAttn + KL (exp76-80, config: `foa_crossattn`)
| Exp | Name | LR | fw | kl | hw |
|-----|------|----|----|----|----|
| 76 | exp76_crossattn_lr1e3_fw0.1_kl0.02 | 0.001 | 0.1 | 0.02 | - |
| 77 | exp77_crossattn_lr5e4_fw0.2_kl0.005 | 0.0005 | 0.2 | 0.005 | - |
| 78 | exp78_crossattn_lr1e3_fw0.05_kl0.01 | 0.001 | 0.05 | 0.01 | - |
| 79 | exp79_crossattn_lr5e4_fw0.1_hw0.2_kl0.01 | 0.0005 | 0.1 | 0.01 | 0.2 |
| 80 | exp80_crossattn_lr1e3_fw0.3_kl0.01 | 0.001 | 0.3 | 0.01 | - |

### Group B — FOA FeatBank + KL (exp81-85, config: `foa_featbank`)
| Exp | Name | LR | fw | kl | hw |
|-----|------|----|----|----|----|
| 81 | exp81_featbank_lr1e3_fw0.1_kl0.02 | 0.001 | 0.1 | 0.02 | - |
| 82 | exp82_featbank_lr5e4_fw0.2_kl0.005 | 0.0005 | 0.2 | 0.005 | - |
| 83 | exp83_featbank_lr1e3_fw0.05_kl0.01 | 0.001 | 0.05 | 0.01 | - |
| 84 | exp84_featbank_lr5e4_fw0.1_hw0.2_kl0.01 | 0.0005 | 0.1 | 0.01 | 0.2 |
| 85 | exp85_featbank_lr1e3_fw0.3_kl0.01 | 0.001 | 0.3 | 0.01 | - |

### Group B — FOA MSAttn + KL (exp86-90, config: `foa_msattn`)
| Exp | Name | LR | fw | kl | hw |
|-----|------|----|----|----|----|
| 86 | exp86_msattn_lr1e3_fw0.1_kl0.02 | 0.001 | 0.1 | 0.02 | - |
| 87 | exp87_msattn_lr5e4_fw0.2_kl0.005 | 0.0005 | 0.2 | 0.005 | - |
| 88 | exp88_msattn_lr1e3_fw0.05_kl0.01 | 0.001 | 0.05 | 0.01 | - |
| 89 | exp89_msattn_lr5e4_fw0.1_hw0.2_kl0.01 | 0.0005 | 0.1 | 0.01 | 0.2 |
| 90 | exp90_msattn_lr1e3_fw0.3_kl0.01 | 0.001 | 0.3 | 0.01 | - |

### Group B — FOA ChannelAttn + KL (exp91-95, config: `foa_channelattn`)
| Exp | Name | LR | fw | kl | hw |
|-----|------|----|----|----|----|
| 91 | exp91_channelattn_lr1e3_fw0.1_kl0.02 | 0.001 | 0.1 | 0.02 | - |
| 92 | exp92_channelattn_lr5e4_fw0.2_kl0.005 | 0.0005 | 0.2 | 0.005 | - |
| 93 | exp93_channelattn_lr1e3_fw0.05_kl0.01 | 0.001 | 0.05 | 0.01 | - |
| 94 | exp94_channelattn_lr5e4_fw0.1_hw0.2_kl0.01 | 0.0005 | 0.1 | 0.01 | 0.2 |
| 95 | exp95_channelattn_lr1e3_fw0.3_kl0.01 | 0.001 | 0.3 | 0.01 | - |

### Group C — FOA Main wider search (exp96-120, config: `foa`)
| Exp | Name | LR | dw | fw | hw | freeze |
|-----|------|----|----|----|----|--------|
| 96 | exp96_foa_lr2e4_dw1.0_fw0.1_hw0.1 | 0.0002 | 1.0 | 0.1 | 0.1 | - |
| 97 | exp97_foa_lr3e4_dw1.0_fw0.1_hw0.1 | 0.0003 | 1.0 | 0.1 | 0.1 | - |
| 98 | exp98_foa_lr7e4_dw1.0_fw0.1_hw0.1 | 0.0007 | 1.0 | 0.1 | 0.1 | - |
| 99 | exp99_foa_lr1e3_dw1.5_fw0.1_hw0.1 | 0.001 | 1.5 | 0.1 | 0.1 | - |
| 100 | exp100_foa_lr1e3_fw0.15_hw0.1 | 0.001 | 1.0 | 0.15 | 0.1 | - |
| 101 | exp101_foa_lr1e3_fw0.1_hw0.15 | 0.001 | 1.0 | 0.1 | 0.15 | - |
| 102 | exp102_foa_lr5e4_dw1.5_fw0.1_hw0.1 | 0.0005 | 1.5 | 0.1 | 0.1 | - |
| 103 | exp103_foa_lr5e4_fw0.15_hw0.1 | 0.0005 | 1.0 | 0.15 | 0.1 | - |
| 104 | exp104_foa_lr5e4_fw0.1_hw0.15 | 0.0005 | 1.0 | 0.1 | 0.15 | - |
| 105 | exp105_foa_lr1e3_fw0.3_hw0.1 | 0.001 | 1.0 | 0.3 | 0.1 | - |
| 106 | exp106_foa_lr1e3_fw0.1_hw0.3 | 0.001 | 1.0 | 0.1 | 0.3 | - |
| 107 | exp107_foa_lr5e4_fw0.3_hw0.1 | 0.0005 | 1.0 | 0.3 | 0.1 | - |
| 108 | exp108_foa_lr5e4_dw2.0_fw0.1_hw0.1 | 0.0005 | 2.0 | 0.1 | 0.1 | - |
| 109 | exp109_foa_lr1e3_dw1.0_fw0.2_hw0.2_freeze5 | 0.001 | 1.0 | 0.2 | 0.2 | 5 |
| 110 | exp110_foa_lr5e4_fw0.2_hw0.1_freeze5 | 0.0005 | 1.0 | 0.2 | 0.1 | 5 |
| 111 | exp111_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze15 | 0.001 | 1.0 | 0.1 | 0.1 | 15 |
| 112 | exp112_foa_lr5e4_dw1.0_fw0.1_hw0.1_freeze10 | 0.0005 | 1.0 | 0.1 | 0.1 | 10 |
| 113 | exp113_foa_lr1e3_dw1.0_fw0.05_hw0.05 | 0.001 | 1.0 | 0.05 | 0.05 | - |
| 114 | exp114_foa_lr5e4_dw0.5_fw0.1_hw0.2 | 0.0005 | 0.5 | 0.1 | 0.2 | - |
| 115 | exp115_foa_lr1e3_dw2.0_fw0.2_hw0.1 | 0.001 | 2.0 | 0.2 | 0.1 | - |
| 116 | exp116_foa_lr3e4_fw0.2_hw0.1 | 0.0003 | 1.0 | 0.2 | 0.1 | - |
| 117 | exp117_foa_lr2e4_dw1.0_fw0.2_hw0.2 | 0.0002 | 1.0 | 0.2 | 0.2 | - |
| 118 | exp118_foa_lr7e4_fw0.15_hw0.15 | 0.0007 | 1.0 | 0.15 | 0.15 | - |
| 119 | exp119_foa_lr1e3_fw0.1_hw0.2_freeze3 | 0.001 | 1.0 | 0.1 | 0.2 | 3 |
| 120 | exp120_foa_lr5e4_fw0.05_hw0.1_freeze5 | 0.0005 | 1.0 | 0.05 | 0.1 | 5 |

### EchoDiff + Wav2Vec (exp125, config: `echodiffusion`)
| Exp | Name | LR | BS |
|-----|------|----|----|
| 125 | exp125_echodiff_wav2vec_lr1e4_bs8 | 0.0001 | 8 |

---

## How to Test After Training Completes

### Step 1: Verify training is done
```bash
# Check which experiments now have best_model.pth
for i in $(seq 70 120) 125; do
    found=$(ls checkpoints/*exp${i}_*/best_model.pth 2>/dev/null)
    if [ -n "$found" ]; then
        echo "READY: exp$i -> $found"
    else
        echo "MISSING: exp$i"
    fi
done
```

### Step 2: Run test.py with --checkpoint-path
The modified `test.py` now supports:
- `--checkpoint-path`: direct path to best_model.pth (bypasses path construction)
- `--batch-size`: override test batch size (default: 1)
- Automatic visualization saving (3 batches) to `results/{checkpoint_dir}/`
- Stats saved to `eval/soundspaces/test/stats_{exp_name}.pt`

Example single test:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --config foa \
    --experiment-name exp96_foa_lr2e4_dw1.0_fw0.1_hw0.1 \
    --checkpoint-path checkpoints/<full_dir_name>/best_model.pth \
    --eval-on test
```

### Step 3: Use bulk0410_120exps.sh as template
The script `scripts/bulk0410_120exps.sh` auto-discovers ALL checkpoints with `best_model.pth` and tests them across 4 GPUs. Simply re-running it after training will pick up the new experiments:

```bash
bash scripts/bulk0410_120exps.sh
```

It determines the config automatically from the checkpoint directory name:
- `echodiffusion_*` -> echodiffusion
- `echonet_*` -> echonet
- `batvision_*` -> batvision
- `unet_256_*_crossattn_*` -> foa_crossattn
- `unet_256_*_featbank_*` -> foa_featbank
- `unet_256_*_msattn_*` -> foa_msattn
- `unet_256_*_channelattn_*` -> foa_channelattn
- `unet_256_*_foa_*` -> foa

### Step 4: Update results.md and table.md
After testing, update:
1. `results.md` — add new experiment results to appropriate tables, update status summary
2. `table.md` — update LaTeX table if any new method beats current best

---

## Config-to-Experiment Mapping (for --kl-weight flag)

The Group B experiments (exp76-95) use `--kl-weight` which is specific to FOA variant configs. These configs already support kl_weight in their YAML definitions and the `--kl-weight` CLI flag is handled by train.py.

The original training command format from `bulk0408_65exps.sh`:
```bash
python train.py --config foa_crossattn --experiment-name exp76_crossattn_lr1e3_fw0.1_kl0.02 \
    --lr 0.001 --foa-weight 0.1 --kl-weight 0.02 --epochs 40 --num-workers 4
```

---

## Server Info

- 4x NVIDIA RTX 4090 (24GB each), GPUs 0-3
- Training uses 2 GPUs per experiment (DataParallel)
- Testing uses 1 GPU per experiment
- Batch size during test: 1 (default from config)
- Test output: 7 depth metrics (ABS_REL, RMSE, Delta1, Delta2, Delta3, Log10, MAE) + FOA metrics for FOA models
- Visualizations: 3 test batches saved as PNG to results/{exp_dir}/

---

## Current Best Results (for reference when comparing new experiments)

| Rank | Model | Score | RMSE | ABS_REL |
|------|-------|-------|------|---------|
| 1 | FOA (exp40) | 0.9802 | 1.2223 | 0.4153 |
| 2 | FOA (exp53) | 0.9818 | 1.2248 | 0.4147 |
| 3 | Pretrained ViT (exp62) | 0.9818 | 1.2350 | 0.3909 |
| 4 | FOA CrossAttn (exp18) | 0.9822 | 1.2198 | 0.4280 |
| 5 | FOA MSAttn (exp27) | 0.9830 | 1.2317 | 0.4028 |

# Sequence Holdout & Best Model Tracking Guide

This guide explains the new features added for overfitting detection and model tracking.

## 🎯 Features

### 1. W&B Best Model Tracking

Automatically saves the best model based on validation metrics and logs to Weights & Biases.

**Benefits:**
- Tracks best performing model during training
- Prevents need to manually find best checkpoint
- Logs best model metrics to W&B for easy comparison
- Supports multiple metrics (RMSE, ABS_REL, Delta1, MAE, Loss)

**Usage:**
```bash
# Save best model based on validation RMSE (lower is better)
python train.py --use_wandb --save_best_model --best_metric rmse

# Use ABS_REL as metric
python train.py --use_wandb --save_best_model --best_metric abs_rel

# Use Delta1 (higher is better)
python train.py --use_wandb --save_best_model --best_metric delta1
```

**Output:**
- Best model saved to: `./checkpoints/{experiment_name}/best_model.pth`
- Logs `best_model_epoch` and `best_{metric}` to W&B
- Console shows: `🎯 New best model! rmse=X.XXXX at epoch Y`

---

### 2. Sequence-Level Holdout

Exclude entire sequences from training to detect overfitting on specific environments.

**Benefits:**
- Test generalization to unseen sequences/locations
- Detect overfitting to training environments
- Separate test and eval holdout sets
- Track holdout performance in W&B alongside validation

**Why Sequence-Level?**
- Audio-based depth estimation may overfit to specific room acoustics
- Standard train/val split shuffles frames from same sequence
- Sequence holdout tests true generalization to new environments

---

## 📋 Available Sequences (BatvisionV2)

From the dataset directory:
- `2ndFloorLuxembourg`
- `3rd_Floor_Luxembourg` ✓ (Good for holdout)
- `Attic`
- `Outdoor_Cobblestone_Path`
- `Salle_Chevalier` ✓ (Good for holdout)
- `Salle_des_Colonnes`
- `V119_Cake_Corridors`
- And more...

**Recommended holdout sequences:**
- `Salle_Chevalier` - Indoor room with specific acoustics
- `3rd_Floor_Luxembourg` - Different building, different characteristics

---

## 🚀 Usage Examples

### Basic Best Model Tracking
```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --save_best_model \
  --best_metric rmse \
  --experiment_name my_experiment
```

### Sequence Holdout (Single Sequence)
```bash
python train.py \
  --sequence_holdout \
  --holdout_test_seq Salle_Chevalier \
  --experiment_name holdout_test
```

### Full Overfitting Check (Recommended)
```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --save_best_model \
  --best_metric rmse \
  --sequence_holdout \
  --holdout_test_seq Salle_Chevalier \
  --holdout_eval_seq 3rd_Floor_Luxembourg \
  --batch_size 256 \
  --learning_rate 0.002 \
  --experiment_name overfitting_check
```

### Combined with Other Features
```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --save_best_model \
  --sequence_holdout \
  --holdout_test_seq Salle_Chevalier \
  --l1_weight 0.8 \
  --silog_weight 0.2 \
  --audio_format mel_spectrogram \
  --experiment_name mel_spec_holdout
```

---

## 📊 W&B Metrics Logged

### Standard Metrics (existing)
- `train/loss` - Training loss
- `train/epoch_time` - Time per epoch
- `val/loss`, `val/rmse`, `val/abs_rel`, `val/delta1`, etc. - Validation metrics
- `val/visualization` - Depth map visualizations

### New Best Model Metrics
- `best_model_epoch` - Epoch when best model was found
- `best_{metric}` - Best metric value (e.g., `best_rmse`)

### New Holdout Metrics
- `holdout_test/rmse`, `holdout_test/abs_rel`, `holdout_test/delta1` - Test sequence performance
- `holdout_eval/rmse`, `holdout_eval/abs_rel`, `holdout_eval/delta1` - Eval sequence performance

---

## 🔍 Interpreting Results

### Checking for Overfitting

Compare these metrics in W&B:

1. **Validation vs Holdout Gap**
   - Small gap: Good generalization ✅
   - Large gap: Overfitting to training environments ⚠️

2. **Metrics to Watch**
   ```
   val/rmse        vs  holdout_test/rmse
   val/abs_rel     vs  holdout_test/abs_rel
   val/delta1      vs  holdout_test/delta1
   ```

3. **Healthy Pattern**
   ```
   val/rmse:           2.5
   holdout_test/rmse:  2.8  (slightly higher is okay)
   holdout_eval/rmse:  2.7
   ```

4. **Overfitting Pattern**
   ```
   val/rmse:           2.5
   holdout_test/rmse:  4.5  (much higher - problem!)
   holdout_eval/rmse:  4.3
   ```

### Best Model Selection

- Best model is automatically saved based on validation performance
- Check W&B for `best_model_epoch` to see when it was found
- Load best model for final evaluation:
  ```python
  checkpoint = torch.load('./checkpoints/{experiment_name}/best_model.pth')
  model.load_state_dict(checkpoint['state_dict'])
  ```

---

## 🔧 Command Line Arguments

### Best Model Tracking
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--save_best_model` | flag | False | Enable best model tracking (requires `--use_wandb`) |
| `--best_metric` | str | 'rmse' | Metric for best model: `rmse`, `abs_rel`, `delta1`, `mae`, `loss` |

### Sequence Holdout
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sequence_holdout` | flag | False | Enable sequence-level holdout |
| `--holdout_test_seq` | str | None | Sequence name to hold out for testing |
| `--holdout_eval_seq` | str | None | Sequence name to hold out for evaluation |

---

## 📝 Implementation Details

### How Sequence Holdout Works

1. **Dataset Filtering**
   - Training/validation datasets exclude blacklisted sequences via `location_blacklist`
   - Separate loaders created for holdout sequences
   - Uses `pandas.DataFrame.str.contains()` to filter by sequence name

2. **Evaluation**
   - Holdout sequences evaluated every validation epoch
   - Same metrics as validation (RMSE, ABS_REL, Delta1, etc.)
   - Results logged to W&B with prefix `holdout_test/` and `holdout_eval/`

3. **Experiment Naming**
   - Holdout sequences automatically added to experiment name
   - Example: `unet_256_batvisionv2_BS256_Lr0.002_AdamW_holdout_Salle_Chevalier_3rd_Floor_Luxembourg_exp1`

### Best Model Logic

```python
# For metrics where lower is better (rmse, abs_rel, mae, loss)
if current_metric < best_metric_value:
    save_best_model()

# For metrics where higher is better (delta1)
if current_metric > best_metric_value:
    save_best_model()
```

---

## 🐛 Troubleshooting

### "No valid locations found"
- Check that sequences exist in dataset directory
- Verify sequence names match exactly (case-sensitive)
- Use `ls /path/to/BatvisionV2/` to see available sequences

### "Warning: --save_best_model requires --use_wandb"
- Add `--use_wandb` flag to enable W&B logging
- Set up W&B: `wandb login`

### Holdout metrics not appearing in W&B
- Check that `--sequence_holdout` is enabled
- Verify holdout sequences have data (check console output)
- Ensure validation is enabled (`--validation true`)

### Best model not being saved
- Check console for "🎯 New best model!" messages
- Verify model is improving on chosen metric
- Check `./checkpoints/{experiment_name}/` directory

---

## 🎓 Advanced Usage

### Custom Metrics for Best Model

Currently supported metrics:
- `rmse`: Root Mean Squared Error (most common)
- `abs_rel`: Absolute Relative Error
- `delta1`: Threshold accuracy δ < 1.25 (higher is better)
- `mae`: Mean Absolute Error
- `loss`: Validation loss

### Multiple Holdout Sequences

You can specify both test and eval sequences:
```bash
--sequence_holdout \
--holdout_test_seq Salle_Chevalier \
--holdout_eval_seq 3rd_Floor_Luxembourg
```

Or just one:
```bash
--sequence_holdout --holdout_test_seq Salle_Chevalier
```

### Resume Training with Best Model

```bash
# First run with best model tracking
python train.py --use_wandb --save_best_model --experiment_name exp1

# Load best model for continued training or evaluation
python train.py --checkpoints best --experiment_name exp1  # (future feature)
```

---

## 📚 References

- **W&B Documentation**: https://docs.wandb.ai/
- **Overfitting Detection**: Compare train/val/holdout metrics
- **Best Practices**: Use holdout sequences from different environments

---

## ✅ Checklist for Experiments

Before running long experiments:

- [ ] W&B is configured (`wandb login`)
- [ ] Best model tracking enabled (`--save_best_model`)
- [ ] Appropriate metric selected (`--best_metric`)
- [ ] Holdout sequences chosen (`--sequence_holdout`)
- [ ] Experiment name is descriptive (`--experiment_name`)
- [ ] Results directory has space
- [ ] GPU memory is sufficient for batch size

---

**Happy training! 🚀**

For questions or issues, check the console output and W&B dashboard.



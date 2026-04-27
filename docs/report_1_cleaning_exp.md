# Report 1: Experiment Cleaning & Organization
**Date:** 2026-04-16

---

## Objective

Clean and organize the experiment infrastructure by:
1. Auditing every experiment against three criteria: (1) training done, (2) checkpoint exists, (3) execution script exists
2. Consolidating completed experiments into summary scripts
3. Consolidating logs into unified directories
4. Removing redundant/stale artifacts
5. Gathering unfinished experiments into TBD scripts

---

## Before Cleaning

### Logs (scattered across 9+ directories)

| Directory | Files | Contents |
|-----------|-------|----------|
| `logs/bulk0407/` | 60 | Training logs exp01-60 |
| `logs/bulk0408/` | 60 | Training logs exp56r-125 |
| `logs/bulk0410/` | 80 | Test logs (78 exps + summary) |
| `logs/bulk0410_test_41/` | 41 | Test logs (extended batch) |
| `logs/bulk0410_sum_revised/` | 4 | EchoNet retrain attempt logs |
| `logs/bulk0415_train_foa25/` | 25 | FOA 0415 v1-v5 training logs |
| `logs/bulk0416_train_vit/` | 8 | ViT+FOA training logs |
| `logs/bulk0416_test_vit/` | 6 | ViT+FOA test logs |
| `logs/test_CW_fire/` | 9 | FOA 0415 spot-check test logs |
| Standalone `train_*.log` | 6 | CW rotation training |
| Standalone `test_*.log` | 6 | CW rotation testing |

### Scripts (11 bulk + standalone)

| Script | Purpose |
|--------|---------|
| `bulk0410_train_41.sh` | Train 41 experiments |
| `bulk0410_train_21_revised.sh` | Train 21 missing/failed experiments |
| `bulk0410_test_78exps.sh` | Test 78 experiments |
| `bulk0410_test_41exps_revised.sh` | Test 41 new experiments |
| `bulk0415_train_foa25.sh` | Train 25 FOA 0415 variants |
| `bulk0415_test_foa25.sh` | Test 25 FOA 0415 variants |
| `bulk0416_train_vit.sh` | Train 10 ViT+FOA variants |
| `bulk0416_test_vit.sh` | Test 10 ViT+FOA variants |
| `test_CW_fire.sh` | Test FOA 0415 spot checks |
| `train_CW.sh`, `test_CW.sh` | CW rotation standalone |
| `train_JS.sh`, `test_JS.sh` | JS experiments |
| `train.sh`, `test.sh` | Single-run helpers |

### Checkpoints (181 directories)

| Category | Count | best_model.pth |
|----------|-------|----------------|
| Completed experiments | 159 | Yes |
| Incomplete (stalled/never started) | 8 | No |
| JS-related | 14 | 5 yes, 9 no |
| **Total** | **181** | **164** |

---

## Cleaning Actions Performed

### 1. Checkpoint Cleanup

**Removed (no best_model.pth, non-functional):**

| Checkpoint | Reason |
|-----------|--------|
| `exp78_crossattn_lr1e3_fw0.05_kl0.01` | Stalled at 4/40 epochs, no best_model.pth |
| `exp01_foa0415v1_lr1e3_lsh0.1` | Old naming scheme, no best_model.pth (replaced by exp130) |
| `exp06_foa0415v2_lr1e3_lsh0.1` | Old naming scheme, no best_model.pth (replaced by exp135) |
| `exp11_foa0415v3_lr1e3_lsh0.1` | Old naming scheme, no best_model.pth (replaced by exp140) |
| `exp16_foa0415v4_lr1e3_lsh0.1` | Old naming scheme, no best_model.pth (replaced by exp145) |
| `exp21_foa0415v5_lr1e3_lsh0.1` | Old naming scheme, no best_model.pth (replaced by exp150) |
| `exp160_pvitfoa_lr1e4_lsh0.1` | Legacy ViT FOA naming, no best_model.pth (replaced by exp160_pvitfoav1) |
| `exp163_pvitfoa_lr5e5_lsh0.5` | Legacy ViT FOA naming, no best_model.pth (replaced by exp163_pvitfoav2) |
| `test_dryrun` | Test artifact, no best_model.pth |

**Accidentally removed (JS-related — should have been kept):**

| Checkpoint | Had best_model.pth | Status |
|-----------|-------------------|--------|
| `feat_attn_foa_v2_js` | Yes | Needs retraining |
| `foa_basic_js_foa_v2_js` | Yes | Needs retraining |
| `foa_feat_attn_v2_foa_v2_js` | Yes | Needs retraining |
| `full_run_foa_v2_js` | Yes | Needs retraining |
| `sh_coeff_hierarch_foa_v2_js` | Yes | Needs retraining |
| `full_run_foa` (non-JS) | Yes | Needs retraining |
| 8 JS dirs without best_model.pth | No | No action needed |

### 2. Log Consolidation

All completed training and test logs were copied into two unified directories, then originals were removed after verification (zero missing files).

| Source | Files | Destination |
|--------|-------|-------------|
| `logs/bulk0407/` (60) | 60 training | `logs/summary_train/` |
| `logs/bulk0408/` (60) | 55 training (5 incomplete skipped) | `logs/summary_train/` |
| `logs/bulk0415_train_foa25/` (25) | 25 training | `logs/summary_train/` |
| `logs/bulk0416_train_vit/` (8) | 6 training (2 incomplete skipped) | `logs/summary_train/` |
| Standalone `train_CW_rot_*.log` (6) | 6 training | `logs/summary_train/` |
| `logs/bulk0410/` (80) | 80 test | `logs/summary_test/` |
| `logs/bulk0410_test_41/` (41) | 41 test | `logs/summary_test/` |
| `logs/bulk0416_test_vit/` (6) | 6 test | `logs/summary_test/` |
| `logs/test_CW_fire/` (9) | 9 test | `logs/summary_test/` |
| Standalone `test_CW_rot_*.log` (6) | 6 test | `logs/summary_test/` |

**Skipped (incomplete, not moved):**
- `exp66-69_echonet` (training stalled at 17-20/40)
- `exp78_crossattn` (training stalled at 4/40)
- `exp166-167_pvitfoav4` (training stalled at 4-6/40)
- `bulk0410_sum_revised/` (EchoNet retrain attempt, 4 incomplete logs)

### 3. Script Consolidation

All experiment definitions from 9 old bulk scripts were verified present in the new summary scripts, then originals were removed.

**Removed scripts (9):**

| Script | Experiments | Covered by |
|--------|-------------|------------|
| `bulk0410_train_41.sh` | 41 train | `summary_train.sh` |
| `bulk0410_train_21_revised.sh` | 21 train | `summary_train.sh` + `summary_train_tbd.sh` |
| `bulk0410_test_78exps.sh` | 78 test | `summary_test.sh` (auto-discovery) |
| `bulk0410_test_41exps_revised.sh` | 41 test | `summary_test.sh` (auto-discovery) |
| `bulk0415_train_foa25.sh` | 25 train | `summary_train.sh` |
| `bulk0415_test_foa25.sh` | 25 test | `summary_test.sh` (auto-discovery) |
| `bulk0416_train_vit.sh` | 10 train | `summary_train.sh` + `summary_train_tbd.sh` |
| `bulk0416_test_vit.sh` | 10 test | `summary_test.sh` (auto-discovery) |
| `test_CW_fire.sh` | 9 test | `summary_test.sh` (auto-discovery) |

**Verification method:** For each old script, every experiment name (exp ID prefix) was confirmed present in at least one of: `summary_train.sh`, `summary_test.sh`, `summary_train_tbd.sh`, or `summary_test_tbd.sh`.

---

## After Cleaning

### Logs

```
logs/
├── summary_train/   (152 completed training logs)
└── summary_test/    (142 completed test logs)
```

### Scripts

```
scripts/
├── summary_train.sh       # All 152 completed training experiments (13 groups)
├── summary_test.sh        # Auto-discovers & tests all 158 checkpoints
├── summary_train_tbd.sh   # 19 unfinished training experiments (4 groups)
├── summary_test_tbd.sh    # ~45 untested/failed test experiments (2 groups)
├── train.sh               # Single-experiment runner
├── test.sh                # Single-experiment runner
├── train_CW.sh            # CW rotation standalone
├── test_CW.sh             # CW rotation standalone
├── train_JS.sh            # JS experiments (untouched)
└── test_JS.sh             # JS experiments (untouched)
```

### Checkpoints

```
checkpoints/   158 directories, ALL with best_model.pth (639 GB)
```

### Summary Script Design

**`summary_train.sh`** — 13 selectable groups:

| Group | Experiments | Epochs |
|-------|-------------|--------|
| `baseline` | exp01-05 | 40 |
| `vit` | exp06-10 | 40 |
| `echodiff` | exp11-15, exp121-125 | 40 |
| `foa_variant` | exp16-35, exp76-95 (crossattn/featbank/msattn/channelattn) | 40 |
| `foa` | exp36-55, exp96-120 | 40 |
| `foav2` | exp56-60 | 40 |
| `resnet` | exp56r-60r | 40 |
| `previt` | exp61-65 | 40 |
| `echonet` | exp70 | 40 |
| `batvision` | exp71-75 | 40 |
| `foa0415` | exp130-154 | 60 |
| `pvitvoa` | exp160-165 | 40 |
| `cw` | CW_rot_foa* (6 variants) | 40 |

Usage: `GPUS="0,1" bash summary_train.sh [GROUP]`

**`summary_test.sh`** — Auto-discovers all checkpoints, maps to config, runs test.py.

Usage: `GPUS="0,1" bash summary_test.sh`

**`summary_train_tbd.sh`** — 19 unfinished experiments in 4 groups:

| Group | Experiments | Issue |
|-------|-------------|-------|
| `echonet` | exp66-69 | Stalled at 17-20/40 epochs |
| `crossattn` | exp78 | Stalled at 4/40 epochs |
| `foa_missing` | exp82,86,90,94,98,102,106,110,114,118 | Never started (bulk0408 killed) |
| `pvitvoa` | exp166-169 | v4 stalled, v5 never started |

**`summary_test_tbd.sh`** — 2 groups:

| Group | Experiments | Issue |
|-------|-------------|-------|
| `foa0415` | 20 of exp130-154 | Trained but never tested |
| `failed_variants` | 25 (exp16-35, exp56-60 foav2) | Test failed in bulk0410 (ckpt loading) |

---

## Disk Space Impact

| Item | Before | After | Saved |
|------|--------|-------|-------|
| Log directories | 9 dirs + 12 standalone files | 2 dirs | ~3 MB (small) |
| Scripts | 15 .sh files | 10 .sh files | Negligible |
| Checkpoints | 181 dirs (est. ~660 GB) | 158 dirs (639 GB) | ~21 GB |

---

## Items Requiring Attention

### Must Retrain (accidentally deleted)

| Experiment | Config | Priority |
|-----------|--------|----------|
| `full_run_foa` | foa | Medium — regular FOA, can re-run via `summary_train.sh foa` |
| `feat_attn_foa_v2_js` | foa_v2_js | Per JS schedule |
| `foa_basic_js_foa_v2_js` | foa_v2_js | Per JS schedule |
| `foa_feat_attn_v2_foa_v2_js` | foa_v2_js | Per JS schedule |
| `full_run_foa_v2_js` | foa_v2_js | Per JS schedule |
| `sh_coeff_hierarch_foa_v2_js` | foa_v2_js | Per JS schedule |

### Must Complete Training (19 experiments)

Run: `GPUS="X,Y" bash scripts/summary_train_tbd.sh`

### Must Run Tests (~45 experiments)

Run: `GPUS="X,Y" bash scripts/summary_test_tbd.sh`

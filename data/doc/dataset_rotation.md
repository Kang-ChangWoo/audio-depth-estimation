# Canonical-frame FOA rotation — change log & rollback guide

This document records every file touched to add the canonical-frame FOA
rotation feature, so the change can be reverted cleanly later.

## Why this change exists

Habitat-Sim Ambisonics are effectively **world-frame** aligned. Each position
is captured with 4 yaw rotations in a fixed cyclic order:

```
raw_sample_index % 4
    0 -> front   (rotate by   0 deg)
    1 -> right   (rotate by -90 deg)
    2 -> back    (rotate by -180 deg)
    3 -> left    (rotate by -270 deg)
```

To make FOA inputs ego-consistent with the RGB / depth views, the dataset
now rotates FOA into a canonical agent-centered frame before any per-channel
statistic (RMS target, covariance, ERP energy map) is computed. `view_mod` is
derived from the **filename-parsed raw index**, not the filtered dataset
position, so it stays correct even when the depth filter drops samples.

The rotation is applied at the **raw IR** level. 90 deg yaw is an exact
sign-swap on the (Y, X) pair in ACN order `[W, Y, Z, X]`; W and Z are
invariant under yaw.

## Feature is opt-in

Nothing changes unless the caller passes `--rotate-canonical` (train.py /
test.py) or sets `cfg.dataset.rotate_canonical = True`. Default behavior is
identical to before the change.

## Files added

| Path                                                   | Purpose                                              |
|--------------------------------------------------------|------------------------------------------------------|
| `data/dataset_rotated.py`                              | `SoundSpacesDatasetRotated` subclass + helpers       |
| `scripts/train_CW.sh`                                  | Train 6 FOA models with rotation, 2x4 GPU parallel   |
| `scripts/test_CW.sh`                                   | Test the same 6 models, mirroring the layout        |
| `data/doc/dataset_rotation.md` (this file)             | Change log / rollback notes                          |

`data/dataset_rotated.py` defines:

- `get_view_mod_from_sample_idx(sample_idx) -> int`
- `rotate_foa_to_canonical(foa, view_mod) -> np.ndarray`
- `class SoundSpacesDatasetRotated(SoundSpacesDataset)` — overrides only
  `_load_foa_ir` to rotate the raw IR before it reaches the RMS / covariance
  / energy-map code path. Every other method (sample listing, depth filter,
  audio / depth / waveform loading) is inherited unchanged.

## Files modified

All edits are small and localized. Exact before -> after shown below.

### 1. `data/dataset.py`

Two edits inside `SoundSpacesDataset`.

**Edit A — route IR load through a hook (inside `__getitem__`, ambi branch):**
```diff
-            ir = np.load(ambi_path).astype(np.float64)
+            ir = self._load_foa_ir(ambi_path, sample_idx)
```

**Edit B — default hook implementation (added just above `_get_spectrogram`):**
```diff
+    def _load_foa_ir(self, ambi_path, sample_idx):
+        """Load raw FOA impulse response. Overridden by rotated subclass."""
+        return np.load(ambi_path).astype(np.float64)
+
     def _get_spectrogram(self, waveform, n_fft=512, power=1.0, win_length=64, hop_length=16):
```

Behavior is byte-identical for the base class — it does exactly what the
original inline `np.load(...).astype(np.float64)` did.

### 2. `data/dataloader.py`

Added an import and a dataset-class selector; the rest of `make_dataloader`
is untouched except for one line that picks the class.

```diff
 from torch.utils.data import DataLoader
 from .dataset import SoundSpacesDataset
+from .dataset_rotated import SoundSpacesDatasetRotated
+
+
+def _select_dataset_class(cfg):
+    """Pick the FOA-rotated dataset when rotate_canonical is enabled."""
+    if (getattr(cfg.dataset, 'rotate_canonical', False)
+            and getattr(cfg.dataset, 'use_ambisonic', False)):
+        return SoundSpacesDatasetRotated
+    return SoundSpacesDataset
```
```diff
-    dataset = SoundSpacesDataset(cfg, split=split)
+    dataset_cls = _select_dataset_class(cfg)
+    dataset = dataset_cls(cfg, split=split)
```

### 3. `train.py`

CLI flag and cfg propagation.

```diff
     p.add_argument('--foa-consistency-weight', type=float, default=None)
+    p.add_argument('--rotate-canonical', action='store_true',
+                   help='Rotate FOA into a canonical listener frame (dataset_rotated.py).')
     args = p.parse_args()
```
```diff
     if args.foa_consistency_weight is not None: cfg.model.foa_consistency_weight = args.foa_consistency_weight
+    if args.rotate_canonical: cfg.dataset.rotate_canonical = True
```

### 4. `test.py`

Same two-line addition.

```diff
     p.add_argument('--vis-per-scene', type=int, default=100,
                    help='Number of visualizations per scene (default: 100)')
+    p.add_argument('--rotate-canonical', action='store_true',
+                   help='Rotate FOA into a canonical listener frame (dataset_rotated.py).')
     args = p.parse_args()
```
```diff
     cfg.mode.vis_per_scene = args.vis_per_scene
+    if args.rotate_canonical: cfg.dataset.rotate_canonical = True
```

## Files NOT modified

- `config/*.yaml` (none of the 6 FOA configs were touched; rotation is
  enabled only via the CLI flag or an explicit cfg override)
- `models/*.py`
- `utils/train_utils.py`, `utils/test_utils.py`, `utils/config.py`,
  `utils/metrics.py`, `utils/visualization.py`
- `data/sh_basis.py`
- Existing scripts `scripts/train_JS.sh`, `scripts/test_JS.sh`, bulk
  scripts — untouched

## Full rollback procedure

If you want to revert to the exact pre-change state, do all of the following:

1. **Delete added files:**
   ```bash
   rm baseline/data/dataset_rotated.py
   rm baseline/scripts/train_CW.sh
   rm baseline/scripts/test_CW.sh
   rm baseline/data/doc/dataset_rotation.md
   rmdir baseline/data/doc   # if empty
   ```

2. **Revert `data/dataset.py`:**
   - In the ambisonic branch of `__getitem__`, change
     `ir = self._load_foa_ir(ambi_path, sample_idx)` back to
     `ir = np.load(ambi_path).astype(np.float64)`.
   - Delete the `_load_foa_ir` method added just above `_get_spectrogram`.

3. **Revert `data/dataloader.py`:**
   - Remove the `from .dataset_rotated import SoundSpacesDatasetRotated` line.
   - Remove the `_select_dataset_class` helper.
   - Change `dataset = dataset_cls(cfg, split=split)` (and the preceding
     `dataset_cls = _select_dataset_class(cfg)` line) back to the single
     line `dataset = SoundSpacesDataset(cfg, split=split)`.

4. **Revert `train.py`:**
   - Remove the `--rotate-canonical` argparse line.
   - Remove the `if args.rotate_canonical: cfg.dataset.rotate_canonical = True`
     line.

5. **Revert `test.py`:**
   - Same two-line removal as in `train.py`.

After rolling back, no config file needs to change; the code will be
byte-equivalent to the pre-change state (the `_load_foa_ir` hook is a pure
refactor wrapping the same `np.load`).

## Partial rollback (keep the hook, disable the feature)

If the feature just needs to be turned **off** (not deleted), simply do not
pass `--rotate-canonical`. The default path uses the base class exactly as
before. This is the recommended path for an A/B comparison.

## Sanity checks after any rollback

```bash
python3 -c "import ast; [ast.parse(open(p).read()) for p in [
    'baseline/data/dataset.py',
    'baseline/data/dataloader.py',
    'baseline/train.py',
    'baseline/test.py',
]]; print('ok')"
bash -n baseline/scripts/train_JS.sh
bash -n baseline/scripts/test_JS.sh
```

And a quick smoke test that `make_dataloader` still loads a batch of the
normal (non-rotated) pipeline.

## Startup speedup (separate change)

To make cold starts fast, `_build_sample_list` now persists the already-
filtered sample list as `samples_{split}_{depth_type}_{hash}.json` in the
dataset root. The hash encodes the scene list + depth_type + ambisonic
flag, so any split change invalidates it automatically. On cache hit,
zero filesystem stats are performed and the walk is skipped entirely.
`train_CW.sh` also primes the FS dentry cache with a single `find` and
staggers the two parallel jobs by 20 s so CUDA/NCCL init doesn't collide.

To roll this back:
- In `data/dataset.py`: remove the `import hashlib` line, the fast-cache
  branch at the top of `_build_sample_list`, and the `json.dump` block
  near the end.
- In `scripts/train_CW.sh`: remove the `find` priming line and the
  `sleep 20` between the paired launches.
- Delete any `samples_*_*.json` files left in the dataset root (they are
  harmless but no longer read).

## Notes

- `_load_foa_ir` is the **only** extensibility hook added to the base
  dataset. It returns a numpy array with the exact same dtype and shape as
  the previous inline `np.load(...).astype(np.float64)`, so any future
  consumer of the IR sees an unchanged interface.
- The rotation is exact (no trig), numerically stable, and O(T) per sample.
- If the front/back convention ever needs to flip, change the signs in
  `rotate_foa_to_canonical` in `data/dataset_rotated.py` — no other file
  needs to be touched.

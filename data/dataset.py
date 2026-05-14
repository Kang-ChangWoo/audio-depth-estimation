"""SoundSpaces dataset: binaural echoes -> ERP depth."""

import hashlib
import json
import os
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchaudio
import torchaudio.transforms as T

warnings.filterwarnings("ignore", message=".*torchaudio.*torchcodec.*", category=UserWarning)

from .sh_basis import sh_basis_matrix, compute_covariance, energy_map_from_cov

SPLIT_FILENAME = 'scene_split.json'


def get_scene_split(dataset_dir, split_ratio, seed=42):
    """Load or create a deterministic train/val/test scene split.

    On first call, generates the split from split_ratio and seed, then saves
    it as scene_split.json inside dataset_dir. On subsequent calls, loads the
    saved split directly so the assignment is always explicit.
    """
    split_path = os.path.join(dataset_dir, SPLIT_FILENAME)

    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            split = json.load(f)
        print(f"Loaded scene split from {split_path}")
        print(f"  train: {len(split['train'])}, "
              f"val: {len(split['val'])}, test: {len(split['test'])}")
        return split

    # First run: generate and persist
    scenes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    rng = np.random.RandomState(seed)
    rng.shuffle(scenes)
    n = len(scenes)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    split = {
        'train': sorted(scenes[:n_train]),
        'val': sorted(scenes[n_train:n_train + n_val]),
        'test': sorted(scenes[n_train + n_val:]),
    }

    with open(split_path, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"Created and saved scene split to {split_path}")
    print(f"  train: {len(split['train'])}, "
          f"val: {len(split['val'])}, test: {len(split['test'])}")
    return split


class SoundSpacesDataset(Dataset):
    """Sound-Spaces dataset: binaural echoes -> ERP depth."""

    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.root_dir = cfg.dataset.dataset_dir
        self.audio_format = cfg.dataset.audio_format
        self.depth_type = cfg.dataset.depth_type
        # Optional override for the depth-directory name inside each
        # scene (file names inside stay '{depth_type}_depth_{idx}.npy').
        # Used to load radial depth from 'erp_depth_radial/' while
        # keeping depth_type='erp'. Safe no-op when unset.
        self.depth_dir_name = (getattr(cfg.dataset, 'depth_dir', None)
                               or f'{cfg.dataset.depth_type}_depth')
        self.max_depth = cfg.dataset.max_depth
        self.min_depth = cfg.dataset.min_depth
        self.use_ambisonic = getattr(cfg.dataset, 'use_ambisonic', False)
        self.use_waveform = getattr(cfg.dataset, 'use_waveform', False)
        # n9_0424: per-distance-bin FOA representatives via Method-E
        # eigendecomposition of the FOA covariance inside each round-trip
        # time window. Requires use_ambisonic=True.
        self.use_distance_bins = getattr(cfg.dataset, 'use_distance_bins', False)
        if self.use_distance_bins and not self.use_ambisonic:
            raise ValueError("use_distance_bins=True requires use_ambisonic=True")
        # ERP RGB image input (formerly the SoundSpacesDatasetRGB wrapper).
        # Now handled inline in __getitem__ via an if-branch.
        self.use_rgb = getattr(cfg.dataset, 'use_rgb', False)
        if self.use_rgb and not self.use_ambisonic:
            raise ValueError("use_rgb=True requires use_ambisonic=True")
        # The RGB branch is at the end of the ambisonic path in __getitem__;
        # use_waveform and use_distance_bins return earlier, so combining
        # them with use_rgb would silently drop the RGB tensor. Fail loud.
        if self.use_rgb and (self.use_waveform or self.use_distance_bins):
            raise ValueError(
                "use_rgb is incompatible with use_waveform / use_distance_bins "
                "(those return before the RGB branch in __getitem__)")
        # Sample-rate assumption for time↔distance conversion. Matches the
        # hardcoded early/mid window boundaries elsewhere in this file
        # (2600 samples ≈ 59ms ≈ 10m round-trip at 343 m/s).
        self._distance_bins_sr = int(getattr(
            cfg.dataset, 'distance_bins_sr', 44100))
        # n3_0425: configurable FOA target shape and computation kind.
        #   rep_kind ∈ {'eigen', 'rms'}  (default 'eigen' — preserves n9_0424)
        #   rep_K    ∈ ℕ                 (default 8     — preserves n9_0424)
        # K=8 + eigen retains the existing geometric distance-bin layout
        # used by n9_0424. Other K values use equal-time bins over the same
        # round-trip range (0 → 10 m, ~2570 samples at 44.1 kHz).
        self._rep_kind = str(getattr(cfg.dataset, 'rep_kind', 'eigen')).lower()
        if self._rep_kind not in ('eigen', 'rms'):
            raise ValueError(f"rep_kind must be 'eigen' or 'rms', got {self._rep_kind!r}")
        self._rep_K = int(getattr(cfg.dataset, 'rep_K', 8))
        self._distance_bins_edges = self._compute_distance_bin_edges(self._rep_K)

        scene_split = get_scene_split(
            self.root_dir, cfg.dataset.split_ratio, seed=cfg.dataset.split_seed)
        self.scenes = scene_split[split]

        self.samples = self._build_sample_list(split)

        if self.use_ambisonic:
            h, w = cfg.dataset.images_size
            h, w = int(h), int(w)
            jj, ii = np.meshgrid(np.arange(w), np.arange(h))
            az_grid = (jj + 0.5) / w * 2 * np.pi - np.pi
            el_grid = np.pi / 2 - (ii + 0.5) / h * np.pi
            self._sh_basis = sh_basis_matrix(1, el_grid, az_grid)
            self._erp_shape = (h, w)
            self._sh_n_ch = 4
            print(f"  Precomputed SH basis matrix: {self._sh_basis.shape} (order=1)")

    def _build_sample_list(self, split):
        """Build sample list with cached depth-validity filter.

        Two levels of caching:
          1. Fast path: a pre-filtered sample list persisted per
             (split, depth_type, use_ambisonic, scenes_signature). When hit,
             this skips every filesystem stat call and the depth-cache parse.
          2. Slow path (cache miss): walk the filesystem as before, using the
             per-sample depth-validity cache, then write the fast-path file.
        The fast-path filename encodes the current scene list, so any change
        to the split or scene set automatically invalidates the cache.
        """
        scenes_sig = hashlib.md5(
            ('|'.join(self.scenes)
             + f'|{self.depth_type}|ambi={int(self.use_ambisonic)}').encode()
        ).hexdigest()[:12]
        fast_cache_path = os.path.join(
            self.root_dir,
            f'samples_{split}_{self.depth_type}_{scenes_sig}.json')

        if os.path.exists(fast_cache_path):
            with open(fast_cache_path, 'r') as f:
                samples = [tuple(s) for s in json.load(f)]
            print(f"[{split}] {len(samples)} samples (fast cache: "
                  f"{os.path.basename(fast_cache_path)})"
                  f"{' [ambisonic=ON]' if self.use_ambisonic else ''}"
                  f"{' [waveform=ON]' if self.use_waveform else ''}")
            return samples

        cache_path = os.path.join(self.root_dir, f'sample_cache_{self.depth_type}.json')

        # Load or build validity cache
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                valid_cache = json.load(f)
        else:
            valid_cache = {}

        samples = []
        skipped = 0
        cache_dirty = False
        n_scenes = len(self.scenes)
        print(f"[{split}] building sample list: {n_scenes} scenes, "
              f"depth_type={self.depth_type}"
              f"{' [ambisonic=ON]' if self.use_ambisonic else ''}"
              f" (no fast cache — this may take a few minutes)",
              flush=True)
        t_start = time.time()

        for si, scene in enumerate(self.scenes, start=1):
            audio_dir = os.path.join(self.root_dir, scene, 'audio_wav')
            depth_dir = os.path.join(self.root_dir, scene, self.depth_dir_name)
            if not os.path.isdir(audio_dir) or not os.path.isdir(depth_dir):
                print(f"  [{si:3d}/{n_scenes}] {scene}: skipped (missing dirs)",
                      flush=True)
                continue
            if self.use_ambisonic:
                ambi_dir = os.path.join(self.root_dir, scene, 'ambi1_npy')
                if not os.path.isdir(ambi_dir):
                    print(f"  [{si:3d}/{n_scenes}] {scene}: skipped (no ambi1_npy)",
                          flush=True)
                    continue

            scene_before = len(samples)
            audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
            for af in audio_files:
                idx = af.replace('audio_', '').replace('.wav', '')
                depth_file = f'{self.depth_type}_depth_{idx}.npy'
                depth_path = os.path.join(depth_dir, depth_file)
                if not os.path.exists(depth_path):
                    continue
                if self.use_ambisonic:
                    ambi_path = os.path.join(self.root_dir, scene, 'ambi1_npy', f'ambi1_{idx}.npy')
                    if not os.path.exists(ambi_path):
                        continue

                cache_key = f'{scene}/{idx}'
                if cache_key in valid_cache:
                    is_valid = valid_cache[cache_key]
                else:
                    depth = np.load(depth_path).astype(np.float32)
                    is_valid = bool(np.mean(depth <= 0) <= 0.1)
                    valid_cache[cache_key] = is_valid
                    cache_dirty = True

                if not is_valid:
                    skipped += 1
                    continue
                samples.append((scene, idx))

            scene_kept = len(samples) - scene_before
            elapsed = time.time() - t_start
            print(f"  [{si:3d}/{n_scenes}] {scene}: +{scene_kept} samples "
                  f"(total {len(samples)}, skipped {skipped}, {elapsed:.0f}s)",
                  flush=True)

        if cache_dirty:
            with open(cache_path, 'w') as f:
                json.dump(valid_cache, f)
            print(f"  Saved sample cache: {cache_path}")

        # Persist pre-filtered sample list so future runs skip the FS walk.
        try:
            with open(fast_cache_path, 'w') as f:
                json.dump(samples, f)
            print(f"  Saved fast sample list: {os.path.basename(fast_cache_path)}")
        except OSError as e:
            print(f"  (could not write fast sample cache: {e})")

        print(f"[{split}] {len(samples)} samples from {len(self.scenes)} scenes "
              f"(filtered {skipped} with >10% no-depth)"
              f"{' [ambisonic=ON]' if self.use_ambisonic else ''}"
              f"{' [waveform=ON]' if self.use_waveform else ''}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene, sample_idx = self.samples[idx]

        # Load binaural audio
        audio_path = os.path.join(self.root_dir, scene, 'audio_wav', f'audio_{sample_idx}.wav')
        waveform, sr = torchaudio.load(audio_path, backend="soundfile")
        waveform = waveform.clone()

        n_fft, hop_length, win_length = 512, 160, 400
        cut = int((2 * 20.0 / 340) * sr)
        waveform = waveform[:, :cut]

        if 'spectrogram' in self.audio_format:
            audio = self._get_spectrogram(waveform, n_fft=n_fft, power=1.0,
                                          win_length=win_length, hop_length=hop_length)
            images_size = self.cfg.dataset.images_size
            target_size = tuple(int(x) for x in images_size)
            audio = F.interpolate(audio.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)
        else:
            audio = waveform

        # Load ERP depth
        depth_path = os.path.join(
            self.root_dir, scene, self.depth_dir_name,
            f'{self.depth_type}_depth_{sample_idx}.npy')
        depth = np.load(depth_path).astype(np.float32)
        # nan_to_num maps NaN->0 and +/-inf->+/-float32_max; the clamps below
        # then pull those into [0, max_depth] (so +inf ends up at max_depth,
        # -inf at 0). Explicit `== inf` checks were dead here — nan_to_num
        # leaves no inf values behind.
        depth = np.nan_to_num(depth)
        depth[depth < 0.0] = 0.0
        depth[depth > self.max_depth] = self.max_depth
        gt_depth = torch.from_numpy(depth).unsqueeze(0)

        if 'resize' in self.cfg.dataset.preprocess:
            h, w = self.cfg.dataset.images_size
            gt_depth = F.interpolate(gt_depth.unsqueeze(0), size=(int(h), int(w)),
                                     mode='nearest').squeeze(0)
        if self.cfg.dataset.depth_norm:
            gt_depth = gt_depth / self.max_depth

        if self.use_waveform:
            # Pad/truncate waveform to fixed length for batching
            wave_len = getattr(self.cfg.dataset, 'waveform_len', 960)
            if waveform.shape[1] < wave_len:
                waveform = F.pad(waveform, (0, wave_len - waveform.shape[1]))
            else:
                waveform = waveform[:, :wave_len]
            waveform = waveform.contiguous()
            # Legacy 3-tuple early return for echodiffusion (binaural+CIDE only).
            # When use_distance_bins=True is also set (echodiffusion_ambi+CIDE
            # variant), fall through to the ambisonic+bins path below and
            # return the 6-tuple with the padded waveform appended.
            if not self.use_distance_bins:
                return (audio.contiguous(), gt_depth.contiguous(), waveform)

        if self.use_ambisonic:
            ambi_path = os.path.join(
                self.root_dir, scene, 'ambi1_npy', f'ambi1_{sample_idx}.npy')
            ir = self._load_foa_ir(ambi_path, sample_idx)
            h, w = self._erp_shape
            n_ch = self._sh_n_ch

            # FOA target: channel RMS from IR -> (4,)
            rms = np.sqrt(np.mean(ir[:n_ch] ** 2, axis=1)).astype(np.float32)
            rms_max = np.abs(rms).max()
            if rms_max > 0:
                rms = rms / rms_max
            foa_target = torch.from_numpy(rms)

            # Covariance-based energy map -> (1, H, W)
            # Optional time-window cut (matches BINS_3 boundaries at 44.1 kHz):
            #   'full'     — full IR (default, legacy)
            #   'early'    — [0, 2600)      ~0–59 ms   first arrivals
            #   'mid'      — [2600, 13000)  ~59–295 ms early reflections
            #   'early_mid'— [0, 13000)     drops diffuse late tail
            _window = getattr(self.cfg.dataset, 'gt_energy_window', 'full')
            if _window == 'early':
                ir_cov = ir[:n_ch, :2600]
            elif _window == 'mid':
                ir_cov = ir[:n_ch, 2600:13000]
            elif _window == 'early_mid':
                ir_cov = ir[:n_ch, :13000]
            else:
                ir_cov = ir[:n_ch]
            if ir_cov.shape[1] == 0:
                ir_cov = ir[:n_ch]
            R = compute_covariance(ir_cov)
            energy = energy_map_from_cov(R, self._sh_basis, h, w).astype(np.float32)
            emax = np.abs(energy).max()
            if emax > 0:
                energy = energy / emax
            energy_map = torch.from_numpy(energy).unsqueeze(0)

            if self.use_distance_bins:
                # Per-bin reps computed from the same (possibly rotated) IR
                # so it's consistent with foa_target/energy_map. Shape
                # (rep_K, 4); rep_kind=eigen reproduces the n9_0424 Method-E
                # layout with K=8.
                rep_gt = self._compute_rep_gt(ir, self._rep_K, self._rep_kind)
                rep_gt_t = torch.from_numpy(rep_gt)  # (rep_K, 4)
                if self.use_waveform:
                    # 6-tuple for echodiffusion_ambi + CIDE: append the padded
                    # binaural waveform (truncated above to waveform_len). Used
                    # by _train_step_foa_0415's len(batch)==6 branch.
                    return (audio.contiguous(), gt_depth.contiguous(),
                            foa_target.contiguous(), energy_map.contiguous(),
                            rep_gt_t.contiguous(), waveform)
                return (audio.contiguous(), gt_depth.contiguous(),
                        foa_target.contiguous(), energy_map.contiguous(),
                        rep_gt_t.contiguous())

            if self.use_rgb:
                # ERP RGB image -> float32 [0,1], (3, H, W), resized to images_size.
                rgb_path = os.path.join(
                    self.root_dir, scene, 'erp_rgb', f'erp_{sample_idx}.png')
                img = Image.open(rgb_path).convert('RGB')
                rgb = torch.from_numpy(
                    np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
                if rgb.shape[1] != h or rgb.shape[2] != w:
                    rgb = F.interpolate(
                        rgb.unsqueeze(0), size=(h, w), mode='bilinear',
                        align_corners=False).squeeze(0)
                return (audio.contiguous(), gt_depth.contiguous(),
                        foa_target.contiguous(), energy_map.contiguous(),
                        rgb.contiguous())

            return (audio.contiguous(), gt_depth.contiguous(),
                    foa_target.contiguous(), energy_map.contiguous())

        return audio.contiguous(), gt_depth.contiguous()

    def _load_foa_ir(self, ambi_path, sample_idx):
        """Load raw FOA impulse response. Overridden by rotated subclass."""
        return np.load(ambi_path).astype(np.float64)

    # n9_0424 / n3_0425 -------------------------------------------------------
    # Round-trip distance bins in metres. K=8 default → 9 edges (matches
    # n9_0424). Times are derived as t = 2·d / c where c=343 m/s.
    # End-of-range = 10 m round-trip ≈ 2570 samples at 44.1 kHz; this is the
    # span for non-default K (equal-time linear edges).
    _DISTANCE_BINS_M = (0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
    _SPEED_OF_SOUND = 343.0

    def _compute_distance_bin_edges(self, K=None):
        """Round-trip-time edges (in samples) for K bins.

        K=8 (default) → geometric distance bins (0.2, 0.5, …, 10.0 m), the
        original layout used by n9_0424.
        Other K → equal-time linear edges over the SAME total range used
        by n9 (round-trip 0.2 m → 10.0 m, ≈ samples 51 → 2571 at 44.1 kHz).
        Samples [0, 51) are skipped — sound hasn't reached anything yet —
        and samples beyond 2571 (long reverb tail) are excluded, matching
        n9_0424's exclusion. This keeps the input window identical across
        K values so the predictability-vs-N comparison is apples-to-apples.
        """
        sr = self._distance_bins_sr
        if K is None:
            K = len(self._DISTANCE_BINS_M) - 1
        if K == 8:
            return tuple(int(round(2.0 * d * sr / self._SPEED_OF_SOUND))
                         for d in self._DISTANCE_BINS_M)
        T_min = int(round(2.0 * self._DISTANCE_BINS_M[0] * sr
                          / self._SPEED_OF_SOUND))
        T_max = int(round(2.0 * self._DISTANCE_BINS_M[-1] * sr
                          / self._SPEED_OF_SOUND))
        span = T_max - T_min
        return tuple(int(round(T_min + i * span / K))
                     for i in range(K + 1))

    def _compute_rep_gt(self, ir: np.ndarray, K: int = None,
                        kind: str = 'eigen') -> np.ndarray:
        """Per-bin FOA representatives.

        Parameters
        ----------
        ir   : (≥4, T_total) FOA impulse response in ACN [W, Y, Z, X].
        K    : bin count. None → use ``self._rep_K`` (default 8).
        kind : 'eigen' (Method-E top eigenvector × √λ) or
               'rms'   (per-channel RMS within each bin).

        Returns ``(K, 4) float32``. Empty / rank-deficient bins are zeros.
        """
        n_ch = 4
        T_total = ir.shape[1]
        if K is None:
            K = self._rep_K
        # Bin edges may have been precomputed for the configured K. If a
        # different K is requested at call time (e.g. from a derived class),
        # recompute on the fly.
        if K == self._rep_K:
            edges = self._distance_bins_edges
        else:
            edges = self._compute_distance_bin_edges(K)
        rep = np.zeros((K, n_ch), dtype=np.float32)
        for k in range(K):
            s_k, e_k = edges[k], edges[k + 1]
            if s_k >= T_total:
                continue
            e_k = min(e_k, T_total)
            A_k = ir[:n_ch, s_k:e_k]
            T_k = A_k.shape[1]
            if T_k < 1:
                continue
            if kind == 'rms':
                # √(mean(a^2)) per channel — channel-wise energy summary.
                rep[k] = np.sqrt(np.mean(A_k ** 2, axis=1)).astype(np.float32)
            else:  # 'eigen' — dominant eigenvector × √λ_max
                if T_k < 2:
                    continue
                R_k = (A_k @ A_k.T) / T_k
                lam, V = np.linalg.eigh(R_k)
                lam_max = float(lam[-1])
                v_max = V[:, -1]
                if v_max[0] < 0:
                    v_max = -v_max
                if lam_max <= 0:
                    continue
                rep[k] = (np.sqrt(lam_max) * v_max).astype(np.float32)
        return rep

    def _get_spectrogram(self, waveform, n_fft=512, power=1.0, win_length=64, hop_length=16):
        spectrogram = T.Spectrogram(n_fft=n_fft, win_length=win_length,
                                    power=power, hop_length=hop_length)
        return spectrogram(waveform)


# ---------------------------------------------------------------------------
# Canonical-frame FOA rotation wrapper (formerly data/dataset_rotated.py)
# ---------------------------------------------------------------------------
#
# Habitat-Sim Ambisonics are effectively *world-frame*, not listener-frame.
# Each position is captured with 4 yaw rotations in a fixed cyclic order:
#
#     view_mod = raw_sample_index % 4
#         0 -> front (no rotation)
#         1 -> right (yaw -90 deg)
#         2 -> back  (yaw -180 deg)
#         3 -> left  (yaw -270 deg)
#
# To make FOA inputs ego-consistent with the RGB / depth views, we rotate the
# FOA channels into a canonical agent-centered frame before any per-channel
# statistics (RMS, covariance, ERP energy) are computed.
#
# Why `sample_idx % 4` and NOT `dataset_idx % 4`
# ----------------------------------------------
# SoundSpacesDataset drops samples whose ERP depth is >10% invalid. After this
# filter, the position in self.samples no longer preserves the raw 4-view
# cycle: a missing entry shifts every subsequent sample. The only reliable
# source of the view index is the raw capture index parsed from the filename
# (e.g. `audio_023.wav` -> 23 -> view_mod=3). We compute view_mod from that.
#
# What is rotated
# ---------------
# We rotate the raw FOA impulse response *before* deriving any target. Doing
# the rotation at the IR level is the most principled choice: it transforms
# exactly one quantity, and every downstream target (RMS target vector,
# covariance matrix, ERP energy map) is then computed from a self-consistent
# canonical-frame signal. Rotating the RMS target alone is ill-defined
# (channel-wise RMS is invariant under sign flips so 180 deg rotation is a
# no-op), and rotating the covariance/energy-map would require extra matrix
# machinery for no benefit. 90 deg yaw is just a sign-swap on (Y, X), so the
# rotation is exact and O(T) per sample.
#
# Assumed channel order (ACN): [W, Y, Z, X]. W and Z are invariant under yaw;
# only the horizontal (Y, X) pair mixes.


def get_view_mod_from_sample_idx(sample_idx) -> int:
    """Return view_mod in {0,1,2,3} from the raw capture index (filename)."""
    return int(sample_idx) % 4


def rotate_foa_to_canonical(foa: np.ndarray, view_mod: int) -> np.ndarray:
    """Rotate 1st-order ACN ambisonics by -90 deg * view_mod (yaw only).

    Args:
        foa: (n_ch, T) array with n_ch >= 4 in ACN order [W, Y, Z, X].
        view_mod: 0, 1, 2, or 3.
    Returns:
        A new array of the same shape, rotated into the canonical frame.
    """
    if view_mod == 0:
        return foa
    out = foa.copy()
    Y = foa[1]
    X = foa[3]
    if view_mod == 1:       # yaw -90 deg:   Y' =  X, X' = -Y
        out[1] = X
        out[3] = -Y
    elif view_mod == 2:     # yaw -180 deg:  Y' = -Y, X' = -X
        out[1] = -Y
        out[3] = -X
    elif view_mod == 3:     # yaw -270 deg:  Y' = -X, X' =  Y
        out[1] = -X
        out[3] = Y
    else:
        raise ValueError(f"view_mod must be in {{0,1,2,3}}, got {view_mod}")
    return out


class SoundSpacesDatasetRotated(SoundSpacesDataset):
    """SoundSpacesDataset with FOA rotated to a canonical listener frame.

    Only the ambisonic IR loader is overridden; the non-ambisonic code path
    is inherited unchanged. The depth-validity filter in the base class is
    preserved, and `view_mod` is still computed from the raw filename index,
    so it stays correct even when samples are dropped.
    """

    def _load_foa_ir(self, ambi_path, sample_idx):
        ir = np.load(ambi_path).astype(np.float64)
        view_mod = get_view_mod_from_sample_idx(sample_idx)
        return rotate_foa_to_canonical(ir, view_mod)

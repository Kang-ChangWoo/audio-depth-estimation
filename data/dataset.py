"""SoundSpaces dataset: binaural echoes -> ERP depth."""

import hashlib
import json
import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
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
        self.max_depth = cfg.dataset.max_depth
        self.min_depth = cfg.dataset.min_depth
        self.use_ambisonic = getattr(cfg.dataset, 'use_ambisonic', False)
        self.use_waveform = getattr(cfg.dataset, 'use_waveform', False)

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

        for scene in self.scenes:
            audio_dir = os.path.join(self.root_dir, scene, 'audio_wav')
            depth_dir = os.path.join(self.root_dir, scene, f'{self.depth_type}_depth')
            if not os.path.isdir(audio_dir) or not os.path.isdir(depth_dir):
                continue
            if self.use_ambisonic:
                ambi_dir = os.path.join(self.root_dir, scene, 'ambi1_npy')
                if not os.path.isdir(ambi_dir):
                    continue

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
            self.root_dir, scene, f'{self.depth_type}_depth',
            f'{self.depth_type}_depth_{sample_idx}.npy')
        depth = np.load(depth_path).astype(np.float32)
        depth = np.nan_to_num(depth)
        depth[depth == -np.inf] = 0
        depth[depth == np.inf] = 0
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
            return (audio.contiguous(), gt_depth.contiguous(),
                    waveform.contiguous())

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
            R = compute_covariance(ir[:n_ch])
            energy = energy_map_from_cov(R, self._sh_basis, h, w).astype(np.float32)
            emax = np.abs(energy).max()
            if emax > 0:
                energy = energy / emax
            energy_map = torch.from_numpy(energy).unsqueeze(0)

            return (audio.contiguous(), gt_depth.contiguous(),
                    foa_target.contiguous(), energy_map.contiguous())

        return audio.contiguous(), gt_depth.contiguous()

    def _load_foa_ir(self, ambi_path, sample_idx):
        """Load raw FOA impulse response. Overridden by rotated subclass."""
        return np.load(ambi_path).astype(np.float64)

    def _get_spectrogram(self, waveform, n_fft=512, power=1.0, win_length=64, hop_length=16):
        spectrogram = T.Spectrogram(n_fft=n_fft, win_length=win_length,
                                    power=power, hop_length=hop_length)
        return spectrogram(waveform)

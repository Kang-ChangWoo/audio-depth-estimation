"""SoundSpaces dataset: binaural echoes -> ERP depth."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

from .sh_basis import sh_basis_matrix, reconstruct_per_component_maps


def get_scene_split(dataset_dir, split_ratio, seed=42):
    """Deterministic train/val/test scene splitting."""
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
    print(f"Scene split — train: {len(split['train'])}, "
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

        scene_split = get_scene_split(
            self.root_dir, cfg.dataset.split_ratio, seed=cfg.dataset.split_seed)
        self.scenes = scene_split[split]

        self.samples = []
        skipped = 0
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
                depth = np.load(depth_path).astype(np.float32)
                if np.mean(depth <= 0) > 0.1:
                    skipped += 1
                    continue
                self.samples.append((scene, idx))

        print(f"[{split}] {len(self.samples)} samples from {len(self.scenes)} scenes "
              f"(filtered {skipped} with >10% no-depth)"
              f"{' [ambisonic=ON]' if self.use_ambisonic else ''}")

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene, sample_idx = self.samples[idx]

        # Load binaural audio
        audio_path = os.path.join(self.root_dir, scene, 'audio_wav', f'audio_{sample_idx}.wav')
        waveform, sr = torchaudio.load(audio_path)
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

        if self.use_ambisonic:
            ambi_path = os.path.join(
                self.root_dir, scene, 'ambi1_npy', f'ambi1_{sample_idx}.npy')
            sh_coeffs = np.load(ambi_path).astype(np.float64)
            h, w = self._erp_shape
            component_maps = reconstruct_per_component_maps(sh_coeffs, self._sh_basis)
            component_maps = component_maps.reshape(4, h, w).astype(np.float32)
            for ch in range(4):
                cmax = np.abs(component_maps[ch]).max()
                if cmax > 0:
                    component_maps[ch] = component_maps[ch] / cmax
            ambi_erp = torch.from_numpy(component_maps)
            return audio.contiguous(), gt_depth.contiguous(), ambi_erp.contiguous()

        return audio.contiguous(), gt_depth.contiguous()

    def _get_spectrogram(self, waveform, n_fft=512, power=1.0, win_length=64, hop_length=16):
        spectrogram = T.Spectrogram(n_fft=n_fft, win_length=win_length,
                                    power=power, hop_length=hop_length)
        return spectrogram(waveform)

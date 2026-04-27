"""N2 dataset: extends rotated dataset with temporal FOA features.

Returns a 7-tuple when use_n2_features is enabled:
    (audio, gt_depth, foa_target, energy_map,
     foa_spec, temporal_rms, temporal_energies)

- foa_spec:           (4, H, W)       spectrogram of the first 4 FOA IR channels
- temporal_rms:       (K*4,)          per-bin RMS: K bins × 4 channels, concatenated
- temporal_energies:  (K, H, W)       energy maps from K temporal windows

Bin presets (at 44.1 kHz):

  n_bins=3 (default):
    bin 0: 0–2600      direct + first reflections   (~0–59 ms)
    bin 1: 2600–13000   early reverb                 (~59–295 ms)
    bin 2: 13000–end    late / diffuse               (~295 ms+)

  n_bins=6:
    bin 0: 0–650        direct sound only            (~0–15 ms)
    bin 1: 650–2600     first-order reflections       (~15–59 ms)
    bin 2: 2600–5200    second/third-order            (~59–118 ms)
    bin 3: 5200–9000    early reverb (discrete)       (~118–204 ms)
    bin 4: 9000–18000   transition to diffuse         (~204–408 ms)
    bin 5: 18000–end    late diffuse field            (~408 ms+)

Set cfg.dataset.n_temporal_bins (3 or 6) to choose. Default is 3.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from .dataset_rotated import SoundSpacesDatasetRotated
from .sh_basis import compute_covariance, energy_map_from_cov

BINS_3 = [(0, 2600), (2600, 13000), (13000, None)]

BINS_6 = [
    (0, 650),          # direct sound only
    (650, 2600),       # first-order reflections
    (2600, 5200),      # second/third-order reflections
    (5200, 9000),      # early reverb (discrete)
    (9000, 18000),     # transition to diffuse
    (18000, None),     # late diffuse field
]

BINS_3_OVERLAP = [(0, 13000), (2600, 18000), (8000, None)]

BINS_4_OVERLAP = [(0, 8800), (2600, 11000), (5400, 15000), (8800, None)]

PRESET = {
    3: BINS_3,
    6: BINS_6,
    'overlap3': BINS_3_OVERLAP,
    'overlap4': BINS_4_OVERLAP,
}


class SoundSpacesDatasetN2(SoundSpacesDatasetRotated):

    def __init__(self, cfg, split='train'):
        super().__init__(cfg, split)
        raw_bins = getattr(cfg.dataset, 'n_temporal_bins', 3)
        key = raw_bins if isinstance(raw_bins, str) else int(raw_bins)
        if key not in PRESET:
            raise ValueError(
                f"n_temporal_bins must be one of {list(PRESET.keys())}, got {key}")
        self._temporal_bins = PRESET[key]
        self._n_bins = len(self._temporal_bins)
        if self.use_ambisonic:
            print(f"  N2 dataset: {self._n_bins} temporal bins (key={key})")

    def __getitem__(self, idx):
        if not self.use_ambisonic:
            return super().__getitem__(idx)

        base = super().__getitem__(idx)
        audio, gt_depth, foa_target, energy_map = base

        scene, sample_idx = self.samples[idx]
        ambi_path = os.path.join(
            self.root_dir, scene, 'ambi1_npy', f'ambi1_{sample_idx}.npy')
        ir = self._load_foa_ir(ambi_path, sample_idx)

        foa_spec = self._compute_foa_spectrogram(ir)
        temporal_rms = self._compute_temporal_rms(ir)
        temporal_energies = self._compute_temporal_energies(ir)

        return (audio.contiguous(), gt_depth.contiguous(),
                foa_target.contiguous(), energy_map.contiguous(),
                foa_spec.contiguous(), temporal_rms.contiguous(),
                temporal_energies.contiguous())

    def _compute_foa_spectrogram(self, ir):
        sr = 44100
        cut = int((2 * 20.0 / 340) * sr)
        n_ch = min(ir.shape[0], 4)
        ir_cut = ir[:n_ch, :cut].astype(np.float32)
        ir_tensor = torch.from_numpy(ir_cut)

        spec_fn = T.Spectrogram(n_fft=512, win_length=400,
                                power=1.0, hop_length=160)
        foa_spec = spec_fn(ir_tensor)

        target_size = tuple(int(x) for x in self.cfg.dataset.images_size)
        foa_spec = F.interpolate(
            foa_spec.unsqueeze(0), size=target_size, mode='nearest'
        ).squeeze(0)
        return foa_spec

    def _compute_temporal_rms(self, ir, n_ch=4):
        rms_parts = []
        for start, end in self._temporal_bins:
            seg = ir[:n_ch, start:end]
            if seg.shape[1] == 0:
                rms_parts.append(np.zeros(n_ch, dtype=np.float32))
                continue
            rms = np.sqrt(np.mean(seg ** 2, axis=1)).astype(np.float32)
            mx = np.abs(rms).max()
            if mx > 0:
                rms = rms / mx
            rms_parts.append(rms)
        return torch.from_numpy(np.concatenate(rms_parts))

    def _compute_temporal_energies(self, ir):
        h, w = self._erp_shape
        n_ch = self._sh_n_ch
        maps = []
        for start, end in self._temporal_bins:
            seg = ir[:n_ch, start:end]
            if seg.shape[1] == 0:
                maps.append(np.zeros((h, w), dtype=np.float32))
                continue
            R = compute_covariance(seg)
            energy = energy_map_from_cov(R, self._sh_basis, h, w).astype(np.float32)
            mx = np.abs(energy).max()
            if mx > 0:
                energy = energy / mx
            maps.append(energy)
        return torch.from_numpy(np.stack(maps))

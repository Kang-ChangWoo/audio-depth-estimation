"""Dataset wrapper that also loads ERP RGB images for teacher-guided training."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset_rotated import SoundSpacesDatasetRotated


class SoundSpacesDatasetRGB(SoundSpacesDatasetRotated):
    """Extends the rotated-FOA dataset with ERP RGB images.

    Returns a 5-tuple: (audio, gt_depth, foa_target, energy_map, rgb).
    The RGB image is loaded from erp_rgb/erp_{idx}.png, converted to
    float32 [0, 1], and resized to match cfg.dataset.images_size.
    """

    def __getitem__(self, idx):
        base = super().__getitem__(idx)
        audio, gt_depth, foa_target, energy_map = base

        scene, sample_idx = self.samples[idx]
        rgb_path = os.path.join(
            self.root_dir, scene, 'erp_rgb', f'erp_{sample_idx}.png')

        img = Image.open(rgb_path).convert('RGB')
        rgb = torch.from_numpy(
            np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        h, w = int(self.cfg.dataset.images_size[0]), int(self.cfg.dataset.images_size[1])
        if rgb.shape[1] != h or rgb.shape[2] != w:
            rgb = F.interpolate(
                rgb.unsqueeze(0), size=(h, w), mode='bilinear',
                align_corners=False).squeeze(0)

        return (audio.contiguous(), gt_depth.contiguous(),
                foa_target.contiguous(), energy_map.contiguous(),
                rgb.contiguous())

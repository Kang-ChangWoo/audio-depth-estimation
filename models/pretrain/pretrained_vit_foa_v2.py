"""v2 — ViT + FOA with histogram alignment (energy-map supervised).

Extends the v1 aux-SH-head base with the full SH machinery from
``unet_foa``: an ``sh_basis`` buffer, energy-map <-> SH projection
helpers, a learnable ``DeepScaleShift`` affine on predicted SH
coefficients, and a dense histogram-alignment loss tying the
model-predicted SH directional energy to the ambisonic-derived energy
map (``foa_v2_js``-style: energy-map GT, not depth-projected GT).

Output dict keeps the same schema as ``AudioDepthFOAGenerator`` so we
reuse the existing ``_train_step_js`` routing and ``AudioDepthFOALoss``
with ``hist_weight > 0`` unchanged.
"""

import math
import torch
import torch.nn as nn

from ..unet_foa import sh_basis_erp, DeepScaleShift
from .pretrained_vit_foa import ViTFOABackbone, SHHead


class PretrainedViTFOAV2(ViTFOABackbone):
    """ViT backbone + depth decoder + SH branch + histogram alignment."""

    DEFAULT_SH_ORDER = 5
    DEFAULT_FOA_DIM = 4
    DEFAULT_PROJ_DIM = 128
    DEFAULT_SCALE_SHIFT_HIDDEN = 256
    DEFAULT_SCALE_SHIFT_LAYERS = 4

    def __init__(self, cfg, input_nc=2, pretrained=True, freeze_encoder=False,
                 proj_dim: int = None, foa_dim: int = None, sh_order: int = None,
                 scale_shift_hidden: int = None, scale_shift_layers: int = None,
                 H_erp: int = None, W_erp: int = None, **_unused):
        super().__init__(cfg, input_nc=input_nc, pretrained=pretrained,
                         freeze_encoder=freeze_encoder)

        self.sh_order = int(sh_order if sh_order is not None else self.DEFAULT_SH_ORDER)
        self.sh_dim = (self.sh_order + 1) ** 2
        self.foa_dim = int(foa_dim if foa_dim is not None else self.DEFAULT_FOA_DIM)
        self.hoa_dim = self.sh_dim - self.foa_dim
        proj_dim = int(proj_dim if proj_dim is not None else self.DEFAULT_PROJ_DIM)
        ss_hidden = int(scale_shift_hidden if scale_shift_hidden is not None
                        else self.DEFAULT_SCALE_SHIFT_HIDDEN)
        ss_layers = int(scale_shift_layers if scale_shift_layers is not None
                        else self.DEFAULT_SCALE_SHIFT_LAYERS)

        H_erp = int(H_erp if H_erp is not None else cfg.dataset.images_size[0])
        W_erp = int(W_erp if W_erp is not None else cfg.dataset.images_size[1])
        basis = sh_basis_erp(self.sh_order, H_erp, W_erp, dtype=torch.float32)
        self.register_buffer("sh_basis", basis, persistent=False)

        self.scale_shift = DeepScaleShift(
            n_channels=self.sh_dim,
            hidden_dim=ss_hidden,
            n_hidden_layers=ss_layers,
        )

        # SH branch — operates on [cls, mean(patch)] from the final transformer
        feat_dim = 2 * self.EMBED_DIM
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )
        self.foa_head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, self.foa_dim),
        )
        self.hoa_head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, self.hoa_dim),
        )

    # ── SH projection helpers (mirror unet_foa) ──────────────────
    def reconstruct_from_coeffs(self, coeffs):
        return (coeffs[:, :, None, None] * self.sh_basis[None]).sum(dim=1, keepdim=True)

    def project_depth_to_sh(self, depth, eps=1e-6):
        """Integrate a spherical map (depth or energy) against the SH basis.

        Shared helper — ``compute_gt_energy_sh`` in train_utils calls this
        on the GT energy map to produce the histogram-alignment target.
        """
        basis = self.sh_basis
        d = depth[:, 0:1, :, :]
        H = basis.shape[1]
        theta = torch.linspace(0, math.pi, H, device=basis.device, dtype=basis.dtype)
        sin_weight = torch.sin(theta)[:, None]
        w = sin_weight[None, None, :, :]
        num = (w * d * basis[None]).sum(dim=(2, 3))
        den = (w * basis[None] ** 2).sum(dim=(2, 3)) + eps
        return num / den

    def forward(self, x, return_hist_maps=False, **_unused):
        orig_h, orig_w = x.shape[2], x.shape[3]
        tokens = self._encode(x)                         # (B, 1+N, D)
        cls_feat = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        patch_mean = patch_tokens.mean(dim=1)
        sh_feat = torch.cat([cls_feat, patch_mean], dim=1)  # (B, 2D)

        foa_latent = self.audio_proj(sh_feat)            # (B, proj_dim)
        pred_foa = self.foa_head(foa_latent)             # (B, 4)
        pred_hoa = self.hoa_head(foa_latent)             # (B, sh_dim-4)
        pred_sh = torch.cat([pred_foa, pred_hoa], dim=1) # (B, sh_dim)

        pred_depth = self._decode_depth(patch_tokens, orig_h, orig_w)

        out = {
            "pred_depth": pred_depth,
            "foa_latent": foa_latent,
            "pred_foa": pred_foa,
            "pred_hoa": pred_hoa,
            "pred_sh": pred_sh,
        }

        if return_hist_maps:
            sh_aligned = self.scale_shift(pred_sh)
            energy_recon_aligned = self.reconstruct_from_coeffs(sh_aligned)
            depth_sh_coeffs = self.project_depth_to_sh(pred_depth)
            depth_sh = self.reconstruct_from_coeffs(depth_sh_coeffs)
            out["energy_recon_aligned"] = energy_recon_aligned
            out["depth_sh"] = depth_sh
            out["depth_sh_coeffs"] = depth_sh_coeffs
            out["sh_aligned"] = sh_aligned

        return out

"""FOAv2: improved FOA baseline with FiLM decoder conditioning and gradient consistency loss."""

import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_foa import AudioDepthFOAGenerator, sh_basis_erp, DeepScaleShift


class FiLMConditioner(nn.Module):
    """Modulate spatial features using a conditioning vector."""
    def __init__(self, cond_dim, feat_channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.ones_(self.proj.bias[:feat_channels])

    def forward(self, feat, cond):
        """feat: (B, C, H, W), cond: (B, cond_dim)"""
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta


class FOAv2Generator(AudioDepthFOAGenerator):
    """Improved FOA baseline with FiLM decoder conditioning and FOA-depth gradient consistency."""

    def __init__(self, cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, proj_dim=128, foa_dim=4, sh_order=5,
                 scale_shift_hidden=256, scale_shift_layers=4,
                 H_erp=256, W_erp=512, **kwargs):
        super().__init__(
            cfg, input_nc=input_nc, output_nc=output_nc, num_downs=num_downs,
            ngf=ngf, use_dropout=use_dropout, proj_dim=proj_dim, foa_dim=foa_dim,
            sh_order=sh_order, scale_shift_hidden=scale_shift_hidden,
            scale_shift_layers=scale_shift_layers, H_erp=H_erp, W_erp=W_erp,
        )
        feat_dim = ngf * 8
        # FiLM conditioning: inject FOA latent into first decoder block
        self.film = FiLMConditioner(proj_dim, feat_dim)

    def _gradient_consistency(self, pred_depth, energy_recon):
        """Spatial gradient alignment between depth and FOA energy.
        Returns a scalar tensor in [0, 2] (0 = perfect alignment)."""
        dx_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                 dtype=pred_depth.dtype,
                                 device=pred_depth.device).view(1, 1, 3, 3)
        dy_kernel = dx_kernel.transpose(2, 3)

        d_dx = F.conv2d(pred_depth, dx_kernel, padding=1)
        d_dy = F.conv2d(pred_depth, dy_kernel, padding=1)
        e_dx = F.conv2d(energy_recon, dx_kernel, padding=1)
        e_dy = F.conv2d(energy_recon, dy_kernel, padding=1)

        d_grad = torch.cat([d_dx, d_dy], dim=1)  # (B, 2, H, W)
        e_grad = torch.cat([e_dx, e_dy], dim=1)

        # Cosine similarity of gradient fields
        cos = F.cosine_similarity(d_grad, e_grad, dim=1).mean()
        return 1.0 - cos  # 0 = perfect alignment

    def forward(self, x, return_hist_maps=False):
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        # SH branch
        pooled = self.pool(bottleneck)
        foa_latent = self.audio_proj(pooled)
        pred_foa = self.foa_head(foa_latent)
        pred_hoa = self.hoa_head(foa_latent)
        pred_sh = torch.cat([pred_foa, pred_hoa], dim=1)

        # Decode with FiLM conditioning from FOA latent
        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)
        h = self.film(h, foa_latent)  # FOA cue injected into decoder
        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

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
            # FOA-depth gradient consistency loss
            out["foa_depth_consistency"] = self._gradient_consistency(
                pred_depth, energy_recon_aligned)
        else:
            out["foa_depth_consistency"] = torch.tensor(
                0.0, device=pred_depth.device)

        return out

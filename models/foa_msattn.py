"""FOA Multi-Scale Attention: aggregate features from multiple encoder levels."""

import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_foa import AudioDepthFOAGenerator, sh_basis_erp, DeepScaleShift


class MultiScaleAttention(nn.Module):
    """Attention-weighted aggregation of features from multiple encoder scales."""

    def __init__(self, channel_list, out_dim):
        """
        channel_list: list of channel dimensions at each encoder level to aggregate
        out_dim: output feature dimension
        """
        super().__init__()
        self.num_scales = len(channel_list)
        # Project each scale to common dimension
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, out_dim),
                nn.ReLU(inplace=True),
            )
            for c in channel_list
        ])
        # Attention over scales
        self.attn_proj = nn.Linear(out_dim, 1)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, feature_list):
        """
        feature_list: list of (B, C_i, H_i, W_i) tensors from different encoder levels
        Returns: aggregated (B, out_dim), attention weights (B, num_scales)
        """
        projected = []
        for feat, proj in zip(feature_list, self.scale_projs):
            projected.append(proj(feat))  # each (B, out_dim)

        stacked = torch.stack(projected, dim=1)  # (B, num_scales, out_dim)
        attn_logits = self.attn_proj(stacked).squeeze(-1)  # (B, num_scales)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, num_scales)

        aggregated = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, out_dim)
        aggregated = self.out_norm(aggregated)
        return aggregated, attn_weights


class FOAMultiScaleAttnGenerator(AudioDepthFOAGenerator):
    """UNet FOA with multi-scale attention aggregation across encoder levels."""

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

        # Build channel list for encoder levels: enc0 output + each encoder output + bottleneck
        channel_list = [ngf]  # enc0
        in_ch = ngf
        for i in range(1, num_downs - 1):
            out_ch = min(ngf * (2 ** i), ngf * 8)
            channel_list.append(out_ch)
            in_ch = out_ch
        channel_list.append(feat_dim)  # bottleneck

        self.ms_attn = MultiScaleAttention(channel_list, feat_dim)

        # Fusion: combine pooled bottleneck + multi-scale aggregated
        self.ms_fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
        )

    def _ms_kl_loss(self, attn_weights):
        """KL divergence between multi-scale attention weights and uniform prior."""
        B, S = attn_weights.shape
        uniform = torch.ones_like(attn_weights) / S
        kl = (attn_weights * (torch.log(attn_weights + 1e-8) - torch.log(uniform))).sum(dim=-1).mean()
        return kl

    def forward(self, x, return_hist_maps=False):
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        # Multi-scale attention across all encoder levels + bottleneck
        all_features = enc_features + [bottleneck]
        ms_aggregated, ms_attn_weights = self.ms_attn(all_features)

        # KL loss on multi-scale attention
        kl_loss = self._ms_kl_loss(ms_attn_weights)

        # Fuse with pooled bottleneck
        pooled = self.pool(bottleneck)
        pooled_flat = pooled.view(pooled.size(0), -1)
        fused = self.ms_fusion(torch.cat([pooled_flat, ms_aggregated], dim=1))

        # SH branch
        foa_latent = self.audio_proj(fused.unsqueeze(-1).unsqueeze(-1))
        pred_foa = self.foa_head(foa_latent)
        pred_hoa = self.hoa_head(foa_latent)
        pred_sh = torch.cat([pred_foa, pred_hoa], dim=1)

        # Decode
        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)
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
            "kl_loss": kl_loss,
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

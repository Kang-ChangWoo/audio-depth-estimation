"""FOA Feature Bank: learnable memory bank of prototype features."""

import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_foa import AudioDepthFOAGenerator, sh_basis_erp, DeepScaleShift


class FeatureBank(nn.Module):
    """Learnable feature bank with attention-based retrieval."""

    def __init__(self, feat_dim, num_prototypes=64, temperature=1.0):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.bank = nn.Parameter(torch.randn(num_prototypes, feat_dim) * 0.02)
        self.query_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, query):
        """
        query: (B, D) -- pooled encoder features
        Returns: retrieved features (B, D), attention distribution (B, K)
        """
        B, D = query.shape
        q = self.query_proj(query)  # (B, D)
        # Compute attention over prototypes
        bank_normed = F.normalize(self.bank, dim=-1)  # (K, D)
        q_normed = F.normalize(q, dim=-1)  # (B, D)
        logits = torch.matmul(q_normed, bank_normed.T) / self.temperature  # (B, K)
        attn = F.softmax(logits, dim=-1)  # (B, K)
        retrieved = torch.matmul(attn, self.bank)  # (B, D)
        retrieved = self.out_proj(self.norm(retrieved))
        return retrieved, attn


class FOAFeatBankGenerator(AudioDepthFOAGenerator):
    """UNet FOA with learnable feature bank bridge."""

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
        num_prototypes = kwargs.get('num_prototypes', 64)
        temperature = kwargs.get('bank_temperature', 1.0)
        self.feature_bank = FeatureBank(feat_dim, num_prototypes, temperature)

        # Fusion layer: combine original pooled + retrieved features
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
        )

    def _bank_kl_loss(self, attn):
        """KL divergence between attention distribution and uniform prior (encourage diversity)."""
        B, K = attn.shape
        uniform = torch.ones_like(attn) / K
        # KL(attn || uniform) = sum attn * log(attn / uniform)
        log_attn = torch.log(attn + 1e-8)
        log_uniform = torch.log(uniform)
        kl = (attn * (log_attn - log_uniform)).sum(dim=-1).mean()
        return kl

    def forward(self, x, return_hist_maps=False):
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        # Feature bank retrieval
        pooled = self.pool(bottleneck)
        pooled_flat = pooled.view(pooled.size(0), -1)  # (B, feat_dim)
        retrieved, bank_attn = self.feature_bank(pooled_flat)

        # KL loss on bank attention
        kl_loss = self._bank_kl_loss(bank_attn)

        # Fuse original + retrieved
        fused = self.fusion(torch.cat([pooled_flat, retrieved], dim=1))

        # SH branch on fused features (re-use audio_proj but feed fused through it)
        # We bypass the pool+flatten in audio_proj since we already have flat features
        # Replicate the audio_proj logic: it expects pooled input
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

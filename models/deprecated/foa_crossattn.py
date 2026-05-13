"""FOA Cross-Attention Bridge: cross-attention between encoder bottleneck and SH branch."""

import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_foa import AudioDepthFOAGenerator, sh_basis_erp, DeepScaleShift


class CrossAttentionBridge(nn.Module):
    """Cross-attention where SH branch features query encoder features."""

    def __init__(self, feat_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        assert feat_dim % num_heads == 0

        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_kv = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        query: (B, D) -- SH branch features
        key_value: (B, D, H, W) -- encoder spatial features
        Returns: attended features (B, D), attention weights (B, num_heads, 1, HW)
        """
        B, D, H, W = key_value.shape
        # Flatten spatial dims for KV
        kv = key_value.view(B, D, H * W).permute(0, 2, 1).contiguous()  # (B, HW, D)
        q = query.unsqueeze(1).contiguous()  # (B, 1, D)

        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        Q = self.q_proj(q).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, heads, 1, HW)
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        out = torch.matmul(attn_weights_drop, V)  # (B, heads, 1, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        out = self.out_proj(out).squeeze(1)  # (B, D)

        return out, attn_weights.squeeze(2)  # (B, heads, HW)


class FiLMConditioner(nn.Module):
    """Modulate spatial features using a conditioning vector."""
    def __init__(self, cond_dim, feat_channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_channels * 2)
        # Initialize near identity: gamma≈1, beta≈0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.ones_(self.proj.bias[:feat_channels])

    def forward(self, feat, cond):
        """feat: (B, C, H, W), cond: (B, cond_dim)"""
        params = self.proj(cond)  # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta


class FOACrossAttnGenerator(AudioDepthFOAGenerator):
    """UNet FOA with cross-attention bridge between encoder bottleneck and SH branch."""

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
        num_heads = kwargs.get('cross_attn_heads', 8)
        self.cross_attn = CrossAttentionBridge(feat_dim, num_heads=num_heads)

        # Project proj_dim -> feat_dim for cross-attn query, and back
        self.query_up = nn.Linear(proj_dim, feat_dim)
        self.attn_down = nn.Linear(feat_dim, proj_dim)

        # Latent distribution heads for KL
        self.mu_head = nn.Linear(proj_dim, proj_dim)
        self.logvar_head = nn.Linear(proj_dim, proj_dim)

        # FiLM conditioning: inject FOA latent into decoder
        self.film = FiLMConditioner(proj_dim, feat_dim)

    def _kl_divergence(self, mu, logvar):
        """KL(N(mu, sigma) || N(0,1)). Returns a scalar tensor."""
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()  # guaranteed 0-dim scalar

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_hist_maps=False):
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        # SH branch with cross-attention
        pooled = self.pool(bottleneck)
        foa_latent_det = self.audio_proj(pooled)

        # Cross-attention: SH features query spatial encoder features
        query_expanded = self.query_up(foa_latent_det)  # (B, feat_dim)
        attended, _attn_w = self.cross_attn(query_expanded, bottleneck)
        attended_proj = self.attn_down(attended)  # (B, proj_dim)
        foa_latent_enhanced = foa_latent_det + attended_proj

        # Latent distribution for KL
        mu = self.mu_head(foa_latent_enhanced)
        logvar = self.logvar_head(foa_latent_enhanced)
        kl_loss = self._kl_divergence(mu, logvar)

        if self.training:
            foa_latent = self._reparameterize(mu, logvar)
        else:
            foa_latent = mu

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

"""FOA Channel Attention (SE): Squeeze-and-Excitation on bottleneck and skip connections."""

import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_foa import AudioDepthFOAGenerator, sh_basis_erp, DeepScaleShift


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: recalibrated x, channel attention distribution (B, C)
        """
        B, C, H, W = x.shape
        s = self.squeeze(x).view(B, C)
        e = self.excitation(s)  # (B, C) -- logits
        attn = torch.sigmoid(e)
        out = x * attn.view(B, C, 1, 1)
        return out, attn


class FOAChannelAttnGenerator(AudioDepthFOAGenerator):
    """UNet FOA with SE channel attention on bottleneck and decoder skip connections."""

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
        reduction = kwargs.get('se_reduction', 16)

        # SE on bottleneck
        self.bottleneck_se = SqueezeExcitation(feat_dim, reduction)

        # SE on each skip connection (encoder features used in decoder)
        # enc_features has num_downs-1 levels (enc0 + encoders)
        # Build channel list matching encoder outputs
        skip_channels = [ngf]
        in_ch = ngf
        for i in range(1, num_downs - 1):
            out_ch = min(ngf * (2 ** i), ngf * 8)
            skip_channels.append(out_ch)
            in_ch = out_ch

        self.skip_se = nn.ModuleList([
            SqueezeExcitation(c, reduction) for c in skip_channels
        ])

    def _channel_kl_loss(self, attn_list):
        """KL divergence on channel attention distributions.

        Each attn is sigmoid output in [0,1]. We treat them as Bernoulli
        parameters and compute KL(Bernoulli(attn) || Bernoulli(0.5))
        to encourage diversity (not all on or all off).
        """
        total_kl = torch.tensor(0.0, device=attn_list[0].device)
        for attn in attn_list:
            # KL(Bernoulli(p) || Bernoulli(0.5)) = p*log(2p) + (1-p)*log(2(1-p))
            p = torch.clamp(attn, 1e-6, 1 - 1e-6)
            kl = p * torch.log(2 * p) + (1 - p) * torch.log(2 * (1 - p))
            total_kl = total_kl + kl.mean()
        return total_kl / len(attn_list)

    def forward(self, x, return_hist_maps=False):
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        # SE on bottleneck
        all_attns = []
        bottleneck_se, bn_attn = self.bottleneck_se(bottleneck)
        all_attns.append(bn_attn)

        # SH branch (from SE-enhanced bottleneck)
        pooled = self.pool(bottleneck_se)
        foa_latent = self.audio_proj(pooled)
        pred_foa = self.foa_head(foa_latent)
        pred_hoa = self.hoa_head(foa_latent)
        pred_sh = torch.cat([pred_foa, pred_hoa], dim=1)

        # Apply SE to skip connections
        se_enc_features = []
        for feat, se in zip(enc_features, self.skip_se):
            se_feat, se_attn = se(feat)
            se_enc_features.append(se_feat)
            all_attns.append(se_attn)

        # KL loss on all channel attention distributions
        kl_loss = self._channel_kl_loss(all_attns)

        # Decode using SE-enhanced features
        enc_reversed = se_enc_features[::-1]
        h = self.decoders[0](bottleneck_se)
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

"""
n3 hybrid — MultiScale SH + EnergyAttention (report_d exp213).

Combines:
    * Multi-scale SH heads at enc[2, 4, 6] + bottleneck  (from n3_multiscale_sh)
    * Energy attention on decoder[4]                     (from n3_energy_attn)

Multi-scale SH supervises directional information from multiple encoder
depths; EnergyAttn applies a learned spatial attention on the late decoder.
The two mechanisms read from different feature stacks (SH from encoder,
Energy from bottleneck) and inject at different depths (SH as loss,
Energy as multiplicative attention) — fully orthogonal.

forward() returns:
    pred_depth      -- (B, 1, H, W)
    pred_sh         -- (B, sh_dim)   fused SH prediction
    pred_sh_scales  -- list of per-scale SH predictions
    pred_energy     -- (B, 1, H, W)
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.foa_0415_v1 import SHHead
from models.n3_0417.n3_energy_attn import EnergyHead


class N3MSSHEnergyAttnGenerator(nn.Module):
    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256
    SH_TAP_INDICES = [2, 4, 6]

    def __init__(self, cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, sh_dim: int = None, head_hidden: int = None,
                 **_unused):
        super().__init__()

        self.num_downs = num_downs
        self.ngf = ngf
        self.depth_norm = cfg.dataset.depth_norm
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        head_hidden = int(head_hidden if head_hidden is not None else self.DEFAULT_HEAD_HIDDEN)

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        use_bias = False

        # ---- Shared encoder ----
        self.enc0 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        encoder_layers = []
        in_ch = ngf
        enc_channels = [ngf]
        for i in range(1, num_downs - 1):
            out_ch = min(ngf * (2 ** i), ngf * 8)
            encoder_layers.append(nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=use_bias),
                norm_layer(out_ch),
            ))
            enc_channels.append(out_ch)
            in_ch = out_ch
        self.encoders = nn.ModuleList(encoder_layers)
        self.enc_inner = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_ch, ngf * 8, 4, 2, 1),
        )

        feat_dim = ngf * 8

        # ---- Multi-scale SH heads + fusion ----
        self.sh_heads = nn.ModuleList()
        for idx in self.SH_TAP_INDICES:
            self.sh_heads.append(SHHead(in_channels=enc_channels[idx],
                                        sh_dim=self.sh_dim, hidden=head_hidden))
        self.sh_heads.append(SHHead(in_channels=feat_dim,
                                    sh_dim=self.sh_dim, hidden=head_hidden))
        num_heads = len(self.SH_TAP_INDICES) + 1
        self.sh_mix = nn.Linear(self.sh_dim * num_heads, self.sh_dim)

        # ---- Energy head ----
        self.energy_head = EnergyHead(ngf)

        # ---- Depth decoder (identical to v1) ----
        decoder_layers = []
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 8),
        ))
        for _ in range(num_downs - 5):
            layers = [
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=use_bias),
                norm_layer(ngf * 8),
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            decoder_layers.append(nn.Sequential(*layers))
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 4),
        ))
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 2),
        ))
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=use_bias),
            norm_layer(ngf),
        ))
        self.decoders = nn.ModuleList(decoder_layers)

        if self.depth_norm:
            self.dec_outer = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.dec_outer = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1),
                nn.ReLU(),
            )

    def forward(self, x, **_unused):
        input_h, input_w = x.shape[2], x.shape[3]

        # ---- Encode ----
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        # ---- Multi-scale SH ----
        pred_sh_scales = []
        for head_idx, tap_idx in enumerate(self.SH_TAP_INDICES):
            pred_sh_scales.append(self.sh_heads[head_idx](enc_features[tap_idx]))
        pred_sh_scales.append(self.sh_heads[-1](bottleneck))
        sh_cat = torch.cat(pred_sh_scales, dim=1)
        pred_sh = self.sh_mix(sh_cat)

        # ---- Energy head ----
        pred_energy = self.energy_head(bottleneck.clone(), input_h, input_w)

        # ---- Decode with Energy attn at decoder[4] ----
        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)
        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
            if i == 3:
                energy_attn = F.interpolate(pred_energy, size=h.shape[2:],
                                            mode='bilinear', align_corners=False)
                h = h.clone() * (1 + energy_attn)
        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

        return {
            "pred_depth": pred_depth,
            "pred_sh": pred_sh,
            "pred_sh_scales": pred_sh_scales,
            "pred_energy": pred_energy,
        }

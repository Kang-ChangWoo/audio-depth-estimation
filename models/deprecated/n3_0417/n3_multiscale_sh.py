"""
n3 variant — Variant 3: Multi-Scale SH Heads.

Same UNet encoder/decoder as v1, but SH predictions are tapped at
multiple encoder stages (indices 2, 4, 6 of enc_features plus
bottleneck) and fused via a learnable linear mix.

forward() returns:
    pred_depth      -- (B, 1, H, W)
    pred_sh         -- (B, sh_dim)  fused prediction (used by training)
    pred_sh_scales  -- list of 4 per-scale SH predictions
"""

import functools
import torch
import torch.nn as nn

from models.foa_0415_v1 import SHHead


class N3MultiScaleSHGenerator(nn.Module):
    """Pix2Pix-style UNet with multi-scale SH heads."""

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    # Indices into enc_features (0 = enc0 output) to tap for SH heads
    SH_TAP_INDICES = [2, 4, 6]

    def __init__(self, cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, sh_dim: int = None, head_hidden: int = None,
                 **_unused):
        super().__init__()

        self.num_downs = num_downs
        self.depth_norm = cfg.dataset.depth_norm
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        head_hidden = int(head_hidden if head_hidden is not None else self.DEFAULT_HEAD_HIDDEN)

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        use_bias = False

        # ---- Shared encoder ----
        self.enc0 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        encoder_layers = []
        in_ch = ngf
        # Track channel dims for each enc_feature index
        enc_channels = [ngf]  # index 0 = enc0 output
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

        # ---- Multi-scale SH heads ----
        # One head per tap index + one on bottleneck
        self.sh_heads = nn.ModuleList()
        for idx in self.SH_TAP_INDICES:
            self.sh_heads.append(SHHead(in_channels=enc_channels[idx],
                                        sh_dim=self.sh_dim, hidden=head_hidden))
        # Bottleneck head (last)
        self.sh_heads.append(SHHead(in_channels=feat_dim,
                                    sh_dim=self.sh_dim, hidden=head_hidden))

        # Learnable fusion: concatenate all 4 SH predictions -> fused
        num_heads = len(self.SH_TAP_INDICES) + 1  # taps + bottleneck
        self.sh_mix = nn.Linear(self.sh_dim * num_heads, self.sh_dim)

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
        # ---- Encode ----
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        # ---- Multi-scale SH predictions ----
        pred_sh_scales = []
        for head_idx, tap_idx in enumerate(self.SH_TAP_INDICES):
            pred_sh_scales.append(self.sh_heads[head_idx](enc_features[tap_idx]))
        # Bottleneck head is last in sh_heads
        pred_sh_scales.append(self.sh_heads[-1](bottleneck))

        # Fuse via learned linear mix
        sh_cat = torch.cat(pred_sh_scales, dim=1)  # (B, sh_dim * 4)
        pred_sh = self.sh_mix(sh_cat)               # (B, sh_dim)

        # ---- Decode (identical to v1) ----
        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)
        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

        return {
            "pred_depth": pred_depth,
            "pred_sh": pred_sh,
            "pred_sh_scales": pred_sh_scales,
        }

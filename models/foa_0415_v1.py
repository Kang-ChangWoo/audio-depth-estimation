"""
foa_0415_v1 — Variant 1: Auxiliary SH Head Only (base).

Shared UNet encoder -> bottleneck:
    branch A: depth decoder -> pred_depth
    branch B: aux SH head   -> pred_sh

SH prediction is NOT injected back into the depth decoder.

Losses used at the training site (see train.py::_train_step_foa_0415):
    L = L_berhu + lambda_si * L_silog + lambda_sh * L_sh
    L_sh = L1 between predicted SH and GT SH (FOA, 4 dims)

sh_dim is configurable. When sh_dim > 4 only the first 4 dims are
supervised (the dataset provides 4-dim FOA ground truth). Higher dims
remain free for later extension into variants 2-5.

The 5 files (foa_0415_v{1..5}.py) are placeholder siblings that all
implement Variant 1 today; each will later be forked into:
    v1 -> Variant 1: Auxiliary SH Head Only (current)
    v2 -> Variant 2: SH latent + FiLM conditioning
    v3 -> Variant 3: band-wise SH prediction
    v4 -> Variant 4: bottleneck cross-attention
    v5 -> Variant 5: multi-scale SH conditioning
"""

import functools
import torch
import torch.nn as nn


class SHHead(nn.Module):
    """AdaptiveAvgPool2d(1) -> Flatten -> Linear -> GELU -> Linear."""

    def __init__(self, in_channels: int, sh_dim: int, hidden: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, sh_dim)

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        x = self.pool(bottleneck)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class FOA0415V1Generator(nn.Module):
    """Pix2Pix-style UNet encoder/decoder with an auxiliary SH head."""

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, sh_dim: int = None, head_hidden: int = None,
                 # Ignored kwargs kept for build_model compatibility with other FOA models.
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
        for i in range(1, num_downs - 1):
            out_ch = min(ngf * (2 ** i), ngf * 8)
            encoder_layers.append(nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=use_bias),
                norm_layer(out_ch),
            ))
            in_ch = out_ch
        self.encoders = nn.ModuleList(encoder_layers)
        self.enc_inner = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_ch, ngf * 8, 4, 2, 1),
        )

        feat_dim = ngf * 8

        # ---- Auxiliary SH head (Variant 1: off bottleneck, no decoder injection) ----
        self.sh_head = SHHead(in_channels=feat_dim, sh_dim=self.sh_dim, hidden=head_hidden)

        # ---- Depth decoder ----
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
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        pred_sh = self.sh_head(bottleneck)

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
        }

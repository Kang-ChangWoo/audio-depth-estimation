"""
n3 variant — Variant 2: FiLM Conditioning.

Same UNet encoder/decoder as v1, but the SH head prediction is
projected through a FiLM layer (Feature-wise Linear Modulation) that
modulates the first decoder feature map.

    bottleneck --> SH head --> pred_sh
                          \--> FiLMProjector --> (gamma, beta)
    decoder[0](bottleneck) --> h * (1 + gamma) + beta --> rest of decoder
"""

import functools
import torch
import torch.nn as nn

from models.foa_0415_v1 import SHHead


class FiLMProjector(nn.Module):
    """Project SH embedding to FiLM parameters (gamma, beta)."""

    def __init__(self, sh_dim: int, film_hidden: int, feat_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sh_dim, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, 2 * feat_channels),
        )
        self.feat_channels = feat_channels

    def forward(self, sh: torch.Tensor):
        """Return (gamma, beta) each of shape (B, C)."""
        out = self.net(sh)  # (B, 2*C)
        gamma, beta = out.split(self.feat_channels, dim=1)
        return gamma, beta


class N3FiLMGenerator(nn.Module):
    """Pix2Pix-style UNet with FiLM conditioning from SH prediction."""

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, sh_dim: int = None, head_hidden: int = None,
                 film_hidden: int = 128, **_unused):
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

        # ---- Auxiliary SH head ----
        self.sh_head = SHHead(in_channels=feat_dim, sh_dim=self.sh_dim, hidden=head_hidden)

        # ---- FiLM projector ----
        self.film = FiLMProjector(sh_dim=self.sh_dim, film_hidden=film_hidden,
                                  feat_channels=feat_dim)

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
        # ---- Encode ----
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        # ---- SH head ----
        pred_sh = self.sh_head(bottleneck)

        # ---- FiLM modulation ----
        gamma, beta = self.film(pred_sh)  # (B, C)
        gamma = gamma[:, :, None, None]   # (B, C, 1, 1) for spatial broadcast
        beta = beta[:, :, None, None]

        # ---- Decode ----
        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)

        # Apply FiLM after first decoder step
        h = h * (1 + gamma) + beta

        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

        return {
            "pred_depth": pred_depth,
            "pred_sh": pred_sh,
        }

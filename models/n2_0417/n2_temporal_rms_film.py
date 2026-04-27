"""
N2 — Temporal RMS FiLM.

Like n3_film but the FiLM conditioning vector is the 12-dim temporal RMS
(3 bins × 4 FOA channels) instead of the 4-dim global RMS. This provides
the decoder with richer temporal-directional context: which directions carry
energy at which latency.

The SH head also predicts 12 dims and is supervised against temporal_rms.

forward(audio, temporal_rms=None) returns {pred_depth, pred_sh}.
"""

import functools
import torch
import torch.nn as nn

from models.foa_0415_v1 import SHHead


class FiLMProjector(nn.Module):
    def __init__(self, cond_dim, film_hidden, feat_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, film_hidden),
            nn.ReLU(True),
            nn.Linear(film_hidden, feat_channels * 2),
        )

    def forward(self, h, cond):
        gb = self.net(cond).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = gb.chunk(2, dim=1)
        return h * (1.0 + gamma) + beta


class N2TemporalRMSFiLMGenerator(nn.Module):
    """UNet with FiLM from predicted 12-dim temporal RMS."""

    DEFAULT_SH_DIM = 12
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, sh_dim=None, head_hidden=None, **_):
        super().__init__()
        self.num_downs = num_downs
        self.depth_norm = cfg.dataset.depth_norm
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        head_hidden = int(head_hidden if head_hidden is not None else self.DEFAULT_HEAD_HIDDEN)

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True,
                                       track_running_stats=True)
        use_bias = False

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
        self.sh_head = SHHead(in_channels=feat_dim, sh_dim=self.sh_dim,
                              hidden=head_hidden)

        self.film = FiLMProjector(cond_dim=self.sh_dim, film_hidden=128,
                                  feat_channels=feat_dim)

        decoder_layers = []
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 8),
        ))
        for _ in range(num_downs - 5):
            layers = [nn.ReLU(True),
                      nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1,
                                         bias=use_bias),
                      norm_layer(ngf * 8)]
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

    def forward(self, audio, temporal_rms=None, **_):
        enc_features = []
        h = self.enc0(audio)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        pred_sh = self.sh_head(bottleneck)

        cond = pred_sh if temporal_rms is None else temporal_rms
        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)
        h = self.film(h, cond)

        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

        return {"pred_depth": pred_depth, "pred_sh": pred_sh}

"""
N2 — FOA STFT FiLM.

The FOA spectrogram (4ch) is processed by a lightweight ConvNet to produce
a compact feature vector, which is then injected as FiLM conditioning at
multiple decoder levels.

This tests direct ambisonics feature learning — the FOA spectrogram is never
reduced to a hand-crafted statistic (RMS, energy map); the ConvNet learns
its own features.

forward(audio, foa_spec) returns {pred_depth, pred_sh}.
"""

import functools
import torch
import torch.nn as nn

from models.foa_0415_v1 import SHHead


class FOAFeatureExtractor(nn.Module):
    """Small ConvNet: (B, 4, H, W) → (B, feat_dim)."""

    def __init__(self, feat_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1),  nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class FiLMBlock(nn.Module):
    """Projects a feature vector to (gamma, beta) for a given channel dim."""

    def __init__(self, cond_dim, feat_channels):
        super().__init__()
        self.fc = nn.Linear(cond_dim, feat_channels * 2)

    def forward(self, h, cond):
        gb = self.fc(cond).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = gb.chunk(2, dim=1)
        return h * (1.0 + gamma) + beta


class N2FOASTFTFiLMGenerator(nn.Module):
    """UNet with FiLM conditioning from a learned FOA spectrogram feature."""

    DEFAULT_SH_DIM = 4
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
        self.foa_extractor = FOAFeatureExtractor(feat_dim=feat_dim)

        self.sh_head = SHHead(in_channels=feat_dim, sh_dim=self.sh_dim,
                              hidden=head_hidden)

        self.film_blocks = nn.ModuleList([
            FiLMBlock(feat_dim, ngf * 8),
            FiLMBlock(feat_dim, ngf * 4),
            FiLMBlock(feat_dim, ngf * 2),
        ])
        self.film_inject_indices = [0, 4, 5]

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

    def forward(self, audio, foa_spec=None, **_):
        enc_features = []
        h = self.enc0(audio)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        if foa_spec is not None:
            foa_feat = self.foa_extractor(foa_spec)
        else:
            foa_feat = torch.zeros(audio.shape[0], bottleneck.shape[1],
                                   device=audio.device)

        pred_sh = self.sh_head(bottleneck)

        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)

        film_idx = 0
        if film_idx < 3 and 0 in self.film_inject_indices:
            h = self.film_blocks[film_idx](h, foa_feat)
            film_idx += 1

        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
            dec_idx = i + 1
            if film_idx < 3 and dec_idx in self.film_inject_indices:
                h = self.film_blocks[film_idx](h, foa_feat)
                film_idx += 1

        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

        return {"pred_depth": pred_depth, "pred_sh": pred_sh}

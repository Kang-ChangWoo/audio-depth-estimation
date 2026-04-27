"""
N2 — Dual Encoder (binaural + FOA spectrogram).

Two separate encoders process the binaural spectrogram (2ch) and the FOA
spectrogram (4ch) independently. Their bottleneck features are concatenated
and linearly projected, then fed to a single decoder with skip connections
from the binaural encoder only.

forward(audio, foa_spec) returns {pred_depth, pred_sh}.
"""

import functools
import torch
import torch.nn as nn

from models.foa_0415_v1 import SHHead


class N2DualEncGenerator(nn.Module):
    """Dual-encoder UNet: binaural (2ch) + FOA spec (4ch)."""

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

        def make_encoder(in_nc):
            enc0 = nn.Conv2d(in_nc, ngf, 4, 2, 1)
            layers = []
            in_ch = ngf
            for i in range(1, num_downs - 1):
                out_ch = min(ngf * (2 ** i), ngf * 8)
                layers.append(nn.Sequential(
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=use_bias),
                    norm_layer(out_ch),
                ))
                in_ch = out_ch
            enc_inner = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_ch, ngf * 8, 4, 2, 1),
            )
            return enc0, nn.ModuleList(layers), enc_inner

        self.bin_enc0, self.bin_encoders, self.bin_enc_inner = make_encoder(2)
        self.foa_enc0, self.foa_encoders, self.foa_enc_inner = make_encoder(4)

        feat_dim = ngf * 8
        self.bottleneck_proj = nn.Linear(feat_dim * 2, feat_dim)

        self.sh_head = SHHead(in_channels=feat_dim, sh_dim=self.sh_dim,
                              hidden=head_hidden)

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

    def _run_encoder(self, x, enc0, encoders, enc_inner):
        feats = []
        h = enc0(x)
        feats.append(h)
        for enc in encoders:
            h = enc(h)
            feats.append(h)
        bn = enc_inner(h)
        return feats, bn

    def forward(self, audio, foa_spec=None, **_):
        bin_feats, bin_bn = self._run_encoder(
            audio, self.bin_enc0, self.bin_encoders, self.bin_enc_inner)

        if foa_spec is None:
            bottleneck = bin_bn
        else:
            _, foa_bn = self._run_encoder(
                foa_spec, self.foa_enc0, self.foa_encoders, self.foa_enc_inner)
            bn_cat = torch.cat([bin_bn, foa_bn], dim=1)
            B, C2, Hb, Wb = bn_cat.shape
            bn_flat = bn_cat.permute(0, 2, 3, 1).reshape(-1, C2)
            bn_fused = self.bottleneck_proj(bn_flat)
            bottleneck = bn_fused.reshape(B, Hb, Wb, C2 // 2).permute(0, 3, 1, 2)

        pred_sh = self.sh_head(bottleneck)

        enc_reversed = bin_feats[::-1]
        h = self.decoders[0](bottleneck)
        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

        return {"pred_depth": pred_depth, "pred_sh": pred_sh}

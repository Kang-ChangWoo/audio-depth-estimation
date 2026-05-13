"""
N2 — Temporal Energy Attention.

Same UNet as v1, plus 3 lightweight energy heads that predict spatial energy
maps for each temporal bin (early / mid / late).  During decoding, each
predicted map is resized and applied as residual multiplicative attention at
a specific decoder level:

    bin 0 (early, direct)  → decoder[4]  (32×64)
    bin 1 (mid, reverb)    → decoder[5]  (64×128)
    bin 2 (late, diffuse)  → decoder[6]  (128×256)

The predicted maps are also returned for explicit supervision against GT
temporal energy maps (computed in dataset_n2.py).

forward() returns: {pred_depth, pred_sh, pred_temporal_energies}
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.foa_0415_v1 import SHHead


class EnergyHead(nn.Module):
    """Bottleneck or decoder feature → predicted spatial energy map (B, 1, H_target, W_target)."""

    def __init__(self, in_channels, hidden=256):
        super().__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.spatial = nn.Sequential(
            nn.Linear(hidden, 16 * 32),
            nn.Unflatten(1, (1, 16, 32)),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat, target_h, target_w):
        x = self.net(feat)
        x = self.spatial(x)
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear',
                          align_corners=False)
        return self.refine(x)


class N2TemporalEnergyGenerator(nn.Module):
    """UNet + temporal energy attention at decoder levels."""

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    @staticmethod
    def _decoder_channels(num_downs, ngf):
        """Output channels at each decoder index (0=first after bottleneck)."""
        chs = [ngf * 8]
        for _ in range(num_downs - 5):
            chs.append(ngf * 8)
        chs.append(ngf * 4)
        chs.append(ngf * 2)
        chs.append(ngf)
        return chs

    def __init__(self, cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, sh_dim=None, head_hidden=None, **_):
        super().__init__()
        self.num_downs = num_downs
        self.depth_norm = cfg.dataset.depth_norm
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        head_hidden = int(head_hidden if head_hidden is not None else self.DEFAULT_HEAD_HIDDEN)

        n_bins = int(getattr(cfg.model, 'n_bins', 3))
        self.gain_mode = getattr(cfg.model, 'gain_mode', 'monotone')
        self.cond_source = getattr(cfg.model, 'cond_source', 'bottleneck')

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

        if n_bins == 4:
            self.attn_inject_indices = [3, 4, 5, 6]
        else:
            self.attn_inject_indices = [4, 5, 6][:n_bins]

        if self.cond_source == 'decoder_level':
            dec_channels = self._decoder_channels(num_downs, ngf)
            self.energy_heads = nn.ModuleList([
                EnergyHead(dec_channels[idx], hidden=256)
                for idx in self.attn_inject_indices
            ])
        else:
            self.energy_heads = nn.ModuleList([
                EnergyHead(feat_dim, hidden=256) for _ in range(n_bins)
            ])

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

    def forward(self, x, **_):
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

        energy_maps = []
        attn_idx = 0
        n_bins = len(self.energy_heads)

        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
            dec_idx = i + 1
            if attn_idx < n_bins and dec_idx == self.attn_inject_indices[attn_idx]:
                cond_input = h if self.cond_source == 'decoder_level' else bottleneck
                emap = self.energy_heads[attn_idx](
                    cond_input, h.shape[2], h.shape[3])
                energy_maps.append(emap)
                if self.gain_mode == 'signed':
                    h = h * (1.0 + 2.0 * emap - 1.0)
                else:
                    h = h * (1.0 + emap)
                attn_idx += 1

        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

        target_h, target_w = pred_depth.shape[2], pred_depth.shape[3]
        resized = [F.interpolate(e, size=(target_h, target_w),
                                 mode='bilinear', align_corners=False)
                   for e in energy_maps]
        pred_temporal_energies = torch.cat(resized, dim=1)

        return {
            "pred_depth": pred_depth,
            "pred_sh": pred_sh,
            "pred_temporal_energies": pred_temporal_energies,
        }

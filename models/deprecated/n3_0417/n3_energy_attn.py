"""
n3 variant — Variant 4: Predicted Energy Map Attention.

Same UNet encoder/decoder as v1, but adds an EnergyHead that predicts
a spatial energy-like attention map from the bottleneck. This map is
used as multiplicative attention on a late decoder feature.

forward() returns:
    pred_depth   -- (B, 1, H, W)
    pred_sh      -- (B, sh_dim)
    pred_energy  -- (B, 1, H, W)  predicted energy map (can be supervised)
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.foa_0415_v1 import SHHead


class EnergyHead(nn.Module):
    """Small decoder: bottleneck -> predicted (B, 1, H, W) energy map."""

    def __init__(self, ngf: int):
        super().__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            norm_layer(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, bottleneck: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        x = self.net(bottleneck)
        # Resize to match the full input spatial resolution
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return x


class N3EnergyAttnGenerator(nn.Module):
    """Pix2Pix-style UNet with predicted energy map attention."""

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

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

        # ---- Energy head ----
        self.energy_head = EnergyHead(ngf)

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
        input_h, input_w = x.shape[2], x.shape[3]

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

        # ---- Energy head (clone bottleneck to avoid in-place ReLU conflict) ----
        pred_energy = self.energy_head(bottleneck.clone(), input_h, input_w)

        # ---- Decode with energy attention ----
        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)
        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)

            # After the 5th decoder step (index 4, which outputs ngf*2 features),
            # apply energy attention. The 5th step is decoders[4] which is called
            # when i == 3 (decoders[i+1] = decoders[4]).
            if i == 3:
                energy_attn = F.interpolate(pred_energy, size=h.shape[2:],
                                            mode='bilinear', align_corners=False)
                h = h.clone() * (1 + energy_attn)

        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

        return {
            "pred_depth": pred_depth,
            "pred_sh": pred_sh,
            "pred_energy": pred_energy,
        }

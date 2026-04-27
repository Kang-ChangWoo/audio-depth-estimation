"""
n3 variant — Variant 5: Temporal Windowed Input.

Same UNet encoder/decoder as v1, but the input spectrogram is split
into 3 overlapping temporal windows. Each window is zero-padded to
the original size and passed through a SHARED encoder. The 3 bottleneck
features are concatenated and projected to a single bottleneck. Skip
connections come from window 0 (the earliest / most complete window).

forward() returns same dict as v1.
"""

import functools
import torch
import torch.nn as nn

from models.foa_0415_v1 import SHHead


class N3TemporalWindowGenerator(nn.Module):
    """Pix2Pix-style UNet with temporal windowed input processing."""

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

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

        # ---- Bottleneck fusion: 3 windows concatenated then projected ----
        self.bottleneck_proj = nn.Linear(feat_dim * 3, feat_dim)

        # ---- Auxiliary SH head (on fused bottleneck) ----
        self.sh_head = SHHead(in_channels=feat_dim, sh_dim=self.sh_dim, hidden=head_hidden)

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

    def _encode(self, x):
        """Run shared encoder, return (enc_features, bottleneck)."""
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)
        return enc_features, bottleneck

    @staticmethod
    def _make_windows(x):
        """Split input (B, 2, 256, T=512) into 3 overlapping temporal windows,
        each zero-padded back to (B, 2, 256, 512)."""
        T = x.shape[3]  # 512

        # Window 0: early [0, 170)  — pad right
        w0 = x[:, :, :, :170]
        w0 = torch.nn.functional.pad(w0, (0, T - 170))  # right-pad

        # Window 1: mid [85, 341)  — pad left and right
        w1 = x[:, :, :, 85:341]
        w1 = torch.nn.functional.pad(w1, (85, T - 341))  # left 85, right 171

        # Window 2: late [256, 512)  — pad left
        w2 = x[:, :, :, 256:]
        w2 = torch.nn.functional.pad(w2, (256, 0))  # left-pad

        return [w0, w1, w2]

    def forward(self, x, **_unused):
        # ---- Temporal windowing ----
        windows = self._make_windows(x)

        # ---- Shared encoder on each window ----
        all_enc_features = []
        all_bottlenecks = []
        for w in windows:
            enc_feats, bn = self._encode(w)
            all_enc_features.append(enc_feats)
            all_bottlenecks.append(bn)

        # ---- Fuse bottlenecks ----
        # Each bottleneck is (B, C, H_bn, W_bn). Concatenate along channel,
        # project per spatial location via Linear.
        bn_cat = torch.cat(all_bottlenecks, dim=1)  # (B, C*3, H_bn, W_bn)
        B, C3, Hb, Wb = bn_cat.shape
        # Reshape to (B*Hb*Wb, C*3) for linear, then back
        bn_flat = bn_cat.permute(0, 2, 3, 1).reshape(-1, C3)
        bn_fused = self.bottleneck_proj(bn_flat)
        feat_dim = C3 // 3
        bottleneck = bn_fused.reshape(B, Hb, Wb, feat_dim).permute(0, 3, 1, 2)

        # ---- SH head on fused bottleneck ----
        pred_sh = self.sh_head(bottleneck)

        # ---- Decode using skip connections from window 0 ----
        enc_features = all_enc_features[0]
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

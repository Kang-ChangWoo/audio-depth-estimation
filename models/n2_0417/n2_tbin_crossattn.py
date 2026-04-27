"""N2 E8: Temporal Bin Cross-Attention — most expressive temporal variant.

Binaural spectrogram goes through the main UNet encoder.
Each temporal energy map bin goes through a shared lightweight conv encoder.
Cross-attention at the bottleneck: binaural features (Q) attend to per-bin
energy features (K, V).

Tests whether explicit temporal factoring of ambisonics spatial energy,
combined with learned attention over time bins, captures depth-relevant
directional structure better than concatenation or FiLM.
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..foa_0415_v1 import SHHead


class TemporalBinEncoder(nn.Module):
    """Shared lightweight encoder for a single energy map (1, H, W) -> (C, h, w)."""

    def __init__(self, out_channels=128):
        super().__init__()
        norm = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            norm(32), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            norm(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, out_channels, 4, 2, 1),
            norm(out_channels), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((4, 8)),
        )

    def forward(self, x):
        return self.net(x)


class TemporalCrossAttention(nn.Module):
    """Cross-attention: binaural bottleneck (Q) attends to temporal bin features (K, V)."""

    def __init__(self, d_model=512, n_heads=4, bin_dim=128):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(bin_dim, d_model)
        self.v_proj = nn.Linear(bin_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, kv_list):
        """
        query: (B, C, h, w) — binaural bottleneck
        kv_list: list of K tensors, each (B, bin_dim, h_bin, w_bin)
        Returns: (B, C, h, w) — residual-added output
        """
        B, C, h, w = query.shape
        q = query.permute(0, 2, 3, 1).reshape(B, h * w, C)
        q = self.q_proj(q)

        kvs = []
        for kv in kv_list:
            B2, Ck, hk, wk = kv.shape
            kv_flat = kv.permute(0, 2, 3, 1).reshape(B, hk * wk, Ck)
            kvs.append(kv_flat)
        kv_cat = torch.cat(kvs, dim=1)

        k = self.k_proj(kv_cat)
        v = self.v_proj(kv_cat)

        q = q.view(B, h * w, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, h * w, C)
        out = self.out_proj(out)

        residual = query.permute(0, 2, 3, 1).reshape(B, h * w, C)
        out = self.norm(out + residual)
        return out.reshape(B, h, w, C).permute(0, 3, 1, 2)


class N2TBinCrossAttnGenerator(nn.Module):
    """UNet with temporal-bin cross-attention at the bottleneck."""

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

    def __init__(self, cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, sh_dim=None, head_hidden=None,
                 n_bins=3, bin_enc_dim=128, n_heads=4, **_unused):
        super().__init__()
        self.num_downs = num_downs
        self.depth_norm = cfg.dataset.depth_norm
        self.sh_dim = int(sh_dim if sh_dim is not None else self.DEFAULT_SH_DIM)
        head_hidden = int(head_hidden if head_hidden is not None else self.DEFAULT_HEAD_HIDDEN)
        self.n_bins = n_bins

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        use_bias = False
        feat_dim = ngf * 8

        # Main UNet encoder
        self.enc0 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        encoder_layers = []
        in_ch = ngf
        for i in range(1, num_downs - 1):
            out_ch = min(ngf * (2 ** i), feat_dim)
            encoder_layers.append(nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=use_bias),
                norm_layer(out_ch),
            ))
            in_ch = out_ch
        self.encoders = nn.ModuleList(encoder_layers)
        self.enc_inner = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_ch, feat_dim, 4, 2, 1),
        )

        # Temporal bin encoder (shared across bins)
        self.bin_encoder = TemporalBinEncoder(out_channels=bin_enc_dim)

        # Cross-attention
        self.cross_attn = TemporalCrossAttention(
            d_model=feat_dim, n_heads=n_heads, bin_dim=bin_enc_dim)

        # SH head
        self.sh_head = SHHead(in_channels=feat_dim, sh_dim=self.sh_dim, hidden=head_hidden)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(feat_dim, feat_dim, 4, 2, 1, bias=use_bias),
            norm_layer(feat_dim),
        ))
        for _ in range(num_downs - 5):
            layers = [
                nn.ReLU(True),
                nn.ConvTranspose2d(feat_dim * 2, feat_dim, 4, 2, 1, bias=use_bias),
                norm_layer(feat_dim),
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            decoder_layers.append(nn.Sequential(*layers))
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(feat_dim * 2, ngf * 4, 4, 2, 1, bias=use_bias),
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

    def forward(self, x, temporal_energies=None, **_unused):
        # Encode binaural
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        # Cross-attend to temporal bins if available
        if temporal_energies is not None:
            bin_feats = []
            for k in range(temporal_energies.shape[1]):
                bin_input = temporal_energies[:, k:k+1]
                bin_feats.append(self.bin_encoder(bin_input))
            bottleneck = self.cross_attn(bottleneck, bin_feats)

        pred_sh = self.sh_head(bottleneck)

        # Decode
        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)
        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

        return {"pred_depth": pred_depth, "pred_sh": pred_sh}

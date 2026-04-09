"""AudioDepthFOA: UNet encoder-decoder with SH auxiliary branch."""

import math
import functools

import numpy as np
import torch
import torch.nn as nn

from data.sh_basis import _acn_to_nm, _sn3d_norm, _real_sh_sn3d_np


def sh_basis_erp(max_order, H, W, dtype=torch.float32):
    """Compute real SH basis functions up to given order on ERP grid.
    Returns tensor of shape [(max_order+1)^2, H, W]."""
    n_ch = (max_order + 1) ** 2
    theta = np.linspace(0, np.pi, H)
    phi = np.linspace(-np.pi, np.pi, W)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    elevation = np.pi / 2 - theta_grid
    azimuth = phi_grid

    basis = np.zeros((n_ch, H, W), dtype=np.float64)
    for q in range(n_ch):
        basis[q] = _real_sh_sn3d_np(q, elevation, azimuth)
    return torch.from_numpy(basis).to(dtype)


class DeepScaleShift(nn.Module):
    """MLP-based per-channel affine transform with residual + gating."""

    def __init__(self, n_channels=36, hidden_dim=256, n_hidden_layers=4, dropout=0.1):
        super().__init__()
        layers = [nn.LayerNorm(n_channels)]
        in_dim = n_channels
        for i in range(n_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            if dropout > 0 and i < n_hidden_layers - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_channels))
        self.mlp = nn.Sequential(*layers)

        self.gamma = nn.Parameter(torch.ones(n_channels))
        self.beta = nn.Parameter(torch.zeros(n_channels))
        self.gate = nn.Parameter(torch.zeros(n_channels))
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = x * self.gamma.unsqueeze(0) + self.beta.unsqueeze(0)
        mlp_out = self.mlp(x)
        alpha = torch.sigmoid(self.gate).unsqueeze(0)
        return (1 - alpha) * residual + alpha * mlp_out


class AudioDepthFOAGenerator(nn.Module):
    """UNet encoder-decoder with SH auxiliary branch."""

    def __init__(self, cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, proj_dim=128, foa_dim=4, sh_order=5,
                 scale_shift_hidden=256, scale_shift_layers=4,
                 H_erp=256, W_erp=512):
        super().__init__()
        self.num_downs = num_downs
        self.depth_norm = cfg.dataset.depth_norm
        self.sh_order = sh_order
        self.sh_dim = (sh_order + 1) ** 2

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        use_bias = False

        basis = sh_basis_erp(sh_order, H_erp, W_erp, dtype=torch.float32)
        self.register_buffer("sh_basis", basis, persistent=False)

        self.scale_shift = DeepScaleShift(
            n_channels=self.sh_dim,
            hidden_dim=scale_shift_hidden,
            n_hidden_layers=scale_shift_layers,
        )

        # Encoder
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

        # SH Branch
        feat_dim = ngf * 8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.audio_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )
        self.foa_head = nn.Sequential(
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, foa_dim),
        )
        hoa_dim = self.sh_dim - foa_dim
        self.hoa_head = nn.Sequential(
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, hoa_dim),
        )

        # Decoder
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

    def reconstruct_from_coeffs(self, coeffs):
        return (coeffs[:, :, None, None] * self.sh_basis[None]).sum(dim=1, keepdim=True)

    def project_depth_to_sh(self, depth, eps=1e-6):
        basis = self.sh_basis
        d = depth[:, 0:1, :, :]
        H = basis.shape[1]
        theta = torch.linspace(0, math.pi, H, device=basis.device, dtype=basis.dtype)
        sin_weight = torch.sin(theta)[:, None]
        w = sin_weight[None, None, :, :]
        num = (w * d * basis[None]).sum(dim=(2, 3))
        den = (w * basis[None] ** 2).sum(dim=(2, 3)) + eps
        return num / den

    def forward(self, x, return_hist_maps=False):
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        # SH branch
        pooled = self.pool(bottleneck)
        foa_latent = self.audio_proj(pooled)
        pred_foa = self.foa_head(foa_latent)
        pred_hoa = self.hoa_head(foa_latent)
        pred_sh = torch.cat([pred_foa, pred_hoa], dim=1)

        # Decode
        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)
        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

        out = {
            "pred_depth": pred_depth,
            "foa_latent": foa_latent,
            "pred_foa": pred_foa,
            "pred_hoa": pred_hoa,
            "pred_sh": pred_sh,
        }

        if return_hist_maps:
            sh_aligned = self.scale_shift(pred_sh)
            energy_recon_aligned = self.reconstruct_from_coeffs(sh_aligned)
            depth_sh_coeffs = self.project_depth_to_sh(pred_depth)
            depth_sh = self.reconstruct_from_coeffs(depth_sh_coeffs)
            out["energy_recon_aligned"] = energy_recon_aligned
            out["depth_sh"] = depth_sh
            out["depth_sh_coeffs"] = depth_sh_coeffs
            out["sh_aligned"] = sh_aligned

        return out

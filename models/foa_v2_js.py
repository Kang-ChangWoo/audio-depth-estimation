"""FOAv2: improved FOA baseline with FiLM decoder conditioning and gradient consistency loss."""

import math
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .unet_foa import AudioDepthFOAGenerator, sh_basis_erp, DeepScaleShift
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
            nn.BatchNorm1d(proj_dim+foa_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim+foa_dim, hoa_dim),
        )

        # Img Decoder
        img_decoder_layers = []
        img_decoder_layers.append(nn.Sequential(
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
            img_decoder_layers.append(nn.Sequential(*layers))
        img_decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 4),
        ))
        img_decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 2),
        ))
        img_decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=use_bias),
            norm_layer(ngf),
        ))
        self.img_decoders = nn.ModuleList(img_decoder_layers)

        # Audio Decoder
        aud_decoder_layers = []
        aud_decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf * 8, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 8),
        ))
        for _ in range(num_downs - 5):
            layers = [
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=use_bias),
                norm_layer(ngf * 8),
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            aud_decoder_layers.append(nn.Sequential(*layers))
        aud_decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 4),
        ))
        aud_decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 2),
        ))
        aud_decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=use_bias),
            norm_layer(ngf),
        ))
        self.aud_decoders = nn.ModuleList(aud_decoder_layers)

        # Align layere
        align_layers = []
        align_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(2 * ngf * 8, ngf * 8, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        ))
        for _ in range(num_downs - 5):
            layers = [
                nn.ReLU(True),
                nn.Conv2d(ngf * 8 * 2, ngf * 8, kernel_size=3, padding=1, bias=use_bias),
                norm_layer(ngf * 8),
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            align_layers.append(nn.Sequential(*layers))
        align_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(ngf * 8 * 2, ngf * 8, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        ))
        align_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(ngf * 4 * 2, ngf * 4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
        ))
        align_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(ngf*2),
        ))
        self.aligner = nn.ModuleList(align_layers)

        self.energy_head = nn.ConvTranspose2d(ngf, 1, 4, 2, 1)

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





class FiLMConditioner(nn.Module):
    """Modulate spatial features using a conditioning vector."""
    def __init__(self, cond_dim, feat_channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.ones_(self.proj.bias[:feat_channels])

    def forward(self, feat, cond):
        """feat: (B, C, H, W), cond: (B, cond_dim)"""
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta


class FOAv2Generator(AudioDepthFOAGenerator):
    """Improved FOA baseline with FiLM decoder conditioning and FOA-depth gradient consistency."""

    def __init__(self, cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, proj_dim=128, foa_dim=4, sh_order=5,
                 scale_shift_hidden=256, scale_shift_layers=4,
                 H_erp=256, W_erp=512, **kwargs):
        super().__init__(
            cfg, input_nc=input_nc, output_nc=output_nc, num_downs=num_downs,
            ngf=ngf, use_dropout=use_dropout, proj_dim=proj_dim, foa_dim=foa_dim,
            sh_order=sh_order, scale_shift_hidden=scale_shift_hidden,
            scale_shift_layers=scale_shift_layers, H_erp=H_erp, W_erp=W_erp,
        )
        feat_dim = ngf * 8
        # FiLM conditioning: inject FOA latent into first decoder block
        self.film = FiLMConditioner(proj_dim, feat_dim)

    def _gradient_consistency(self, pred_depth, energy_recon):
        """Spatial gradient alignment between depth and FOA energy.
        Returns a scalar tensor in [0, 2] (0 = perfect alignment)."""
        dx_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                 dtype=pred_depth.dtype,
                                 device=pred_depth.device).view(1, 1, 3, 3)
        dy_kernel = dx_kernel.transpose(2, 3)

        d_dx = F.conv2d(pred_depth, dx_kernel, padding=1)
        d_dy = F.conv2d(pred_depth, dy_kernel, padding=1)
        e_dx = F.conv2d(energy_recon, dx_kernel, padding=1)
        e_dy = F.conv2d(energy_recon, dy_kernel, padding=1)

        d_grad = torch.cat([d_dx, d_dy], dim=1)  # (B, 2, H, W)
        e_grad = torch.cat([e_dx, e_dy], dim=1)

        # Cosine similarity of gradient fields
        cos = F.cosine_similarity(d_grad, e_grad, dim=1).mean()
        return 1.0 - cos  # 0 = perfect alignment

    def forward(self, x, return_hist_maps=False):
        enc_features = []
        h = self.enc0(x)    # Input shape: [256 x 512]
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)  # [B, 512, 1, 2]
        
        # SH branch
        pooled = self.pool(bottleneck)          # [B, 512, 1, 1]
        foa_latent = self.audio_proj(pooled)    # [B, 128]
        pred_foa = self.foa_head(foa_latent)
        pred_hoa = self.hoa_head(torch.cat([pred_foa, foa_latent], dim=1))
        pred_sh = torch.cat([pred_foa, pred_hoa], dim=1)
        
        # Decoder
        foa_latent = (foa_latent.unsqueeze(-1).unsqueeze(-1)).expand(-1, -1, -1, 2)
        enc_reversed = enc_features[::-1]

        h = self.img_decoders[0](bottleneck)
        a = self.aud_decoders[0](foa_latent.clone())
        
        for i in range(len(self.img_decoders)-1):
            align = torch.cat([h, a], dim=1)
            align = self.aligner[i+1](align)
            
            h = torch.cat([enc_reversed[i], h+align], dim=1)  # [B, C, H, W]
            h = self.img_decoders[i+1](h)
            a = self.aud_decoders[i+1](a+align)
        
        pred_energy = self.energy_head(a)

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
            # FOA-depth gradient consistency loss
            out["foa_depth_consistency"] = self._gradient_consistency(
                pred_depth, energy_recon_aligned)
        else:
            out["foa_depth_consistency"] = torch.tensor(
                0.0, device=pred_depth.device)

        return out

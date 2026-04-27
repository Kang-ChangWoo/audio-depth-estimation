"""FOAv2: improved FOA baseline with FiLM decoder conditioning and gradient consistency loss."""

import math
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .unet_foa import AudioDepthFOAGenerator, sh_basis_erp, DeepScaleShift
from data.sh_basis import _acn_to_nm, _sn3d_norm, _real_sh_sn3d_np
from timm.models.vision_transformer import _cfg
from typing import Any, Dict, Optional, Tuple
from .foa_js_swin import SwinTransformer
from functools import partial
from einops import rearrange


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


class AttentionLayer(nn.Module):
    def __init__(
        self,
        sink_dim: int,
        hidden_dim: int,
        source_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        pre_norm: bool = True,
        norm_layer=nn.LayerNorm,
        sink_competition: bool = False,
        qkv_bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.pre_norm = pre_norm
        assert (
            hidden_dim % num_heads
        ) == 0, "hidden_dim and num_heads are not divisible"
        self.scale = (hidden_dim // num_heads) ** -0.5
        self.num_heads = num_heads

        self.norm = norm_layer(sink_dim, eps=eps)
        self.norm_context = (
            norm_layer(source_dim, eps=eps) if source_dim is not None else None
        )

        self.to_q = nn.Linear(sink_dim, hidden_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(
            sink_dim if source_dim is None else source_dim,
            hidden_dim * 2,
            bias=qkv_bias,
        )
        self.to_out = nn.Linear(
            hidden_dim, sink_dim if output_dim is None else output_dim
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.sink_competition = sink_competition

    def forward(
        self, sink: torch.Tensor, source: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.pre_norm:
            sink = self.norm(sink)
            if source is not None:
                source = self.norm_context(source)

        q = self.to_q(sink)
        source = source if source is not None else sink
        k, v = self.to_kv(source).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.num_heads),
            (q, k, v),
        )

        similarity_matrix = torch.einsum("bid, bjd -> bij", q, k) * self.scale

        if self.sink_competition:
            attn = F.softmax(similarity_matrix, dim=-2) + self.eps
            attn = attn / torch.sum(attn, dim=(-1,), keepdim=True)
        else:
            attn = F.softmax(similarity_matrix, dim=-1)

        attn = self.dropout(attn)

        out = torch.einsum("bij, bjd -> bid", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.num_heads)
        out = self.to_out(out)
        if not self.pre_norm:
            out = self.norm(out)

        return out


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


''' Added by Syniez '''
class SH_Coeff_Extractor(nn.Module):
    def __init__(self):
        super(SH_Coeff_Extractor, self).__init__()
        self.dense1 = nn.Conv2d(128, 64, kernel_size=5, stride=2, bias=False)
        self.dense2 = nn.Conv2d(64, 32, kernel_size=5, stride=2, bias=False)
        self.dense3 = nn.Conv2d(32, 16, kernel_size=5, stride=2, bias=False)
        self.final_dense = nn.Linear(16*13*29, 4)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.dense1(x)))
        x = self.relu(self.bn2(self.dense2(x)))
        x = self.relu(self.bn3(self.dense3(x)))
        
        x = torch.flatten(x, 1)
        coeffs = self.final_dense(x)
        return coeffs


def swin_large_22k(pretrained: bool = True, **kwargs):
    if pretrained:
        pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dims=[192 * (2**i) for i in range(4)],
        num_heads=[6, 12, 24, 48],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 18, 2],
        drop_path_rate=0.2,
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model



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
            scale_shift_layers=scale_shift_layers, H_erp=H_erp, W_erp=W_erp,)

        feat_dim = ngf * 8

        self.SH_basis = sh_basis_erp(max_order=1, H=128, W=256)
        self.PE = nn.Parameter(torch.zeros(128, 128))
        self.MLP = nn.Sequential(nn.Linear(128, 256),
                                nn.GELU(),
                                nn.Linear(256, 128))
        self.depth_head = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 1, kernel_size=3, padding=1))

        self.patch_emb = nn.Conv2d(128, 128, kernel_size=16, stride=16, padding=0)
        self.SH_patch_emb = nn.Conv2d(4, 4, kernel_size=16, stride=16, padding=0)
        self.coeff_net = SH_Coeff_Extractor()
        self.Cross_Attn = AttentionLayer(sink_dim=128,
                                        hidden_dim=128,
                                        source_dim=128,
                                        output_dim=128,
                                        num_heads=1,
                                        dropout=0.0,
                                        pre_norm=True,
                                        sink_competition=True)

        self.Swin_encoder = swin_large_22k(pretrained=True)
        self.Deconv = nn.ModuleList()

        self.Deconv.append(nn.Sequential(nn.ConvTranspose2d(1536, 768, 4, 2, 1, bias=True),
                                            nn.BatchNorm2d(768),
                                            nn.ReLU(inplace=True)))
        for i in range(3):
            self.Deconv.append(nn.Sequential(nn.ConvTranspose2d(2 * 768 // (2**i), 768 // (2**(i+1)), 4, 2, 1, bias=True),
                                            nn.BatchNorm2d(768 // (2**(i+1))),
                                            nn.ReLU(inplace=True)))

        self.Deconv.append(nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=True))
        

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

        cos = F.cosine_similarity(d_grad, e_grad, dim=1).mean()
        return 1.0 - cos

    def invert_encoder_output_order(self, xs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return tuple(xs[::-1])

    def filter_decoder_relevant_resolutions(self, decoder_outputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return tuple(decoder_outputs[1:])

    def audio_enc(self, x):
        enc_features = []
        h = self.enc0(x)    # Input shape: [256 x 512]
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)  # [B, 512, 1, 2]

        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)
        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
        h = torch.cat([enc_reversed[-1], h], dim=1)
        return h

    def img_enc(self, x):
        encoder_outputs = self.Swin_encoder(x)
        feat = self.Deconv[0](encoder_outputs[-1])

        encoder_outputs = encoder_outputs[::-1]
        for i in range(len(self.Deconv)-2):
            feat = torch.cat([encoder_outputs[i+1], feat], dim=1)
            feat = self.Deconv[i+1](feat)
        
        feat = self.Deconv[-1](feat)
        return feat

    def feature_fusion(self, q, kv):
        B, C, H, W = q.shape
        PE = self.PE.unsqueeze(0).expand(B, -1, -1)

        q_emb = self.patch_emb(q.clone()).flatten(2)
        kv_emb = self.SH_patch_emb(kv).flatten(2)

        update = self.Cross_Attn(q_emb + PE, kv_emb)
        q_emb = q_emb + update
        
        q_flat = rearrange(q.clone(), "b c h w -> b (h w) c")
        update = self.Cross_Attn(q_flat, q_emb)

        q_flat = q_flat + update
        q_flat = q_flat + self.MLP(q_flat)
        output = rearrange(q_flat, "b (h w) c -> b c h w", h=H)
        return output


    def forward(self, audio, rgb=None, mode='train', return_hist_maps=False):
        audio_feat = self.audio_enc(audio)
        SH_coeffs = self.coeff_net(audio_feat)
        SH = torch.einsum("bn, nhw -> bnhw", SH_coeffs, self.SH_basis.to(SH_coeffs.device))

        if mode == 'train' and rgb is not None:
            rgb_feat = self.img_enc(rgb)
            feat = self.feature_fusion(rgb_feat, SH)
        else:
            rgb_feat = None
            feat = self.feature_fusion(audio_feat, SH)

        pred_depth = self.depth_head(feat)

        out = {"pred_depth": pred_depth,
               "foa_latent": None,
               "pred_foa": SH_coeffs,
               "pred_hoa": None,
               "pred_sh": None,
               "audio_feat": audio_feat,
               "rgb_feat": rgb_feat,}

        if return_hist_maps:
            sh_aligned = self.scale_shift(SH_coeffs)
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
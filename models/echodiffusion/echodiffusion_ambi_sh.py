"""EchoDiffusionAmbiSH — EchoDiffusion + SH coarse-layout prior fusion.

Pipeline (matches the spec in the design notes):

    binaural spec (B, 2, H, W)
        │
        ├──► ASPP+ASFF + DiffusionUNet → encoder features
        │       │
        │       ▼
        │   Decoder → (B, 192, H', W')   feature map
        │       │
        │       ▼  ─────── gated residual fusion ───────┐
        │   fused features                              │
        │       │                                       │
        │       ▼                                       │
        │   Depth head → pred_depth (B, 1, H, W)        │
        │                                               │
    rep_gt (B, K, 4) ──► AudioToSHCoeff (MLP) ──► SH coeff (B, C_sh)
                                                        │
                                                        ▼
                                          SHRenderer (real SH basis order L)
                                                        │
                                                        ▼
                                              SH layout (B, 1, H, W)
                                                        │
                                                        └── softplus → ≥0 ──┘

The diffusion-UNet's transformer cross-attention also receives the SH
coefficients as a single context token (B, 1, 768), so the audio prior
conditions the backbone too.

Forward returns dict (matches n4_0425 contract → routes via
_train_step_foa_0415):
    pred_depth   (B, 1, H, W)
    sh_coeff     (B, (L+1)^2)            predicted SH coefficients
    sh_layout    (B, 1, H, W)            rendered coarse layout (≥0)
    rep_pred     (B, K, 4)               = rep_gt (oracle), so the
                                          legacy weighted_rep_loss is 0
    pred_sh      (B, 4)                  rep_gt[:, 0, :], legacy stub

Construction kwargs (passed via cfg.model.* + CLI overrides):
    K            : int    distance bins (must match dataset rep_K)
    sh_order     : int    SH order L; coeffs = (L+1)^2
    sh_hidden    : int    hidden width of the audio→SH MLP
    max_depth    : float  output rescale (matches EchoDiffusion)
    embed_dim    : int    decoder width (same default as EchoDiffusion: 192)
    emb_dim      : int    cross-attention context dim (768)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.sh_basis import sh_basis_matrix
from .aspp_asff import UNetASPPASFF
from .diffusion_unet import DiffusionUNet
from .echodiffusion import Decoder, CIDE


# ---------------------------------------------------------------------------
# Tapered decoder — replaces EchoDiffusion's [32, 32, 32] bottleneck stack.
# ---------------------------------------------------------------------------
class TaperedDecoder(nn.Module):
    """Same shape as EchoDiffusion.Decoder but with a *tapered* deconv stack.

    EchoDiffusion's original Decoder used [32, 32, 32] channels through 3
    deconv layers — a 32-ch bottleneck right after the 1536-ch features.
    This variant tapers gradually (e.g., [256, 128, 64]) so the upsampling
    path retains more capacity and the SH layout has more channels to write
    into during gated fusion.

    Args:
        in_channels    : input feature channels (e.g., 1536 for embed_dim=192)
        out_channels   : output feature channels (e.g., 192 = embed_dim)
        deconv_channels: list of K deconv output widths, K=3 by default.
                         Default [256, 128, 64] reverses EchoDiffusion's
                         [32, 32, 32] severe bottleneck.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 deconv_channels=(256, 128, 64)):
        super().__init__()
        self.deconv_layers = self._make_deconv_layer(
            list(deconv_channels), in_channels)
        last_ch = deconv_channels[-1]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(last_ch, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        out = self.deconv_layers(x)
        out = self.conv_layers(out)
        out = self.up(self.up(out))
        return out

    @staticmethod
    def _make_deconv_layer(channels, in_planes):
        layers = []
        for ch in channels:
            layers.extend([
                nn.ConvTranspose2d(in_planes, ch, 4, stride=2,
                                   padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ])
            in_planes = ch
        return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# SH building blocks
# ---------------------------------------------------------------------------
def _build_real_sh_basis(H: int, W: int, order: int) -> torch.Tensor:
    """Real SH basis on an ERP grid using the project's `sh_basis_matrix`.

    Returns a (C_sh, H, W) float32 tensor with C_sh = (order+1)^2.
    Grid convention matches data/dataset.py (lines 113-115):
        az = (j + 0.5) / W * 2π − π
        el = π/2 − (i + 0.5) / H * π
    """
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    az = (jj + 0.5) / W * 2 * np.pi - np.pi
    el = np.pi / 2 - (ii + 0.5) / H * np.pi
    basis = sh_basis_matrix(order, el, az)               # (H*W, C_sh) float64
    basis = basis.reshape(H, W, -1).transpose(2, 0, 1)   # (C_sh, H, W)
    return torch.from_numpy(basis.astype(np.float32))


class AudioToSHCoeff(nn.Module):
    """rep_gt (B, K, 4) → SH coefficients (B, (L+1)^2).

    The input rep is itself a per-bin order-1 SH summary (4 coeffs each).
    This MLP upgrades that low-order per-bin info to a single global
    higher-order SH expansion.
    """

    def __init__(self, K: int = 8, bin_dim: int = 4,
                 sh_order: int = 5, hidden: int = 256):
        super().__init__()
        self.num_coeff = (sh_order + 1) ** 2
        self.net = nn.Sequential(
            nn.Linear(K * bin_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_coeff),
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        return self.net(rep.flatten(1))                  # (B, C_sh)


class SHRenderer(nn.Module):
    """Render SH coefficients into an ERP layout map.

    layout(i, j) = softplus( Σ_c coeff_c · Y_c(el_i, az_j) )
    softplus keeps the layout non-negative so it's interpretable as a
    coarse depth-like prior. (We don't *force* it to be calibrated depth
    — the gated fusion learns how to use it.)
    """

    def __init__(self, H: int, W: int, sh_order: int = 5):
        super().__init__()
        basis = _build_real_sh_basis(H, W, sh_order)
        self.register_buffer('basis', basis, persistent=False)

    def forward(self, coeff: torch.Tensor) -> torch.Tensor:
        layout = torch.einsum('bc,chw->bhw', coeff, self.basis)
        return F.softplus(layout).unsqueeze(1)           # (B, 1, H, W)


class SHGatedFusion(nn.Module):
    """Gated residual fusion of an N-channel feature map with the 1-ch SH layout.

    feat        : (B, C, h, w)   from the depth decoder
    sh_layout   : (B, 1, H, W)   resized to (h, w) inside this module
    output      : (B, C, h, w)   feat + gate ⊙ proj(sh_layout)
    """

    def __init__(self, feat_ch: int):
        super().__init__()
        self.layout_proj = nn.Sequential(
            nn.Conv2d(1, feat_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor,
                sh_layout: torch.Tensor) -> torch.Tensor:
        if sh_layout.shape[-2:] != feat.shape[-2:]:
            sh_layout = F.interpolate(sh_layout, size=feat.shape[-2:],
                                      mode='bilinear', align_corners=False)
        sh_feat = self.layout_proj(sh_layout)
        gate = self.gate(torch.cat([feat, sh_feat], dim=1))
        return feat + gate * sh_feat


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class EchoDiffusionAmbiSH(nn.Module):
    """EchoDiffusion-style depth network + SH coarse-layout prior fusion."""

    def __init__(self, cfg, max_depth: float = 10.0,
                 embed_dim: int = 192, emb_dim: int = 768,
                 K: int = 8, sh_order: int = 5, sh_hidden: int = 256,
                 unet_channels: int = 64,
                 decoder_channels=(256, 128, 64),
                 use_cide: bool = False,
                 **_unused):
        super().__init__()
        self.max_depth = float(max_depth)
        self.K = int(K)
        self.sh_order = int(sh_order)
        self.num_sh_coeff = (self.sh_order + 1) ** 2
        self.use_cide = bool(use_cide)

        H, W = (int(v) for v in cfg.dataset.images_size)

        # ---------- Spec encoder (binaural only, 2-ch input) ----------
        self.aspp_asff = UNetASPPASFF(in_channels=2, base_c=64)
        self.latent_proj = nn.Sequential(
            nn.Conv2d(128, 512, 1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
        )

        # ---------- Audio → SH coefficients → SH layout ----------
        self.audio_to_sh = AudioToSHCoeff(K=self.K, bin_dim=4,
                                          sh_order=self.sh_order,
                                          hidden=sh_hidden)
        self.sh_renderer = SHRenderer(H, W, sh_order=self.sh_order)

        # ---------- SH coeff → cross-attention context token ----------
        # SH coefficients are projected to a single (B, 1, emb_dim) cross-
        # attention token. If use_cide=True, a CIDE token (binaural waveform
        # → Wav2Vec2 → embedding) is PREPENDED → context = (B, 2, emb_dim).
        self.coeff_to_ctx = nn.Linear(self.num_sh_coeff, emb_dim)

        # ---------- Optional CIDE / Wav2Vec2 conditioning ----------
        if self.use_cide:
            self.cide_module = CIDE(emb_dim=emb_dim)
        else:
            self.cide_module = None

        # ---------- Diffusion UNet feature extractor (fixed t=1) ----------
        # model_channels=64 (was 32 in EchoDiffusion). Inner widths become
        # 64, 128, 256, 256 → ~40M params, ~4× the original 10.5M. Closer
        # to a usable depth-decoding capacity.
        self.unet = DiffusionUNet(
            in_channels=512, out_channels=4, model_channels=int(unet_channels),
            channel_mult=(1, 2, 4, 4), num_res_blocks=2,
            attention_resolutions=(4, 2, 1), num_heads=8,
            context_dim=emb_dim, transformer_depth=1,
        )

        # ---------- Multi-scale aggregator (matches scaled UNet widths) ----
        # ldm_prior must mirror the UNet's output widths at the 3 returned
        # scales: (model_channels*1, model_channels*2, model_channels*8).
        # See DiffusionUNet's `outs` ordering.
        ldm_prior = (int(unet_channels), int(unet_channels) * 2,
                     int(unet_channels) * 8)
        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]), nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), embed_dim * 8, 1),
            nn.GroupNorm(16, embed_dim * 8), nn.ReLU(),
        )

        # ---------- Tapered decoder + SH fusion + depth head ----------
        # decoder_channels=(256, 128, 64) replaces EchoDiffusion's
        # (32, 32, 32) bottleneck → ~3M params for the decoder vs the
        # original 0.26M.
        self.decoder = TaperedDecoder(embed_dim * 8, embed_dim,
                                      deconv_channels=tuple(decoder_channels))
        self.sh_fusion = SHGatedFusion(feat_ch=embed_dim)
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(embed_dim, 1, 3, 1, 1),
        )

        n_total = sum(p.numel() for p in self.parameters()) / 1e6
        n_train = sum(p.numel() for p in self.parameters()
                      if p.requires_grad) / 1e6
        print(f"  [echodiff_ambi_sh] K={self.K} sh_order={self.sh_order} "
              f"C_sh={self.num_sh_coeff}  unet_ch={unet_channels}  "
              f"dec_ch={list(decoder_channels)}  use_cide={self.use_cide}  "
              f"total={n_total:.2f}M  trainable={n_train:.2f}M")

    def forward(self, x: torch.Tensor, rep_gt: torch.Tensor = None,
                audio_wave: torch.Tensor = None,
                **_unused) -> dict:
        if rep_gt is None:
            raise ValueError(
                "EchoDiffusionAmbiSH requires rep_gt. Set "
                "cfg.dataset.use_distance_bins=True so the dataloader returns "
                "the 5-tuple (audio, depth, foa, em, rep_gt).")

        B, _, H_in, W_in = x.shape

        # 1. Audio → SH coeffs → SH layout (full ERP resolution).
        sh_coeff = self.audio_to_sh(rep_gt)                 # (B, C_sh)
        sh_layout = self.sh_renderer(sh_coeff)              # (B, 1, H, W)

        # 2. Spec → ASPP+ASFF latent (resized internally to 128×128 for the
        #    ASPP+ASFF working resolution, same as EchoDiffusion).
        spec_input = x
        if spec_input.shape[2] != 128 or spec_input.shape[3] != 128:
            spec_input = F.interpolate(spec_input, size=(128, 128),
                                       mode='bilinear', align_corners=False)
        latents = self.aspp_asff(spec_input)                # (B, 128, 32, 32)
        latents = self.latent_proj(latents)                 # (B, 512, 32, 32)

        # 3. Build cross-attention context.
        sh_token = self.coeff_to_ctx(sh_coeff).unsqueeze(1)  # (B, 1, emb_dim)
        if self.cide_module is not None:
            if audio_wave is None:
                raise ValueError(
                    "use_cide=True but forward() received audio_wave=None. "
                    "Set cfg.dataset.use_waveform=True so the dataloader "
                    "returns the 6-tuple including the binaural waveform.")
            cide_token = self.cide_module(audio_wave)        # (B, 1, emb_dim)
            ctx = torch.cat([cide_token, sh_token], dim=1)   # (B, 2, emb_dim)
        else:
            ctx = sh_token                                    # (B, 1, emb_dim)

        # 4. Diffusion UNet at fixed t=1, conditioned on the context tokens.
        t = torch.ones((B,), device=x.device).long()
        outs = self.unet(latents, t, context=ctx)

        # 4. Multi-scale aggregator (matches EchoDiffusion encoder output).
        feats = [
            outs[0],
            outs[1],
            torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1),
        ]
        agg = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]),
                         feats[2]], dim=1)
        agg = self.out_layer(agg)                            # (B, 1536, ?, ?)

        # 5. Decoder → 192-ch feature; SH layout fuses here as residual prior.
        dec_feat = self.decoder(agg)                         # (B, 192, h, w)
        fused = self.sh_fusion(dec_feat, sh_layout)          # (B, 192, h, w)

        # 6. Depth head + final resize to input resolution.
        depth = torch.sigmoid(self.last_layer_depth(fused)) * self.max_depth
        if depth.shape[-2:] != (H_in, W_in):
            depth = F.interpolate(depth, size=(H_in, W_in), mode='nearest')

        return {
            'pred_depth': depth,
            'sh_coeff':   sh_coeff,
            'sh_layout':  sh_layout,
            # Train-step compatibility (oracle: rep_pred = rep_gt → loss 0).
            'rep_pred':   rep_gt,
            'pred_sh':    rep_gt[:, 0, :].contiguous(),
        }


if __name__ == '__main__':
    from types import SimpleNamespace as NS
    cfg = NS(dataset=NS(images_size=[256, 512], depth_norm=True))
    net = EchoDiffusionAmbiSH(cfg, K=8, sh_order=5).eval()
    x = torch.randn(2, 2, 256, 512)
    rep = torch.randn(2, 8, 4)
    with torch.no_grad():
        out = net(x, rep_gt=rep)
    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f'  {k:12s}: {tuple(v.shape)}')

"""EchoDiffusionSHSide: EchoDiffusion backbone + SH side prior (concise).

Design goals
------------
- Keep the strong binaural main path from EchoDiffusion untouched.
- Use SH/FOA only as a *side prior*, never as a bottleneck.
- Support both oracle upper-bound experiments (rep_gt) and predicted SH.
- Stay simple: no CIDE, no second ViT/UNet branch, no teacher forcing.

Main path
---------
    binaural spectrogram
      -> ASPP+ASFF
      -> latent_proj
      -> DiffusionUNet
      -> multi-scale aggregation
      -> decoder
      -> depth

SH side path
------------
    pooled latent/audio feature
      -> rep_head -> rep_pred (B, K, 4)
      -> gate_head -> gate (B, K)
      -> SH-project + log1p(square) -> gated energy map
      -> small per-scale adapters -> additive fusion into UNet features
      -> gated residual fusion after decoder

Outputs
-------
    pred_depth   : (B, 1, H, W)
    pred_energy  : (B, 1, H, W)   gated SH-derived energy prior
    rep_pred     : (B, K, 4)      model-predicted per-bin FOA reps (ungated)
    rep_used     : (B, K, 4)      actual rep used by the side renderer
                                 (= rep_pred or rep_gt, then gate-applied)
    pred_sh      : (B, 4)         gate-weighted global FOA proxy for eval
    gate         : (B, K)

Training tip
------------
Recommended losses:
    L = L_depth
      + 0.03 * L_rep_dir
      + 0.01 * L_rep_mag
      + 0.02 * L_energy
where L_rep_dir / L_rep_mag compare rep_pred against rep_gt.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.sh_basis import sh_basis_matrix
from models.echodiffusion.aspp_asff import UNetASPPASFF
from models.echodiffusion.diffusion_unet import DiffusionUNet
from models.echodiffusion.echodiffusion import Decoder


def _build_foa_basis_erp(H: int, W: int) -> torch.Tensor:
    """Return order-1 real SH basis on ERP grid: (4, H, W)."""
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    az = (jj + 0.5) / W * 2 * np.pi - np.pi
    el = np.pi / 2 - (ii + 0.5) / H * np.pi
    basis = sh_basis_matrix(1, el, az)  # (H*W, 4)
    basis = basis.reshape(H, W, 4).transpose(2, 0, 1)
    return torch.from_numpy(basis.astype(np.float32))


class SHRepHead(nn.Module):
    def __init__(self, in_dim: int, K: int = 8, hidden: int = 512):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, K * 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.shape[0], self.K, 4)


class GateHead(nn.Module):
    def __init__(self, in_dim: int, K: int = 8, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, K),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class SideAdapter(nn.Module):
    """1ch energy prior -> feature map matching one UNet scale."""

    def __init__(self, out_ch: int):
        super().__init__()
        n_groups = 16 if out_ch % 16 == 0 else 8
        self.net = nn.Sequential(
            nn.Conv2d(1, out_ch, 3, padding=1),
            nn.GroupNorm(n_groups, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, em: torch.Tensor, size) -> torch.Tensor:
        x = F.interpolate(em, size=size, mode='bilinear', align_corners=False)
        return self.net(x)


class SHGatedFusion(nn.Module):
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

    def forward(self, feat: torch.Tensor, sh_layout: torch.Tensor) -> torch.Tensor:
        if sh_layout.shape[-2:] != feat.shape[-2:]:
            sh_layout = F.interpolate(sh_layout, size=feat.shape[-2:],
                                      mode='bilinear', align_corners=False)
        sh_feat = self.layout_proj(sh_layout)
        gate = self.gate(torch.cat([feat, sh_feat], dim=1))
        return feat + gate * sh_feat


class EchoDiffusionSHSide(nn.Module):
    def __init__(self, cfg, max_depth: float = 10.0, embed_dim: int = 192,
                 K: int = 8, rep_hidden: int = 512,
                 side_fusion: bool = True, oracle_mode: bool = False,
                 sh_grid=None, **_unused):
        super().__init__()
        self.max_depth = float(max_depth)
        self.K = int(K)
        self.side_fusion = bool(side_fusion)
        self.oracle_mode = bool(oracle_mode)

        in_h, in_w = (int(v) for v in cfg.dataset.images_size)
        sh_h, sh_w = sh_grid if sh_grid is not None else (in_h, in_w)

        # ----- EchoDiffusion main path (without CIDE) -----
        self.aspp_asff = UNetASPPASFF(in_channels=2, base_c=64)
        self.latent_proj = nn.Sequential(
            nn.Conv2d(128, 512, 1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
        )
        self.unet = DiffusionUNet(
            in_channels=512, out_channels=4, model_channels=32,
            channel_mult=(1, 2, 4, 4), num_res_blocks=2,
            attention_resolutions=(4, 2, 1), num_heads=8,
            context_dim=None, transformer_depth=1,
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.GroupNorm(16, 32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(32 + 64 + 256, embed_dim * 8, 1),
            nn.GroupNorm(16, embed_dim * 8),
            nn.ReLU(),
        )
        self.decoder = Decoder(embed_dim * 8, embed_dim)
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(embed_dim, 1, 3, 1, 1),
        )

        # ----- SH side path -----
        self.rep_head = SHRepHead(in_dim=512, K=K, hidden=rep_hidden)
        self.gate_head = GateHead(in_dim=512, K=K, hidden=max(128, rep_hidden // 2))
        self.register_buffer('foa_basis', _build_foa_basis_erp(sh_h, sh_w), persistent=False)
        self.side1 = SideAdapter(32)
        self.side2 = SideAdapter(64)
        self.side3 = SideAdapter(256)
        self.post_fusion = SHGatedFusion(embed_dim)

        # Start from near-baseline behavior; let SH help only if useful.
        self.alpha1 = nn.Parameter(torch.tensor(0.0))
        self.alpha2 = nn.Parameter(torch.tensor(0.0))
        self.alpha3 = nn.Parameter(torch.tensor(0.0))

    def _render_energy(self, rep: torch.Tensor, gate: torch.Tensor, out_hw) -> torch.Tensor:
        """rep: (B,K,4), gate: (B,K) -> energy map (B,1,H,W)."""
        rep_g = rep * gate.unsqueeze(-1)
        signed = torch.einsum('bkc,chw->bkhw', rep_g, self.foa_basis)
        em = torch.log1p(signed.square()).sum(dim=1, keepdim=True)
        if em.shape[-2:] != out_hw:
            em = F.interpolate(em, size=out_hw, mode='bilinear', align_corners=False)
        return em

    @staticmethod
    def _global_foa(rep: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        w = gate / (gate.sum(dim=1, keepdim=True) + 1e-6)
        return (rep * w.unsqueeze(-1)).sum(dim=1)

    def forward(self, audio_spec: torch.Tensor, rep_gt: torch.Tensor = None,
                use_oracle_rep: bool = None, **_unused) -> dict:
        orig_h, orig_w = audio_spec.shape[2], audio_spec.shape[3]
        use_oracle = self.oracle_mode if use_oracle_rep is None else bool(use_oracle_rep)

        # EchoDiffusion works on 128x128 internally.
        spec_input = audio_spec
        if spec_input.shape[2] != 128 or spec_input.shape[3] != 128:
            spec_input = F.interpolate(spec_input, size=(128, 128),
                                       mode='bilinear', align_corners=False)

        latents = self.aspp_asff(spec_input)            # (B,128,32,32)
        latents = self.latent_proj(latents)             # (B,512,32,32)
        pooled = F.adaptive_avg_pool2d(latents, 1).flatten(1)  # (B,512)

        rep_pred = self.rep_head(pooled)                # (B,K,4)
        gate = self.gate_head(pooled)                   # (B,K)

        rep_source = rep_gt if (use_oracle and rep_gt is not None) else rep_pred
        rep_used = rep_source * gate.unsqueeze(-1)

        t = torch.ones((audio_spec.shape[0],), device=audio_spec.device).long()
        outs = self.unet(latents, t, context=None)

        # SH-derived side prior (render at input size, then resize per scale).
        pred_energy = self._render_energy(rep_source, gate, out_hw=(orig_h, orig_w))

        feat0 = outs[0]
        feat1 = outs[1]
        feat2 = torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)

        if self.side_fusion:
            feat0 = feat0 + torch.tanh(self.alpha1) * self.side1(pred_energy, feat0.shape[-2:])
            feat1 = feat1 + torch.tanh(self.alpha2) * self.side2(pred_energy, feat1.shape[-2:])
            feat2 = feat2 + torch.tanh(self.alpha3) * self.side3(pred_energy, feat2.shape[-2:])

        agg = torch.cat([self.layer1(feat0), self.layer2(feat1), feat2], dim=1)
        agg = self.out_layer(agg)

        dec = self.decoder(agg)
        if self.side_fusion:
            dec = self.post_fusion(dec, pred_energy)
        depth = torch.sigmoid(self.last_layer_depth(dec)) * self.max_depth
        if depth.shape[-2:] != (orig_h, orig_w):
            depth = F.interpolate(depth, size=(orig_h, orig_w), mode='nearest')

        pred_sh = self._global_foa(rep_pred, gate)
        return {
            'pred_depth': depth,
            'pred_energy': pred_energy,
            'rep_pred': rep_pred,
            'rep_used': rep_used,
            'pred_sh': pred_sh,
            'gate': gate,
        }


if __name__ == '__main__':
    from types import SimpleNamespace as NS

    cfg = NS(dataset=NS(images_size=[256, 512], depth_norm=True))
    net = EchoDiffusionSHSide(cfg, K=8, oracle_mode=True).eval()
    x = torch.randn(2, 2, 256, 512)
    rep = torch.randn(2, 8, 4)
    with torch.no_grad():
        out = net(x, rep_gt=rep)
    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f'{k:12s}: {tuple(v.shape)}')

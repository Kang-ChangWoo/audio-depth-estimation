import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.sh_basis import sh_basis_matrix
from .aspp_asff import UNetASPPASFF
from .diffusion_unet import DiffusionUNet


def _build_foa_basis_erp(H: int, W: int) -> torch.Tensor:
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    az = (jj + 0.5) / W * 2 * np.pi - np.pi
    el = np.pi / 2 - (ii + 0.5) / H * np.pi
    basis = sh_basis_matrix(1, el, az)  # (H*W, 4)
    basis = basis.reshape(H, W, 4).transpose(2, 0, 1)
    return torch.from_numpy(basis.astype(np.float32))


class TaperedDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 deconv_channels=(256, 128, 64)):
        super().__init__()
        self.deconv_layers = self._make_deconv_layer(list(deconv_channels), in_channels)
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
                nn.ConvTranspose2d(in_planes, ch, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ])
            in_planes = ch
        return nn.Sequential(*layers)


class SHRepSplitHead(nn.Module):
    """Predict per-bin FOA as direction * magnitude for stabler optimization."""
    def __init__(self, in_dim: int, K: int = 8, hidden: int = 512):
        super().__init__()
        self.K = K
        self.pre = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.dir_head = nn.Linear(hidden, K * 4)
        self.mag_head = nn.Linear(hidden, K)
        self.gate_head = nn.Linear(hidden, K)
        self.ctx_head = nn.Linear(hidden, 768)

    def forward(self, x: torch.Tensor):
        h = self.pre(x)
        raw_dir = self.dir_head(h).view(x.shape[0], self.K, 4)
        dir_unit = raw_dir / (raw_dir.norm(dim=-1, keepdim=True) + 1e-6)
        mag = F.softplus(self.mag_head(h)).unsqueeze(-1)
        rep_pred = dir_unit * mag
        gate = torch.sigmoid(self.gate_head(h))
        ctx = self.ctx_head(h).unsqueeze(1)
        return rep_pred, gate, ctx


class SideAdapter(nn.Module):
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
            sh_layout = F.interpolate(sh_layout, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        sh_feat = self.layout_proj(sh_layout)
        gate = self.gate(torch.cat([feat, sh_feat], dim=1))
        return feat + gate * sh_feat


class EchoDiffusionSHSidePlus(nn.Module):
    """
    Stronger version of EchoDiffusionSHSide.

    Changes vs the concise model:
    - larger DiffusionUNet width (64 instead of 32)
    - SH token injected into DiffusionUNet cross-attention early
    - direction/magnitude split SH head for stabler supervision
    - tapered decoder for higher-capacity upsampling
    - additive multi-scale SH fusion + zero-init post fusion
    - explicit oracle gate mode for clean upper-bound experiments
    """
    def __init__(self, cfg, max_depth: float = 10.0, embed_dim: int = 192,
                 K: int = 8, rep_hidden: int = 512,
                 unet_channels: int = 64,
                 decoder_channels=(256, 128, 64),
                 side_fusion: bool = True,
                 oracle_mode: bool = False,
                 oracle_gate_mode: str = 'ones',
                 sh_grid=None,
                 **_unused):
        super().__init__()
        self.max_depth = float(max_depth)
        self.K = int(K)
        self.side_fusion = bool(side_fusion)
        self.oracle_mode = bool(oracle_mode)
        self.oracle_gate_mode = str(oracle_gate_mode)

        in_h, in_w = (int(v) for v in cfg.dataset.images_size)
        sh_h, sh_w = sh_grid if sh_grid is not None else (in_h, in_w)

        self.aspp_asff = UNetASPPASFF(in_channels=2, base_c=64)
        self.latent_proj = nn.Sequential(
            nn.Conv2d(128, 512, 1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
        )

        self.sh_head = SHRepSplitHead(in_dim=512, K=K, hidden=rep_hidden)
        self.unet = DiffusionUNet(
            in_channels=512, out_channels=4, model_channels=int(unet_channels),
            channel_mult=(1, 2, 4, 4), num_res_blocks=2,
            attention_resolutions=(4, 2, 1), num_heads=8,
            context_dim=768, transformer_depth=1,
        )

        ldm_prior = (int(unet_channels), int(unet_channels) * 2, int(unet_channels) * 8)
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
            nn.GroupNorm(16, embed_dim * 8),
            nn.ReLU(),
        )
        self.decoder = TaperedDecoder(embed_dim * 8, embed_dim, deconv_channels=tuple(decoder_channels))
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(embed_dim, 1, 3, 1, 1),
        )

        self.register_buffer('foa_basis', _build_foa_basis_erp(sh_h, sh_w), persistent=False)
        self.side1 = SideAdapter(ldm_prior[0])
        self.side2 = SideAdapter(ldm_prior[1])
        self.side3 = SideAdapter(ldm_prior[2])
        self.post_fusion = SHGatedFusion(embed_dim)

        self.alpha1 = nn.Parameter(torch.tensor(0.0))
        self.alpha2 = nn.Parameter(torch.tensor(0.0))
        self.alpha3 = nn.Parameter(torch.tensor(0.0))
        self.alpha_post = nn.Parameter(torch.tensor(0.0))

    def _oracle_gate(self, pred_gate: torch.Tensor) -> torch.Tensor:
        if self.oracle_gate_mode == 'ones':
            return torch.ones_like(pred_gate)
        if self.oracle_gate_mode == 'pred':
            return pred_gate
        raise ValueError(f'Unsupported oracle_gate_mode: {self.oracle_gate_mode}')

    def _render_energy(self, rep: torch.Tensor, gate: torch.Tensor, out_hw) -> torch.Tensor:
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

        spec_input = audio_spec
        if spec_input.shape[2] != 128 or spec_input.shape[3] != 128:
            spec_input = F.interpolate(spec_input, size=(128, 128), mode='bilinear', align_corners=False)

        latents = self.aspp_asff(spec_input)
        latents = self.latent_proj(latents)
        pooled = F.adaptive_avg_pool2d(latents, 1).flatten(1)

        rep_pred, gate_pred, sh_ctx = self.sh_head(pooled)
        if use_oracle and rep_gt is not None:
            rep_source = rep_gt
            gate_used = self._oracle_gate(gate_pred)
        else:
            rep_source = rep_pred
            gate_used = gate_pred
        rep_used = rep_source * gate_used.unsqueeze(-1)

        t = torch.ones((audio_spec.shape[0],), device=audio_spec.device).long()
        outs = self.unet(latents, t, context=sh_ctx)

        pred_energy = self._render_energy(rep_source, gate_used, out_hw=(orig_h, orig_w))

        feat0 = outs[0]
        feat1 = outs[1]
        feat2 = torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2.0, mode='nearest')], dim=1)

        if self.side_fusion:
            feat0 = feat0 + torch.tanh(self.alpha1) * self.side1(pred_energy, feat0.shape[-2:])
            feat1 = feat1 + torch.tanh(self.alpha2) * self.side2(pred_energy, feat1.shape[-2:])
            feat2 = feat2 + torch.tanh(self.alpha3) * self.side3(pred_energy, feat2.shape[-2:])

        agg = torch.cat([self.layer1(feat0), self.layer2(feat1), feat2], dim=1)
        agg = self.out_layer(agg)

        dec = self.decoder(agg)
        if self.side_fusion:
            dec = dec + torch.tanh(self.alpha_post) * (self.post_fusion(dec, pred_energy) - dec)
        depth = torch.sigmoid(self.last_layer_depth(dec)) * self.max_depth
        if depth.shape[-2:] != (orig_h, orig_w):
            depth = F.interpolate(depth, size=(orig_h, orig_w), mode='nearest')

        pred_sh = self._global_foa(rep_source, gate_used)
        return {
            'pred_depth': depth,
            'pred_energy': pred_energy,
            'rep_pred': rep_pred,
            'rep_used': rep_used,
            'pred_sh': pred_sh,
            'gate': gate_used,
            'gate_pred': gate_pred,
        }


if __name__ == '__main__':
    from types import SimpleNamespace as NS

    cfg = NS(dataset=NS(images_size=[256, 512], depth_norm=True))
    net = EchoDiffusionSHSidePlus(cfg, K=8, oracle_mode=True).eval()
    x = torch.randn(2, 2, 256, 512)
    rep = torch.randn(2, 8, 4)
    with torch.no_grad():
        out = net(x, rep_gt=rep)
    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f'{k:12s}: {tuple(v.shape)}')

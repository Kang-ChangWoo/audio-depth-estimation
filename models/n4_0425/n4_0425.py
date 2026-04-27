"""n4_0425 — Binaural UNet + bin-gated multi-scale energy fusion.

The oracle bin selection (rep_gt × learnable gate) is now MERGED into
the energy-map encoder pathway: instead of feeding the dataset's GT
energy_map into a separate stream and conditioning the bottleneck with
a bin-gate MLP, we synthesize a gate-filtered per-bin SH-projected
energy map from rep_gt and use THAT as the multi-scale energy input.

Concretely:

    rep_gt (B, K, 4) × g       (gate g = sigmoid(gate_logit))
                │
                │  einsum('bkc,chw → bkhw') with foa_basis (4, H, W)
                ▼
        signed (B, K, H, W)            ← per-bin SH-projection of FOA coefs
                │
                │  log1p(·²)
                ▼
       energy_per_bin (B, K, H, W)
                │
                │  Σ_k                  ← bin gate is in rep_gated already
                ▼
        gated_em (B, 1, H, W)
                │
                │  peak-normalize per sample to [0, 1]
                ▼
        gated_em (B, 1, H, W)  ◄── INPUT to the EM encoder
                │
                ▼
   8-level parallel EM encoder ──► m1, m2, …, m8

Binaural pathway is unchanged:

    spec (B, 2, H, W) ── 8-level UNet enc ──► e1, e2, …, e8
                                                │+   │+        │+
                                              (m_i added per level)
                                                │
                                       UNet decoder + skips ──► depth

The GT energy_map from the dataset is NOT used — the bin gate's effect
flows entirely through the synthesized gated_em input. Drop-bin / single-
bin retraining therefore removes that bin's spatial energy signature from
everything the model sees, not just from a bottleneck bias term.

forward returns dict:
    pred_depth   (B, 1, H, W)
    gate         (K,)         — sigmoid(gate_logit), for sparsity loss + analysis
    rep_pred     (B, K, 4)    — rep_gt * gate, kept for compat with test_utils
    pred_sh      (B, 4)       — rep_pred[:, 0, :], legacy compat stub

Drop-bin / single-bin retraining: pass `gate_mask` (length-K float list) at
construction time. The gate is then a frozen buffer set to inv_sigmoid(mask)
so sigmoid(gate) == mask exactly.
"""

import torch
import torch.nn as nn

from models.n9_0424 import make_foa_basis_erp


def _down(c_in, c_out, first=False):
    if first:
        return nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False)
    return nn.Sequential(
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False),
        nn.BatchNorm2d(c_out),
    )


def _down_innermost(c_in, c_out):
    return nn.Sequential(
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False),
    )


def _up(c_in, c_out, dropout=False):
    layers = [
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False),
        nn.BatchNorm2d(c_out),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class N4_0425Net(nn.Module):
    """Binaural UNet + bin-gated SH-reconstructed energy → multi-scale fusion.

    Parameters
    ----------
    cfg          : config object — uses cfg.dataset.images_size, depth_norm.
    input_nc     : int — audio channels (default 2 = binaural).
    K            : int — number of distance bins (default 8).
    ngf          : int — base UNet width (default 64).
    gate_init    : float — initial gate_logit value. sigmoid(2.0) ≈ 0.88.
    gate_mask    : list[float] | None — fixed per-bin mask in [0, 1]. When set,
                   the gate is a frozen buffer (inv-sigmoid of the mask).
    """

    def __init__(self, cfg, input_nc=2, K=8, ngf=64, gate_init=2.0,
                 gate_mask=None, **_unused):
        super().__init__()
        self.K = int(K)
        self.depth_norm = bool(getattr(cfg.dataset, 'depth_norm', True))

        # ---------- 8-level binaural UNet encoder ----------
        self.enc1 = _down(input_nc, ngf,    first=True)   # ngf,    H/2,  W/2
        self.enc2 = _down(ngf,      ngf*2)                # 2ngf,   H/4
        self.enc3 = _down(ngf*2,    ngf*4)                # 4ngf,   H/8
        self.enc4 = _down(ngf*4,    ngf*8)                # 8ngf,   H/16
        self.enc5 = _down(ngf*8,    ngf*8)                # 8ngf,   H/32
        self.enc6 = _down(ngf*8,    ngf*8)                # 8ngf,   H/64
        self.enc7 = _down(ngf*8,    ngf*8)                # 8ngf,   H/128
        self.enc8 = _down_innermost(ngf*8, ngf*8)         # 8ngf,   H/256

        # ---------- 8-level parallel energy-map encoder ----------
        # Input is the gate-filtered SH-projected energy map (1 channel).
        self.em_enc1 = _down(1,        ngf,    first=True)
        self.em_enc2 = _down(ngf,      ngf*2)
        self.em_enc3 = _down(ngf*2,    ngf*4)
        self.em_enc4 = _down(ngf*4,    ngf*8)
        self.em_enc5 = _down(ngf*8,    ngf*8)
        self.em_enc6 = _down(ngf*8,    ngf*8)
        self.em_enc7 = _down(ngf*8,    ngf*8)
        self.em_enc8 = _down_innermost(ngf*8, ngf*8)

        # ---------- Bin gate ----------
        if gate_mask is None:
            self.gate_logit = nn.Parameter(torch.full((K,), float(gate_init)))
            self.gate_learnable = True
        else:
            mask = torch.tensor(list(gate_mask), dtype=torch.float32)
            assert mask.numel() == K, \
                f"gate_mask length {mask.numel()} != K={K}"
            mask_c = mask.clamp(1e-6, 1 - 1e-6)
            logit = torch.log(mask_c / (1 - mask_c))      # inv-sigmoid
            self.register_buffer('gate_logit', logit, persistent=True)
            self.gate_learnable = False

        # FOA basis at the input resolution — used to SH-project rep_gt into
        # per-bin spatial energy maps inside forward(). No learnable params.
        H_erp, W_erp = (int(v) for v in cfg.dataset.images_size)
        basis = make_foa_basis_erp(H_erp, W_erp)               # (4, H, W)
        self.register_buffer('foa_basis', basis, persistent=False)

        # ---------- 8-level UNet decoder ----------
        self.dec8 = _up(ngf*8,        ngf*8, dropout=True)
        self.dec7 = _up(ngf*8 * 2,    ngf*8, dropout=True)
        self.dec6 = _up(ngf*8 * 2,    ngf*8, dropout=True)
        self.dec5 = _up(ngf*8 * 2,    ngf*8)
        self.dec4 = _up(ngf*8 * 2,    ngf*4)
        self.dec3 = _up(ngf*4 * 2,    ngf*2)
        self.dec2 = _up(ngf*2 * 2,    ngf)
        self.dec1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1),
            nn.Sigmoid() if self.depth_norm else nn.ReLU(inplace=True),
        )

    def _build_gated_em(self, rep_gt: torch.Tensor,
                        g: torch.Tensor) -> torch.Tensor:
        """Synthesize the gate-filtered per-bin energy map.

        rep_gt: (B, K, 4)   oracle FOA representatives per bin
        g:      (K,)        sigmoid-applied gate

        Returns (B, 1, H, W), peak-normalized per sample.
        """
        rep_gated = rep_gt * g.view(1, -1, 1)                  # (B, K, 4)
        signed = torch.einsum(
            'bkc,chw->bkhw', rep_gated, self.foa_basis)        # (B, K, H, W)
        energy_per_bin = torch.log1p(signed.square())
        em = energy_per_bin.sum(dim=1, keepdim=True)           # (B, 1, H, W)
        emax = em.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return em / emax

    def forward(self, x: torch.Tensor, rep_gt: torch.Tensor = None,
                **_unused) -> dict:
        """
        x      : (B, input_nc, H, W) binaural spectrogram.
        rep_gt : (B, K, 4) oracle per-bin FOA representatives.
                 Required at train time; if None at eval, the EM-encoder
                 pathway is skipped (binaural-only baseline).
        """
        # Binaural encoder.
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        g = torch.sigmoid(self.gate_logit)                     # (K,)

        # Gate-filtered energy map → multi-scale features → additive fusion.
        if rep_gt is not None:
            gated_em = self._build_gated_em(rep_gt, g)
            m1 = self.em_enc1(gated_em)
            m2 = self.em_enc2(m1)
            m3 = self.em_enc3(m2)
            m4 = self.em_enc4(m3)
            m5 = self.em_enc5(m4)
            m6 = self.em_enc6(m5)
            m7 = self.em_enc7(m6)
            m8 = self.em_enc8(m7)
            e1 = e1 + m1
            e2 = e2 + m2
            e3 = e3 + m3
            e4 = e4 + m4
            e5 = e5 + m5
            e6 = e6 + m6
            e7 = e7 + m7
            e8 = e8 + m8

        # Decoder.
        d = self.dec8(e8)
        d = self.dec7(torch.cat([d, e7], dim=1))
        d = self.dec6(torch.cat([d, e6], dim=1))
        d = self.dec5(torch.cat([d, e5], dim=1))
        d = self.dec4(torch.cat([d, e4], dim=1))
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = self.dec2(torch.cat([d, e2], dim=1))
        depth = self.dec1(torch.cat([d, e1], dim=1))           # (B, 1, H, W)

        out = {'pred_depth': depth, 'gate': g}
        if rep_gt is not None:
            rep_pred = rep_gt * g.view(1, -1, 1)
            out['rep_pred'] = rep_pred
            out['pred_sh'] = rep_pred[:, 0, :].contiguous()
        return out


if __name__ == '__main__':
    from types import SimpleNamespace as NS
    cfg = NS(dataset=NS(images_size=[256, 512], depth_norm=True))
    net = N4_0425Net(cfg, input_nc=2, K=8).eval()
    x = torch.randn(2, 2, 256, 512)
    rep = torch.randn(2, 8, 4)
    with torch.no_grad():
        out = net(x, rep_gt=rep)
        for k, v in out.items():
            print(f'{k:12s}: {tuple(v.shape)}')
    print(f'params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M')

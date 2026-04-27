"""n9_0425 — Binaural UNet + bin-gated multi-scale energy fusion,
but the per-bin FOA reps are PREDICTED by a pre-trained n3_0425 model
instead of taken from the dataset's oracle rep_gt.

This is the prediction-cascade analogue of n4_0425. The architecture is
identical except for one substitution at the input of the SH-projection
step:

    n4_0425:  rep_in  =  rep_gt          (oracle, dataset 5-tuple)
    n9_0425:  rep_in  =  n3_0425(spec)   (predicted from binaural)

Pipeline:

    spec (B, 2, H, W)
      ├──► n3_0425 (frozen OR finetuned) ──► rep_pred_n3 (B, K, 4)
      │                                              │
      │                                              │  × g (sigmoid(gate_logit))
      │                                              ▼
      │                                       gated rep_pred (B, K, 4)
      │                                              │
      │                                       einsum(.., foa_basis)
      │                                              │
      │                                          log1p(·²)
      │                                              │
      │                                          Σ_k → peak-norm
      │                                              ▼
      │                                       gated_em (B, 1, H, W)
      │                                              │
      │                                              ▼
      │                                  8-level parallel EM encoder
      │                                              │
      └──► 8-level binaural UNet enc                │
                          e1..e8       +            m1..m8
                                       ▼
                                 UNet decoder + skips ──► depth

Forward returns dict:
    pred_depth   (B, 1, H, W)
    gate         (K,)            sigmoid(gate_logit)
    rep_pred     (B, K, 4)       n3 RAW output (NOT gated). When
                                 lambda_sh > 0 in cfg, the train step
                                 applies weighted_rep_loss(rep_pred,
                                 rep_gt) to fine-tune n3 toward oracle.
    pred_sh      (B, 4)          rep_pred[:, 0, :], legacy compat stub.

Construction kwargs (passed via cfg.model.* + a couple of CLI overrides):
    n3_checkpoint : str|None   path to a trained n3_0425 best_model.pth.
                                If None, n3 starts from random init —
                                effectively a "scratch joint" baseline.
    freeze_n3     : bool        whether n3 weights are frozen and n3 is
                                kept in eval(). Default True.

Drop-bin / single-bin retraining is identical to n4_0425: pass
`gate_mask` (length-K float list) at construction time. Gate becomes a
frozen buffer set to inv_sigmoid(mask).
"""

import os
import torch
import torch.nn as nn

from models.n3_0425 import N3_0425Net
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


def _load_n3_state_dict(n3: nn.Module, ckpt_path: str) -> None:
    """Load pre-trained n3_0425 weights, robust to DataParallel prefixes."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"n3 checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    sd = {(k[len('module.'):] if k.startswith('module.') else k): v
          for k, v in sd.items()}
    missing, unexpected = n3.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [n9_0425] n3 ckpt loaded with {len(missing)} missing keys "
              f"(first: {missing[:3]})")
    if unexpected:
        print(f"  [n9_0425] n3 ckpt loaded with {len(unexpected)} unexpected "
              f"keys (first: {unexpected[:3]})")
    if not missing and not unexpected:
        print(f"  [n9_0425] n3 ckpt loaded cleanly from {ckpt_path}")


class N9_0425Net(nn.Module):
    """Binaural UNet + bin-gated SH-reconstructed energy from PREDICTED FOA.

    Parameters
    ----------
    cfg            : config object — uses cfg.dataset.images_size, depth_norm.
    input_nc       : int — audio channels (default 2 = binaural).
    K              : int — number of distance bins (default 8).
    ngf            : int — base UNet width (default 64).
    gate_init      : float — initial gate_logit value. sigmoid(2.0) ≈ 0.88.
    gate_mask      : list[float] | None — fixed per-bin mask in [0, 1]. When
                      set, the gate is a frozen buffer (inv-sigmoid of the
                      mask). Same semantics as n4_0425.
    n3_checkpoint  : str | None — path to a pre-trained n3_0425 best_model.pth.
    freeze_n3      : bool — if True, n3 parameters require_grad=False AND n3
                      is kept in eval() across train()/eval() switches so
                      its BatchNorm stats are stable. Default True.
    n3_ngf         : int — n3 inner UNet width. Should match the n3 ckpt's
                      training cfg (default 64).
    """

    def __init__(self, cfg, input_nc=2, K=8, ngf=64, gate_init=2.0,
                 gate_mask=None,
                 n3_checkpoint=None, freeze_n3=True, n3_ngf=64,
                 **_unused):
        super().__init__()
        self.K = int(K)
        self.depth_norm = bool(getattr(cfg.dataset, 'depth_norm', True))
        self.freeze_n3 = bool(freeze_n3)

        # ---------- Inner n3_0425 (binaural → rep_pred) ----------
        self.n3 = N3_0425Net(cfg, input_nc=input_nc, K=self.K, ngf=int(n3_ngf))
        if n3_checkpoint:
            _load_n3_state_dict(self.n3, n3_checkpoint)
        else:
            print('  [n9_0425] no n3 checkpoint provided — n3 starts random')
        if self.freeze_n3:
            for p in self.n3.parameters():
                p.requires_grad = False
            self.n3.eval()
            print('  [n9_0425] n3 frozen (eval mode, no_grad in forward)')
        else:
            print('  [n9_0425] n3 trainable (gradients flow back through n3)')

        # ---------- 8-level binaural UNet encoder ----------
        self.enc1 = _down(input_nc, ngf,    first=True)
        self.enc2 = _down(ngf,      ngf*2)
        self.enc3 = _down(ngf*2,    ngf*4)
        self.enc4 = _down(ngf*4,    ngf*8)
        self.enc5 = _down(ngf*8,    ngf*8)
        self.enc6 = _down(ngf*8,    ngf*8)
        self.enc7 = _down(ngf*8,    ngf*8)
        self.enc8 = _down_innermost(ngf*8, ngf*8)

        # ---------- 8-level parallel energy-map encoder ----------
        self.em_enc1 = _down(1,        ngf,    first=True)
        self.em_enc2 = _down(ngf,      ngf*2)
        self.em_enc3 = _down(ngf*2,    ngf*4)
        self.em_enc4 = _down(ngf*4,    ngf*8)
        self.em_enc5 = _down(ngf*8,    ngf*8)
        self.em_enc6 = _down(ngf*8,    ngf*8)
        self.em_enc7 = _down(ngf*8,    ngf*8)
        self.em_enc8 = _down_innermost(ngf*8, ngf*8)

        # ---------- Bin gate (same semantics as n4_0425) ----------
        if gate_mask is None:
            self.gate_logit = nn.Parameter(torch.full((self.K,), float(gate_init)))
            self.gate_learnable = True
        else:
            mask = torch.tensor(list(gate_mask), dtype=torch.float32)
            assert mask.numel() == self.K, \
                f"gate_mask length {mask.numel()} != K={self.K}"
            mask_c = mask.clamp(1e-6, 1 - 1e-6)
            logit = torch.log(mask_c / (1 - mask_c))
            self.register_buffer('gate_logit', logit, persistent=True)
            self.gate_learnable = False

        # FOA basis at the input ERP resolution (4, H, W). Used to project
        # the gated rep_pred into per-bin spatial energy maps.
        H_erp, W_erp = (int(v) for v in cfg.dataset.images_size)
        basis = make_foa_basis_erp(H_erp, W_erp)
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

    # Keep n3 in eval() across train()/eval() toggles when frozen — critical
    # for stable BatchNorm running stats (the inner n3 has BN layers).
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_n3:
            self.n3.eval()
        return self

    def _build_gated_em(self, rep: torch.Tensor,
                        g: torch.Tensor) -> torch.Tensor:
        """Synthesize the gate-filtered per-bin energy map.

        rep: (B, K, 4)   per-bin FOA representatives (predicted, here).
        g:   (K,)        sigmoid-applied gate.
        Returns (B, 1, H, W), peak-normalized per sample.
        """
        rep_gated = rep * g.view(1, -1, 1)
        signed = torch.einsum(
            'bkc,chw->bkhw', rep_gated, self.foa_basis)
        energy_per_bin = torch.log1p(signed.square())
        em = energy_per_bin.sum(dim=1, keepdim=True)
        emax = em.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return em / emax

    def forward(self, x: torch.Tensor, **_unused) -> dict:
        """
        x : (B, input_nc, H, W) binaural spectrogram.
        Any rep_gt the train step passes is ignored — n9_0425 generates
        its own rep from the inner n3.
        """
        # ---- 1. Predict per-bin FOA reps from binaural via n3 ----
        if self.freeze_n3:
            with torch.no_grad():
                rep_pred = self.n3(x)['pred_rep']     # (B, K, 4)
        else:
            rep_pred = self.n3(x)['pred_rep']

        # ---- 2. Binaural UNet encoder ----
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # ---- 3. Gate-filtered energy-map → multi-scale → additive fusion ----
        g = torch.sigmoid(self.gate_logit)
        gated_em = self._build_gated_em(rep_pred, g)
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

        # ---- 4. UNet decoder ----
        d = self.dec8(e8)
        d = self.dec7(torch.cat([d, e7], dim=1))
        d = self.dec6(torch.cat([d, e6], dim=1))
        d = self.dec5(torch.cat([d, e5], dim=1))
        d = self.dec4(torch.cat([d, e4], dim=1))
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = self.dec2(torch.cat([d, e2], dim=1))
        depth = self.dec1(torch.cat([d, e1], dim=1))

        return {
            'pred_depth': depth,
            'gate':       g,
            # rep_pred is the RAW n3 output (NOT gated). When lambda_sh > 0
            # in cfg, _train_step_foa_0415 applies weighted_rep_loss against
            # rep_gt — useful for fine-tuning n3 toward the oracle target.
            'rep_pred':   rep_pred,
            'pred_sh':    rep_pred[:, 0, :].contiguous(),
        }


if __name__ == '__main__':
    from types import SimpleNamespace as NS
    cfg = NS(dataset=NS(images_size=[256, 512], depth_norm=True))
    net = N9_0425Net(cfg, input_nc=2, K=8,
                     n3_checkpoint=None, freeze_n3=True).eval()
    x = torch.randn(2, 2, 256, 512)
    with torch.no_grad():
        out = net(x)
        for k, v in out.items():
            print(f'{k:12s}: {tuple(v.shape)}')
    n_total = sum(p.numel() for p in net.parameters()) / 1e6
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    print(f'params: {n_total:.2f}M total, {n_train:.2f}M trainable')

"""n3_0425 — Binaural-only FOA representation prediction (no depth).

Implements `docs/experiment_plan_predict_FOA.md`. The model takes a
binaural spectrogram and predicts a compressed FOA target of shape
(K, 4): either a K-bin top-eigenvector summary (Method-E) or a K-bin
channel-wise RMS summary. K ∈ {1, 8, 100} per the experimental matrix.

Pipeline (256x512 input, ngf=64):

    spec (B, 2, H, W)
      ── 8-level UNet encoder ──► bottleneck b (B, 8ngf, 1, 2)
                                            │
                       AdaptiveAvgPool ──► (B, 8ngf)
                                            │
                       LayerNorm → Linear → GELU → Linear ─► (B, K·4)
                                            │
                                      reshape (B, K, 4)
                                            │
                                       pred_rep

Forward returns dict:
    pred_rep   (B, K, 4)

The training step (``_train_step_n3_0425``) consumes rep_gt from the
5-tuple batch (use_distance_bins=True) and applies:

    L = L1(pred_rep, rep_gt) + lambda_dir · (1 − cos(pred_rep, rep_gt))

The cosine term is applied per (B, K) over the 4-channel axis, and is
skipped when rep_kind='rms' (channel-wise energy has no direction).
"""

import torch
import torch.nn as nn


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


class N3_0425Net(nn.Module):
    """Binaural UNet encoder + small head → (B, K, 4) FOA representation.

    Parameters
    ----------
    cfg       : config object — only used for sanity logs.
    input_nc  : int — audio channels (default 2 = binaural).
    K         : int — number of temporal bins. Should match
                ``cfg.dataset.rep_K`` so the loss target shape lines up.
    ngf       : int — base UNet width (default 64).
    """

    def __init__(self, cfg, input_nc=2, K=8, ngf=64, **_unused):
        super().__init__()
        self.K = int(K)
        # ---------- 8-level UNet encoder (decoder dropped — head only) ----------
        self.enc1 = _down(input_nc, ngf,    first=True)   # ngf,    H/2
        self.enc2 = _down(ngf,      ngf*2)                # 2ngf,   H/4
        self.enc3 = _down(ngf*2,    ngf*4)                # 4ngf,   H/8
        self.enc4 = _down(ngf*4,    ngf*8)                # 8ngf,   H/16
        self.enc5 = _down(ngf*8,    ngf*8)                # 8ngf,   H/32
        self.enc6 = _down(ngf*8,    ngf*8)                # 8ngf,   H/64
        self.enc7 = _down(ngf*8,    ngf*8)                # 8ngf,   H/128
        self.enc8 = _down_innermost(ngf*8, ngf*8)         # 8ngf,   H/256

        # ---------- Prediction head ----------
        # AdaptiveAvgPool → 8ngf vector → MLP → K*4. Pool keeps the head
        # input shape independent of the spec H×W (any input divisible
        # by 256 works).
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(ngf * 8),
            nn.Linear(ngf * 8, ngf * 8),
            nn.GELU(),
            nn.Linear(ngf * 8, self.K * 4),
        )

    def forward(self, x: torch.Tensor, **_unused) -> dict:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)                                   # bottleneck

        flat = self.head(e8)                                 # (B, K*4)
        rep = flat.view(-1, self.K, 4)                       # (B, K, 4)
        return {'pred_rep': rep}


if __name__ == '__main__':
    from types import SimpleNamespace as NS
    cfg = NS(dataset=NS(images_size=[256, 512], depth_norm=True))
    for K in (1, 8, 100):
        net = N3_0425Net(cfg, input_nc=2, K=K).eval()
        x = torch.randn(2, 2, 256, 512)
        with torch.no_grad():
            out = net(x)
        print(f'K={K:3d}  pred_rep={tuple(out["pred_rep"].shape)}  '
              f'params={sum(p.numel() for p in net.parameters())/1e6:.2f}M')

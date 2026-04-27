"""n9_0426 — n9_0425 cascade with a pretrained-ViT/ResNet outer hourglass.

Same I/O contract as n9_0425. Architectural change: the outer 8-level
UNet (binaural encoder + parallel EM encoder + UNet decoder) is replaced
by a pretrained ViT-B/16 or ResNet-50 hourglass that consumes
``concat([binaural, gated_em], dim=1)`` as a 3-channel input.

Inner FOA predictor is unchanged from n9_0425: the dataset's oracle
``rep_gt`` is NOT used; instead a frozen-or-finetuned ``n3_0425``
produces ``rep_pred`` from the binaural spectrogram.

Pipeline:

    spec (B, 2, H, W)
      ├──► n3_0425 (frozen OR finetuned) ──► rep_pred (B, K, 4)
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
      └──► concat([x, gated_em], dim=1) ────────────┘
                          │
                          ▼
                Pretrained ViT-B/16  OR  ResNet-50
                (input_nc = 3, ImageNet weights, optional freeze)
                          │
                          ▼
                    pred_depth (B, 1, H, W)

Forward returns dict (matches n9_0425 for downstream compatibility):
    pred_depth   (B, 1, H, W)
    gate         (K,)            sigmoid(gate_logit)
    rep_pred     (B, K, 4)       n3 RAW output (NOT gated). When
                                 lambda_sh > 0 the train step applies
                                 weighted_rep_loss(rep_pred, rep_gt) for
                                 fine-tuning n3 toward the oracle target.
    pred_sh      (B, 4)          rep_pred[:, 0, :], legacy compat stub.

Construction kwargs (passed via cfg.model.* + CLI overrides):
    backbone           : str         'vit' or 'resnet' — outer hourglass type.
    freeze_backbone    : bool        freeze pretrained backbone weights
                                     (input_adapter and decoder remain trainable).
    pretrained         : bool        load ImageNet weights (default True).
    n3_checkpoint      : str | None  path to a trained n3_0425 best_model.pth.
    freeze_n3          : bool        freeze inner n3 (default True).
    n3_ngf             : int         inner n3 base width (must match ckpt).
    K                  : int         distance-bin count (must match n3).
    gate_init          : float       initial gate_logit value.
    gate_mask          : list[float] | None  fixed per-bin mask, frozen.
"""

import os
import torch
import torch.nn as nn

from models.n3_0425 import N3_0425Net
from models.n9_0424 import make_foa_basis_erp
from models.pretrain.pretrained_vit import PretrainedViT
from models.pretrain.pretrained_resnet import PretrainedResNet


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
        print(f"  [n9_0426] n3 ckpt loaded with {len(missing)} missing keys "
              f"(first: {missing[:3]})")
    if unexpected:
        print(f"  [n9_0426] n3 ckpt loaded with {len(unexpected)} unexpected "
              f"keys (first: {unexpected[:3]})")
    if not missing and not unexpected:
        print(f"  [n9_0426] n3 ckpt loaded cleanly from {ckpt_path}")


class N9_0426Net(nn.Module):
    """Pretrained-backbone outer hourglass + n3_0425 inner FOA cascade.

    Parameters
    ----------
    cfg              : config object — uses cfg.dataset.images_size, depth_norm.
    input_nc         : int — audio channels (default 2 = binaural).
    K                : int — number of distance bins (default 8).
    gate_init        : float — initial gate_logit value. sigmoid(2.0) ≈ 0.88.
    gate_mask        : list[float] | None — fixed per-bin mask in [0, 1]. When
                        set, the gate is a frozen buffer (inv-sigmoid of the
                        mask). Same semantics as n4_0425/n9_0425.
    n3_checkpoint    : str | None — path to a pre-trained n3_0425 best_model.pth.
    freeze_n3        : bool — freeze inner n3 (eval-mode + no_grad). Default True.
    n3_ngf           : int — inner n3 UNet width (must match ckpt; default 64).
    backbone         : str — outer backbone, 'vit' or 'resnet'.
    freeze_backbone  : bool — freeze the pretrained ViT/ResNet encoder weights
                        (input_adapter + decoder always trainable).
    pretrained       : bool — load ImageNet weights for the outer backbone.
    """

    def __init__(self, cfg, input_nc=2, K=8, gate_init=2.0,
                 gate_mask=None,
                 n3_checkpoint=None, freeze_n3=True, n3_ngf=64,
                 backbone='vit', freeze_backbone=False, pretrained=True,
                 **_unused):
        super().__init__()
        self.K = int(K)
        self.depth_norm = bool(getattr(cfg.dataset, 'depth_norm', True))
        self.freeze_n3 = bool(freeze_n3)
        self.backbone_name = str(backbone).lower()

        # ---------- Inner n3_0425 (binaural → rep_pred) ----------
        self.n3 = N3_0425Net(cfg, input_nc=input_nc, K=self.K, ngf=int(n3_ngf))
        if n3_checkpoint:
            _load_n3_state_dict(self.n3, n3_checkpoint)
        else:
            print('  [n9_0426] no n3 checkpoint provided — n3 starts random')
        if self.freeze_n3:
            for p in self.n3.parameters():
                p.requires_grad = False
            self.n3.eval()
            print('  [n9_0426] n3 frozen (eval mode, no_grad in forward)')
        else:
            print('  [n9_0426] n3 trainable (gradients flow back through n3)')

        # ---------- Bin gate (same semantics as n9_0425) ----------
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

        # FOA basis (4, H, W) for projecting gated rep into the energy map.
        H_erp, W_erp = (int(v) for v in cfg.dataset.images_size)
        basis = make_foa_basis_erp(H_erp, W_erp)
        self.register_buffer('foa_basis', basis, persistent=False)

        # ---------- Outer pretrained hourglass ----------
        # Input = concat(binaural, gated_em) along channel dim.
        outer_input_nc = int(input_nc) + 1
        if self.backbone_name == 'vit':
            self.outer = PretrainedViT(
                cfg, input_nc=outer_input_nc,
                pretrained=bool(pretrained),
                freeze_encoder=bool(freeze_backbone),
            )
        elif self.backbone_name == 'resnet':
            self.outer = PretrainedResNet(
                cfg, input_nc=outer_input_nc,
                pretrained=bool(pretrained),
                freeze_encoder=bool(freeze_backbone),
            )
        else:
            raise ValueError(
                f"backbone must be 'vit' or 'resnet', got {backbone!r}")
        print(f"  [n9_0426] outer backbone={self.backbone_name} "
              f"(input_nc={outer_input_nc}, pretrained={pretrained}, "
              f"freeze_encoder={freeze_backbone})")

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
        Any rep_gt the train step passes is ignored — n9_0426 generates
        its own rep from the inner n3.
        """
        # ---- 1. Predict per-bin FOA reps from binaural via n3 ----
        if self.freeze_n3:
            with torch.no_grad():
                rep_pred = self.n3(x)['pred_rep']     # (B, K, 4)
        else:
            rep_pred = self.n3(x)['pred_rep']

        # ---- 2. Build gate-filtered energy map ----
        g = torch.sigmoid(self.gate_logit)
        gated_em = self._build_gated_em(rep_pred, g)  # (B, 1, H, W)

        # ---- 3. Outer hourglass on concat(binaural, gated_em) ----
        outer_in = torch.cat([x, gated_em], dim=1)    # (B, input_nc+1, H, W)
        depth = self.outer(outer_in)                  # (B, 1, H, W)

        return {
            'pred_depth': depth,
            'gate':       g,
            'rep_pred':   rep_pred,
            'pred_sh':    rep_pred[:, 0, :].contiguous(),
        }


if __name__ == '__main__':
    from types import SimpleNamespace as NS
    cfg = NS(dataset=NS(images_size=[256, 512], depth_norm=True))
    for backbone in ('vit', 'resnet'):
        net = N9_0426Net(cfg, input_nc=2, K=8,
                         n3_checkpoint=None, freeze_n3=True,
                         backbone=backbone, freeze_backbone=True,
                         pretrained=False).eval()
        x = torch.randn(2, 2, 256, 512)
        with torch.no_grad():
            out = net(x)
        n_total = sum(p.numel() for p in net.parameters()) / 1e6
        n_train = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
        print(f'backbone={backbone}: pred_depth={tuple(out["pred_depth"].shape)} '
              f'rep_pred={tuple(out["rep_pred"].shape)} '
              f'params={n_total:.2f}M (trainable {n_train:.2f}M)')

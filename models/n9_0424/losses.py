"""Auxiliary losses for n9_0424.

weighted_rep_loss : per-bin weighted SmoothL1 on rep coefficients.
energy_map_loss   : projection-consistency loss on log1p(signed²) maps.

Both apply the same per-bin weights (near strong, far weak — near bins
carry the informative early reflections; far-bin covariance is noise-
dominated, per EDA DIAGNOSTIC_REPORT §2 already in the repo).
"""

import torch
import torch.nn.functional as F


# Per-bin weights, keyed by K. K=1 collapses all bins into a single round-trip
# window (uniform weight 1.0); K=8 uses the eigen-geometric layout where near
# bins carry early reflections and far bins are noise-dominated.
_BIN_WEIGHTS_BY_K = {
    1: (1.0,),
    8: (1.0, 1.0, 0.85, 0.70, 0.40, 0.25, 0.10, 0.10),
}


def _bin_weights_for(K: int):
    if K not in _BIN_WEIGHTS_BY_K:
        raise ValueError(
            f"no _BIN_WEIGHTS registered for K={K}; "
            f"known K: {sorted(_BIN_WEIGHTS_BY_K)}")
    return _BIN_WEIGHTS_BY_K[K]


def _bin_weight_tensor(ref: torch.Tensor, K: int, shape):
    return torch.tensor(_bin_weights_for(K), device=ref.device,
                        dtype=ref.dtype).view(*shape)


def weighted_rep_loss(rep_pred: torch.Tensor,
                      rep_gt: torch.Tensor) -> torch.Tensor:
    """SmoothL1(rep_pred, rep_gt) with per-bin weights.

    rep_pred, rep_gt: (B, K, 4). Returns scalar.
    """
    K = rep_pred.shape[1]
    w = _bin_weight_tensor(rep_pred, K, (1, K, 1))
    loss = F.smooth_l1_loss(rep_pred, rep_gt, reduction='none')
    return (loss * w).mean()


def energy_map_loss(rep_pred: torch.Tensor,
                    rep_gt: torch.Tensor,
                    basis: torch.Tensor) -> torch.Tensor:
    """Projection-consistency loss on log1p(signed²) maps.

    rep_pred, rep_gt: (B, K, 4)
    basis:            (4, H, W)  — FOA basis at feature-grid resolution.

    Operation:
      signed_{pred,gt}  = einsum('bkc,chw->bkhw', rep, basis)
      energy_{pred,gt}  = log1p(signed²)
      loss              = weighted SmoothL1(energy_pred, energy_gt)
    """
    K = rep_pred.shape[1]
    w = _bin_weight_tensor(rep_pred, K, (1, K, 1, 1))

    signed_pred = torch.einsum('bkc,chw->bkhw', rep_pred, basis)
    signed_gt = torch.einsum('bkc,chw->bkhw', rep_gt, basis)

    energy_pred = torch.log1p(signed_pred.square())
    energy_gt = torch.log1p(signed_gt.square())

    loss = F.smooth_l1_loss(energy_pred, energy_gt, reduction='none')
    return (loss * w).mean()

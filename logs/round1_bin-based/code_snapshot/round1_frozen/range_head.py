"""RangeDepthHead — per-pixel depth range distribution.

Replaces a scalar depth head with a softmax over Br depth bins. Final
depth is recovered by expectation (default) or median over the
distribution. Bins are log-spaced by default, matching the
distance-perception literature (denser near r_min).

For background and design rationale see the EchoRange spec.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RangeDepthHead(nn.Module):
    """Convolutional head that outputs a per-pixel range distribution.

    Parameters
    ----------
    in_channels  : int — channel count of the decoder feature map.
    num_bins     : int — number of range bins (Br). Common: 64 (MVP), 48.
    r_min, r_max : float — depth-range endpoints (metres).
    spacing      : 'log' (default, denser near r_min) or 'linear'.
    output_mode  : 'expectation' (default) or 'median'.
    eps          : numerical floor for log/clamp ops.

    Forward returns a dict:
        pred_depth     (B, 1, H, W)  expected (or median) depth
        range_logits   (B, Br, H, W) raw logits
        range_prob     (B, Br, H, W) softmax probabilities
        range_entropy  (B, 1, H, W)  normalized entropy ∈ [0, 1]
        range_bins     (Br,)          buffer (also exposed for reference)
    """

    def __init__(self, in_channels: int, num_bins: int = 64,
                 r_min: float = 0.1, r_max: float = 20.0,
                 spacing: str = "log",
                 output_mode: str = "expectation",
                 eps: float = 1e-8):
        super().__init__()
        if spacing not in ("log", "linear"):
            raise ValueError(f"spacing must be 'log' or 'linear', got {spacing!r}")
        if output_mode not in ("expectation", "median"):
            raise ValueError(
                f"output_mode must be 'expectation' or 'median', got {output_mode!r}")
        if num_bins < 2:
            raise ValueError(f"num_bins must be ≥ 2, got {num_bins}")
        if r_min <= 0 or r_max <= r_min:
            raise ValueError(f"need 0 < r_min < r_max, got ({r_min}, {r_max})")

        self.num_bins = int(num_bins)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.spacing = spacing
        self.output_mode = output_mode
        self.eps = float(eps)

        if spacing == "log":
            bins = torch.exp(torch.linspace(
                math.log(r_min), math.log(r_max), num_bins))
        else:
            bins = torch.linspace(r_min, r_max, num_bins)
        self.register_buffer("range_bins", bins)            # (Br,)

        # 3-conv stack mirroring echodiffusion's last_layer_depth shape
        # so the receptive field/capacity is comparable to the scalar head.
        self.logit_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, num_bins, 3, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> dict:
        logits = self.logit_conv(feat)                     # (B, Br, H, W)
        prob = F.softmax(logits, dim=1)
        bins = self.range_bins.view(1, -1, 1, 1)           # (1, Br, 1, 1)

        if self.output_mode == "expectation":
            pred_depth = (prob * bins).sum(dim=1, keepdim=True)
        else:                                              # median
            cdf = torch.cumsum(prob, dim=1)                # (B, Br, H, W)
            mask = (cdf >= 0.5)                            # bool
            # First bin where cdf >= 0.5 — argmax of bool returns first True.
            # If no bin reaches 0.5 (numerical edge case), fall back to argmax.
            idx = mask.float().argmax(dim=1, keepdim=True)
            # Replace zero-positions where mask is all-False with last index.
            no_hit = ~mask.any(dim=1, keepdim=True)
            idx = torch.where(no_hit, torch.full_like(idx, self.num_bins - 1), idx)
            pred_depth = bins.expand(-1, -1, *prob.shape[-2:]).gather(1, idx)

        log_p = torch.log(prob.clamp(min=self.eps))
        entropy = -(prob * log_p).sum(dim=1, keepdim=True) / math.log(self.num_bins)

        return {
            "pred_depth": pred_depth,
            "range_logits": logits,
            "range_prob": prob,
            "range_entropy": entropy,
            "range_bins": self.range_bins,
        }


def soft_range_nll_loss(logits: torch.Tensor,
                        target_depth: torch.Tensor,
                        range_bins: torch.Tensor,
                        valid_mask: torch.Tensor = None,
                        sigma: float = 0.08,
                        eps: float = 1e-8,
                        weights: torch.Tensor = None) -> torch.Tensor:
    """Soft-bin NLL loss for range distributions.

    For each valid GT depth D, builds a Gaussian-in-log-space soft label
    over the range bins:

        q_j ∝ exp( - (log r_j - log D)^2 / (2σ^2) )

    and computes ``L = -Σ_j q_j log p_j`` averaged over valid pixels.

    Parameters
    ----------
    logits       : (B, Br, H, W) raw logits — softmax along dim=1.
    target_depth : (B, 1, H, W) or (B, H, W) GT depth (metres).
    range_bins   : (Br,) bin centres.
    valid_mask   : optional bool/float mask (B, 1, H, W). NaN/Inf and
                   non-positive depths are always excluded.
    sigma        : log-space soft-label width.
    eps          : numerical floor.
    weights      : optional (B, 1, H, W) or broadcastable per-pixel weight
                   for the final mean. Used for ERP cos(lat) reweighting
                   so polar pixels — which are oversampled in equirectangular
                   projection — don't dominate the loss.
    """
    if target_depth.dim() == 3:
        target_depth = target_depth.unsqueeze(1)
    if target_depth.shape[1] != 1:
        target_depth = target_depth[:, :1]

    valid = torch.isfinite(target_depth) & (target_depth > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.to(dtype=torch.bool)

    if not valid.any():
        # No valid pixels → return a zero loss that still keeps the graph
        # connected (multiply logits by 0).
        return logits.sum() * 0.0

    r_min = float(range_bins[0])
    r_max = float(range_bins[-1])
    target_clamped = target_depth.clamp(min=r_min, max=r_max)

    log_target = torch.log(target_clamped.clamp(min=eps))               # (B, 1, H, W)
    log_bins = torch.log(range_bins.clamp(min=eps)).view(1, -1, 1, 1)   # (1, Br, 1, 1)

    # Build soft labels in log-prob form, then normalize via logsumexp.
    log_q = -(log_bins - log_target).pow(2) / (2.0 * sigma * sigma)     # (B, Br, H, W)
    log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True)          # log q normalized

    log_p = F.log_softmax(logits, dim=1)                                 # (B, Br, H, W)

    # Per-pixel CE: -Σ_j q_j log p_j  =  -Σ_j exp(log_q) * log_p
    ce = -(log_q.exp() * log_p).sum(dim=1, keepdim=True)                 # (B, 1, H, W)

    if weights is None:
        return ce[valid].mean()

    w = weights.to(dtype=ce.dtype)
    if w.dim() == 1:                              # (H,) → (1, 1, H, 1)
        w = w.view(1, 1, -1, 1)
    w = w * valid.to(dtype=ce.dtype)              # zero-out invalid
    denom = w.sum().clamp(min=eps)
    return (ce * w).sum() / denom

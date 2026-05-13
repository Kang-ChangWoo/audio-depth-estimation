"""Bin-based depth heads.

Two heads share the same I/O contract:

  • RangeDepthHead          — softmax over Br log-spaced depth bins. Final
                              depth is recovered as the bin expectation
                              (default) or median.
  • HazardRangeDepthHead    — per-bin hazard alpha_j = P(first hit at j |
                              no previous hit). Final depth is rendered by
                              the differentiable first-hit weighting

                                w_j   = T_j * alpha_j
                                T_j   = prod_{k<j} (1 - alpha_k)
                                w_bg  = prod_j   (1 - alpha_j)
                                d_hat = sum_j w_j * r_j  +  w_bg * max_depth

                              No expectation blur: the renderer commits to
                              the first non-zero hit instead of averaging
                              across bins. The background term keeps the
                              output well-defined when the network believes
                              all bins are free space.

Both heads register a ``range_bins`` buffer so the surrounding training /
evaluation code can introspect the bin grid identically. Both expose
``pred_depth`` in the same units.

For background and design rationale see the EchoRange spec.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared bin-grid utility
# ---------------------------------------------------------------------------

def _make_range_bins(num_bins: int, r_min: float, r_max: float,
                     spacing: str) -> torch.Tensor:
    """Build a (Br,) tensor of bin centres.

    The returned bins are a 1-D grid over a single distance axis. The axis
    interpretation (radial vs horizontal-cylinder vs vertical-cylinder) is
    decided externally — the head treats whichever axis it is given as a
    scalar distance, and the train step is responsible for converting GT
    depth to the matching axis when ``range_bin_axis != 'radial'``.
    """
    if spacing == "log":
        return torch.exp(torch.linspace(
            math.log(r_min), math.log(r_max), num_bins))
    if spacing == "linear":
        return torch.linspace(r_min, r_max, num_bins)
    raise ValueError(f"spacing must be 'log' or 'linear', got {spacing!r}")


# ---------------------------------------------------------------------------
# Range posterior representatives (eval / inference) and a differentiable
# soft-quantile estimator (training).
# ---------------------------------------------------------------------------

# Allowed mode names for ``range_point_estimate`` and the
# ``RangeDepthHead.output_mode`` field. ``median`` is preserved for
# backward-compat (it is exactly ``quantile`` with q=0.5, but the original
# RangeDepthHead path used a separate code branch).
_POINT_ESTIMATE_MODES = (
    "expectation",
    "median",
    "map",
    "quantile",
    "temperature_expectation",
)


def range_point_estimate(logits: torch.Tensor,
                         range_bins: torch.Tensor,
                         mode: str = "expectation",
                         q: float = 0.5,
                         temperature: float = 1.0,
                         eps: float = 1e-8):
    """Convert range logits to a scalar depth via the chosen representative.

    Used in two places:
      * ``RangeDepthHead.forward`` for the head's primary output.
      * Eval-time ``range_eval_mode`` override (post-hoc representative
        sweep without retraining).

    Parameters
    ----------
    logits      : (B, R, H, W) raw logits.
    range_bins  : (R,) bin centres (metres on whichever axis was chosen).
    mode        : one of ``_POINT_ESTIMATE_MODES``.
                  - ``expectation``           : Σ_j p_j r_j with T=1.
                  - ``temperature_expectation``: Σ_j softmax(logits/T)_j r_j.
                  - ``map``                    : r_{argmax_j logits_j}.
                  - ``quantile``               : r_{first_j with cdf_j ≥ q}.
                  - ``median``                 : alias of quantile, q=0.5.
    q           : target quantile in (0, 1) (used by ``quantile`` /
                  ``median``).
    temperature : softmax temperature T > 0. Only honoured by
                  ``temperature_expectation``; for the other modes T=1.

    Returns
    -------
    pred_depth : (B, 1, H, W) — depth on the bin's axis.
    prob       : (B, R, H, W) — softmax probabilities used to compute it.
                 (For ``map``/``quantile``/``median`` we still return the
                 T=1 probabilities so the caller can compute entropy etc.)
    """
    if mode not in _POINT_ESTIMATE_MODES:
        raise ValueError(
            f"mode must be one of {_POINT_ESTIMATE_MODES}, got {mode!r}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if range_bins.dim() != 1:
        raise ValueError(f"range_bins must be 1-D, got shape {range_bins.shape}")
    bins = range_bins.view(1, -1, 1, 1).to(dtype=logits.dtype, device=logits.device)
    R = logits.shape[1]

    if mode == "temperature_expectation":
        prob_T = F.softmax(logits / float(temperature), dim=1)
        pred = (prob_T * bins).sum(dim=1, keepdim=True)
        # Always return T=1 prob too so entropy / diagnostics stay comparable.
        prob = F.softmax(logits, dim=1)
        return pred, prob

    prob = F.softmax(logits, dim=1)

    if mode == "expectation":
        pred = (prob * bins).sum(dim=1, keepdim=True)
    elif mode == "map":
        idx = prob.argmax(dim=1, keepdim=True)
        pred = bins.expand(prob.shape[0], -1, *prob.shape[-2:]).gather(1, idx)
    elif mode in ("quantile", "median"):
        q_use = 0.5 if mode == "median" else float(q)
        if not (0.0 < q_use < 1.0):
            raise ValueError(f"q must be in (0, 1), got {q_use}")
        cdf = torch.cumsum(prob, dim=1)
        mask = (cdf >= q_use)
        idx = mask.float().argmax(dim=1, keepdim=True)
        no_hit = ~mask.any(dim=1, keepdim=True)
        idx = torch.where(no_hit, torch.full_like(idx, R - 1), idx)
        pred = bins.expand(prob.shape[0], -1, *prob.shape[-2:]).gather(1, idx)
    else:                                       # unreachable
        raise ValueError(f"Unknown mode: {mode}")

    return pred, prob


def soft_quantile_depth(logits: torch.Tensor,
                        range_bins: torch.Tensor,
                        q: float = 0.5,
                        tau: float = 0.05,
                        eps: float = 1e-8) -> torch.Tensor:
    """Differentiable quantile-like depth.

    Hard quantile (``cumsum → argmax → gather``) cuts gradient flow from
    the depth loss back to the logits. We approximate it with a softmax
    over the bin axis whose un-normalized log-weights are
    ``-|cdf_j - q| / tau``: bins whose CDF sits closest to the target
    quantile get the most weight, and the result is a smooth interpolation
    between bin centres.

    As ``tau → 0`` this approaches the hard quantile; at the recommended
    ``tau = 0.05`` it remains differentiable while staying within ~1 bin
    of the hard pick. Used as an auxiliary train loss alongside the
    expectation-based BerHu/SILog so we can imitate the round-2 exp907
    *median* gain in a gradient-friendly way.

    Parameters
    ----------
    logits     : (B, R, H, W) raw logits.
    range_bins : (R,) bin centres.
    q          : target quantile in (0, 1).
    tau        : softness > 0. Smaller = closer to hard quantile.

    Returns
    -------
    depth_q : (B, 1, H, W) on the bin's axis.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q}")
    if range_bins.dim() != 1:
        raise ValueError(f"range_bins must be 1-D, got shape {range_bins.shape}")

    prob = F.softmax(logits, dim=1)
    cdf = torch.cumsum(prob, dim=1)
    weights = F.softmax(-(cdf - float(q)).abs() / float(tau), dim=1)
    bins = range_bins.view(1, -1, 1, 1).to(dtype=logits.dtype, device=logits.device)
    return (weights * bins).sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Softmax range head (existing — preserved verbatim in behaviour)
# ---------------------------------------------------------------------------

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
                 output_quantile: float = 0.5,
                 output_temperature: float = 1.0,
                 eps: float = 1e-8):
        super().__init__()
        if output_mode not in _POINT_ESTIMATE_MODES:
            raise ValueError(
                f"output_mode must be one of {_POINT_ESTIMATE_MODES}, "
                f"got {output_mode!r}")
        if num_bins < 2:
            raise ValueError(f"num_bins must be ≥ 2, got {num_bins}")
        if r_min <= 0 or r_max <= r_min:
            raise ValueError(f"need 0 < r_min < r_max, got ({r_min}, {r_max})")
        if not (0.0 < float(output_quantile) < 1.0):
            raise ValueError(
                f"output_quantile must be in (0, 1), got {output_quantile}")
        if float(output_temperature) <= 0:
            raise ValueError(
                f"output_temperature must be > 0, got {output_temperature}")

        self.num_bins = int(num_bins)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.spacing = spacing
        self.output_mode = output_mode
        self.output_quantile = float(output_quantile)
        self.output_temperature = float(output_temperature)
        self.eps = float(eps)

        self.register_buffer("range_bins",
                             _make_range_bins(num_bins, r_min, r_max, spacing))

        # 3-conv stack mirroring echodiffusion's last_layer_depth shape
        # so the receptive field/capacity is comparable to the scalar head.
        self.logit_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, num_bins, 3, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> dict:
        logits = self.logit_conv(feat)                     # (B, Br, H, W)
        # Always materialise the T=1 softmax — entropy/diagnostics use it,
        # and the eval-time representative override needs it as a fallback.
        pred_depth, prob = range_point_estimate(
            logits=logits,
            range_bins=self.range_bins,
            mode=self.output_mode,
            q=self.output_quantile,
            temperature=self.output_temperature,
            eps=self.eps,
        )

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


# ---------------------------------------------------------------------------
# Hazard range head (new — A-Hazard)
# ---------------------------------------------------------------------------

class HazardRangeDepthHead(nn.Module):
    """First-hit hazard rendering head.

    Predicts per-bin hazards ``alpha_j = sigmoid(logits_j) ∈ (0, 1)``,
    interpreted as ``P(first hit at j | no previous hit)``. Differentiable
    first-hit rendering recovers a single depth per pixel:

        T_j  = prod_{k<j} (1 - alpha_k)        (transmittance, T_0 = 1)
        w_j  = T_j * alpha_j                   (first-hit weights)
        w_bg = prod_j (1 - alpha_j)            (no-hit / background)
        d̂   = sum_j w_j * r_j + w_bg * max_depth

    Compared to the softmax range head, this avoids expectation blur
    when the bin distribution is multimodal: each hit absorbs all
    remaining "ray probability", so a confident near-bin fully short-
    circuits any far-bin contribution. The background term keeps d̂
    well-defined when alpha is uniformly small.

    Parameters
    ----------
    in_channels : int   — decoder feature channels.
    num_bins    : int   — Br.
    r_min, r_max: float — bin endpoints (metres).
    spacing     : 'log' (default) or 'linear'.
    max_depth   : float — background depth value used by w_bg. Should
                          match the dataset clamp so background pixels
                          decode to the metric's far-clip.
    bias_init   : float — initial bias of the final 1×1 conv. Drives the
                          starting alpha toward sigmoid(bias_init); the
                          default −4.6 → α≈0.01 keeps T_j ≈ 1 across all
                          bins at initialization, so warmup gradients
                          come almost entirely from L_hit / L_free
                          rather than from a near-degenerate render.
    eps         : float — numerical floor.

    Forward returns a dict with the same ``pred_depth`` and ``range_bins``
    keys as RangeDepthHead, plus hazard-specific diagnostics:

        pred_depth        (B, 1, H, W)   rendered depth (metres)
        range_logits      (B, Br, H, W)  raw logits (kept for compat)
        range_bins        (Br,)          bin centres
        hazard_alpha      (B, Br, H, W)  α_j  (after clamp)
        hazard_weights    (B, Br, H, W)  w_j  (first-hit weights)
        hazard_bg_weight  (B, 1, H, W)   w_bg (no-hit weight)
        range_entropy     (B, 1, H, W)   normalized entropy of [w_j, w_bg]
                                         over Br+1 categories ∈ [0, 1]
        free_length       (B, 1, H, W)   Σ_j T_j — proxy for the
                                         expected free-space distance
                                         (in bins); useful diagnostic
                                         for whether the head learned a
                                         confident first hit.
    """

    def __init__(self, in_channels: int, num_bins: int = 32,
                 r_min: float = 0.1, r_max: float = 10.0,
                 spacing: str = "log",
                 max_depth: float = 10.0,
                 bias_init: float = -4.6,
                 eps: float = 1e-8):
        super().__init__()
        if num_bins < 2:
            raise ValueError(f"num_bins must be ≥ 2, got {num_bins}")
        if r_min <= 0 or r_max <= r_min:
            raise ValueError(f"need 0 < r_min < r_max, got ({r_min}, {r_max})")
        if max_depth < r_max:
            # Soft warning — render still works, but background absorbs at a
            # depth shorter than the farthest bin centre, which can pull
            # near-saturating predictions slightly inwards. The intended
            # configuration is max_depth >= r_max.
            pass

        self.num_bins = int(num_bins)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.spacing = spacing
        self.max_depth = float(max_depth)
        self.bias_init = float(bias_init)
        self.eps = float(eps)

        self.register_buffer("range_bins",
                             _make_range_bins(num_bins, r_min, r_max, spacing))

        # Same 3-conv stack as RangeDepthHead for capacity parity. Final
        # bias is initialized so that the starting α is small.
        self.logit_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, num_bins, 3, 1, 1),
        )
        nn.init.constant_(self.logit_conv[-1].bias, self.bias_init)

    def forward(self, feat: torch.Tensor) -> dict:
        logits = self.logit_conv(feat)                          # (B, R, H, W)
        # Clamp α away from {0, 1} so log/cumprod stay finite.
        alpha = torch.sigmoid(logits).clamp(min=self.eps,
                                            max=1.0 - self.eps)

        one_minus = 1.0 - alpha                                  # (B, R, H, W)
        # T_inclusive[..., j] = prod_{k<=j} (1 - alpha_k).
        T_inclusive = torch.cumprod(one_minus, dim=1)
        # T_prefix[..., j]    = prod_{k<j}  (1 - alpha_k); T_0 = 1.
        T_prefix = torch.cat(
            [torch.ones_like(T_inclusive[:, :1]), T_inclusive[:, :-1]], dim=1)
        weights = T_prefix * alpha                               # (B, R, H, W)
        bg_weight = T_inclusive[:, -1:]                          # (B, 1, H, W)

        bins = self.range_bins.view(1, -1, 1, 1)                 # (1, R, 1, 1)
        pred_depth = (weights * bins).sum(dim=1, keepdim=True) \
            + bg_weight * self.max_depth                          # (B, 1, H, W)

        # Diagnostics ---------------------------------------------------
        # Full categorical (w_1, ..., w_R, w_bg) sums to 1 by construction
        # (telescoping: 1 = T_0 = T_R + Σ_j w_j after expanding).
        full = torch.cat([weights, bg_weight], dim=1)            # (B, R+1, H, W)
        log_full = torch.log(full.clamp(min=self.eps))
        entropy = -(full * log_full).sum(dim=1, keepdim=True) \
            / math.log(self.num_bins + 1)                         # ∈ [0, 1]
        # Σ_j T_prefix_j ∈ [0, R]. Larger ⇒ ray traveled further before
        # being absorbed (or never absorbed at all).
        free_length = T_prefix.sum(dim=1, keepdim=True)

        return {
            "pred_depth":       pred_depth,
            "range_logits":     logits,
            "range_bins":       self.range_bins,
            "hazard_alpha":     alpha,
            "hazard_weights":   weights,
            "hazard_bg_weight": bg_weight,
            "range_entropy":    entropy,
            "free_length":      free_length,
        }


def hazard_supervision_loss(logits: torch.Tensor,
                            target_depth: torch.Tensor,
                            range_bins: torch.Tensor,
                            log_delta: float = None,
                            valid_mask: torch.Tensor = None,
                            weights: torch.Tensor = None,
                            use_hit: bool = True,
                            use_free: bool = True,
                            eps: float = 1e-8) -> dict:
    """Per-pixel hit/free supervision for the hazard head.

    For each pixel with valid GT depth ``D`` and bin centres ``r_j``:

      • free   :  bins with  log(r_j) - log(D) < -log_delta  → α target = 0
      • hit    :  bins with |log(r_j) - log(D)| ≤  log_delta → α target = 1
      • behind :  bins with  log(r_j) - log(D) >  log_delta  → not supervised

    The supervision is computed in log-depth space so the hit window has a
    constant *bin-count* width regardless of where ``D`` falls along the
    log-spaced grid (with log spacing, log(r_{j+1}) - log(r_j) is constant).

    Both losses are computed via ``binary_cross_entropy_with_logits`` for
    numerical stability and reduced by a weighted mean. ``weights`` is the
    ERP cos(latitude) per-row weight reused from the softmax NLL path so
    the hazard head shares the same anti-aliasing behaviour.

    Returns
    -------
    dict with keys ``'hit'`` and ``'free'`` (each a 0-d tensor). Missing
    keys when the corresponding flag is disabled.
    """
    target = target_depth
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if target.shape[1] != 1:
        target = target[:, :1]

    # Default hit-window width: half the log-bin spacing → contiguous,
    # non-overlapping windows for adjacent GT depths on a log grid.
    if log_delta is None:
        if range_bins.numel() >= 2:
            ratio = (range_bins[1] / range_bins[0]).clamp(min=eps).item()
            log_delta = math.log(ratio) * 0.5
        else:
            log_delta = 0.1
    log_delta = float(log_delta)

    bins = range_bins.view(1, -1, 1, 1)                          # (1, R, 1, 1)
    log_bins = torch.log(bins.clamp(min=eps))
    log_target = torch.log(target.clamp(min=eps))                # (B, 1, H, W)

    # log(r_j) - log(D) per (pixel, bin).
    diff = log_bins - log_target                                 # (B, R, H, W)

    valid = torch.isfinite(target) & (target > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.to(dtype=torch.bool)

    free_mask = (diff < -log_delta) & valid                      # (B, R, H, W)
    hit_mask  = (diff.abs() <= log_delta) & valid                # (B, R, H, W)

    # Per-pixel weights (cos-lat). Each bin in a pixel inherits its weight.
    if weights is not None:
        w = weights.to(dtype=logits.dtype)
        if w.dim() == 1:                                         # (H,) → (1,1,H,1)
            w = w.view(1, 1, -1, 1)
    else:
        w = None

    losses = {}
    if use_hit:
        target_hit = torch.ones_like(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, target_hit, reduction='none')                # (B, R, H, W)
        m = hit_mask.to(dtype=bce.dtype)
        if w is not None:
            m = m * w
        denom = m.sum().clamp(min=eps)
        # Keep the graph connected even if no hit pixels were sampled in
        # this micro-batch (rare, but possible at very high masking rates).
        losses['hit'] = (bce * m).sum() / denom \
            if hit_mask.any() else logits.sum() * 0.0

    if use_free:
        target_free = torch.zeros_like(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, target_free, reduction='none')               # (B, R, H, W)
        m = free_mask.to(dtype=bce.dtype)
        if w is not None:
            m = m * w
        denom = m.sum().clamp(min=eps)
        losses['free'] = (bce * m).sum() / denom \
            if free_mask.any() else logits.sum() * 0.0

    return losses


# ===========================================================================
# Round-4 hazard losses (rendered first-hit NLL / survival / soft-hit BCE).
#
# Motivation: in round 3 the raw α-hit BCE (`hazard_supervision_loss(use_hit=
# True)` above) saturated sigmoid α at near-1, killing gradients via
# ∂α/∂logit = α(1-α). The losses below supervise the *rendered* quantities
# (w_j or T_j) which depend on the entire α prefix, so each gradient is
# distributed across many bins and a single saturated bin can't trap the
# whole pixel. See logs/n9_0427_train/E20260428-haz_consult_request.md.
#
# All three new losses share two design rules:
#   • Numerical stability via logsigmoid + cumsum-in-log-space (never
#     materialize alpha and 1-alpha at near-{0,1} and then take log).
#   • Far-censored pixels (gt >= far_thresh, default 9.8) are EXCLUDED
#     from event/hit supervision but kept as "no-hit so far" evidence in
#     survival supervision (asserts "we know D > r_j for r_j < far_thresh").
# ===========================================================================


def _broadcast_pixel_weights(weights, ref):
    """Bring per-pixel weight tensor to (B, 1, H, W) broadcastable to ref."""
    if weights is None:
        return None
    w = weights.to(dtype=ref.dtype, device=ref.device)
    if w.dim() == 1:                              # (H,) → (1, 1, H, 1)
        w = w.view(1, 1, -1, 1)
    return w


def rendered_event_nll(logits: torch.Tensor,
                       target_depth: torch.Tensor,
                       range_bins: torch.Tensor,
                       valid_mask: torch.Tensor = None,
                       weights: torch.Tensor = None,
                       sigma_bins: float = 1.0,
                       far_thresh: float = 9.8,
                       eps: float = 1e-8) -> torch.Tensor:
    """NLL of the *rendered first-hit weight* w_j against a soft GT target.

    Replaces the round-3 raw-α-hit BCE. The supervision target is a
    log-Gaussian over the bin grid centred at log(GT), with σ proportional
    to the log-bin spacing (default σ = 1 bin):

        log_α_j = logsigmoid(logits_j)
        log(1−α_j) = logsigmoid(−logits_j)
        log T_j = Σ_{k<j} log(1−α_k)            (exclusive cumsum)
        log w_j = log T_j + log α_j             (rendered first-hit log-prob)

        q_j  ∝ exp( −0.5 ((log r_j − log D)/σ)² )       (normalize over j)
        L    = −Σ_j q_j · log w_j                       (weighted mean)

    Because w_j is a *product* across the entire α prefix, gradients on
    each α_k are bounded (∂log w_j / ∂logit_k is finite even when α_k → 1
    — it's just −α_k or −(1−α_k), no α(1−α) bottleneck).

    Far-censored pixels (target ≥ far_thresh) are excluded from the
    valid set: their GT depth is a clamped sentinel, not a real surface,
    so giving them an event target would push α to the last bin
    artificially.

    Parameters
    ----------
    logits       : (B, Br, H, W) raw hazard logits.
    target_depth : (B, 1, H, W) or (B, H, W) GT depth in metres.
    range_bins   : (Br,) bin centres in metres.
    valid_mask   : optional (B, 1, H, W) bool/float mask.
    weights      : optional (H,) or (B, 1, H, W) per-pixel weight (cos-lat).
    sigma_bins   : σ in *units of bin-log-spacing*. Default 1.0 means the
                   soft target spans roughly ±1 bin.
    far_thresh   : metres. Pixels with GT ≥ this are excluded.
    """
    target = target_depth
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if target.shape[1] != 1:
        target = target[:, :1]

    log_alpha = F.logsigmoid(logits)               # log α
    log_1ma   = F.logsigmoid(-logits)              # log(1 − α)

    # Exclusive log-transmittance:  log T_j = Σ_{k<j} log(1 − α_k);  T_0 = 1.
    zero = torch.zeros_like(log_1ma[:, :1])
    log_T = torch.cumsum(
        torch.cat([zero, log_1ma[:, :-1]], dim=1), dim=1)   # (B, R, H, W)
    log_w = log_T + log_alpha                                # (B, R, H, W)

    log_bins = torch.log(range_bins.view(1, -1, 1, 1).clamp(min=eps))
    log_target = torch.log(target.clamp(min=eps))            # (B, 1, H, W)

    if range_bins.numel() < 2:
        raise ValueError("rendered_event_nll requires ≥ 2 bins.")
    bin_log_step = torch.log(
        (range_bins[1] / range_bins[0]).clamp(min=eps))
    sigma = float(sigma_bins) * bin_log_step

    # Soft target distribution q_j (Gaussian in log-depth space).
    q = torch.exp(-0.5 * ((log_bins - log_target) / sigma).pow(2))   # (B, R, H, W)

    valid = torch.isfinite(target) & (target > 0) & (target < float(far_thresh))
    if valid_mask is not None:
        valid = valid & valid_mask.to(dtype=torch.bool)
    valid_f = valid.to(dtype=q.dtype)                                 # (B, 1, H, W)

    # Zero-out invalid pixels (broadcasts over R) and renormalize.
    q = q * valid_f
    q = q / q.sum(dim=1, keepdim=True).clamp(min=eps)

    # Per-pixel CE on the rendered weight: −Σ_j q_j log w_j  (B, 1, H, W).
    loss_map = -(q * log_w).sum(dim=1, keepdim=True)
    loss_map = loss_map * valid_f                                    # zero invalid

    pix_w = _broadcast_pixel_weights(weights, loss_map)
    if pix_w is not None:
        loss_map = loss_map * pix_w
        denom = (valid_f * pix_w).sum().clamp(min=eps)
    else:
        denom = valid_f.sum().clamp(min=eps)

    if not valid.any():
        return logits.sum() * 0.0
    return loss_map.sum() / denom


def survival_loss(logits: torch.Tensor,
                  target_depth: torch.Tensor,
                  range_bins: torch.Tensor,
                  valid_mask: torch.Tensor = None,
                  weights: torch.Tensor = None,
                  tau_bins: float = 1.0,
                  far_thresh: float = 9.8,
                  eps: float = 1e-8) -> torch.Tensor:
    """Survival / ordinal supervision: BCE between predicted and target
    survival probabilities ``S_j = P(D > r_j) = ∏_{k≤j}(1 − α_k)``.

    Target:
        S_gt(r_j) = sigmoid( (log D − log r_j) / τ )
            → 1 when r_j < D  (bin in front of the surface)
            → 0 when r_j > D  (bin behind the surface)

    Numerically stable via:
        log S_j  = Σ_{k≤j} logsigmoid(−logits_k)            (inclusive cumsum)
        log(1−S) = log1p(−exp(log_S))                        (clamped)

    Far-censored handling (target_depth ≥ far_thresh):
        The pixel's true depth is unknown beyond far_thresh. We still
        know D > r_j for any bin r_j < far_thresh, so we supervise those
        bins with target S_gt = 1; bins with r_j ≥ far_thresh are masked.

    Returns the weighted mean BCE over (pixel × bin).
    """
    target = target_depth
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if target.shape[1] != 1:
        target = target[:, :1]

    if range_bins.numel() < 2:
        raise ValueError("survival_loss requires ≥ 2 bins.")
    bin_log_step = torch.log(
        (range_bins[1] / range_bins[0]).clamp(min=eps))
    tau = float(tau_bins) * bin_log_step

    log_bins = torch.log(range_bins.view(1, -1, 1, 1).clamp(min=eps))
    log_target = torch.log(target.clamp(min=eps))

    # Predicted survival in log-space.
    log_1ma = F.logsigmoid(-logits)                          # (B, R, H, W)
    log_S = torch.cumsum(log_1ma, dim=1)                     # log S_j

    # Clamp away from 0 (i.e. log_S = 0 ⇔ S = 1) for log1p safety.
    log_S_for_neg = log_S.clamp(max=-eps)
    log_1mS = torch.log1p(-torch.exp(log_S_for_neg))         # log(1 − S)

    # GT survival: 1 if r_j < D, smoothly transitioning at r_j = D.
    S_gt = torch.sigmoid((log_target - log_bins) / tau)      # (B, R, H, W)

    # BCE_j = −[S_gt · log_S + (1 − S_gt) · log(1 − S)]
    bce = -(S_gt * log_S + (1.0 - S_gt) * log_1mS)           # (B, R, H, W)

    # Pixel validity (any positive finite GT).
    pixel_valid = torch.isfinite(target) & (target > 0)
    if valid_mask is not None:
        pixel_valid = pixel_valid & valid_mask.to(dtype=torch.bool)

    # Bin-level mask:
    #   • non-censored pixel  →  all bins valid.
    #   • censored pixel      →  only bins with r_j < far_thresh valid.
    far_censored = (target >= float(far_thresh))             # (B, 1, H, W)
    bins_below_far = (range_bins.view(1, -1, 1, 1)
                      < float(far_thresh)).to(dtype=bce.dtype)   # (1, R, 1, 1)
    bin_mask = torch.where(
        far_censored, bins_below_far.expand_as(bce),
        torch.ones_like(bce)
    )
    bin_mask = bin_mask * pixel_valid.to(dtype=bce.dtype)

    pix_w = _broadcast_pixel_weights(weights, bce)
    if pix_w is not None:
        bin_mask = bin_mask * pix_w

    denom = bin_mask.sum().clamp(min=eps)
    if not pixel_valid.any():
        return logits.sum() * 0.0
    return (bce * bin_mask).sum() / denom


def soft_hit_bce_loss(logits: torch.Tensor,
                      target_depth: torch.Tensor,
                      range_bins: torch.Tensor,
                      log_delta: float = None,
                      soft_target: float = 0.75,
                      valid_mask: torch.Tensor = None,
                      weights: torch.Tensor = None,
                      far_thresh: float = 9.8,
                      eps: float = 1e-8) -> torch.Tensor:
    """Like the round-3 raw-hit BCE but with a target < 1 to keep sigmoid
    away from saturation. The hit window selection is identical to
    ``hazard_supervision_loss`` (±half-log-bin around log(GT)).

    With target_alpha = 0.75 (logit ≈ 1.10), the optimum α is bounded —
    once the bin's α matches 0.75, gradient flips sign, so α can't run
    away to 1. Used as a "salvage" diagnostic for raw-α supervision.
    """
    target = target_depth
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if target.shape[1] != 1:
        target = target[:, :1]

    if log_delta is None:
        if range_bins.numel() >= 2:
            ratio = (range_bins[1] / range_bins[0]).clamp(min=eps).item()
            log_delta = math.log(ratio) * 0.5
        else:
            log_delta = 0.1
    log_delta = float(log_delta)

    log_bins = torch.log(range_bins.view(1, -1, 1, 1).clamp(min=eps))
    log_target = torch.log(target.clamp(min=eps))
    diff = log_bins - log_target                                  # (B, R, H, W)

    valid = torch.isfinite(target) & (target > 0) & (target < float(far_thresh))
    if valid_mask is not None:
        valid = valid & valid_mask.to(dtype=torch.bool)
    hit_mask = (diff.abs() <= log_delta) & valid

    target_t = torch.full_like(logits, float(soft_target))
    bce = F.binary_cross_entropy_with_logits(logits, target_t, reduction='none')

    m = hit_mask.to(dtype=bce.dtype)
    pix_w = _broadcast_pixel_weights(weights, bce)
    if pix_w is not None:
        m = m * pix_w
    denom = m.sum().clamp(min=eps)
    if not hit_mask.any():
        return logits.sum() * 0.0
    return (bce * m).sum() / denom


def hazard_free_loss(logits: torch.Tensor,
                     target_depth: torch.Tensor,
                     range_bins: torch.Tensor,
                     log_delta: float = None,
                     valid_mask: torch.Tensor = None,
                     weights: torch.Tensor = None,
                     far_thresh: float = 9.8,
                     eps: float = 1e-8) -> torch.Tensor:
    """Standalone free-space loss (raw α=0 BCE on bins clearly closer
    than GT). Same logic as ``hazard_supervision_loss(use_free=True)`` but
    extracted so it can be combined with arbitrary primary losses.

    Far-censored pixels are still useful for free supervision (any bin
    closer than the censoring threshold is unambiguously free), so we
    keep them in the valid set with no special handling.
    """
    target = target_depth
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if target.shape[1] != 1:
        target = target[:, :1]

    if log_delta is None:
        if range_bins.numel() >= 2:
            ratio = (range_bins[1] / range_bins[0]).clamp(min=eps).item()
            log_delta = math.log(ratio) * 0.5
        else:
            log_delta = 0.1
    log_delta = float(log_delta)

    log_bins = torch.log(range_bins.view(1, -1, 1, 1).clamp(min=eps))
    log_target = torch.log(target.clamp(min=eps))
    diff = log_bins - log_target

    valid = torch.isfinite(target) & (target > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.to(dtype=torch.bool)
    free_mask = (diff < -log_delta) & valid

    bce = F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction='none')
    m = free_mask.to(dtype=bce.dtype)
    pix_w = _broadcast_pixel_weights(weights, bce)
    if pix_w is not None:
        m = m * pix_w
    denom = m.sum().clamp(min=eps)
    if not free_mask.any():
        return logits.sum() * 0.0
    return (bce * m).sum() / denom


# ---------------------------------------------------------------------------
# Self-test (CPU smoke check) — run with: `python -m models.bin_based.range_head`
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)
    B, C, H, W, R = 2, 16, 8, 16, 32
    feat = torch.randn(B, C, H, W, requires_grad=True)
    gt = torch.rand(B, 1, H, W) * 9.0 + 0.5                       # (0.5, 9.5) m

    # ---------------- RangeDepthHead: 5 representatives ----------------
    print("--- RangeDepthHead point-estimate modes smoke test ---")
    for mode in _POINT_ESTIMATE_MODES:
        head_r = RangeDepthHead(
            C, num_bins=R, r_min=0.1, r_max=10.0, spacing='log',
            output_mode=mode, output_quantile=0.5, output_temperature=0.75)
        feat_r = torch.randn(B, C, H, W, requires_grad=True)
        out_r = head_r(feat_r)
        pd = out_r['pred_depth']
        assert pd.shape == (B, 1, H, W)
        assert (pd >= 0.1 - 1e-4).all() and (pd <= 10.0 + 1e-4).all(), \
            f"[{mode}] pred_depth out of range: min={pd.min()} max={pd.max()}"
        # Entropy ∈ [0, 1] in nats / log(R).
        ent = out_r['range_entropy']
        assert ent.shape == (B, 1, H, W) and \
            (ent >= -1e-4).all() and (ent <= 1.0 + 1e-4).all(), \
            f"[{mode}] entropy out of [0,1]: min={ent.min()} max={ent.max()}"
        # Backward must produce a finite gradient on the feat for ALL modes.
        # MAP/quantile/median (hard) won't gradient through bin selection,
        # but the entropy term is differentiable, so feat.grad is non-None.
        loss = pd.mean() + ent.mean()
        loss.backward()
        assert feat_r.grad is not None and torch.isfinite(feat_r.grad).all(), \
            f"[{mode}] backward produced NaN/None grad"
        # For expectation / temperature_expectation, also check pred_depth-only
        # backward is non-zero (gradient should flow through soft argmax).
        if mode in ("expectation", "temperature_expectation"):
            head2 = RangeDepthHead(
                C, num_bins=R, r_min=0.1, r_max=10.0, spacing='log',
                output_mode=mode, output_quantile=0.5, output_temperature=0.75)
            f2 = torch.randn(B, C, H, W, requires_grad=True)
            head2(f2)['pred_depth'].mean().backward()
            assert f2.grad is not None and f2.grad.abs().sum().item() > 0, \
                f"[{mode}] pred_depth-only gradient is zero"
        print(f"  {mode:24s} pd∈[{pd.min().item():.3f}, {pd.max().item():.3f}] OK")

    # ---------------- soft_quantile_depth: differentiable median ----------
    print("--- soft_quantile_depth smoke test ---")
    head_sq = RangeDepthHead(C, num_bins=R, r_min=0.1, r_max=10.0,
                             spacing='log', output_mode='expectation')
    feat_sq = torch.randn(B, C, H, W, requires_grad=True)
    out_sq = head_sq(feat_sq)
    for q_val in (0.35, 0.50, 0.65):
        depth_q = soft_quantile_depth(
            out_sq['range_logits'], head_sq.range_bins,
            q=q_val, tau=0.05)
        assert depth_q.shape == (B, 1, H, W)
        assert (depth_q >= 0.1 - 1e-4).all() and (depth_q <= 10.0 + 1e-4).all()
        # Differentiability: pred-only backward must produce non-zero feat grad.
        f3 = torch.randn(B, C, H, W, requires_grad=True)
        out3 = head_sq(f3)
        d3 = soft_quantile_depth(
            out3['range_logits'], head_sq.range_bins, q=q_val, tau=0.05)
        d3.mean().backward()
        assert f3.grad is not None and f3.grad.abs().sum().item() > 0, \
            f"[soft_quantile q={q_val}] gradient is zero (broken)"
        print(f"  q={q_val:.2f} tau=0.05 → "
              f"pd∈[{depth_q.min().item():.3f}, {depth_q.max().item():.3f}] OK")

    # ---------------- range_point_estimate: standalone path -----------------
    print("--- range_point_estimate (eval-side override) smoke test ---")
    head_e = RangeDepthHead(C, num_bins=R, r_min=0.1, r_max=10.0,
                            spacing='log', output_mode='expectation')
    feat_e = torch.randn(B, C, H, W)
    with torch.no_grad():
        out_e = head_e(feat_e)
        for mode in _POINT_ESTIMATE_MODES:
            for T in (0.5, 1.0, 1.5):
                pred_T, _ = range_point_estimate(
                    out_e['range_logits'], head_e.range_bins,
                    mode=mode, q=0.5, temperature=T)
                assert pred_T.shape == (B, 1, H, W)
                assert (pred_T >= 0.1 - 1e-4).all() and \
                       (pred_T <= 10.0 + 1e-4).all()
    print("  all (mode × temperature) combinations OK")

    print("--- HazardRangeDepthHead smoke test ---")
    head = HazardRangeDepthHead(C, num_bins=R, r_min=0.1, r_max=10.0,
                                spacing='log', max_depth=10.0,
                                bias_init=-4.6)
    out = head(feat)
    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f"  {k:18s} shape={tuple(v.shape)}  "
                  f"min={v.min().item():.4f} max={v.max().item():.4f}")
    # Assertions
    assert out['pred_depth'].shape == (B, 1, H, W)
    pd = out['pred_depth']
    assert (pd >= 0).all() and (pd <= 10.0 + 1e-4).all(), \
        f"pred_depth out of range [0, 10]: min={pd.min()}, max={pd.max()}"
    full = torch.cat([out['hazard_weights'], out['hazard_bg_weight']], dim=1)
    s = full.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4), \
        f"weights+bg do not sum to 1: max-deviation={(s - 1).abs().max()}"

    losses = hazard_supervision_loss(
        out['range_logits'], gt, head.range_bins,
        weights=torch.linspace(0.1, 1.0, H))
    print(f"  L_hit  = {losses['hit'].item():.4f}")
    print(f"  L_free = {losses['free'].item():.4f}")
    (losses['hit'] + losses['free'] + pd.mean()).backward()
    g = feat.grad
    assert g is not None and torch.isfinite(g).all(), \
        "Backward pass produced NaN/None gradients."
    print("  raw_hit backward OK")

    print("--- R4 losses smoke test ---")
    feat2 = torch.randn(B, C, H, W, requires_grad=True)
    out2 = head(feat2)
    cos_w = torch.linspace(0.1, 1.0, H)
    L_event = rendered_event_nll(
        out2['range_logits'], gt, head.range_bins,
        weights=cos_w, sigma_bins=1.0, far_thresh=9.8)
    L_surv = survival_loss(
        out2['range_logits'], gt, head.range_bins,
        weights=cos_w, tau_bins=1.0, far_thresh=9.8)
    L_soft = soft_hit_bce_loss(
        out2['range_logits'], gt, head.range_bins,
        weights=cos_w, soft_target=0.75, far_thresh=9.8)
    L_free = hazard_free_loss(
        out2['range_logits'], gt, head.range_bins,
        weights=cos_w, far_thresh=9.8)
    print(f"  rendered_event_nll  = {L_event.item():.4f}")
    print(f"  survival_loss       = {L_surv.item():.4f}")
    print(f"  soft_hit_bce_loss   = {L_soft.item():.4f}")
    print(f"  hazard_free_loss    = {L_free.item():.4f}")
    for name, l in (('event', L_event), ('surv', L_surv),
                    ('soft', L_soft), ('free', L_free)):
        assert torch.isfinite(l), f"{name} produced non-finite loss"
    (L_event + L_surv + L_soft + L_free + out2['pred_depth'].mean()).backward()
    assert feat2.grad is not None and torch.isfinite(feat2.grad).all(), \
        "R4 losses backward produced NaN/None grad."
    print("  R4 backward OK")

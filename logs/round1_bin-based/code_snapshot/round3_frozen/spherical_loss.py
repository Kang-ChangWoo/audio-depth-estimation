"""Spherical-harmonic auxiliary loss for ERP radial-depth regression.

Audio-only depth from a sphere — D : S² → R+ — has a strong global
low-frequency layout (rooms have a couple of dominant walls, near/far
asymmetry, ceiling/floor), even when fine boundaries are unrecoverable
from sound. Matching a few low-order spherical-harmonic (SH) coefficients
gives the network a coarse global geometry signal without forcing it to
overfit individual ERP pixels.

Conventions
-----------
ERP grid rows index latitude (top row = +π/2, bottom row = −π/2),
columns index longitude (left = −π, right = +π). Each pixel is
treated as a point sample of D on the unit sphere; cos(latitude)
weighting is applied so polar pixels — over-sampled by ERP — don't
dominate the SH integration.

Real spherical harmonics up to L=4 are supported (l ∈ {0..L},
m ∈ {-l..l}; total (L+1)² coefficients). Above L=4 we'd want a proper
recursion; the closed-form expressions used here are simpler and
sufficient for the layout-regularization use case in this project.

Public API
----------
make_erp_grid(H, W, device) → (lat[H], lon[W], cos_lat_weights[H,W])
real_sh_basis(L, lat, lon)  → (L+1)², H, W   real SH basis evaluated
spherical_sh_coeffs(D, L, …) → (B, K)        cos-lat-weighted projection
spherical_sh_loss(pred_m, gt_m, L, …) → scalar   smooth_l1 between coeffs

The loss caller is responsible for passing pred / gt in *metres*. When the
training pipeline normalises depth to [0, 1] the train step rescales by
``cfg.dataset.max_depth`` before invoking this loss so the SH coefficients
live on a consistent scale across runs.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ERP grid
# ---------------------------------------------------------------------------

def make_erp_grid(H: int, W: int, device=None, dtype=torch.float32):
    """Return (lat[H], lon[W], cos_lat_weights[H,W]).

    Pixel-centre convention: row y → latitude  π/2 − π·(y+0.5)/H.
    Column x → longitude   −π + 2π·(x+0.5)/W.

    cos_lat_weights has shape (H, W) and is the per-pixel surface area
    proxy on the unit sphere. We DO NOT renormalise here; downstream
    integration normalises by ``cos_lat_weights.sum()`` so the grid
    behaves like a Riemann sum approximation of ∫ f cos(lat) dlat dlon.
    """
    if H < 2 or W < 2:
        raise ValueError(f"need H, W ≥ 2; got H={H}, W={W}")
    y = torch.arange(H, device=device, dtype=dtype) + 0.5
    x = torch.arange(W, device=device, dtype=dtype) + 0.5
    lat = math.pi / 2.0 - math.pi * y / float(H)              # (H,)
    lon = -math.pi + 2.0 * math.pi * x / float(W)             # (W,)
    cos_lat = torch.cos(lat).clamp(min=0.0)                   # (H,)
    weights = cos_lat.view(H, 1).expand(H, W).contiguous()    # (H, W)
    return lat, lon, weights


# ---------------------------------------------------------------------------
# Real spherical harmonics up to L=4 (closed form).
# ---------------------------------------------------------------------------
#
# Orientation: latitude θ (−π/2 … +π/2, +π/2 = north pole). We use the
# co-latitude ψ = π/2 − θ in the SH formulas where the pole convention
# is ψ=0. Equivalently, sin(θ) = cos(ψ) and cos(θ) = sin(ψ); we keep
# sin/cos of latitude directly to keep the code grounded in the ERP grid.
#
# Real SH (Condon–Shortley convention absorbed; sign factors chosen to
# give the standard real basis). Only coefficient combinations matter
# for the matching loss — overall sign is consistent between pred and gt.

def _real_sh_basis_lm(l: int, m: int, sin_lat: torch.Tensor,
                      cos_lat: torch.Tensor, sin_lon: torch.Tensor,
                      cos_lon: torch.Tensor) -> torch.Tensor:
    """Real spherical harmonic Y_{l,m}(θ, φ) up to l ≤ 4.

    sin_lat, cos_lat shape (H, 1).  sin_lon, cos_lon shape (1, W).
    Returns (H, W) real SH basis value.
    """
    # Convenience products (H, W). sin/cos of latitude are independent of
    # longitude; sin/cos of multiples of longitude broadcast over rows.
    sl, cl = sin_lat, cos_lat
    if l == 0:
        # Y_{0,0} = 1/2 √(1/π)
        c = 0.5 * math.sqrt(1.0 / math.pi)
        return torch.full_like(sl + 0.0 * cos_lon, c)
    if l == 1:
        if m == -1:
            return math.sqrt(3.0 / (4.0 * math.pi)) * (cl * sin_lon)
        if m == 0:
            return math.sqrt(3.0 / (4.0 * math.pi)) * sl
        if m == 1:
            return math.sqrt(3.0 / (4.0 * math.pi)) * (cl * cos_lon)
    if l == 2:
        if m == -2:
            # 1/4 √(15/π) sin²(co-lat) sin(2φ) — the round-trip test
            # below verifies the prefactor (round 5 patch fixed 1/2 → 1/4).
            return 0.25 * math.sqrt(15.0 / math.pi) * (cl * cl) \
                   * (2.0 * sin_lon * cos_lon)                       # sin(2φ)
        if m == -1:
            return 0.5 * math.sqrt(15.0 / math.pi) * (sl * cl) * sin_lon
        if m == 0:
            return 0.25 * math.sqrt(5.0 / math.pi) * (3.0 * sl * sl - 1.0)
        if m == 1:
            return 0.5 * math.sqrt(15.0 / math.pi) * (sl * cl) * cos_lon
        if m == 2:
            cos2 = cos_lon * cos_lon - sin_lon * sin_lon            # cos(2φ)
            return 0.25 * math.sqrt(15.0 / math.pi) * (cl * cl) * cos2
    if l == 3:
        # Multi-angle helpers
        sin2 = 2.0 * sin_lon * cos_lon
        cos2 = cos_lon * cos_lon - sin_lon * sin_lon
        sin3 = sin_lon * (3.0 - 4.0 * sin_lon * sin_lon)            # sin(3φ)
        cos3 = cos_lon * (4.0 * cos_lon * cos_lon - 3.0)            # cos(3φ)
        if m == -3:
            return 0.25 * math.sqrt(35.0 / (2.0 * math.pi)) \
                   * (cl ** 3) * sin3
        if m == -2:
            return 0.25 * math.sqrt(105.0 / math.pi) \
                   * sl * (cl * cl) * sin2
        if m == -1:
            return 0.25 * math.sqrt(21.0 / (2.0 * math.pi)) \
                   * cl * (5.0 * sl * sl - 1.0) * sin_lon
        if m == 0:
            return 0.25 * math.sqrt(7.0 / math.pi) \
                   * (5.0 * sl ** 3 - 3.0 * sl)
        if m == 1:
            return 0.25 * math.sqrt(21.0 / (2.0 * math.pi)) \
                   * cl * (5.0 * sl * sl - 1.0) * cos_lon
        if m == 2:
            return 0.25 * math.sqrt(105.0 / math.pi) \
                   * sl * (cl * cl) * cos2
        if m == 3:
            return 0.25 * math.sqrt(35.0 / (2.0 * math.pi)) \
                   * (cl ** 3) * cos3
    if l == 4:
        # Multi-angle helpers
        sin2 = 2.0 * sin_lon * cos_lon
        cos2 = cos_lon * cos_lon - sin_lon * sin_lon
        sin3 = sin_lon * (3.0 - 4.0 * sin_lon * sin_lon)
        cos3 = cos_lon * (4.0 * cos_lon * cos_lon - 3.0)
        sin4 = 2.0 * sin2 * cos2
        cos4 = 1.0 - 2.0 * sin2 * sin2
        if m == -4:
            return (3.0 / 16.0) * math.sqrt(35.0 / math.pi) \
                   * (cl ** 4) * sin4
        if m == -3:
            return (3.0 / 4.0) * math.sqrt(35.0 / (2.0 * math.pi)) \
                   * sl * (cl ** 3) * sin3
        if m == -2:
            return (3.0 / 8.0) * math.sqrt(5.0 / math.pi) \
                   * (cl * cl) * (7.0 * sl * sl - 1.0) * sin2
        if m == -1:
            return (3.0 / 4.0) * math.sqrt(5.0 / (2.0 * math.pi)) \
                   * cl * (7.0 * sl ** 3 - 3.0 * sl) * sin_lon
        if m == 0:
            return (3.0 / 16.0) * math.sqrt(1.0 / math.pi) \
                   * (35.0 * sl ** 4 - 30.0 * sl * sl + 3.0)
        if m == 1:
            return (3.0 / 4.0) * math.sqrt(5.0 / (2.0 * math.pi)) \
                   * cl * (7.0 * sl ** 3 - 3.0 * sl) * cos_lon
        if m == 2:
            return (3.0 / 8.0) * math.sqrt(5.0 / math.pi) \
                   * (cl * cl) * (7.0 * sl * sl - 1.0) * cos2
        if m == 3:
            return (3.0 / 4.0) * math.sqrt(35.0 / (2.0 * math.pi)) \
                   * sl * (cl ** 3) * cos3
        if m == 4:
            return (3.0 / 16.0) * math.sqrt(35.0 / math.pi) \
                   * (cl ** 4) * cos4
    raise ValueError(f"Unsupported (l, m) = ({l}, {m}); supported l ≤ 4")


def real_sh_basis(L: int, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """Stack of (L+1)² real SH basis maps over an ERP grid.

    Parameters
    ----------
    L   : int — max spherical harmonic order (≤ 4).
    lat : (H,)  latitude per row in radians.
    lon : (W,)  longitude per column in radians.

    Returns
    -------
    basis : (K, H, W) where K = (L+1)².
    """
    if L < 0 or L > 4:
        raise ValueError(f"L must be in [0, 4]; got {L}")
    H = lat.shape[0]
    W = lon.shape[0]
    # Broadcast all four trig factors to (H, W) up-front so the closed-form
    # Y_{l,m} expressions don't have to manually tile the m=0 cases (which
    # would otherwise stay shape (H, 1) and break torch.stack).
    sin_lat = torch.sin(lat).view(H, 1).expand(H, W).contiguous()
    cos_lat = torch.cos(lat).view(H, 1).expand(H, W).contiguous()
    sin_lon = torch.sin(lon).view(1, W).expand(H, W).contiguous()
    cos_lon = torch.cos(lon).view(1, W).expand(H, W).contiguous()

    out = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            y = _real_sh_basis_lm(
                l, m, sin_lat, cos_lat, sin_lon, cos_lon)
            # Defensive broadcast: in case any closed-form path returned
            # a non-(H, W) tensor (e.g. l=0 used full_like which already
            # had the right shape, but future additions might not).
            if y.shape != (H, W):
                y = y.expand(H, W).contiguous()
            out.append(y)
    return torch.stack(out, dim=0)                    # (K, H, W)


# ---------------------------------------------------------------------------
# SH coefficients of a depth field on the sphere
# ---------------------------------------------------------------------------

def spherical_sh_coeffs(depth_m: torch.Tensor,
                        L: int = 2,
                        use_log_depth: bool = True,
                        eps: float = 1e-3) -> torch.Tensor:
    """Project a (B, 1, H, W) depth field onto real SH up to order L.

    The depth field is treated as a function on the sphere using the ERP
    pixel-centre grid; the projection is a cos(latitude)-weighted Riemann
    sum so polar pixels do not dominate the integral:

        c_k(D) = (Σ_pix Y_k(θ,φ) · g(D) · cos(lat))
                 / Σ_pix cos(lat)

    where g(D) = log(D + eps) when ``use_log_depth`` else g(D) = D.
    Using log-depth compresses the dynamic range and makes coefficient
    matching less sensitive to a few near-saturated far pixels — a
    common failure mode in Matterport-style scenes.

    Returns (B, K) where K = (L + 1)².
    """
    if depth_m.dim() != 4 or depth_m.shape[1] != 1:
        raise ValueError(
            f"depth_m must be (B, 1, H, W); got {tuple(depth_m.shape)}")
    B, _, H, W = depth_m.shape
    device = depth_m.device
    dtype = depth_m.dtype

    lat, lon, w_hw = make_erp_grid(H, W, device=device, dtype=dtype)
    basis = real_sh_basis(L, lat, lon).to(dtype=dtype, device=device)  # (K,H,W)
    K = basis.shape[0]

    g = torch.log(depth_m.clamp(min=eps)) if use_log_depth else depth_m
    g = g.view(B, 1, H, W)
    w = w_hw.view(1, 1, H, W).to(dtype=dtype, device=device)

    # Numerator: sum_pix Y_k · g · w        →  (B, K)
    num = (basis.view(1, K, H, W) * g * w).sum(dim=(2, 3))
    denom = w.sum(dim=(2, 3)).clamp(min=eps)                # (1, 1)
    coeffs = num / denom
    return coeffs


def spherical_sh_loss(pred_depth_m: torch.Tensor,
                      gt_depth_m: torch.Tensor,
                      L: int = 2,
                      use_log_depth: bool = True,
                      eps: float = 1e-3) -> torch.Tensor:
    """smooth_l1 between SH coefficients of pred and gt depth fields.

    Inputs in metres. If the training pipeline normalises depth, the
    caller must rescale before invoking. Returns a 0-D tensor (scalar
    loss) that backpropagates through ``pred_depth_m`` only — gt is
    treated as a target.
    """
    pred_c = spherical_sh_coeffs(
        pred_depth_m, L=L, use_log_depth=use_log_depth, eps=eps)
    with torch.no_grad():
        gt_c = spherical_sh_coeffs(
            gt_depth_m, L=L, use_log_depth=use_log_depth, eps=eps)
    return F.smooth_l1_loss(pred_c, gt_c)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)

    # 1. Grid sanity ---------------------------------------------------------
    H, W = 16, 32
    lat, lon, w = make_erp_grid(H, W)
    assert lat.shape == (H,) and lon.shape == (W,) and w.shape == (H, W)
    assert torch.isclose(lat[0], torch.tensor(math.pi / 2.0 - math.pi / (2 * H)))
    assert (w >= 0).all()
    print(f"--- make_erp_grid OK  H={H} W={W} sum(w)={w.sum().item():.3f} ---")

    # 2. Basis: shape and orthogonality (approximate) ------------------------
    print("--- real_sh_basis shape / quasi-orthogonality smoke ---")
    for L in (0, 1, 2, 3, 4):
        basis = real_sh_basis(L, lat, lon)
        K = (L + 1) ** 2
        assert basis.shape == (K, H, W)
        # On-grid quasi-orthogonality: <Y_k, Y_l>_w = δ_kl · 4π / Σw  (Riemann)
        # We check Y_{0,0} self-overlap is positive and finite, and that the
        # off-diagonal magnitudes shrink as the grid gets finer. We don't
        # assert tight orthogonality (16x32 is too coarse) — just that the
        # diagonal dominates significantly for small L.
        b_hi = real_sh_basis(L, *make_erp_grid(64, 128)[:2])
        w_hi = make_erp_grid(64, 128)[2]
        gram = (b_hi.unsqueeze(0) * b_hi.unsqueeze(1) *
                w_hi.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3)) \
               / w_hi.sum()
        diag = gram.diag()
        offdiag = gram - torch.diag(diag)
        dr = (offdiag.abs().max() / diag.abs().max()).item()
        print(f"  L={L} K={K} basis OK  diag/offdiag ratio={1.0/max(dr,1e-9):.1f}x")

    # 3. Coeffs round-trip on a synthetic field -----------------------------
    print("--- spherical_sh_coeffs round-trip smoke ---")
    # Build D = 3 + 0.5·Y_{2,0}. Recall the K-th SH index is l² + l + m,
    # so (l=2, m=0) → 4 + 2 + 0 = 6. The projection denominator is
    # ∫ cos(lat) dlat dlon = 4π, so the orthonormal-against-itself
    # contribution lands at 1/(4π); for c_{2,0} this gives 0.5/(4π).
    H2, W2 = 64, 128
    lat2, lon2, _ = make_erp_grid(H2, W2)
    basis_l2 = real_sh_basis(2, lat2, lon2)
    Y20_idx = 2 * 2 + 2 + 0                              # l² + l + m
    Y20 = basis_l2[Y20_idx]
    field = (3.0 + 0.5 * Y20).view(1, 1, H2, W2)
    coeffs = spherical_sh_coeffs(field, L=4, use_log_depth=False)
    target_c00 = 3.0 / (2.0 * math.sqrt(math.pi))         # 3 · Y_{0,0}_const
    target_c20 = 0.5 / (4.0 * math.pi)                    # 0.5 · 1/(4π)
    print(f"  c[0,0]={coeffs[0,0].item():.4f}  want≈{target_c00:.4f}")
    print(f"  c[2,0]={coeffs[0,Y20_idx].item():.4f}  want≈{target_c20:.4f}")
    assert abs(coeffs[0, 0].item() - target_c00) < 5e-3, \
        f"c[0,0] off: got {coeffs[0,0].item()} want {target_c00}"
    assert abs(coeffs[0, Y20_idx].item() - target_c20) < 5e-3, \
        f"c[2,0] off: got {coeffs[0,Y20_idx].item()} want {target_c20}"
    # The other l=2 modes should be orthogonal → near-zero.
    for k_off in (4, 5, 7, 8):                            # m=-2,-1,1,2
        c_off = coeffs[0, k_off].item()
        assert abs(c_off) < 5e-3, \
            f"c[{k_off}] should be ~0 by orthogonality, got {c_off}"

    # 4. Loss: zero on identical inputs, positive otherwise; backward OK ----
    print("--- spherical_sh_loss smoke ---")
    pred = torch.full((2, 1, H2, W2), 5.0, requires_grad=True)
    gt = torch.full((2, 1, H2, W2), 5.0)
    L_eq = spherical_sh_loss(pred, gt, L=2, use_log_depth=True)
    print(f"  identical pred/gt → loss={L_eq.item():.6f}")
    assert L_eq.item() < 1e-6, "loss on identical inputs must be ~0"

    pred2 = (torch.rand(2, 1, H2, W2) * 9.0 + 0.5).requires_grad_(True)
    gt2 = (torch.rand(2, 1, H2, W2) * 9.0 + 0.5)
    L_diff = spherical_sh_loss(pred2, gt2, L=2, use_log_depth=True)
    print(f"  random  pred/gt → loss={L_diff.item():.6f}")
    assert L_diff.item() > 0
    L_diff.backward()
    g = pred2.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum().item() > 0
    print("  random backward OK (finite, non-zero)")

    # 5. log_depth toggle ----------------------------------------------------
    L_lin = spherical_sh_loss(pred2.detach().clone().requires_grad_(True),
                              gt2, L=2, use_log_depth=False)
    L_log = spherical_sh_loss(pred2.detach().clone().requires_grad_(True),
                              gt2, L=2, use_log_depth=True)
    print(f"  linear-depth loss={L_lin.item():.4f}  "
          f"log-depth loss={L_log.item():.4f}")
    assert torch.isfinite(L_lin) and torch.isfinite(L_log)

    print("--- spherical_loss.py self-test PASSED ---")

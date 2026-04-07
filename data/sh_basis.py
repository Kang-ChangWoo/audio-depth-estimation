"""Spherical Harmonics basis computation (ACN / SN3D)."""

import math
import numpy as np
from scipy.special import lpmv
from scipy.special import factorial as sp_factorial


def _acn_to_nm(acn):
    n = int(math.floor(math.sqrt(acn)))
    m = acn - n * n - n
    return n, m


def _sn3d_norm(n, m):
    m_abs = abs(m)
    delta = 1.0 if m == 0 else 0.0
    return math.sqrt((2.0 - delta) * sp_factorial(n - m_abs, exact=True)
                     / sp_factorial(n + m_abs, exact=True))


def _real_sh_sn3d_np(acn, elevation, azimuth):
    n, m = _acn_to_nm(acn)
    m_abs = abs(m)
    N = _sn3d_norm(n, m)
    P = (-1)**m_abs * lpmv(m_abs, n, np.sin(elevation))
    if m > 0:
        return N * P * np.cos(m * azimuth)
    elif m == 0:
        return N * P
    else:
        return N * P * np.sin(m_abs * azimuth)


def sh_basis_matrix(max_order, elevation, azimuth):
    """Compute SH basis matrix for given grid. Returns (N_pixels, N_channels)."""
    n_ch = (max_order + 1) ** 2
    el_flat = elevation.ravel()
    az_flat = azimuth.ravel()
    B = np.zeros((el_flat.size, n_ch))
    for q in range(n_ch):
        B[:, q] = _real_sh_sn3d_np(q, el_flat, az_flat)
    return B


def reconstruct_per_component_maps(sh_coeffs, B):
    """Reconstruct per-SH-component energy maps (OLD, kept for reference)."""
    n_ch = B.shape[1]
    A = sh_coeffs[:n_ch]
    rms = np.sqrt(np.mean(A ** 2, axis=1))
    maps = B * rms[None, :]
    return maps.T


def compute_covariance(ir, window=None, sr=48000):
    """Compute inter-channel covariance matrix R from ambisonic IR.

    R_nm = (1/T) sum_t b_n(t) b_m(t)

    Args:
        ir: (n_ch, N_samples) ambisonic impulse response
        window: (start_ms, end_ms) time window, or None for full IR
        sr: sample rate
    Returns:
        R: (n_ch, n_ch) covariance matrix
    """
    if window is not None:
        s0 = int(window[0] * sr / 1000)
        s1 = min(int(window[1] * sr / 1000), ir.shape[1])
        ir_win = ir[:, s0:s1]
    else:
        ir_win = ir

    T = ir_win.shape[1]
    if T == 0:
        return np.zeros((ir.shape[0], ir.shape[0]))
    return (ir_win @ ir_win.T) / T


def energy_map_from_cov(R, B, H, W):
    """Compute directional energy E(Omega) = y(Omega)^T R y(Omega).

    Args:
        R: (n_ch, n_ch) covariance matrix
        B: (H*W, n_ch) SH basis
        H, W: spatial dimensions
    Returns:
        E: (H, W) directional energy map
    """
    BR = B @ R          # (H*W, n_ch)
    E = np.sum(BR * B, axis=1)  # (H*W,)
    return E.reshape(H, W)


def reconstruct_energy_map_cov(ir, B, H, W, sr=48000, window=None):
    """Corrected directional energy map from ambisonic IR using covariance.

    Steps:
      1. Compute covariance R over time window
      2. E(Omega) = y(Omega)^T R y(Omega) for each ERP direction
    """
    n_ch = B.shape[1]
    R = compute_covariance(ir[:n_ch], window=window, sr=sr)
    return energy_map_from_cov(R, B, H, W)

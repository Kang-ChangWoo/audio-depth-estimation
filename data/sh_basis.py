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
    """Reconstruct per-SH-component energy maps."""
    n_ch = B.shape[1]
    A = sh_coeffs[:n_ch]
    rms = np.sqrt(np.mean(A ** 2, axis=1))
    maps = B * rms[None, :]
    return maps.T

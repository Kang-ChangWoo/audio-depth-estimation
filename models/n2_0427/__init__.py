"""n2_0427 — EchoDiffusion + SH side-prior variants (2026-04-27).

Two models:
    EchoDiffusionSHSide      — concise side-prior (lightweight)
    EchoDiffusionSHSidePlus  — wider UNet, tapered decoder, SH cross-attn token

Both are designed to compare:
    Baseline (side_fusion=False)        — pure binaural EchoDiffusion path
    Real oracle upper bound             — side_fusion=True + oracle_mode=True
                                          (Plus: oracle_gate_mode='ones')
"""

from .echodiff_sh_side import EchoDiffusionSHSide
from .echodiff_sh_side_plus import EchoDiffusionSHSidePlus

__all__ = ['EchoDiffusionSHSide', 'EchoDiffusionSHSidePlus']

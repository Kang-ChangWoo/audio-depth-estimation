"""n2_0427 — EchoDiffusion + SH side-prior variants (2026-04-27).

Kept model:
    EchoDiffusionSHSidePlus  — wider UNet, tapered decoder, SH cross-attn token

Phase F1 (2026-05-14): the concise EchoDiffusionSHSide variant had no kept
config and no archived runs; moved to models/deprecated/n2_0427/.

Comparison setup (Plus):
    Baseline (side_fusion=False)        — pure binaural EchoDiffusion path
    Real oracle upper bound             — side_fusion=True + oracle_mode=True
                                          (oracle_gate_mode='ones')
"""

from .echodiff_sh_side_plus import EchoDiffusionSHSidePlus

__all__ = ['EchoDiffusionSHSidePlus']

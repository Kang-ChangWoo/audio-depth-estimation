"""n9_0424 — Binaural ERP depth with Implicit Sound-Field Projection Fusion.

See models/n9_0424/n9_0424.py for the full architecture.

Ablation variants (same N9_0424Net class, different flags via config):
    A. baseline       enable_fusion=False, lambda_sh=0   (pure depth)
    B. aux-only       enable_fusion=False, lambda_sh>0   (rep supervised, no fusion)
    C. proj fusion    enable_fusion=True,  lambda_sh>0
    D. + energy map   Variant C + lambda_energy_map>0
    E. fixed gate     Variant C + gate_learnable=False
"""

from .n9_0424 import N9_0424Net
from .fusion import ImplicitSoundFieldProjectionFusion, make_foa_basis_erp
from .losses import weighted_rep_loss, energy_map_loss

__all__ = [
    'N9_0424Net',
    'ImplicitSoundFieldProjectionFusion',
    'make_foa_basis_erp',
    'weighted_rep_loss',
    'energy_map_loss',
]

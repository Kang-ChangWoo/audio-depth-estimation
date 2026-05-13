"""Bin-based (probabilistic) depth heads.

EchoRange — replaces the scalar depth head with a per-pixel range
distribution over log-spaced depth bins. Default behavior with
``depth_head_type='scalar'`` reproduces the original echodiffusion
scalar output exactly.

See docs and the model docstring for the full motivation.
"""

from .range_head import (
    RangeDepthHead,
    HazardRangeDepthHead,
    soft_range_nll_loss,
    hazard_supervision_loss,
    rendered_event_nll,
    survival_loss,
    soft_hit_bce_loss,
    hazard_free_loss,
    range_point_estimate,
    soft_quantile_depth,
)
from .spherical_loss import (
    make_erp_grid,
    real_sh_basis,
    spherical_sh_coeffs,
    spherical_sh_loss,
)
from .echorange import EchoRangeDepth

__all__ = [
    'RangeDepthHead',
    'HazardRangeDepthHead',
    'soft_range_nll_loss',
    'hazard_supervision_loss',
    'rendered_event_nll',
    'survival_loss',
    'soft_hit_bce_loss',
    'hazard_free_loss',
    'range_point_estimate',
    'soft_quantile_depth',
    'make_erp_grid',
    'real_sh_basis',
    'spherical_sh_coeffs',
    'spherical_sh_loss',
    'EchoRangeDepth',
]

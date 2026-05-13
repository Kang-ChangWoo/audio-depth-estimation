"""N3 hybrid models (2026-04-19) — report_d proposals.

Group I: hybrid architectures that combine N3 (0417) mechanisms.
- N3FiLMEnergyAttnGenerator : FiLM on decoder[0] + energy attention on decoder[4]
- N3MSSHEnergyAttnGenerator : multi-scale SH tapping + energy attention on decoder[4]

Each hybrid reuses the FiLMProjector / SHHead / EnergyHead building blocks
from the 0417 models so there is no new training-path logic needed — the
model output dict ({pred_depth, pred_sh, pred_energy[, pred_sh_scales]})
slots directly into the existing foa_0415 training step.
"""
from .n3_film_energy_attn import N3FiLMEnergyAttnGenerator
from .n3_mssh_energy_attn import N3MSSHEnergyAttnGenerator
from .n3_energy_only import (
    EmapUNetGenerator, EmapUNetTemporalGenerator,
    EmapViTGenerator, EmapViTTemporalGenerator,
)

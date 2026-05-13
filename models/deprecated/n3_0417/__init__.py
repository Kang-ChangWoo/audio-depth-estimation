"""N3 experiment models (2026-04-17).

Group C: deployable (binaural-only inference)
Group D: oracle (GT FOA/energy at test time)
"""
from .n3_film import N3FiLMGenerator
from .n3_multiscale_sh import N3MultiScaleSHGenerator
from .n3_energy_attn import N3EnergyAttnGenerator
from .n3_temporal_window import N3TemporalWindowGenerator
from .n3_oracle import FOAOracleGenerator

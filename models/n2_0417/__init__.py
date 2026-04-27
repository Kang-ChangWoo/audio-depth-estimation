"""N2 experiment models (2026-04-17): temporal FOA decomposition.

E1 (n2_temap_input):      Temporal energy maps as extra input channels (5ch)
E2 (n2_6ch_input):         FOA0415V1Generator with input_nc=6 (binaural + FOA spec)
E3 (n2_temporal_rms_film): 12-dim temporal RMS → FiLM conditioning
E4 (n2_temporal_energy):   Temporal energy maps as spatial attention at 3 decoder levels
E5 (n2_foa_stft_film):     FOA spectrogram → ConvNet → FiLM conditioning
E6 (n2_dual_enc):          Separate binaural and FOA encoders, bottleneck fusion
E7 (n2_temporal_rms):      FOA0415V1Generator with sh_dim=12 (supervision-only)
E8 (n2_tbin_crossattn):    Temporal bin cross-attention at bottleneck
"""
from .n2_temap_input import N2TemapInputGenerator
from .n2_temporal_energy import N2TemporalEnergyGenerator
from .n2_dual_enc import N2DualEncGenerator
from .n2_foa_stft_film import N2FOASTFTFiLMGenerator
from .n2_temporal_rms_film import N2TemporalRMSFiLMGenerator
from .n2_tbin_crossattn import N2TBinCrossAttnGenerator

"""N2 E1: Temporal Energy Map Input — simplest temporal variant.

Concatenates K temporal energy maps (visualization-based) with binaural
spectrogram at the input level: (2+K, 256, 512).

SH head predicts per-bin temporal RMS (K*4 dims) as auxiliary supervision.

This is the most robust variant — if temporal hypothesis is wrong, the model
can simply learn to ignore the extra channels.
"""

from ..foa_0415_v1 import FOA0415V1Generator


class N2TemapInputGenerator(FOA0415V1Generator):

    DEFAULT_SH_DIM = 12
    DEFAULT_HEAD_HIDDEN = 256

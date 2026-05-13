"""Oracle models for upper-bound analysis with GT FOA/energy input.

Group D experiments feed ground-truth spatial audio cues (energy maps, FOA
spectrograms) directly into the UNet encoder at **both** train and test time.
This establishes an oracle upper bound on depth prediction performance.

The architecture is identical to FOA0415V1Generator (shared UNet encoder with
auxiliary SH head); only `input_nc` varies to accommodate concatenated GT
channels.  All input assembly (concatenation of binaural audio with GT cues,
or replacement of binaural audio entirely) is handled by the training step,
not by the model itself.

Oracle modes (controlled by ``model.input_nc`` in config):
    D1  input_nc=3  binaural (2ch) + GT energy map (1ch)
    D2  input_nc=6  binaural (2ch) + GT FOA spectrogram (4ch)  [future]
    D5  input_nc=1  GT energy map only (no binaural audio)
"""

from models.foa_0415_v1 import FOA0415V1Generator  # noqa: F401  (SHHead re-exported via parent)


class FOAOracleGenerator(FOA0415V1Generator):
    """Same UNet + SH architecture as Variant 1, variable input channels.

    Inherits everything from :class:`FOA0415V1Generator`.  The parent
    ``__init__`` already accepts ``input_nc`` so no override is needed.
    """

    DEFAULT_SH_DIM = 4
    DEFAULT_HEAD_HIDDEN = 256

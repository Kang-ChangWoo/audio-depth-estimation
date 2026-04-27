"""n9_0425 — n4_0425 architecture, but the per-bin FOA representatives
come from a pre-trained n3_0425 (binaural→FOA) model instead of the
dataset's oracle ``rep_gt``.

Two operating modes selectable via cfg.model.freeze_n3 (or --freeze-n3):
    True  — n3 weights frozen, n3 always in eval(); only the UNet+gate
            train. End-to-end-fair comparison against n4_0425 (oracle
            rep_gt) — answers "how much do we lose by going from oracle
            FOA to predicted FOA?"
    False — n3 weights also trainable; depth loss flows back into n3
            and (optionally, via lambda_sh > 0) the FOA L1+cosine loss
            against rep_gt fine-tunes n3 toward the oracle target.
"""

from .n9_0425 import N9_0425Net

__all__ = ['N9_0425Net']

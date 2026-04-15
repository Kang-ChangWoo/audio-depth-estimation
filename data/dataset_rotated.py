"""Canonical-frame FOA rotation wrapper for SoundSpacesDataset.

Habitat-Sim Ambisonics are effectively *world-frame*, not listener-frame.
Each position is captured with 4 yaw rotations in a fixed cyclic order:

    view_mod = raw_sample_index % 4
        0 -> front (no rotation)
        1 -> right (yaw -90 deg)
        2 -> back  (yaw -180 deg)
        3 -> left  (yaw -270 deg)

To make FOA inputs ego-consistent with the RGB / depth views, we rotate the
FOA channels into a canonical agent-centered frame before any per-channel
statistics (RMS, covariance, ERP energy) are computed.

Why `sample_idx % 4` and NOT `dataset_idx % 4`
----------------------------------------------
SoundSpacesDataset drops samples whose ERP depth is >10% invalid. After this
filter, the position in self.samples no longer preserves the raw 4-view
cycle: a missing entry shifts every subsequent sample. The only reliable
source of the view index is the raw capture index parsed from the filename
(e.g. `audio_023.wav` -> 23 -> view_mod=3). We compute view_mod from that.

What is rotated
---------------
We rotate the raw FOA impulse response *before* deriving any target. Doing
the rotation at the IR level is the most principled choice: it transforms
exactly one quantity, and every downstream target (RMS target vector,
covariance matrix, ERP energy map) is then computed from a self-consistent
canonical-frame signal. Rotating the RMS target alone is ill-defined
(channel-wise RMS is invariant under sign flips so 180 deg rotation is a
no-op), and rotating the covariance/energy-map would require extra matrix
machinery for no benefit. 90 deg yaw is just a sign-swap on (Y, X), so the
rotation is exact and O(T) per sample.

Assumed channel order (ACN): [W, Y, Z, X]. W and Z are invariant under yaw;
only the horizontal (Y, X) pair mixes.
"""

import numpy as np

from .dataset import SoundSpacesDataset


def get_view_mod_from_sample_idx(sample_idx) -> int:
    """Return view_mod in {0,1,2,3} from the raw capture index (filename)."""
    return int(sample_idx) % 4


def rotate_foa_to_canonical(foa: np.ndarray, view_mod: int) -> np.ndarray:
    """Rotate 1st-order ACN ambisonics by -90 deg * view_mod (yaw only).

    Args:
        foa: (n_ch, T) array with n_ch >= 4 in ACN order [W, Y, Z, X].
        view_mod: 0, 1, 2, or 3.
    Returns:
        A new array of the same shape, rotated into the canonical frame.
    """
    if view_mod == 0:
        return foa
    out = foa.copy()
    Y = foa[1]
    X = foa[3]
    if view_mod == 1:       # yaw -90 deg:   Y' =  X, X' = -Y
        out[1] = X
        out[3] = -Y
    elif view_mod == 2:     # yaw -180 deg:  Y' = -Y, X' = -X
        out[1] = -Y
        out[3] = -X
    elif view_mod == 3:     # yaw -270 deg:  Y' = -X, X' =  Y
        out[1] = -X
        out[3] = Y
    else:
        raise ValueError(f"view_mod must be in {{0,1,2,3}}, got {view_mod}")
    return out


class SoundSpacesDatasetRotated(SoundSpacesDataset):
    """SoundSpacesDataset with FOA rotated to a canonical listener frame.

    Only the ambisonic IR loader is overridden; the non-ambisonic code path
    is inherited unchanged. The depth-validity filter in the base class is
    preserved, and `view_mod` is still computed from the raw filename index,
    so it stays correct even when samples are dropped.
    """

    def _load_foa_ir(self, ambi_path, sample_idx):
        ir = np.load(ambi_path).astype(np.float64)
        view_mod = get_view_mod_from_sample_idx(sample_idx)
        return rotate_foa_to_canonical(ir, view_mod)

"""MINT predicts its own projection K: refit its joints and MANO export from that K."""

import numpy as np

from utils.mano import refit_prediction


def native_prediction(raw, camera, model=None, *, joint_only=False):
    """Refit from MINT's predicted K; its native left hands are already mirrored (no reflection)."""
    k = np.asarray(raw["predicted_projection_K"])
    native = np.stack([k[..., 0, 0], k[..., 1, 1], k[..., 0, 2], k[..., 1, 2]], -1)
    return refit_prediction(
        raw, camera, model, joint_only=joint_only, native_intrinsics=native, reflect_left=False
    )

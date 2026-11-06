"""InfiniHand camera-space hands to canonical RIGHT/mirrored-left MANO."""

import numpy as np
import torch

from utils.mano import native_to_mano

SLOTS = [1, 0]  # native hand slots are [right, left]; the export is [left, right]


def native_prediction(raw, camera, model):
    """Fit the canonical MANO to the native meshes, started from the native rotations.

    The native left hand comes from the LEFT MANO layer: its mirrored pose (S R S) starts the RIGHT-model fit
    (reflect_left=False: native_to_mano leaves it, mesh2mano mirrors side 0 into the canonical frame).
    """
    rotations = np.concatenate([raw["global_orient"], raw["hand_pose"]], axis=2)
    torch.set_num_threads(2)
    return native_to_mano(raw, rotations.astype(np.float32), camera.source_frame_ids, model, reflect_left=False)

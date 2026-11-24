"""Method-owned native mesh and MANO export conversion."""

import numpy as np
import torch
from utils.mano import native_to_mano


def native_prediction(raw, camera, model):
    """Convert a method's native rotations and mesh into the shared MANO dict."""
    rotations = np.tile(
        np.eye(3, dtype=np.float32), (len(raw["frame_ids"]), 2, 16, 1, 1)
    )
    torch.set_num_threads(2)
    parameters = native_to_mano(
        raw, rotations.astype(np.float32), camera.source_frame_ids, model, zero_betas=True
    )
    return parameters

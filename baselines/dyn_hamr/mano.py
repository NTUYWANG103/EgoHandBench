"""Method-local native mesh conversion to canonical RIGHT/mirrored-left MANO."""


import numpy as np
import torch
from utils.mano import native_to_mano


def native_prediction(raw, camera, model):
    """Convert a method's native rotations and mesh into the shared MANO dict."""
    rotations = np.concatenate([raw["global_orient"], raw["hand_pose"]], axis=2).copy()
    torch.set_num_threads(2)
    parameters = native_to_mano(
        raw, rotations.astype(np.float32), camera.source_frame_ids, model, reflect_left=False
    )
    return parameters


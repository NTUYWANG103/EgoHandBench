"""Convert this baseline's native meshes and poses to canonical RIGHT/mirrored-left MANO."""


import numpy as np
from scipy.spatial.transform import Rotation
import torch
from utils.mano import native_to_mano


def rotation_vector(matrix):
    value = np.asarray(matrix)
    return (
        Rotation.from_matrix(value.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(*value.shape[:-2], 3)
    )


def native_prediction(raw, camera, model):
    """Convert a method's native rotations and mesh into the shared MANO dict."""
    rotations = np.concatenate([raw["global_orient"], raw["hand_pose"]], axis=2).copy()
    torch.set_num_threads(2)
    parameters = native_to_mano(
        raw, rotations.astype(np.float32), camera.source_frame_ids, model, reflect_left=False
    )
    return parameters


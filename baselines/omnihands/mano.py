"""Convert this baseline's native meshes and poses to canonical RIGHT/mirrored-left MANO."""


import numpy as np
from scipy.spatial.transform import Rotation
import torch
from utils.mano import native_to_mano


def rotation_matrix(rotvec):
    value = np.asarray(rotvec)
    return (
        Rotation.from_rotvec(value.reshape(-1, 3))
        .as_matrix()
        .reshape(*value.shape[:-1], 3, 3)
    )


def rotation_vector(matrix):
    value = np.asarray(matrix)
    return (
        Rotation.from_matrix(value.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(*value.shape[:-2], 3)
    )


def native_prediction(raw, camera, model, mano_assets):
    """Convert a method's native rotations and mesh into the shared MANO dict."""
    import pickle
    from pathlib import Path

    rotations = np.concatenate([raw["global_orient"], raw["hand_pose"]], axis=2).copy()
    means = raw.get("source_hand_mean")
    if means is None:
        means = []
        for side in (0, 1):
            hand = "RIGHT"
            with (Path(mano_assets) / f"MANO_{hand}.pkl").open("rb") as stream:
                source = pickle.load(stream, encoding="latin1")
            means.append(np.asarray(source["hands_mean"]).reshape(15, 3))
        means = np.asarray(means)
    active = np.asarray(raw["presence"], bool)
    for side in (0, 1):
        rows = active[:, side]
        if rows.any():
            source = raw.get("source_hand_pose_axis_angle")
            aa = (
                source[rows, side].reshape(-1, 15, 3)
                if source is not None
                else rotation_vector(rotations[rows, side, 1:])
            )
            rotations[rows, side, 1:] = rotation_matrix(aa + means[side])
    torch.set_num_threads(2)
    parameters = native_to_mano(
        raw, rotations.astype(np.float32), camera.source_frame_ids, model
    )
    return parameters


"""Method-owned native mesh and MANO export conversion."""

from utils.mano import apply_camera, mano_joint_prediction, native_to_mano
from .camera import source_prediction

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def rotation_matrix(rotvec):
    value = np.asarray(rotvec)
    return (
        Rotation.from_rotvec(value.reshape(-1, 3))
        .as_matrix()
        .reshape(*value.shape[:-1], 3, 3)
    )


def native_prediction(raw, camera, model, mano_assets, *, joint_only=False):
    """Convert a method's native rotations and mesh into the shared MANO dict; joints are decoded from it."""
    import pickle
    from pathlib import Path

    _, delta = source_prediction(raw, camera)
    rotations = np.concatenate([raw["global_orient"], raw["hand_pose"]], axis=2).copy()
    means = []
    for side in (0, 1):
        hand = "LEFT" if side == 0 else "RIGHT"
        with (Path(mano_assets) / f"MANO_{hand}.pkl").open("rb") as stream:
            source = pickle.load(stream, encoding="latin1")
        means.append(np.asarray(source["hands_mean"]).reshape(15, 3))
    means = np.asarray(means)
    active = np.asarray(raw["presence"], bool)
    for side in (0, 1):
        rows = active[:, side]
        if rows.any():
            source = raw["source_hand_pose_axis_angle"]
            aa = source[rows, side].reshape(-1, 15, 3)
            rotations[rows, side, 1:] = rotation_matrix(aa + means[side])
    torch.set_num_threads(2)
    parameters = native_to_mano(
        raw, rotations.astype(np.float32), camera.source_frame_ids, model, reflect_left=False
    )
    mano = apply_camera(parameters, delta)
    return {"mano": None if joint_only else mano, "joint": mano_joint_prediction(mano, model)}

"""Method-local native mesh conversion to canonical RIGHT/mirrored-left MANO."""

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from utils.mano import native_to_mano, OPENPOSE_TO_MANO  # native joints in models/mano_layer.mano_joint_mapping order


def native_joints(raw, source_frame_ids):
    """Return the native camera-space FK joints, including native left geometry."""
    joints = np.asarray(raw["joints_cam"], dtype=np.float32)[:, :, OPENPOSE_TO_MANO].copy()
    valid = np.asarray(raw["presence"], dtype=bool) & np.isfinite(joints).all((2, 3))
    joints[~valid] = np.nan
    return dict(joints3d=joints, valid=valid, source_frame_ids=np.asarray(source_frame_ids).copy())


def rotation_matrix(rotvec):
    value = np.asarray(rotvec)
    return (
        Rotation.from_rotvec(value.reshape(-1, 3))
        .as_matrix()
        .reshape(*value.shape[:-1], 3, 3)
    )


def native_prediction(raw, camera, model):
    """Convert a method's native rotations and mesh into the shared MANO dict."""
    rotations = np.concatenate([raw["global_orient"], raw["hand_pose"]], axis=2).copy()
    means = raw["source_hand_mean"]
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
    return parameters

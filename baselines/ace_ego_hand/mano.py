"""Method-local export of ACE-Ego-Hand's native MANO and direct-head joints."""

import numpy as np
from utils.mano import OPENPOSE_TO_MANO, PARAM_KEYS  # direct-head joints in mano_utils._MANO_TO_OP order


def native_joints(raw, source_frame_ids):
    """Return the independent native joint head, not joints decoded from MANO."""
    joints = np.asarray(raw["direct_joints_cam"], dtype=np.float32)[:, :, OPENPOSE_TO_MANO].copy()
    valid = np.asarray(raw["presence"], dtype=bool) & np.isfinite(joints).all((2, 3))
    joints[~valid] = np.nan
    return dict(joints3d=joints, valid=valid, source_frame_ids=np.asarray(source_frame_ids).copy())


def native_prediction(raw, source_frame_ids):
    """Export the network's MANO as is: it was trained in the canonical convention.

    The egohandbench weights use full local rotations on MANO_RIGHT for both
    slots with the left output x-mirrored, plus an additive cam_trans, which
    is exactly utils.mano.ManoModel; so no pose-mean change and no refit.
    """
    claims = np.asarray(raw["presence"], dtype=bool)
    count = len(claims)
    shapes = dict(global_orient=(1, 3, 3), hand_pose=(15, 3, 3), betas=(10,), cam_trans=(3,))
    arrays = {}
    for key in PARAM_KEYS:
        value = np.asarray(raw[key], dtype=np.float32).reshape(count, 2, *shapes[key]).copy()
        value[~claims] = np.nan
        arrays[key] = value
    return dict(arrays, valid=claims.copy(), source_frame_ids=np.asarray(source_frame_ids).copy())

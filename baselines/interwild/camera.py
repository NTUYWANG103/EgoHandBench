"""Native joint ordering and translation under supplied camera calibration."""

import numpy as np

from utils.mano import OPENPOSE_TO_MANO, refit_translation


def source_prediction(raw, camera):
    """Return native MANO21 and its camera shift without canonical fitting."""
    valid = np.asarray(raw["presence"], dtype=bool)
    joints = np.asarray(raw["joints_cam"], dtype=np.float64)[:, :, OPENPOSE_TO_MANO].copy()
    joints[~valid] = np.nan
    source = raw["projection_intrinsics"]
    target = np.asarray(camera.K, dtype=np.float64)
    target_parameters = target[:, (0, 1, 0, 1), (0, 1, 2, 2)]
    native = np.broadcast_to(source, (*valid.shape, 4))[valid]
    supplied = np.broadcast_to(target_parameters[:, None, :], (*valid.shape, 4))[valid]
    supplied_k = np.broadcast_to(target[:, None], (*valid.shape, 3, 3))[valid]
    points = joints[valid]
    if not np.isfinite(target).all() or np.any(target_parameters[:, :2] <= 0):
        raise ValueError("Finite intrinsics with positive focal lengths required")
    if not np.isfinite(native).all() or np.any(native[:, :2] <= 0):
        raise ValueError("Invalid native camera")
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite native joints on a claimed hand")
    delta = np.zeros((*valid.shape, 3), dtype=np.float64)
    uv = points[..., :2] * native[:, None, :2] + native[:, None, 2:]
    depth = np.sqrt(supplied[:, 0] * supplied[:, 1] / (native[:, 0] * native[:, 1]))
    if not np.isfinite(uv).all() or np.any(depth <= 0):
        raise ValueError("Invalid native joint projection or wrist depth")
    wrist_pixels = np.column_stack((uv[:, 0], np.ones(len(uv))))
    wrist_rays = np.linalg.solve(supplied_k, wrist_pixels[..., None])[..., 0]
    prior = wrist_rays / wrist_rays[:, 2:3] * depth[:, None]
    shifts = np.zeros((len(points), 3), dtype=np.float64)
    for intrinsics in np.unique(supplied_k, axis=0):
        rows = np.all(supplied_k == intrinsics, axis=(-1, -2))
        shifts[rows] = refit_translation(
            points[rows] - points[rows, :1], uv[rows], intrinsics, prior[rows]
        ) - points[rows, 0]
    delta[valid] = shifts
    joints[valid] += shifts[:, None]
    return dict(joints3d=joints.astype(np.float32), valid=valid.copy(),
                source_frame_ids=camera.source_frame_ids.copy()), delta


"""Preserve the demo camera inside WildHands and localize under supplied K."""

import numpy as np

from utils.mano import refit_translation

from .mano import native_joints


def reference_intrinsics(width, height):
    """Released demo camera in original-image pixels, before 840/224 resizing."""
    return np.array([[1000., 0., width / 2], [0., 1000., height / 2], [0., 0., 1.]])


def source_prediction(raw, camera, refit=True):
    """refit=False: the network already ran under supplied K; its joints are in the source camera as is.
    refit=True (demo camera inside the network): centre native image rays on the selected detector box,
    then solve translation under K."""
    native = native_joints(raw, camera.source_frame_ids)
    if not refit:
        return native, np.zeros((*native["valid"].shape, 3), np.float64)
    valid = native["valid"]
    points = native["joints3d"][valid].astype(np.float64)
    target = np.broadcast_to(camera.K[:, None], (*valid.shape, 3, 3))[valid]
    reference = reference_intrinsics(*camera.canvas[0])
    homogeneous = np.einsum("ij,nqj->nqi", reference, points)
    own_uv = homogeneous[..., :2] / homogeneous[..., 2:3]
    own_center = (own_uv.min(1) + own_uv.max(1)) / 2
    boxes = np.asarray(raw["detector_boxes_original"], dtype=np.float64)[valid]
    if not np.isfinite(boxes).all():
        raise ValueError("Invalid selected WildHands detector box")
    detector_center = (boxes[:, :2] + boxes[:, 2:]) / 2
    uv = own_uv + (detector_center - own_center)[:, None]
    depth = np.sqrt(target[:, 0, 0] * target[:, 1, 1]) / 1000 * points[:, 0, 2]
    if not np.isfinite(uv).all() or np.any(depth <= 0):
        raise ValueError("Invalid native WildHands projection or wrist depth")
    wrist_pixels = np.column_stack((uv[:, 0], np.ones(len(uv))))
    rays = np.linalg.solve(target, wrist_pixels[..., None])[..., 0]
    prior = rays / rays[:, 2:3] * depth[:, None]
    shifts = np.zeros((len(points), 3), np.float64)
    for intrinsics in np.unique(target, axis=0):
        rows = np.all(target == intrinsics, axis=(-1, -2))
        shifts[rows] = refit_translation(
            points[rows] - points[rows, :1], uv[rows], intrinsics, prior[rows]
        ) - points[rows, 0]
    delta = np.zeros((*valid.shape, 3), np.float64)
    delta[valid] = shifts
    joints = native["joints3d"].astype(np.float64)
    joints[valid] += shifts[:, None]
    return dict(native, joints3d=joints.astype(np.float32)), delta


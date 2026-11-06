"""Static-camera fallback with supplied K, initial translations at supplied K, and validated optimized intrinsics."""

from pathlib import Path

import cv2
import numpy as np


def write_static_camera(images, folder, intrinsics):
    """Write cameras.npz for a camera that never moves, with supplied [fx, fy, cx, cy].

    Used if DROID-SLAM fails, or for the one retry after an optimizer NaN under the
    SLAM cameras. This is upstream
    preproc/run_slam.py's own fallback for fewer than two DROID keyframes
    (get_frame_cameras: one eye(4) world-to-camera pose per frame), saved as its
    save_cameras saves it, with intrinsics through the same intrinsics_only.txt round trip
    that load_intrins reads (float32) and the image size from get_hwf.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    k_path = folder / "intrinsics_only.txt"
    np.savetxt(k_path, intrinsics, fmt="%.9g")
    intrins = np.loadtxt(k_path).astype(np.float32).reshape(-1, 4)
    height, width = cv2.imread(str(sorted(Path(images).glob("*.jpg"))[0])).shape[:2]
    np.savez(
        folder / "cameras.npz",
        height=np.float32(height),
        width=np.float32(width),
        focal=intrins[:, :2].mean(),
        intrins=intrins,
        w2c=np.repeat(np.eye(4, dtype=np.float32)[None], len(intrins), 0),
    )


def translation_to_intrinsics(raw, matrices):
    """Per-hand translation that keeps native 2D joints under supplied K."""
    matrices = np.asarray(matrices, np.float64)
    valid = np.asarray(raw["presence"], bool)
    supplied = np.broadcast_to(matrices[:, None], (*valid.shape, 3, 3))[valid]
    points = np.asarray(raw["joints_cam"], np.float64)[valid]
    source = np.broadcast_to(raw["projection_intrinsics"], (*valid.shape, 4))[valid]
    if not np.isfinite(source).all() or np.any(source[:, :2] <= 0):
        raise ValueError("Invalid native projection intrinsics")
    if len(points):
        uv = points[..., :2] / points[..., 2:] * source[:, None, :2] + source[:, None, 2:]
        root = points[:, 0]
        relative = points - root[:, None]
        pixels = np.concatenate([uv, np.ones((*uv.shape[:-1], 1))], -1)
        rays = np.einsum("bij,bkj->bki", np.linalg.inv(supplied), pixels)
        depth = root[:, 2] * np.sqrt(
            supplied[:, 0, 0] * supplied[:, 1, 1] / (source[:, 0] * source[:, 1])
        )
        prior = rays[:, 0] * (depth / rays[:, 0, 2])[:, None]
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        projectors = np.eye(3) - rays[..., :, None] * rays[..., None, :]
        information = projectors.mean(1)
        gradient = np.einsum("bkij,bkj->bi", projectors, relative) / points.shape[1]
        delta = np.empty_like(root)
        for i in range(len(points)):
            solution = prior[i]
            if np.linalg.cond(information[i]) <= 1e6:
                candidate = np.linalg.solve(information[i], -gradient[i])
                if np.isfinite(candidate).all() and np.all(relative[i, :, 2] + candidate[2] > 1e-3):
                    solution = candidate
            delta[i] = solution - root[i]
        target = supplied[:, (0, 1, 0, 1), (0, 1, 2, 2)]
        same = np.all(np.isclose(source, target, rtol=1e-6, atol=1e-6), axis=-1)
        same &= (supplied[:, 0, 1] == 0) & (supplied[:, 1, 0] == 0)
        delta[same] = 0
    else:
        delta = np.empty((0, 3))
    return delta


def validate_prediction_intrinsics(raw, camera):
    """Reject mismatched K without modifying the optimized prediction."""
    matrices = np.asarray(camera.K, np.float64)
    valid = np.asarray(raw["presence"], bool)
    supplied = np.broadcast_to(matrices[:, None], (*valid.shape, 3, 3))[valid]
    source = np.broadcast_to(raw["projection_intrinsics"], (*valid.shape, 4))[valid]
    target = supplied[:, (0, 1, 0, 1), (0, 1, 2, 2)]
    if (not np.isfinite(supplied).all() or not np.isfinite(source).all()
            or np.any(target[:, :2] <= 0)
            or np.any(supplied[:, 0, 1] != 0) or np.any(supplied[:, 1, 0] != 0)
            or not np.allclose(source, target, rtol=1e-6, atol=1e-6)):
        raise ValueError("Dyn-HaMR optimized intrinsics differ from supplied K; refusing to recalibrate output")

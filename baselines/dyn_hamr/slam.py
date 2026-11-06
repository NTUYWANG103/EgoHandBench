"""Run the upstream RGB-only DROID camera estimator with intrinsic-only input."""

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import subprocess

# DROID-SLAM's only recognized algorithmic failure: fewer than two keyframes leave the backend without
# proximity edges (droid_slam/factor_graph.py:374 in add_proximity_factors). This is the only failure
# treated as a SLAM failure. Any other failure is an error.
SLAM_FAILURE = "ValueError: not enough values to unpack (expected 2, got 0)"
# run_slam.py:195 prints this; with t <= 1 it returns eye(4) for every frame (:205), upstream's own fallback.
KEYFRAMES = re.compile(r"^(\d+) keyframes in map$", re.M)


class SlamFailed(RuntimeError):
    """DROID-SLAM itself failed on this clip (the caller may use a static camera)."""


def slam_failure(returncode, log):
    """SlamFailed for DROID's own keyframe failure; any other non-zero exit is an error."""
    if returncode > 0 and SLAM_FAILURE in log and "add_proximity_factors" in log:
        return SlamFailed("droid_too_few_keyframes")
    return None


def estimate_camera(adapter, images, folder, intrinsics, work):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "cameras.npz"
    receipt = folder / "rgb_slam.json"
    if path.exists():
        if not receipt.exists() or json.loads(receipt.read_text()) != {
            "camera_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "uses_gt_extrinsics": False,
        }:
            raise ValueError("Unverified camera cache; use a fresh workspace")
    else:
        from utils.io import resource_path

        weights = resource_path(adapter.assets, "dyn_hamr.droid")
        droid = adapter.repo / "third_party/DROID-SLAM"
        try:
            from baselines.dyn_hamr import droid_backends  # noqa: F401
            from baselines.dyn_hamr import lietorch  # noqa: F401
            missing_backend = False
        except (ImportError, OSError):
            missing_backend = True
        if missing_backend or not weights.is_file() or not (droid / "droid_slam/droid.py").is_file():
            from .setup_slam import main as setup_slam

            setup_slam(adapter.assets)
        k_path = folder / "intrinsics_only.txt"
        np.savetxt(k_path, intrinsics, fmt="%.9g")
        try:
            adapter._command(
                [sys.executable, "-m", "baselines.dyn_hamr.slam", adapter.device, str(adapter.seed),
                 "-i", str(images), "--map_dir", str(folder),
                 "--intrins_path", str(k_path), "--weights", str(weights), "--disable_vis"],
                work, "slam.log",
            )
        except subprocess.CalledProcessError as error:
            failure = slam_failure(error.returncode, (Path(work) / "slam.log").read_text(errors="ignore"))
            if failure is None:
                raise
            raise failure from error
    with np.load(path, allow_pickle=False) as data:
        poses, k = data["w2c"], data["intrins"]
        if k.shape != intrinsics.shape or not np.allclose(k, intrinsics):
            raise ValueError("RGB SLAM camera timeline or intrinsics differ")
        if (poses.shape != (len(intrinsics), 4, 4) or not np.isfinite(poses).all()
                or not np.allclose(poses[:, 3], [0, 0, 0, 1])
                or not np.allclose(poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3],
                                   np.eye(3), atol=1e-3)
                or not np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-3)):
            raise SlamFailed("invalid_camera_poses")
    receipt.write_text(json.dumps({
        "camera_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "uses_gt_extrinsics": False,
    }))
    log = Path(work) / "slam.log"
    keyframes = KEYFRAMES.findall(log.read_text(errors="ignore")) if log.exists() else []
    if keyframes and int(keyframes[-1]) <= 1 and np.allclose(poses, np.eye(4)):
        return "droid_identity_fallback_le1_keyframe"
    return None


if __name__ == "__main__":
    import os
    import runpy

    # DROID uses cuda:0 internally; expose only the adapter's selected device.
    device = int(sys.argv.pop(1).split(":")[-1])
    seed = sys.argv.pop(1)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[device] if visible else str(device)
    if seed != "None":
        import random
        import torch

        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    source = Path(__file__).resolve().parent
    sys.path.insert(0, str(source / "third_party/DROID-SLAM/droid_slam"))
    runpy.run_path(str(source / "dyn-hamr/preproc/run_slam.py"), run_name="__main__")

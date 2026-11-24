"""Run the pinned upstream HaWoR SLAM with supplied intrinsics in its own interpreter."""

from glob import glob
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main(video, start, end, focal, cx, cy, droid, metric3d, seed):
    if seed is not None:
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    source = Path(__file__).resolve().parent / "third_party/HaWoR"
    os.chdir(source)
    sys.path[:0] = [str(source), str(source / "thirdparty/DROID-SLAM/droid_slam")]
    from scripts.scripts_test_video import hawor_slam as slam

    calib = [focal, focal, cx, cy]
    slam.args.weights = droid
    slam.est_calib = lambda imgfiles: list(calib)
    metric = slam.Metric3D
    slam.Metric3D = lambda _: metric(metric3d)
    # Use the same calibrated frames as the HaWoR network when they exist.
    slam.glob = lambda pattern: glob(pattern.replace("*.jpg", "*.png")) or glob(pattern)
    slam.hawor_slam(SimpleNamespace(video_path=video, img_focal=focal), start, end)


if __name__ == "__main__":
    video, start, end, focal, cx, cy, droid, metric3d, seed = sys.argv[1:]
    seed = None if seed == "None" else int(seed)
    main(video, int(start), int(end), float(focal), float(cx), float(cy), droid, metric3d, seed)

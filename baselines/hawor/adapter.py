"""Released HaWoR: camera-space estimator, SLAM with supplied K, and temporal infiller."""

import random
import shutil
from tempfile import TemporaryDirectory
from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, weight_path
from utils.mano import ManoModel, native_source_prediction
from .mano import native_prediction
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from imageio_ffmpeg import get_ffmpeg_exe
from .mano import rotation_vector


# Camera trajectory for the world-space infiller when upstream DROID-SLAM + Metric3D
# (scripts_test_video/hawor_slam.py) fails in its own algorithm (too few DROID keyframes on short
# or low-motion clips, a non-finite trajectory or scale, or no result within the time limit),
# where upstream has no fallback: a static camera.
def static_camera(m, intrinsics):
    """A compatible synthetic identity trajectory in hawor_slam.py's output layout (its np.savez keys).

    traj rows are DROID camera-to-world poses [tx ty tz qx qy qz qw] (float32, as
    droid.terminate returns them), here identity for all m frames; scale is the metric
    scale load_slam_cam multiplies traj[:, :3] by, 1 (no Metric3D). Only traj and scale are
    read (lib/eval_utils/custom_utils.py:129-138); tstamp and disps (DROID keyframes) are
    empty and unread.
    """
    traj = np.zeros((m, 7), np.float32)
    traj[:, 6] = 1.0
    return dict(
        tstamp=np.zeros(0, np.int32),
        disps=np.zeros((0,), np.float32),
        traj=traj,
        img_focal=intrinsics["fx"],
        img_center=np.array([intrinsics["cx"], intrinsics["cy"]]),
        scale=np.float64(1.0),
    )


def slam_succeeded(path, m):
    """hawor_slam.py wrote one finite camera per frame (non-degenerate quaternion) and a finite positive
    metric scale, and load_slam_cam (lib/eval_utils/custom_utils.py:129-138) decodes finite cameras."""
    if not path.is_file():
        return False
    with np.load(path) as data:
        traj, scale = data["traj"], data["scale"]
    if not (traj.shape == (m, 7) and np.isfinite(traj).all() and np.isfinite(scale) and scale > 0
            and (np.linalg.norm(traj[:, 3:], axis=1) > 1e-6).all()):
        return False
    import torch
    from baselines.hawor.lib.eval_utils.custom_utils import load_slam_cam

    return all(bool(torch.isfinite(camera).all()) for camera in load_slam_cam(str(path)))


# DROID-SLAM's own failure on too little camera motion: fewer than two keyframes leave the
# backend without proximity edges (droid_slam/factor_graph.py:396 in add_proximity_factors).
SLAM_FAILURE = "ValueError: not enough values to unpack (expected 2, got 0)"
# Environment failures are never a reason to change the camera.
HARD_FAILURES = ("out of memory", "ModuleNotFoundError", "ImportError", "FileNotFoundError")


def slam_failure(done, path, m):
    """Why SLAM failed in its own algorithm (None: it succeeded); raise for any other failure."""
    if done.returncode < 0 or (done.returncode and (
            any(text in done.stderr for text in HARD_FAILURES)
            or SLAM_FAILURE not in done.stderr or "add_proximity_factors" not in done.stderr)):
        raise subprocess.CalledProcessError(done.returncode, done.args, stderr=done.stderr)
    if done.returncode:
        return "slam_too_few_keyframes"
    return None if slam_succeeded(path, m) else "slam_nonfinite"


def fps30_timeline(video, n):
    """Map ffmpeg's `-vf fps=30` output frames to the n source frames, and back.

    ffmpeg's fps filter (round=near, no start_time) rescales each source frame's
    absolute filter-graph pts to 1/30 s, rounding half away from zero, and output
    frame j (pts P_j, read from the filter's output) shows the last source frame whose
    rescaled pts is <= P_j. The frame checksums ffmpeg reports before and after the
    filter must agree; the JPEG sequence starts at t = 0 (see below). Files with more
    than one video stream are rejected (extraction and map both use stream 0:v:0). Returns
    shown[j] (source frame of JPEG j) and native[i] (JPEG read for source frame i:
    the first one showing it, or, for a frame the filter dropped, the one showing the
    nearest kept source frame).
    """
    import re
    from fractions import Fraction

    log = subprocess.run(
        [get_ffmpeg_exe(), "-hide_banner", "-nostats", "-threads", "2", "-i", str(video),
         "-map", "0:v:0", "-vf", "showinfo,fps=30,showinfo", "-f", "null", "-"],
        capture_output=True, encoding="utf-8", errors="replace", check=True,
    ).stderr
    frames = {0: [], 2: []}
    base = {}
    for line in log.splitlines():
        hit = re.search(r"Parsed_showinfo_(\d+) .*config in time_base: (\d+)/(\d+)", line)
        if hit:
            base[int(hit[1])] = Fraction(int(hit[2]), int(hit[3]))
            continue
        hit = re.search(r"Parsed_showinfo_(\d+) .* n: *\d+ pts: *(-?\d+) .*checksum:(\w+)", line)
        if hit:
            frames[int(hit[1])].append((int(hit[2]), hit[3]))
    # Whole unindented lines only: metadata values (also continuation lines) are indented.
    header = re.split(r"(?m)^Stream mapping:$", log, maxsplit=1)[0]
    if len(re.findall(r"(?m)^  Stream #0:\d+\S*: Video:", header)) != 1:
        raise ValueError("HaWoR expects exactly one video stream")
    source, output = frames[0], frames[2]
    if len(source) != n or not output:
        raise ValueError("HaWoR source video and camera timelines differ")

    def near_inf(value):  # av_rescale_q_rnd(..., AV_ROUND_NEAR_INF): half away from zero
        magnitude = (2 * abs(value.numerator) + value.denominator) // (2 * value.denominator)
        return magnitude if value >= 0 else -magnitude

    slots = np.array([near_inf(p * base[0] * 30) for p, _ in source])
    out_pts = [p * base[2] * 30 for p, _ in output]
    if any(p.denominator != 1 for p in out_pts) or np.any(np.diff(slots) < 0):
        raise ValueError("HaWoR 30 fps timestamps are not on the expected grid")
    out_pts = np.array([int(p) for p in out_pts])
    if out_pts[0] < 0 or np.any(np.diff(out_pts) != 1):
        raise ValueError("HaWoR 30 fps output is not a contiguous frame grid")
    shown = np.searchsorted(slots, out_pts, side="right") - 1
    if shown.min() < 0 or any(output[j][1] != source[i][1] for j, i in enumerate(shown)):
        raise ValueError("HaWoR 30 fps frame map disagrees with ffmpeg")
    # The JPEG writer's constant-rate sync starts the image sequence at t = 0, so a video
    # whose first output pts is P0 > 0 (e.g. it starts after an audio stream) gets P0
    # leading copies of its first frame.
    shown = np.concatenate([np.full(out_pts[0], shown[0], int), shown])
    first = {}
    for j, i in enumerate(shown):
        first.setdefault(int(i), j)
    kept = np.array(sorted(first))
    native = np.empty(n, int)
    for i in range(n):
        k = min(np.searchsorted(kept, i), len(kept) - 1)
        if k > 0 and abs(kept[k - 1] - i) <= abs(kept[k] - i):
            k -= 1
        native[i] = first[int(kept[k])]
    return shown, native


class HaWoRAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        if batch_size is not None:
            raise ValueError("HaWoR does not expose a batch_size option")
        native_camera = {"official": True, "egohandbench": False}.get(version)  # True: native input; False: warp
        if native_camera is None:
            raise ValueError(f"Unknown HaWoR version {version!r}")
        method = "hawor"
        assets = checkpoint_dir(method, checkpoints, version=version)
        device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if device.type != "cuda":
            raise ValueError("HaWoR currently requires a CUDA device")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        device = str(device)
        self.assets = Path(assets)
        self.device = device
        if torch.device(self.device).type == "cuda":
            torch.cuda.set_device(self.device)
        self.repo = Path(__file__).resolve().parent
        # Camera input per version. official: upstream's own, one scalar focal (demo.py --img_focal; we pass fx)
        # with the image centre (scripts_test_video/hawor_video.py:54,96,163) and DROID calibrated as
        # [img_focal, img_focal, W/2, H/2] (hawor_slam.py:70,82-84; lib/pipeline/masked_droid_slam.py:59-71).
        # egohandbench: frames warped to K with fy = fx and the supplied principal point, the camera the
        # egohandbench weights were trained with.
        self.native_camera = native_camera
        self._seed = seed
        work = Path(__file__).resolve().parents[2] / "debug/work"
        work.mkdir(parents=True, exist_ok=True)
        self._temporary = TemporaryDirectory(prefix=method + "-", dir=work)
        self.work_dir = Path(self._temporary.name)
        self._mano_model = None
        torch.set_num_threads(2)
        # official demo.py (ThunderVVV/HaWoR@66c7d41) sets no TF32 flag: keep PyTorch defaults
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        os.environ.update(
            {
                "HAWOR_MANO_DIR": str(resource_path(self.assets, "hawor.mano")),
                "HAWOR_MANO_DATA": str(resource_path(self.assets, "hawor.mano").parent),
                "HAWOR_MANO_LEFT_DIR": str(
                    resource_path(self.assets, "hawor.mano_left")
                ),
                "HAWOR_MANO_LEFT_DATA": str(
                    resource_path(self.assets, "hawor.mano_left").parent
                ),
                "HAWOR_MEAN_PARAMS": str(
                    resource_path(self.assets, "hawor.mean_params")
                ),
                "HAWOR_DETECTOR": str(resource_path(self.assets, "hawor.detector")),
            }
        )
        from baselines.hawor.scripts.scripts_test_video.detect_track_video import (
            detect_track_video,
        )
        from baselines.hawor.scripts.scripts_test_video import hawor_video
        from baselines.hawor.hawor.utils.process import run_mano, run_mano_left

        self.detect, self.video = (detect_track_video, hawor_video)
        self.source_decode = {0: run_mano_left, 1: run_mano}
        self.checkpoint = weight_path(assets, "hawor")
        original_load = self.video.load_hawor
        intrinsics = {}

        def load(path):
            model, cfg = original_load(path)
            inference = model.inference

            def calibrated(imgfiles, boxes, img_focal, img_center, **kwargs):
                return inference(
                    imgfiles,
                    boxes,
                    img_focal=intrinsics["fx"],
                    img_center=[intrinsics["cx"], intrinsics["cy"]],
                    **kwargs,
                )

            model.inference = calibrated
            return (model, cfg)

        self._load_model = load
        self.intrinsics = intrinsics
        self.reset()

    def _predict_raw(self, record):
        n = record["num_frames"]
        if not np.isfinite(record["fps"]) or record["fps"] <= 0:
            raise ValueError("Invalid source timeline fps")
        work = self.work_dir / "hawor_paper" / "full" / record["clip"]
        work.mkdir(parents=True, exist_ok=True)
        video = work / (record["clip"] + ".mp4")
        if not video.exists():
            video.symlink_to(Path(record["video"]).resolve())
        images = work / record["clip"] / "extracted_images"
        images.mkdir(parents=True, exist_ok=True)
        if not list(images.glob("*.jpg")):
            # Official detect_track_video.py always extracts at 30 fps, which sets the
            # time span of the 16-frame estimator and 120-frame infiller windows.
            subprocess.run(
                [
                    get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-threads",
                    "2",
                    "-i",
                    str(video),
                    "-map",
                    "0:v:0",
                    "-vf",
                    "fps=30",
                    "-start_number",
                    "0",
                    "-threads",
                    "2",
                    str(images / "%04d.jpg"),
                ],
                check=True,
            )
        m = len(list(images.glob("*.jpg")))
        shown, native = fps30_timeline(video, n)
        if len(shown) != m:
            raise ValueError("HaWoR 30 fps JPEG timeline differs from ffmpeg's frame map")
        if self.native_camera:
            height, width = cv2.imread(str(images / "0000.jpg")).shape[:2]
            focal = float(record["intrinsics_per_frame"][shown][0][0, 0])
            self.intrinsics.update(fx=focal, fy=focal, cx=width / 2, cy=height / 2)
        else:
            from .camera import rectify_frames

            self.intrinsics.update(rectify_frames(images, record["intrinsics_per_frame"][shown]))
        args = SimpleNamespace(
            video_path=str(video),
            input_type="file",
            img_focal=self.intrinsics["fx"],
            img_center=[self.intrinsics["cx"], self.intrinsics["cy"]],
            checkpoint=str(self.checkpoint),
        )
        start, end, folder, _ = self.detect(args)
        frame_chunks, _ = self.video.hawor_motion_estimation(
            args, start, end, folder, load_model=self._load_model
        )
        if end != m or start != 0:
            raise ValueError("HaWoR camera-space timeline differs")
        slam = Path(folder) / f"SLAM/hawor_slam_w_scale_{start}_{end}.npz"
        if not slam.is_file():
            try:
                from baselines.hawor import droid_backends  # noqa: F401
                from baselines.hawor import lietorch  # noqa: F401
                missing_backend = False
            except (ImportError, OSError):
                missing_backend = True
            if (missing_backend or not (self.repo / "third_party/HaWoR/scripts").is_dir()
                    or any(not resource_path(self.assets, key).is_file()
                           for key in ("hawor.droid", "hawor.metric3d", "hawor.infiller"))):
                from .setup_slam import main as setup_slam

                setup_slam(self.assets)
            try:
                done = subprocess.run(
                    [
                        sys.executable, "-m", "baselines.hawor.slam", str(video),
                        str(start), str(end), str(self.intrinsics["fx"]),
                        str(self.intrinsics["cx"]), str(self.intrinsics["cy"]),
                        str(resource_path(self.assets, "hawor.droid")),
                        str(resource_path(self.assets, "hawor.metric3d")),
                        str(self._seed),
                    ],
                    cwd=self.repo.parents[1],
                    stderr=subprocess.PIPE,
                    encoding="utf-8",  # The host locale may be ASCII
                    errors="replace",
                    # hawor_slam.py's `while math.isnan(scale)` (:124-127) can loop forever
                    timeout=1800 + m,
                )
                sys.stderr.write(done.stderr)
                reason = slam_failure(done, slam, m)
            except subprocess.TimeoutExpired:
                reason = "slam_timeout"
        else:  # a trajectory already in this workspace is held to the same check
            reason = None if slam_succeeded(slam, m) else "slam_nonfinite"
        if reason:  # SLAM's own failure: a static camera
            print(json.dumps(dict(camera_fallback="static", method="hawor", clip=record["clip"], reason=reason)), flush=True)
            slam.parent.mkdir(parents=True, exist_ok=True)
            np.savez(slam, **static_camera(m, self.intrinsics))
        args.infiller_weight = str(resource_path(self.assets, "hawor.infiller"))
        world = self.video.hawor_infiller(args, start, end, frame_chunks)
        result = {
            k: np.full((m, 2, *tail), np.nan, np.float32)
            for k, tail in [
                ("vertices_cam", (778, 3)),
                ("joints_cam", (21, 3)),
                ("global_orient", (1, 3, 3)),
                ("hand_pose", (15, 3, 3)),
                ("betas", (10,)),
                ("cam_trans", (3,)),
            ]
        }
        for side in (0, 1):
            # Apply chunks in frame_chunks order, as official hawor_infiller does, so a
            # frame shared by two chunks keeps the later chunk's prediction.
            for frame_ck in frame_chunks[side]:
                ids = np.asarray(frame_ck, int)
                path = Path(folder) / "cam_space" / str(side) / f"{frame_ck[0]}_{frame_ck[-1]}.json"
                data = json.loads(path.read_text())
                go = np.asarray(data["init_root_orient"], np.float32)[0]
                hp = np.asarray(data["init_hand_pose"], np.float32)[0]
                beta = np.asarray(data["init_betas"], np.float32)[0]
                trans = np.asarray(data["init_trans"], np.float32)[0]
                if len(go) != len(ids):
                    raise ValueError("HaWoR chunk length differs from its frames")
                with torch.no_grad():
                    decoded = self.source_decode[side](
                        torch.as_tensor(trans)[None],
                        torch.as_tensor(rotation_vector(go))[None],
                        torch.as_tensor(rotation_vector(hp))[None],
                        betas=torch.as_tensor(beta)[None],
                    )
                result["vertices_cam"][ids, side] = decoded["vertices"][0].cpu().numpy()
                result["joints_cam"][ids, side] = decoded["joints"][0].cpu().numpy()
                result["global_orient"][ids, side, 0] = go
                result["hand_pose"][ids, side] = hp
                result["betas"][ids, side] = beta
                result["cam_trans"][ids, side] = trans
        self._fill_missing(result, world, slam, frame_chunks)
        result = {k: v[native] for k, v in result.items()}
        present = np.isfinite(result["vertices_cam"]).all(axis=(-1, -2))
        result.update(
            presence=present,
            confidence=present.astype(np.float32),
            frame_ids=np.arange(n),
            projection_intrinsics=np.repeat(
                record["intrinsics_per_frame"][:, None, (0, 1, 0, 1), (0, 1, 2, 2)],
                2, axis=1,
            ),
        )
        return result

    def _fill_missing(self, result, world, slam, frame_chunks):
        """Bring infilled world-space hands back through the SLAM cameras."""
        from baselines.hawor.lib.eval_utils.custom_utils import load_slam_cam
        from baselines.hawor.lib.eval_utils.filling_utils import world2canonical_convert
        from baselines.hawor.hawor.utils.rotation import angle_axis_to_rotation_matrix

        trans, rot, pose, betas, valid = world
        rotation, translation, _, _ = load_slam_cam(str(slam))
        observed = np.isfinite(result["vertices_cam"]).all(axis=(-1, -2))
        for side in (0, 1):
            # Like official demo.py, a side that was never detected still takes the
            # infiller output wherever hawor_infiller marks it valid.
            missing = np.flatnonzero(valid[side] & ~observed[:, side])
            if len(missing) == 0:
                continue
            index = torch.as_tensor(missing)
            camera = world2canonical_convert(
                rotation[index].float(),
                translation[index].float(),
                {
                    "init_trans": trans[side : side + 1, index],
                    "init_root_orient": angle_axis_to_rotation_matrix(rot[side : side + 1, index]),
                    "init_hand_pose": angle_axis_to_rotation_matrix(
                        pose[side : side + 1, index].reshape(1, -1, 15, 3)
                    ),
                    "init_betas": betas[side : side + 1, index],
                },
                "right" if side else "left",
            )
            with torch.no_grad():
                decoded = self.source_decode[side](
                    camera["init_trans"],
                    camera["init_root_orient"],
                    camera["init_hand_pose"],
                    betas=camera["init_betas"],
                )
            result["vertices_cam"][missing, side] = decoded["vertices"][0].cpu().numpy()
            result["joints_cam"][missing, side] = decoded["joints"][0].cpu().numpy()
            result["global_orient"][missing, side, 0] = angle_axis_to_rotation_matrix(
                camera["init_root_orient"][0]
            ).numpy()
            result["hand_pose"][missing, side] = angle_axis_to_rotation_matrix(
                camera["init_hand_pose"][0]
            ).numpy()
            result["betas"][missing, side] = camera["init_betas"][0].numpy()
            result["cam_trans"][missing, side] = camera["init_trans"][0].numpy()

    def inference_stream(self, inputs, *, joint_only=False):
        raise NotImplementedError("hawor requires a complete offline clip")

    def inference_offline(self, inputs, *, joint_only=False):
        return self._inference(inputs, "offline", joint_only)

    def _inference(self, inputs, mode, joint_only=False):
        if torch.device(self.device).type == "cuda":
            torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        with InferenceInput(inputs, mode=mode, requires_video_path=True) as (
            record,
            camera,
        ):
            raw = self._predict_raw(record)
        source = native_source_prediction(raw, camera)  # native mano_wrapper.mano_to_openpose order
        if joint_only:
            return {"mano": None, "joint": source}
        if self._mano_model is None:
            self._mano_model = ManoModel(
                resource_path(self.assets, "mano")
            )
        return {"mano": native_prediction(raw, camera, self._mano_model), "joint": source}

    def reset(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)

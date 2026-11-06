"""Dyn-HaMR with HaMeR initialization, supplied K and RGB-estimated extrinsics."""

from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, load_npz, resource_path
import json
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from utils.mano import axis_angle_to_rotmat, ManoModel, native_source_prediction
from .mano import native_prediction
import torch
from .camera import translation_to_intrinsics, write_static_camera
import pickle
import shutil


class DynHaMRAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        import tempfile

        self.assets = checkpoint_dir("dyn_hamr", checkpoints, version=version)
        self.repo = Path(__file__).resolve().parent
        self.seed = seed
        self.batch_size = batch_size
        selected_device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if selected_device.type == "cuda" and selected_device.index is None:
            selected_device = torch.device("cuda", torch.cuda.current_device())
        self.device = str(selected_device)
        if not self.device.startswith("cuda"):
            raise ValueError("The official Dyn-HaMR pipeline requires CUDA")
        torch.cuda.set_device(self.device)
        torch.set_num_threads(2)
        folder = self.repo.parents[1] / "debug/work"
        folder.mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(prefix="dyn_hamr-", dir=folder)
        self.work_dir = Path(self._directory.name)
        self.raw_root = self.work_dir / "initializer_raw"
        self._mano_model = None

        self.work_root = self.work_dir / "dyn_hamr"
        self.reset()

    def _command(self, command, work, log):
        with (work / log).open("a") as stream:
            subprocess.run(
                command,
                cwd=self.repo.parents[1],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def export(self, record, n, work):
        """Decode completed official stages without rerunning their optimizer."""
        work = Path(work)
        clip = record["clip"]
        output = work / "optimization"
        completion = json.loads((work / "optimization_complete.json").read_text())
        if (
            completion.get("pipeline_input_failure")
            == "no_tracks_survive_official_selection"
        ):
            if completion["frames"] != n or completion["optimizer_ran"]:
                raise ValueError("Invalid official empty-track selection receipt")
            result = {
                k: np.full((n, 2, *shape), np.nan, np.float32)
                for k, shape in (
                    ("vertices_cam", (778, 3)),
                    ("joints_cam", (21, 3)),
                    ("global_orient", (1, 3, 3)),
                    ("hand_pose", (15, 3, 3)),
                    ("betas", (10,)),
                    ("cam_trans", (3,)),
                )
            }
            result.update(
                frame_ids=np.arange(n),
                presence=np.zeros((n, 2), bool),
                confidence=np.zeros((n, 2), np.float32),
                projection_intrinsics=np.full((n, 2, 4), np.nan, np.float64),
            )
            return result
        if (
            completion["root_schedule"] != 50
            or completion["smooth_schedule"] != 300
            or completion["run_prior"]
            or completion["exit_code"] != 0
        ):
            raise ValueError(
                "Official pipeline did not finish the released configuration"
            )
        result_path = output / "smooth_fit" / f"{clip}_000300_world_results.npz"
        with np.load(result_path, allow_pickle=False) as data:
            prediction = {k: np.array(v) for k, v in data.items()}
        info = json.loads((output / "track_info.json").read_text())
        start, end = info["meta"]["seq_interval"]
        if info["meta"]["data_interval"] != [0, n] or not 0 <= start < end <= n:
            raise ValueError("Dyn-HaMR optimization timeline differs from pinned input")
        b, t = prediction["trans"].shape[:2]
        if t != end - start or b != len(info["tracks"]):
            raise ValueError("Dyn-HaMR track metadata and optimization output disagree")
        from importlib import import_module

        body_model = import_module("baselines.dyn_hamr.dyn-hamr.body_model")
        MANO, run_mano = body_model.MANO, body_model.run_mano

        model = MANO(
            model_path=str(resource_path(self.assets, "mano")),
            batch_size=b * t,
            pose2rot=True,
            create_body_pose=False,
        ).to(self.device)
        tensors = {
            k: torch.as_tensor(prediction[k], device=self.device, dtype=torch.float32)
            for k in ("trans", "root_orient", "pose_body", "is_right", "betas")
        }
        with torch.no_grad():
            decoded = run_mano(
                model,
                tensors["trans"],
                tensors["root_orient"],
                tensors["pose_body"],
                tensors["is_right"],
                betas=tensors["betas"],
            )
        world = decoded["vertices"].cpu().numpy()
        world_joints = decoded["joints"].cpu().numpy()
        rotation, translation = prediction["cam_R"], prediction["cam_t"]
        if rotation.shape != (b, t, 3, 3) or translation.shape != (b, t, 3):
            raise ValueError("Unexpected official camera shape")
        # The released optimizer exports one fixed [fx, fy, cx, cy] vector.
        # Broadcasting also accepts an explicitly time-varying camera export.
        with np.load(work / "dynhamr/cameras" / clip / "shot-0/cameras.npz") as cameras:
            intrinsics = np.broadcast_to(cameras["intrins"][start:end], (b, t, 4)).copy()
        camera_vertices = (
            np.einsum("btij,btvj->btvi", rotation, world) + translation[:, :, None]
        )
        camera_joints = (
            np.einsum("btij,btvj->btvi", rotation, world_joints)
            + translation[:, :, None]
        )
        go = axis_angle_to_rotmat(
            torch.from_numpy(prediction["root_orient"]).double()
        ).numpy()
        hp = axis_angle_to_rotmat(
            torch.from_numpy(prediction["pose_body"]).double().reshape(b, t, 15, 3)
        ).numpy()
        result = {
            k: np.full((n, 2, *shape), np.nan, np.float32)
            for k, shape in (
                ("vertices_cam", (778, 3)),
                ("joints_cam", (21, 3)),
                ("global_orient", (1, 3, 3)),
                ("hand_pose", (15, 3, 3)),
                ("betas", (10,)),
                ("cam_trans", (3,)),
            )
        }
        presence = np.zeros((n, 2), bool)
        projection = np.full((n, 2, 4), np.nan, np.float64)
        for tid, track in info["tracks"].items():
            index = track["index"]
            side = int(tid)
            if side not in (0, 1) or not np.all(prediction["is_right"][index] == side):
                raise ValueError("Official track handedness changed")
            observed = np.flatnonzero(track["vis_mask"])
            if len(observed) == 0:
                continue
            full = np.arange(max(start, observed[0]), min(end, observed[-1] + 1))
            take = full - start
            reflect = np.diag([-1.0, 1.0, 1.0]) if side == 0 else np.eye(3)
            # The official run_mano reflects the entire world mesh, including
            # translation. Camera rotation acts AFTER this reflection.
            result["vertices_cam"][full, side] = camera_vertices[index, take]
            result["joints_cam"][full, side] = camera_joints[index, take]
            result["global_orient"][full, side, 0] = (
                rotation[index, take] @ reflect @ go[index, take] @ reflect
            )
            result["hand_pose"][full, side] = reflect @ hp[index, take] @ reflect
            result["betas"][full, side] = prediction["betas"][index]
            result["cam_trans"][full, side] = (
                np.einsum(
                    "tij,tj->ti",
                    rotation[index, take],
                    prediction["trans"][index, take] @ reflect,
                )
                + translation[index, take]
            )
            projection[full, side] = intrinsics[index, take]
            presence[full, side] = True
        if not np.isfinite(result["vertices_cam"][presence]).all():
            raise ValueError("Nonfinite official Dyn-HaMR optimized mesh")
        result.update(
            frame_ids=np.arange(n),
            presence=presence,
            confidence=presence.astype(np.float32),
            projection_intrinsics=projection,
        )
        return result

    def _frames(self, record):
        """Every native frame as JPEG, like the official preproc split_frame."""
        n = record["num_frames"]
        work = self.work_root / "full" / record["clip"]
        work.mkdir(parents=True, exist_ok=True)
        images = work / "images" / record["clip"]
        images.mkdir(parents=True, exist_ok=True)
        if not (work / "frames_complete.json").exists():
            cap = cv2.VideoCapture(record["video"])
            try:
                for i in range(n):
                    ok, frame = cap.read()
                    if not ok or not cv2.imwrite(str(images / f"{i+1:06d}.jpg"), frame):
                        raise ValueError("Dyn-HaMR image extraction failed")
            finally:
                cap.release()
            (work / "frames_complete.json").write_text(json.dumps(dict(frames=n)))
        if len(list(images.glob("*.jpg"))) != n:
            raise ValueError("Dyn-HaMR JPEG timeline differs")
        return work, images

    @staticmethod
    def _static_camera(images, folder, intrinsics):
        if folder.exists():
            shutil.rmtree(folder)
        write_static_camera(images, folder, intrinsics[:, (0, 1, 0, 1), (0, 1, 2, 2)])

    def _predict_native(self, record):
        n = record["num_frames"]
        raw = load_npz(self.raw_root / (record["clip"] + ".npz"))
        if not np.array_equal(raw["frame_ids"], np.arange(record["num_frames"])):
            raise ValueError("HaMeR source timeline differs")
        # A fresh workspace per clip (HandDetector resets): padded JPEGs left by an earlier
        # attempt would fail _frames' JPEG timeline check.
        work, images = self._frames(record)
        # Supplied K drives RGB SLAM and temporal optimization.
        intrinsics = record["intrinsics_per_frame"][:n]
        # Upstream optimizes only tracks seen in more than MIN_TRACK_LEN frames
        # (dyn-hamr/data/dataset.py:33,134-142) and drops the rest. A shorter track's last
        # real frame is repeated after the clip end, entirely (image, K and every hand
        # observed in that frame), until the track passes.
        # With two short tracks the later last frame comes first; the other track, if still
        # short, then gets its own last frame repeated. The repeated frames, and each
        # track's frames after its last real observation, are dropped from the output.
        # Clips without a short track run unchanged.
        from importlib import import_module

        minimum = import_module("baselines.dyn_hamr.dyn-hamr.data.dataset").MIN_TRACK_LEN
        presence = raw["presence"][:n].astype(bool)
        counts = presence.sum(axis=0)
        last = [int(np.flatnonzero(presence[:, s])[-1]) if counts[s] else -1 for s in (0, 1)]
        short = [s for s in (0, 1) if 0 < counts[s] <= minimum]
        tail = []  # the real frame each padded frame repeats
        for s in sorted(short, key=lambda s: -last[s]):
            seen = counts[s] + sum(presence[f, s] for f in tail)
            tail += [last[s]] * max(0, int(minimum + 1 - seen))
        pad = len(tail)
        if pad:
            print(json.dumps(dict(track_padding=pad, method="dyn_hamr", clip=record["clip"], short=short,
                                  repeated_frames=sorted(set(tail)))), flush=True)
            for i, f in enumerate(tail, n):
                shutil.copyfile(images / f"{f+1:06d}.jpg", images / f"{i+1:06d}.jpg")
            intrinsics = np.concatenate([intrinsics, intrinsics[tail]])
        total = n + pad
        if not (work / "initial.pkl").exists():
            if "model_anchors2d_openpose" not in raw:
                raise ValueError(
                    "Paper Dyn-HaMR requires saved native HaMeR 2D-head anchors"
                )
            uv = raw["model_anchors2d_openpose"][:n]
            if (
                uv.shape != (n, 2, 21, 2)
                or not np.isfinite(uv[raw["presence"][:n]]).all()
            ):
                raise ValueError("Missing HaMeR model anchors")
            # Start HaMeR's virtual-focal translation at the supplied K.
            delta = np.zeros((n, 2, 3))
            delta[presence] = translation_to_intrinsics(
                {k: raw[k][:n] for k in ("presence", "joints_cam")}
                | {"projection_intrinsics": raw["projection_intrinsics"]},
                intrinsics[:n],
            )
            frames = {}
            for i in range(total):
                j = i if i < n else tail[i - n]
                sides = np.flatnonzero(presence[j]).tolist()
                source = {s: j for s in sides}
                frames[str(images / f"{i+1:06d}.jpg")] = dict(
                    mano=[
                        dict(
                            global_orient=raw["global_orient"][source[s], s],
                            hand_pose=raw["hand_pose"][source[s], s],
                            betas=raw["betas"][source[s], s],
                            is_right=s,
                        )
                        for s in sides
                    ],
                    cam_trans=[raw["cam_trans"][source[s], s] + delta[source[s], s] for s in sides],
                    tid=np.asarray(sides),
                    tracked_ids=sides,
                    tracked_time=[0] * len(sides),
                    extra_data=[
                        np.column_stack([uv[source[s], s], np.ones(21)]).tolist() for s in sides
                    ],
                    shot=0,
                )
            with (work / "initial.pkl").open("wb") as f:
                pickle.dump(frames, f)
        folder = work / "dynhamr/cameras" / record["clip"] / "shot-0"
        from .slam import SlamFailed, estimate_camera

        try:
            native = estimate_camera(
                self, images, folder, intrinsics[:, (0, 1, 0, 1), (0, 1, 2, 2)], work
            )
            if native:  # upstream run_slam's own identity cameras (<= 1 DROID keyframe), kept as is
                print(json.dumps(dict(camera_fallback="static", method="dyn_hamr", clip=record["clip"], reason=native)), flush=True)
        except SlamFailed as error:
            # DROID-SLAM's own failure: its static (identity) fallback.
            print(json.dumps(dict(camera_fallback="static", method="dyn_hamr", clip=record["clip"], reason=str(error))), flush=True)
            self._static_camera(images, folder, intrinsics)
        command = [
            sys.executable,
            "-m",
            "baselines.dyn_hamr",
            "--assets",
            str(self.assets),
            "--work",
            str(work),
            "--clip",
            record["clip"],
            "--frames",
            str(total),
            "--fps",
            str(record["fps"]),
            "--device",
            self.device,
        ]
        if not (work / "optimization_complete.json").exists():
            try:
                self._command(command, work, "optimization.log")
            except subprocess.CalledProcessError as error:
                # worker.py exits with NAN_STOP_STATUS (and writes optimizer_nan.json) only for
                # upstream's NaN stop (optim/optimizers.py:290-295). Under the SLAM cameras, retry
                # once from the same stage-1 input with the static camera, from a clean state (the
                # optimizer resumes from checkpoints it finds, optimizers.py:85-100,258).
                from .worker import NAN_STOP_STATUS

                if error.returncode != NAN_STOP_STATUS or not (
                        (folder / "rgb_slam.json").exists() and (work / "optimizer_nan.json").exists()):
                    raise
                print(json.dumps(dict(camera_fallback="static", method="dyn_hamr", clip=record["clip"], reason="optimizer_nan")), flush=True)
                (work / "optimizer_nan.json").unlink()
                for stale in (work / "dynhamr", work / "optimization"):
                    if stale.exists():
                        shutil.rmtree(stale)
                    assert not stale.exists()
                self._static_camera(images, folder, intrinsics)
                self._command(command, work, "optimization.log")
        result = self.export(record, total, work)
        if pad:  # keep the real frames, and each track only up to its last real observation
            result = {k: v[:n] if k != "frame_ids" else np.arange(n) for k, v in result.items()}
            for s in (0, 1):
                after = slice(last[s] + 1, n)
                for k in ("vertices_cam", "joints_cam", "global_orient", "hand_pose", "betas", "cam_trans"):
                    result[k][after, s] = np.nan
                result["projection_intrinsics"][after, s] = np.nan
                result["presence"][after, s] = False
                result["confidence"][after, s] = 0
        return result

    def inference_stream(self, inputs, *, joint_only=False):
        raise NotImplementedError("Dyn-HaMR requires a complete clip")

    def inference_offline(self, inputs, *, joint_only=False, batch_size=None):
        if str(self.device).startswith("cuda"):
            torch.cuda.set_device(self.device)
        from .initializer import HaMeRInitializer
        from utils.io import save_npz
        import gc

        with InferenceInput(
            inputs, mode="offline", requires_video_path=True
        ) as (record, camera):
            # The official HaMeR stage reads the same JPEG frames as SLAM.
            _, images = self._frames(record)
            initializer = HaMeRInitializer(
                checkpoints=self.assets,
                device=self.device,
                seed=self.seed,
                batch_size=self.batch_size if batch_size is None else batch_size,
            )
            try:
                raw = initializer._predict_raw(record, images)
                save_npz(self.raw_root / (record["clip"] + ".npz"), raw)
            finally:
                initializer.reset()
                del initializer
                gc.collect()
                torch.cuda.empty_cache()
            raw = self._predict_native(record)
        from .camera import validate_prediction_intrinsics

        validate_prediction_intrinsics(raw, camera)
        source = native_source_prediction(raw, camera)  # native mano_wrapper.mano_to_openpose order
        if joint_only:
            return {"mano": None, "joint": source}
        if self._mano_model is None:
            self._mano_model = ManoModel(resource_path(self.assets, "mano"))
        return {"mano": native_prediction(raw, camera, self._mano_model), "joint": source}

    def reset(self):
        import shutil

        if self.work_root.exists():
            shutil.rmtree(self.work_root)
        if self.seed is not None:
            import random

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.random.default_generator.manual_seed(self.seed)
            if str(self.device).startswith("cuda"):
                torch.cuda.set_device(self.device)
                torch.cuda.manual_seed(self.seed)

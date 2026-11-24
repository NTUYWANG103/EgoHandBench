"""Official EgoForce inference, known pinhole K, and transparent FK capture."""

from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, VideoReader, weight_path
import numpy as np
from utils.mano import axis_angle_to_rotmat, ManoModel
from .mano import native_prediction, native_joints
import torch


class CaptureLimb:
    """Observe the official FK inputs/output without changing any model result."""

    def __init__(self, original):
        self.original = original
        self.last = None

    def __getattr__(self, key):
        return getattr(self.original, key)

    def __call__(
        self, betas, go, hp, transl, side, arm_shape, arm_rot, *args, **kwargs
    ):
        from baselines.egoforce.utils.rotations import rotation_6d_to_axis_angle_direct

        output = self.original(
            betas, go, hp, transl, side, arm_shape, arm_rot, *args, **kwargs
        )
        if go.shape[-1] == 6:
            go = rotation_6d_to_axis_angle_direct(go.reshape(-1, 6))
            hp = rotation_6d_to_axis_angle_direct(hp.reshape(-1, 6))
        self.last = dict(
            betas=betas.detach().float().cpu().numpy().reshape(2, 10),
            global_orient_aa=go.detach().float().cpu().numpy().reshape(2, 3),
            hand_pose_aa=hp.detach().float().cpu().numpy().reshape(2, 45),
            base_vertices=output.hand.vertices.detach().float().cpu().numpy().copy(),
            initial_translation=transl.detach().float().cpu().numpy().reshape(2, 3),
        )
        return output


class EgoForceAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        if batch_size is not None:
            raise ValueError("EgoForce uses one frame with its fixed left/right native batch")
        self.assets = checkpoint_dir("egoforce", checkpoints, version=version)
        selected_device = torch.device(
            "cuda:0"
            if device in (None, "auto") and torch.cuda.is_available()
            else "cpu" if device in (None, "auto") else device
        )
        if selected_device.type == "cuda" and selected_device.index is None:
            selected_device = torch.device("cuda", torch.cuda.current_device())
        self.device = str(selected_device)
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
        self.seed = seed
        self._mano_model = None
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if seed is not None:
            import random

            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed(seed)
        if not self.device.startswith("cuda"):
            raise ValueError("Official EgoForce TensorRT inference requires CUDA")
        import os

        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from baselines.egoforce.settings import config
        from baselines.egoforce.demo.inference import Inference

        assets = self.assets
        config.POSE_3D.CHECKPOINT_PATH = str(weight_path(assets, "egoforce"))
        config.DETECTION.HAND_ARM_PATH = str(resource_path(assets, "egoforce.pose"))
        config.DETECTION.HAND_PATH = str(resource_path(assets, "egoforce.detector"))
        config.MANO_PATH = str(resource_path(assets, "egoforce.mano"))
        self.runner = Inference(undistort_inp=True)
        self.runner.limb_model = CaptureLimb(self.runner.limb_model)
        self.runner.device = torch.device(self.device)
        means = []
        for name in ("left", "right"):
            source = getattr(self.runner.limb_model.hand_model, "mano_layer_" + name)
            means.append(source.hand_mean.detach().cpu().numpy().reshape(15, 3))
        self.means = np.array(means, np.float32)
        values = [
            torch.zeros(shape, device=self.runner.device, dtype=torch.float16)
            for shape in (
                (2, 1, 3, 224, 224),
                (2, 1, 3, 6, 2),
                (2, 1, 3, 224, 224),
                (2, 1, 3, 6, 2),
            )
        ]
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=True, dtype=torch.float16
        ):
            self.runner.model(*values)
            torch.cuda.synchronize()
        self.reset()

    def reset(self):
        """Discard tracking/Kalman history while retaining the loaded models."""
        torch.cuda.set_device(self.device)
        if self.seed is not None:
            import random

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.random.default_generator.manual_seed(self.seed)
            if str(self.device).startswith("cuda"):
                torch.cuda.manual_seed(self.seed)
        self._camera_signature = None
        self.runner.reset_runtime_state()
        self.runner.limb_model.last = None
        predictor = getattr(self.runner.hand_detector, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()

    def _configure_camera(self, record, undistort_inp=True, frame_index=0):
        """Update frame calibration without resetting temporal history."""
        k = record["intrinsics_per_frame"][frame_index]
        signature = tuple(k[(0, 1, 0, 1), (0, 1, 2, 2)]) + (
            record["width"],
            record["height"],
            record["fps"],
            undistort_inp,
        )
        if self._camera_signature is not None:
            if signature[4:] != self._camera_signature[4:]:
                raise ValueError(
                    "EgoForce canvas/fps/undistortion changed; call reset() first"
                )
            if signature == self._camera_signature:
                return
        from baselines.egoforce.camera_models.pinhole import PinholeCameraModel

        camera = PinholeCameraModel(
            [k[0, 0], k[1, 1]], [k[0, 2], k[1, 2]], record["width"], record["height"]
        )
        self.runner.set_camera_model(camera, undistort_inp=undistort_inp)
        if self._camera_signature is not None:
            self._camera_signature = signature
            return
        self.runner.set_kalman_filter_frequency(record["fps"])
        self.runner.reset_runtime_state()
        import json
        import zipfile

        with zipfile.ZipFile(
            resource_path(self.assets, "egoforce.detector")
        ) as archive:
            name = next(
                (n for n in archive.namelist() if n.endswith("/extra/config.txt"))
            )
            detector_metadata = json.loads(archive.read(name))
        self.runner.yolo_track_cfg = dict(getattr(self.runner, "yolo_track_cfg", {}))
        self.runner.yolo_track_cfg.update(imgsz=detector_metadata["imgsz"], rect=False)
        self._camera_signature = signature

    def predict(self, record, *, joint_only=False, undistort_inp=True):
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        self._configure_camera(record, undistort_inp)
        n = record["num_frames"]
        shapes = [("joints_cam", (21, 3))]
        if not joint_only:
            shapes += [
                ("vertices_cam", (778, 3)),
                ("global_orient", (1, 3, 3)),
                ("hand_pose", (15, 3, 3)),
                ("betas", (10,)),
                ("cam_trans", (3,)),
                ("source_hand_pose_axis_angle", (45,)),
            ]
        result = {
            key: np.full((n, 2, *shape), np.nan, np.float32)
            for key, shape in shapes
        }
        presence = np.zeros((n, 2), bool)
        cap = VideoReader(record.get("_frames_rgb", record["video"]), color="bgr")
        try:
            for t in range(n):
                self._configure_camera(record, undistort_inp, frame_index=t)
                ok, frame = cap.read()
                if not ok or frame.shape[:2] != (record["height"], record["width"]):
                    raise ValueError(f"RGB timeline/canvas mismatch at frame {t}")
                self.runner.limb_model.last = None
                output = self.runner.run_outputs(frame[:, :, ::-1].copy(), self.device)
                result["joints_cam"][t] = np.asarray(output["pred_j3d"], np.float32).reshape(2, 21, 3)
                presence[t] = np.asarray(output["visible_hand"], bool).reshape(2)
                if joint_only:
                    continue
                captured = self.runner.limb_model.last
                if captured is None:
                    raise ValueError(
                        "Official EgoForce inference omitted its FK capture"
                    )
                vertices = np.asarray(output["pred_vertices"], np.float32).reshape(
                    2, 778, 3
                )
                delta = vertices - captured["base_vertices"].reshape(2, 778, 3)
                translation = delta[:, 0]
                aa = np.concatenate(
                    [
                        captured["global_orient_aa"][:, None],
                        captured["hand_pose_aa"].reshape(2, 15, 3),
                    ],
                    axis=1,
                )
                rotations = (
                    axis_angle_to_rotmat(torch.tensor(aa, dtype=torch.float64))
                    .float()
                    .numpy()
                )
                result["vertices_cam"][t] = vertices
                result["global_orient"][t] = rotations[:, :1]
                result["hand_pose"][t] = rotations[:, 1:]
                result["betas"][t] = captured["betas"]
                result["cam_trans"][t] = captured["initial_translation"] + translation
                result["source_hand_pose_axis_angle"][t] = captured["hand_pose_aa"]
                if (t + 1) % 100 == 0:
                    print(
                        f"EgoForce progress: {record['clip']} {t + 1}/{n}", flush=True
                    )
        finally:
            cap.release()
        result.update(
            frame_ids=np.arange(n), presence=presence, source_hand_mean=self.means
        )
        return result

    def inference_stream(self, inputs, *, joint_only=False, undistort_inp=True):
        with InferenceInput(inputs, mode="stream") as (record, camera):
            self._validate_camera(camera)
            return self._output(
                self.predict(record, joint_only=joint_only, undistort_inp=undistort_inp),
                camera, joint_only=joint_only,
            )

    def inference_offline(self, inputs, *, joint_only=False, undistort_inp=True):
        with InferenceInput(inputs, mode="offline") as (record, camera):
            self._validate_camera(camera)
            return self._output(
                self.predict(record, joint_only=joint_only, undistort_inp=undistort_inp),
                camera, joint_only=joint_only,
            )

    def _validate_camera(self, camera):
        if np.any(camera.K[:, (0, 1), (1, 0)] != 0):
            raise ValueError("EgoForce's native pinhole camera does not support skew")

    def _output(self, raw, camera, *, joint_only=False):
        joints = native_joints(raw, camera.source_frame_ids)
        if joint_only:
            return dict(mano=None, joint=joints)
        if self._mano_model is None:
            self._mano_model = ManoModel(
                resource_path(self.assets, "mano"), device="cpu"
            )
        return dict(mano=native_prediction(raw, camera, self._mano_model), joint=joints)

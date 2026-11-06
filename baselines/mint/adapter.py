"""MINT native camera-space hand decoding."""

from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, VideoReader, weight_path
from pathlib import Path
import numpy as np
from utils.mano import ManoModel
from .mano import native_prediction
import torch
from scipy.spatial.transform import Rotation


def decode_native(prediction, model_dir, height, width):
    """Decode the released hand parameters using the native MANO implementation."""
    from baselines.mint.mint.visualization import geometry, mano

    mano._DEVICE = torch.device("cpu")
    mano._MANO_RIGHT_DIR = mano._MANO_LEFT_DIR = Path(model_dir).resolve()
    raw = {
        key: np.asarray(value)
        for key, value in prediction.items()
        if not key.startswith("_")
    }
    n = len(raw["hand"])
    hand = raw["hand"].reshape(n, 2, 109)
    claims = raw["hand_presence_logits"].reshape(n, 2) >= 0
    if not np.isfinite(hand[claims]).all():
        raise ValueError("Invalid MINT hand/camera timeline")
    _, k = geometry.decode_camera_pose_enc(raw["pose_enc"], height, width, fov_mean=True)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid predicted MINT intrinsics")
    v = np.full((n, 2, 778, 3), np.nan, np.float32)
    j = np.full((n, 2, 21, 3), np.nan, np.float32)
    go = np.full((n, 2, 1, 3, 3), np.nan, np.float32)
    hp = np.full((n, 2, 15, 3, 3), np.nan, np.float32)
    trans = np.full((n, 2, 3), np.nan, np.float32)
    for side in (0, 1):
        for start in range(0, n, 128):
            ix = np.flatnonzero(claims[start : start + 128, side]) + start
            if not len(ix):
                continue
            h = hand[ix, side]
            params = mano.decode_hand_6d(
                h[:, :3], h[:, 3:9], h[:, 9:99], h[:, 99:], bool(side)
            )
            verts, joints = mano.run_mano(**params, is_right=bool(side))
            v[ix, side] = verts
            j[ix, side] = joints
            trans[ix, side] = params["trans"]
            go[ix, side, 0] = Rotation.from_rotvec(params["rot"]).as_matrix()
            hp[ix, side] = (
                Rotation.from_rotvec(params["hand_pose"].reshape(-1, 3))
                .as_matrix()
                .reshape(-1, 15, 3, 3)
            )
    raw.update(
        vertices_cam=v,
        joints_cam=j,
        global_orient=go,
        hand_pose=hp,
        betas=np.where(claims[..., None], hand[..., 99:], np.nan),
        cam_trans=trans,
        presence=claims,
        frame_ids=np.arange(n),
        predicted_projection_K=k,
    )
    return raw


class MINTAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        self.assets = checkpoint_dir("mint", checkpoints, version=version)
        self.repo = Path(__file__).resolve().parent
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
        # official MINT (wuji-ego-mint@1bd21e4, `python -m mint viewer`) sets no TF32 flag: keep PyTorch defaults
        torch.backends.cuda.matmul.allow_tf32 = False
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
        assets, device = (self.assets, self.device)
        from baselines.mint.mint.inference.engine import StudentEngine

        checkpoint = weight_path(assets, "mint")
        self.engine = StudentEngine(
            str(self.repo / "configs/training/mint_step2.yaml"),
            ckpt=str(checkpoint),
            device=str(device),
            compile_mode="off",
            fp8_mode="off",
        )
        self.window_batch_size = 2 if batch_size is None else batch_size
        self.reset()

    def predict(self, record, *, cam_mode="chunked", hand_mode="hard", window_batch_size=None, on_step=None, cancel_check=None):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        n = record["num_frames"]
        frames = np.empty((n, record["height"], record["width"], 3), np.uint8)
        with VideoReader(record.get("_frames_rgb", record.get("video"))) as reader:
            for t in range(n):
                ok, frame = reader.read()
                if not ok or frame.shape != frames.shape[1:]:
                    raise ValueError("MINT RGB timeline/canvas mismatch")
                frames[t] = frame
            if reader.read()[0]:
                raise ValueError("MINT RGB contains extra source frames")
        prediction = self.engine.predict(
            frames,
            cam_mode=cam_mode,
            hand_mode=hand_mode,
            window_batch_size=self.window_batch_size if window_batch_size is None else window_batch_size,
            on_step=on_step,
            cancel_check=cancel_check,
        )
        if len(prediction["hand"]) != len(frames):
            raise ValueError("MINT inference did not cover all source frames")
        return decode_native(
            prediction, resource_path(self.assets, "mano"), record["height"], record["width"]
        )

    def inference_stream(self, inputs, *, joint_only=False):
        raise NotImplementedError("mint requires a complete clip")

    def inference_offline(self, inputs, *, joint_only=False, cam_mode="chunked", hand_mode="hard", window_batch_size=None, on_step=None, cancel_check=None):
        if str(self.device).startswith("cuda"):
            torch.cuda.set_device(self.device)
        with InferenceInput(inputs, mode="offline") as (record, camera):
            return self._canonical(self.predict(
                record, cam_mode=cam_mode, hand_mode=hand_mode,
                window_batch_size=window_batch_size, on_step=on_step, cancel_check=cancel_check,
            ), camera, joint_only=joint_only)

    def _canonical(self, raw, camera, *, joint_only=False):
        if not joint_only and self._mano_model is None:
            self._mano_model = ManoModel(
                resource_path(self.assets, "mano"), device="cpu"
            )
        return native_prediction(raw, camera, self._mano_model, joint_only=joint_only)

    def reset(self):
        if self.seed is not None:
            import random

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.random.default_generator.manual_seed(self.seed)
            if str(self.device).startswith("cuda"):
                torch.cuda.set_device(self.device)
                torch.cuda.manual_seed(self.seed)

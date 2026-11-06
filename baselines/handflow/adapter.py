from pathlib import Path

"""Released HandFlow temporal inference with official real-K input and projection."""

from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, weight_path
import os
import pickle
import numpy as np
from utils.mano import enable_mano_pickle_compatibility, ManoModel, mano_joint_prediction
from .mano import native_prediction
import torch
import cv2
from scipy.spatial.transform import Rotation

# manopth RIGHT-layer fingertips (both tracks run through the right layer).
NATIVE_TIP_IDS = (745, 317, 444, 556, 673)


class HandFlowAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        self.assets = checkpoint_dir("handflow", checkpoints, version=version)
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
        enable_mano_pickle_compatibility()
        self._mano_model = None

        torch.set_num_threads(2)
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
        routes = {
            "HAMER_CKPT": weight_path(self.assets, "hamer"),
            "HAMER_DATA": resource_path(self.assets, "hamer.data"),
            "DETECTOR_CKPT": resource_path(self.assets, "wilor.detector"),
            "MANO_ROOT": resource_path(self.assets, "mano"),
            "HANDFLOW_NORMALIZATION_STATS": resource_path(
                self.assets, "handflow.normalization"
            ),
        }
        os.environ.update({key: str(value) for key, value in routes.items()})
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        device = self.device
        self.device = torch.device(device)
        source = self.repo
        weights = weight_path(self.assets, "handflow")
        from omegaconf import OmegaConf
        from baselines.handflow.utils.checkpoint_utils import load_denoiser_from_ckpt
        from baselines.handflow.utils.inference_utils import (
            run_fm_inference_with_overlap,
            split_sequence_into_windows,
        )
        from baselines.handflow.utils.mano_utils import MANOForwardKinematics
        from baselines.handflow.utils.online_hamer import OnlineHaMeRPipeline

        self.run_overlap = run_fm_inference_with_overlap
        self.split_windows = split_sequence_into_windows
        from baselines.handflow.third_party.hamer.hamer import configs as hc

        get_config = hc.get_config
        mano_path = str(routes["MANO_ROOT"])
        mano_mean_params = str(
            routes["HAMER_DATA"] / "data" / "mano_mean_params.npz"
        )

        def routed_config(*args, **kwargs):
            cfg = get_config(*args, **kwargs)
            cfg.defrost()
            cfg.MANO.MODEL_PATH = mano_path
            cfg.MANO.MEAN_PARAMS = mano_mean_params
            cfg.MODEL.pop("PRETRAINED_WEIGHTS_INIT_REGRESSION", None)
            cfg.freeze()
            return cfg

        hc.get_config = routed_config
        self.cfg = OmegaConf.load(source / "configs/inference.yaml")
        config = OmegaConf.merge(OmegaConf.load(source / self.cfg.model_yaml), self.cfg)
        # Eval mode, as upstream's evaluator (utils/inference_utils.py:407, evaluate_fm_on_loader) sets it; upstream
        # scripts/demo.py never calls .eval() and would leave Dropout 0.1 and the 20% cmask frame masking active.
        self.model = load_denoiser_from_ckpt(config, str(weights), self.device)
        self.model.eval()
        self.online = OnlineHaMeRPipeline(
            device=device, hamer_batch_size=32 if batch_size is None else batch_size  # upstream default
        )
        self.mano = MANOForwardKinematics(os.environ["MANO_ROOT"], self.device)
        checkpoint = torch.load(
            weights, map_location="cpu", weights_only=False, mmap=True
        )
        state = {
            k.removeprefix("module."): v
            for k, v in checkpoint.get("module", checkpoint).items()
        }
        required = {k for k, p in self.model.named_parameters() if p.requires_grad}
        if required - set(state):
            raise ValueError("Missing learned HandFlow checkpoint parameters")
        del checkpoint, state
        with (resource_path(self.assets, "mano") / "MANO_RIGHT.pkl").open(
            "rb"
        ) as stream:
            mano = pickle.load(stream, encoding="latin1")
        self.regressor = mano["J_regressor"].toarray()
        self.reset()

    @torch.inference_mode()
    def predict(self, record, *, ode_steps=3):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        n = record["num_frames"]
        ks = record["intrinsics_per_frame"][:n]
        if np.any(ks[:, 0, 1] != 0) or np.any(ks[:, 1, 0] != 0):
            raise ValueError("HandFlow crop calibration does not support camera skew")
        intrinsics = np.stack(
            [ks[:, 0, 0], ks[:, 1, 1], ks[:, 0, 2], ks[:, 1, 2]], axis=-1
        ).astype(np.float32)
        out = {
            k: np.full((n, 2, *tail), np.nan, np.float32)
            for k, tail in [
                ("vertices_cam", (778, 3)),
                ("joints_cam", (21, 3)),
                ("global_orient", (1, 3, 3)),
                ("hand_pose", (15, 3, 3)),
                ("betas", (10,)),
                ("cam_trans", (3,)),
                ("bbox_xyxy", (4,)),
                ("native_joints2d", (21, 2)),
            ]
        }
        out.update(
            frame_ids=np.arange(n),
            presence=np.zeros((n, 2), bool),
            confidence=np.zeros((n, 2), np.float32),
            native_joints2d_order=np.array("mano16_tips"),
            native_joints2d_source=np.array("handflow_supplied_K_projection"),
            projection_intrinsics=np.repeat(intrinsics[:, None, :], 2, axis=1),
        )
        # Official HandFlow is right-hand only: its HaMeR stage (right-hand ViT) and its
        # denoiser were trained on right hands, and the released README requires that
        # "left-hand videos must be mirrored to right-hand first". The left track is
        # therefore run on the mirrored video as a right hand and mirrored back.
        for side in (0, 1):
            cap = cv2.VideoCapture(record["video"])
            parts = []
            tokens = []
            try:
                for start in range(0, n, 64):
                    frames = []
                    for _ in range(min(64, n - start)):
                        ok, frame = cap.read()
                        if not ok or frame.shape[:2] != (
                            record["height"],
                            record["width"],
                        ):
                            raise ValueError("RGB timeline mismatch")
                        frames.append(cv2.flip(frame, 1) if side == 0 else frame)
                    # Full-image mirroring turns the left track into a right track.
                    crop_inputs = intrinsics[start : start + len(frames)].copy()
                    if side == 0:
                        crop_inputs[:, 2] = record["width"] - 1 - crop_inputs[:, 2]
                    part = self.online.process_sequence(
                        frames, list(crop_inputs), target_side="right"
                    )
                    # Compressed below in one call over the whole sequence, as upstream's
                    # scripts/demo.py does.
                    tokens.append(part.pop("backbone_features"))
                    parts.append(
                        {
                            k: part[k]
                            for k in (
                                "hamer_landmarks",
                                "crop_intrinsics",
                                "hamer_confidence",
                                "detection_valid",
                                "bbox_xyxy",
                                "bbox_conf",
                            )
                        }
                    )
                    print(
                        "HandFlow RGB",
                        record["clip"],
                        side,
                        start + len(frames),
                        "/",
                        n,
                        flush=True,
                    )
            finally:
                cap.release()
            p = {k: torch.cat([part[k] for part in parts]) for k in parts[0]}
            if self.seed is not None:
                torch.manual_seed(self.seed + side)
            features = torch.cat(tokens).to(self.device)
            del tokens
            image_tokens = self.model.frame_compressor(features.unsqueeze(0)).squeeze(0)
            del features
            seq = dict(
                mano_params=torch.zeros(1, n, 48),
                mano_trans=torch.zeros(1, n, 3),
                mano_betas=torch.zeros(1, 10),
                padding_mask=torch.zeros(1, n, dtype=torch.bool),
                image_tokens=image_tokens.unsqueeze(0),
                hamer_landmarks=p["hamer_landmarks"].unsqueeze(0),
                crop_intrinsics=p["crop_intrinsics"].unsqueeze(0),
                hamer_confidence=p["hamer_confidence"].unsqueeze(0),
                side=["right"],
                source=["custom"],
            )
            windows, _ = self.split_windows(seq, 16, 14, self.device)
            windows = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in windows.items()
            }
            pose, trans, beta = self.run_overlap(
                self.model,
                self.model,
                windows,
                16,
                2,
                ode_steps,
                self.device,
                overlap_method="vblend",
            )
            if len(pose) < n:
                raise ValueError("HandFlow truncated the input timeline")
            pose, trans, beta = pose[:n], trans[:n], beta[:n]
            for name, value in dict(
                source_mano_params=pose.cpu().numpy(),
                source_mano_trans=trans.cpu().numpy(),
                **{"source_" + k: v.numpy() for k, v in p.items()}
            ).items():
                if name not in out:
                    out[name] = np.full((n, 2, *value.shape[1:]), np.nan, np.float32)
                out[name][:, side] = value
            verts = (
                self.mano.verts(pose, beta, trans, ["right"] * n).cpu().numpy() / 1000.0
            )
            # Native manopth OpenPose joints of the same right-hand forward pass.
            native_joints = (
                self.mano.joints(pose, beta, trans, ["right"] * n).cpu().numpy() / 1000.0
            )
            mats = (
                Rotation.from_rotvec(pose.cpu().numpy().reshape(-1, 3))
                .as_matrix()
                .astype(np.float32)
                .reshape(n, 16, 3, 3)
            )
            original_trans = trans.cpu().numpy().copy()
            boxes = p["bbox_xyxy"].numpy().copy()
            if side == 0:
                # Back to the unmirrored camera; the rotations stay RIGHT/mirrored-left,
                # the convention native_to_mano expects after its own reflection.
                verts[..., 0] *= -1
                native_joints[..., 0] *= -1
                original_trans[:, 0] *= -1
                boxes[:, [0, 2]] = record["width"] - boxes[:, [2, 0]]  # box edges: x -> W - x
            joints = np.concatenate(
                [
                    np.einsum("jv,tvc->tjc", self.regressor, verts),
                    verts[:, NATIVE_TIP_IDS],
                ],
                axis=1,
            )
            out["native_joints2d"][:, side] = (
                joints[..., :2] / joints[..., 2:] * intrinsics[:, None, :2]
                + intrinsics[:, None, 2:]
            )
            out["vertices_cam"][:, side] = verts
            out["joints_cam"][:, side] = native_joints
            out["cam_trans"][:, side] = original_trans
            out["global_orient"][:, side] = mats[:, :1]
            out["hand_pose"][:, side] = mats[:, 1:]
            out["betas"][:, side] = beta.cpu().numpy()
            out["bbox_xyxy"][:, side] = boxes
            out["presence"][:, side] = p["detection_valid"].numpy()
            out["confidence"][:, side] = p["bbox_conf"].numpy()
        return out

    def inference_stream(self, inputs, *, joint_only=False):
        raise NotImplementedError("handflow requires a complete clip")

    def inference_offline(self, inputs, *, joint_only=False, ode_steps=3):
        if str(self.device).startswith("cuda"):
            torch.cuda.set_device(self.device)
        with InferenceInput(
            inputs, mode="offline", requires_video_path=True
        ) as (record, camera):
            return self._canonical(self.predict(record, ode_steps=ode_steps), camera, joint_only)

    def _canonical(self, raw, camera, joint_only=False):
        # Scored joints are decoded from the MANO export, so joint_only still fits it.
        if self._mano_model is None:
            self._mano_model = ManoModel(
                resource_path(self.assets, "mano"), device="cpu"
            )
        mano = native_prediction(raw, camera, self._mano_model)
        return {"mano": None if joint_only else mano, "joint": mano_joint_prediction(mano, self._mano_model)}

    def reset(self):
        if self.seed is not None:
            import random

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.random.default_generator.manual_seed(self.seed)
            if str(self.device).startswith("cuda"):
                torch.cuda.set_device(self.device)
                torch.cuda.manual_seed(self.seed)

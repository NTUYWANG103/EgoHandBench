"""InfiniHand streaming camera-space hands (upstream infinihand.infer) under the supplied per-frame K."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, weight_path
from utils.mano import ManoModel, mano_joint_prediction
from .mano import native_prediction, SLOTS

ANCHOR_FRAMES, MAX_FRAMES = 4, 8192  # infinihand.infer: four anchor frames, at most 8192 frames per invocation


class InfiniHandAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        if batch_size is not None:
            raise ValueError("infinihand streams one clip (4 anchor frames, then one frame per step); batch_size is unsupported")
        self.assets = checkpoint_dir("infinihand", checkpoints, version=version)
        self.repo = Path(__file__).resolve().parent
        selected = torch.device("cuda:0" if device in (None, "auto") else device)
        if selected.type != "cuda":
            raise ValueError("infinihand runs on CUDA only (two CUDA streams)")
        if selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
        self.device = selected
        torch.cuda.set_device(self.device)
        self.seed = seed
        self._mano_model = None
        self.reset()
        from baselines.infinihand.infinihand.models.hand_model import LingBotMapHand, LingBotMapHandConfig
        from baselines.infinihand.infinihand.weights import load_inference_weights

        # infinihand.infer.build_model, with the bundled code paths and this group's MANO assets
        config = json.loads((self.repo / "configs/infinihand_v1.json").read_text())
        code = self.repo / "infinihand"
        config.update(
            lingbotmap_root=str(code / "lingbot_map"), wilor_root=str(code / "wilor"),
            wilor_cfg=str(code / "wilor/pretrained_models/model_config.yaml"),
            mano_model_path=str(Path(resource_path(self.assets, "mano")).resolve()),
            lingbotmap_ckpt="", wilor_ckpt="", load_wilor_checkpoint=False, step_memory_enabled=True,
            step_memory_scale_frames=4, step_memory_sliding_window=16, step_memory_max_frames=8192,
        )
        self.model = LingBotMapHand(LingBotMapHandConfig(**config))
        state = torch.load(weight_path(self.assets, "infinihand"), map_location="cpu", mmap=True, weights_only=True)
        if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError("Expected a complete exported tensor state_dict")
        load_inference_weights(self.model, state)
        self.model.to(self.device).eval()
        self.mask_threshold = 0.9  # infinihand.infer --mask-threshold default and the evaluation config

    def predict(self, record):
        from baselines.infinihand.infinihand.infer import run_dual_stream, validate_contract

        # infinihand.infer sets no TF32 flag: pin PyTorch's defaults (an environment override such as
        # TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which turns matmul TF32 on, would otherwise change the numerics)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        frames = record["_frames_rgb"]
        n, h, w = len(frames), record["height"], record["width"]
        height, width = self.model.cfg.image_height, self.model.cfg.image_width
        k = np.asarray(record["intrinsics_per_frame"], np.float32).copy()
        k[:, 0] *= width / w
        k[:, 1] *= height / h
        # one upstream invocation takes ANCHOR_FRAMES to MAX_FRAMES frames: a shorter clip repeats its last frame up to
        # ANCHOR_FRAMES (outputs of the repeats dropped), a longer one runs as equal consecutive invocations
        parts = []
        for rows in np.array_split(np.arange(n), -(-n // MAX_FRAMES)):
            rows = np.concatenate([rows, np.full(max(0, ANCHOR_FRAMES - len(rows)), rows[-1])])
            images = np.stack([cv2.resize(frames[i], (width, height)) for i in rows])
            images = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0).to(self.device)
            intrinsics = torch.from_numpy(k[rows]).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                out, stats, _ = run_dual_stream(self.model, images, intrinsics, bbox_source="mask", mask_threshold=self.mask_threshold)
            validate_contract(stats, len(rows))
            parts.append({key: value.float().cpu().numpy() for key, value in out.items()})
            del out, images, intrinsics
            torch.cuda.empty_cache()
        arrays = {key: np.concatenate([part[key] for part in parts])[:n] for key in parts[0]}
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError("Non-finite model output: " + key)
        return dict(
            joints_cam=arrays["camera_joints"][:, SLOTS],
            vertices_cam=arrays["camera_vertices"][:, SLOTS],
            global_orient=arrays["camera_global_orient"][:, SLOTS, None],
            hand_pose=arrays["hand_pose"][:, SLOTS],
            betas=arrays["betas"][:, SLOTS],
            cam_trans=arrays["camera_translation"][:, SLOTS],
            presence=arrays["presence_mask"][:, SLOTS] > 0.5,
        )

    def inference_stream(self, inputs, *, joint_only=False):
        raise NotImplementedError("infinihand needs at least four frames: use inference_offline")

    def inference_offline(self, inputs, *, joint_only=False):
        torch.cuda.set_device(self.device)
        with InferenceInput(inputs, mode="offline") as (record, camera):
            raw = self.predict(record)
        # Left hands come from the LEFT MANO layer, so the joints are decoded from the RIGHT-mirror MANO export.
        if self._mano_model is None:
            self._mano_model = ManoModel(resource_path(self.assets, "mano"))
        mano = native_prediction(raw, camera, self._mano_model)
        return {"mano": None if joint_only else mano, "joint": mano_joint_prediction(mano, self._mano_model)}

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

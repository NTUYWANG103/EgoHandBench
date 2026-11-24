"""ACE-Ego-Hand tiled video inference and learned presence."""

from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, weight_path
from pathlib import Path
import numpy as np
from .mano import native_prediction, native_joints
import torch


class ACEEgoHandAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="egohandbench"):
        if batch_size is not None:
            raise ValueError("ACE-Ego-Hand uses native temporal windows, not a batch_size argument")
        self.assets = checkpoint_dir("ace_ego_hand", checkpoints, version=version)
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
        option = "options/ace_ego_hand_k.yml"
        tile_width = 22
        device = self.device
        import os

        os.environ["ACE_EGO_HAND_MANO_DIR"] = str(
            resource_path(self.assets, "ace_ego_hand.mano")
        )
        os.environ["ACE_EGO_HAND_VAE_ROOT"] = str(
            resource_path(self.assets, "ace_ego_hand.vae").parent
        )
        import yaml

        from baselines.ace_ego_hand.ace_ego_hand.models.geodit_model import GeoDitModel
        from baselines.ace_ego_hand.ace_ego_hand.video_vae import load_vae

        self.option = yaml.safe_load((self.repo / option).read_text())
        self.option["paths"]["model_root"] = str(
            resource_path(self.assets, "ace_ego_hand.backbone").parent
        )
        self.option["paths"]["caption_embed"] = str(
            resource_path(self.assets, "ace_ego_hand.caption")
        )
        self.option["paths"]["videox_config"] = str(
            self.repo / self.option["paths"]["videox_config"]
        )
        self.device = torch.device(device)
        self.model = GeoDitModel(self.option, self.device)
        checkpoint = weight_path(self.assets, "ace_ego_hand")
        self.model.load_inference(str(checkpoint))
        self.vae = load_vae(self.device)
        self.tile_width = tile_width
        self.reset()

    def predict(self, record, *, encode_width=832, inference_mode="tiled"):
        if inference_mode not in ("tiled", "full"):
            raise ValueError("ACE-Ego-Hand inference_mode must be tiled or full")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        from baselines.ace_ego_hand.ace_ego_hand.inference import (
            DUMP_KEYS,
            predict_video,
        )
        from baselines.ace_ego_hand.ace_ego_hand.video_vae import decode_video, encode
        from baselines.ace_ego_hand.infer_video import _encode_size

        n = record["num_frames"]
        count = max(5, 4 * ((n + 2) // 4) + 1)
        w, h = _encode_size(
            record["width"], record["height"], encode_width or None
        )
        matrices = np.asarray(record["intrinsics_per_frame"])
        if matrices.shape != (n, 3, 3):
            raise ValueError("ACE-Ego-Hand camera and RGB timelines differ")
        sx = w / record["width"]
        sy = h / record["height"]
        intr = dict(
            per_frame=[dict(fx=k[0, 0] * sx, fy=k[1, 1] * sy,
                            cx=k[0, 2] * sx, cy=k[1, 2] * sy)
                       for k in matrices],
            image_width=w,
            image_height=h,
        )
        intr["per_frame"].extend(dict(intr["per_frame"][-1]) for _ in range(count - n))
        pixels = decode_video(record["video"], n, resize_hw=(w, h))
        if count > n:
            pixels = torch.cat([pixels, pixels[-1:].repeat(count - n, 1, 1, 1)])
        control = encode(self.vae, pixels, self.device).float()
        prediction = predict_video(
            self.model,
            int(self.option.get("backbone", {}).get("tap_layers", [15])[0]),
            control,
            intr,
            inference_mode,
            self.device,
            tile_w=self.tile_width,
        )
        raw = {key: np.asarray(prediction[key][:n], np.float32) for key in DUMP_KEYS}
        claims = raw["exists_2d"] > 0.5
        raw.update(presence=claims)
        return raw

    def inference_stream(self, inputs, *, joint_only=False, encode_width=832, inference_mode="tiled"):
        raise NotImplementedError("ace_ego_hand requires a complete clip")

    def inference_offline(self, inputs, *, joint_only=False, encode_width=832, inference_mode="tiled"):
        if str(self.device).startswith("cuda"):
            torch.cuda.set_device(self.device)
        with InferenceInput(inputs, mode="offline", requires_video_path=True) as (
            record,
            camera,
        ):
            if np.any(camera.K[:, (0, 1), (1, 0)] != 0):
                raise ValueError("ACE-Ego-Hand's native translation decoder does not support skew")
            return self._output(
                self.predict(record, encode_width=encode_width, inference_mode=inference_mode),
                camera, joint_only=joint_only,
            )

    def _output(self, raw, camera, *, joint_only=False):
        joints = native_joints(raw, camera.source_frame_ids)
        if joint_only:
            return dict(mano=None, joint=joints)
        return dict(mano=native_prediction(raw, camera.source_frame_ids), joint=joints)

    def reset(self):
        if self.seed is not None:
            import random

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.random.default_generator.manual_seed(self.seed)
            if str(self.device).startswith("cuda"):
                torch.cuda.set_device(self.device)
                torch.cuda.manual_seed(self.seed)

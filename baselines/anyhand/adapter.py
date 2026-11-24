"""AnyHand inference using its released WiLoR model and camera configuration."""

import random
import numpy as np
import torch
from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, VideoReader, weight_path
from utils.mano import enable_mano_pickle_compatibility, ManoModel, refit_prediction


class AnyHandAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, version="official"):
        self.method = "anyhand"
        selected_device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if selected_device.type == "cuda" and selected_device.index is None:
            selected_device = torch.device("cuda", torch.cuda.current_device())
        self.device = str(selected_device)
        if not self.device.startswith("cuda"):
            raise ValueError("anyhand requires CUDA")
        torch.cuda.set_device(self.device)
        torch.set_num_threads(2)
        # official scripts/rgb_predictor.py (AnyHand@33a4c4c) sets no TF32 flag (WiLoR demo.py:30 'high' is --fast only): keep PyTorch defaults
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        self.seed = seed
        self.assets = checkpoint_dir(self.method, checkpoints, version=version)
        self._mano_model = None
        enable_mano_pickle_compatibility()
        self.reset()
        from .scripts.rgb_predictor import AnyHandPredictor

        checkpoint = weight_path(self.assets, "model")
        self.predictor = AnyHandPredictor(
            wilor_mano_dir=str(resource_path(self.assets, "anyhand.mano")),
            wilor_ckpt=str(checkpoint),
            wilor_cfg=str(resource_path(self.assets, "wilor.config")),
            detector_pt=str(resource_path(self.assets, "anyhand.detector")),
            device=self.device,
            det_conf=0.3,
            det_iou=0.3,
            rescale_factor=2.0,
            batch_size=16 if batch_size is None else batch_size,
        )
        self.model = self.predictor._wilor_model
        self.cfg = self.predictor._wilor_model_cfg
        self.parameters = []

        def capture(model, inputs, output):
            self.parameters.extend(
                (
                    {
                        key: output["pred_mano_params"][key][index]
                        .detach()
                        .cpu()
                        .numpy()
                        .copy()
                        for key in ("global_orient", "hand_pose", "betas")
                    }
                    | {
                        "joints": output["pred_keypoints_3d"][index]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    }
                    for index in range(len(output["pred_vertices"]))
                )
            )

        self.hook = self.model.register_forward_hook(capture)
        self.reset()

    def inference_stream(self, inputs, *, joint_only=False, batch_size=None):
        """Predict canonical MANO from one RGB frame and its Camera."""
        return self._inference(inputs, "stream", joint_only=joint_only, batch_size=batch_size)

    def inference_offline(self, inputs, *, joint_only=False, batch_size=None):
        return self._inference(inputs, "offline", joint_only=joint_only, batch_size=batch_size)

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def _inference(self, inputs, mode, *, joint_only=False, batch_size=None):
        torch.cuda.set_device(self.device)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        with InferenceInput(inputs, mode=mode) as (record, camera):
            raw = self.predict(record, batch_size=batch_size)
            if not joint_only and self._mano_model is None:
                self._mano_model = ManoModel(resource_path(self.assets, "mano"))
            return refit_prediction(raw, camera, self._mano_model, joint_only=joint_only)

    @torch.inference_mode()
    def predict(self, record, *, batch_size=None):
        count = record["num_frames"]
        raw = {
            key: np.full((count, 2, *shape), np.nan, np.float32)
            for key, shape in (
                ("vertices_cam", (778, 3)),
                ("joints_cam", (21, 3)),
                ("global_orient", (1, 3, 3)),
                ("hand_pose", (15, 3, 3)),
                ("betas", (10,)),
                ("cam_trans", (3,)),
            )
        }
        raw["presence"] = np.zeros((count, 2), bool)
        reader = VideoReader(record.get("_frames_rgb", record["video"]), color="bgr")
        try:
            for frame_index in range(count):
                _, frame = reader.read()
                self.parameters.clear()
                hands = self.predictor.predict(frame, batch_size=batch_size)
                for side in (0, 1):
                    candidates = [
                        index
                        for index, hand in enumerate(hands)
                        if int(hand.is_right) == side
                    ]
                    if not candidates:
                        continue
                    index = max(candidates, key=lambda index: hands[index].score)
                    hand = hands[index]
                    for key in ("global_orient", "hand_pose", "betas"):
                        raw[key][frame_index, side] = self.parameters[index][key]
                    raw["vertices_cam"][frame_index, side] = (
                        hand.vertices + hand.cam_t[None]
                    )
                    raw["cam_trans"][frame_index, side] = hand.cam_t
                    joints = self.parameters[index]["joints"].copy()
                    joints[:, 0] *= 2 * side - 1
                    raw["joints_cam"][frame_index, side] = joints + hand.cam_t[None]
                    raw["presence"][frame_index, side] = True
        finally:
            reader.release()
        focal = (
            float(self.cfg.EXTRA.FOCAL_LENGTH) / int(self.cfg.MODEL.IMAGE_SIZE)
            * max(record["width"], record["height"])
        )
        raw.update(
            frame_ids=np.arange(count),
            projection_intrinsics=np.array(
                [focal, focal, record["width"] / 2, record["height"] / 2], np.float64
            ),
        )
        return raw

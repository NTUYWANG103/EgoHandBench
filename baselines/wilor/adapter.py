"""Official WiLoR inference without publishing or renderer initialization."""

import random
import numpy as np
import torch
import cv2
from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, VideoReader, weight_path
from utils.mano import enable_mano_pickle_compatibility, ManoModel, refit_prediction


class Predictor:

    def __init__(self, batch_size, assets):
        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("The official WiLoR route requires CUDA")
        from baselines.wilor.wilor.models import load_wilor
        from baselines.wilor.wilor.datasets.vitdet_dataset import ViTDetDataset
        from baselines.wilor.wilor.utils import recursive_to
        from baselines.wilor.wilor.utils.renderer import cam_crop_to_full
        from ultralytics import YOLO

        self.batch_size = batch_size
        self.dataset_class = ViTDetDataset
        self.recursive_to = recursive_to
        self.cam_crop_to_full = cam_crop_to_full
        self.model, self.cfg = load_wilor(
            str(weight_path(assets, "wilor")),
            str(resource_path(assets, "wilor.config")),
            mano_dir=str(resource_path(assets, "wilor.mano")),
        )
        self.model = self.model.float().cuda().eval()
        self.detector = YOLO(str(resource_path(assets, "wilor.detector"))).to("cuda")

    @torch.inference_mode()
    def predict(self, frames, *, batch_size=None):
        samples, locations = ([], [])
        for fi, image in enumerate(frames):
            boxes = (
                self.detector(image, conf=0.3, verbose=False)[0]
                .boxes.data.cpu()
                .numpy()
            )
            chosen = []
            for side in (0, 1):
                candidates = boxes[boxes[:, 5].astype(int) == side]
                if len(candidates):
                    chosen.append(candidates[np.argmax(candidates[:, 4])])
            if not chosen:
                continue
            chosen = np.stack(chosen)
            dataset = self.dataset_class(
                self.cfg,
                image,
                chosen[:, :4],
                chosen[:, 5],
                rescale_factor=2.0,
                fp16=False,
            )
            for i in range(len(dataset)):
                samples.append(dataset[i])
                locations.append((fi, int(chosen[i, 5])))
        outputs = [[] for _ in frames]
        batch_size = self.batch_size if batch_size is None else batch_size
        for start in range(0, len(samples), batch_size):
            batch = torch.utils.data.default_collate(
                samples[start : start + batch_size]
            )
            batch = self.recursive_to(batch, "cuda")
            pred = self.model(batch)
            cam = pred["pred_cam"].clone()
            cam[:, 1] *= 2 * batch["right"] - 1
            focal = (
                self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE
                * batch["img_size"].max(dim=1).values
            )
            trans = self.cam_crop_to_full(
                cam,
                batch["box_center"].float(),
                batch["box_size"].float(),
                batch["img_size"].float(),
                focal,
            )
            params = {k: v.cpu().numpy() for k, v in pred["pred_mano_params"].items()}
            verts = pred["pred_vertices"].cpu().numpy()
            joints = pred["pred_keypoints_3d"].cpu().numpy()
            trans = trans.cpu().numpy()
            for i, (fi, side) in enumerate(locations[start : start + batch_size]):
                mirror = 2 * side - 1
                verts[i, :, 0] *= mirror
                joints[i, :, 0] *= mirror
                result = dict(
                    side=side,
                    global_orient=params["global_orient"][i].reshape(1, 3, 3),
                    hand_pose=params["hand_pose"][i].reshape(15, 3, 3),
                    betas=params["betas"][i].reshape(10),
                    cam_trans=trans[i],
                    verts=verts[i] + trans[i],
                    joints=joints[i] + trans[i],
                )
                if not all(
                    (
                        np.isfinite(result[k]).all()
                        for k in (
                            "global_orient",
                            "hand_pose",
                            "betas",
                            "cam_trans",
                            "verts",
                            "joints",
                        )
                    )
                ):
                    raise ValueError("Non-finite model prediction")
                outputs[fi].append(result)
        return outputs


class WiLoRAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        self.method = "wilor"
        selected_device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if selected_device.type == "cuda" and selected_device.index is None:
            selected_device = torch.device("cuda", torch.cuda.current_device())
        self.device = str(selected_device)
        if not self.device.startswith("cuda"):
            raise ValueError("wilor requires CUDA")
        torch.cuda.set_device(self.device)
        torch.set_num_threads(2)
        # official demo.py:30 (rolpotamias/WiLoR@fcb9113) sets 'high' only under --fast (FP16); this full-precision path keeps PyTorch defaults
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        self.seed = seed
        self.assets = checkpoint_dir(self.method, checkpoints, version=version)
        self._mano_model = None
        enable_mano_pickle_compatibility()
        self.reset()
        self.model = Predictor(
            batch_size=16 if batch_size is None else batch_size, assets=self.assets
        )
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

    def predict(self, record, *, batch_size=None):
        n = record["num_frames"]
        raw = {
            k: np.full((n, 2, *tail), np.nan, np.float32)
            for k, tail in [
                ("vertices_cam", (778, 3)),
                ("joints_cam", (21, 3)),
                ("global_orient", (1, 3, 3)),
                ("hand_pose", (15, 3, 3)),
                ("betas", (10,)),
                ("cam_trans", (3,)),
            ]
        }
        raw["presence"] = np.zeros((n, 2), bool)
        cap = VideoReader(record.get("_frames_rgb", record["video"]), color="bgr")
        try:
            for begin in range(0, n, 8):
                frames = []
                for _ in range(min(8, n - begin)):
                    ok, frame = cap.read()
                    if not ok or frame.shape[:2] != (record["height"], record["width"]):
                        raise ValueError("Input timeline/size mismatch")
                    frames.append(frame)
                predictions = self.model.predict(frames, batch_size=batch_size)
                for j, hands in enumerate(predictions):
                    for hand in hands:
                        t, side = (begin + j, hand["side"])
                        for k in ("global_orient", "hand_pose", "betas", "cam_trans"):
                            raw[k][t, side] = hand[k]
                        raw["vertices_cam"][t, side] = hand["verts"]
                        raw["joints_cam"][t, side] = hand["joints"]
                        raw["presence"][t, side] = True
            if cap.read()[0]:
                raise ValueError("Unexpected extra RGB frames")
        finally:
            cap.release()
        focal = (
            float(self.model.cfg.EXTRA.FOCAL_LENGTH) / int(self.model.cfg.MODEL.IMAGE_SIZE)
            * max(record["width"], record["height"])
        )
        raw.update(
            frame_ids=np.arange(n),
            projection_intrinsics=np.array(
                [focal, focal, record["width"] / 2, record["height"] / 2], np.float64
            ),
        )
        return raw

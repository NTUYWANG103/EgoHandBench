"""Released HandDGP, detector crops and native DGP translation, without fitting."""

import random
import numpy as np
import torch
from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, VideoReader, weight_path
from utils.mano import ManoModel, mano_joint_prediction
from .mano import native_prediction


class HandDGPAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        self.batch_size = 1 if batch_size is None else batch_size
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("HandDGP batch_size must be a positive integer")
        self.method = "handdgp"
        selected_device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if selected_device.type == "cuda" and selected_device.index is None:
            selected_device = torch.device("cuda", torch.cuda.current_device())
        self.device = str(selected_device)
        if not self.device.startswith("cuda"):
            raise ValueError("handdgp requires CUDA")
        torch.cuda.set_device(self.device)
        torch.set_num_threads(2)
        # official src/run.py (nianticlabs/HandDGP@e173034) sets no TF32 flag: keep PyTorch defaults
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        self.seed = seed
        self.assets = checkpoint_dir(self.method, checkpoints, version=version)
        self._mano_model = None
        self.reset()
        import os

        os.environ["HANDDGP_TEMPLATE_DIR"] = str(
            resource_path(self.assets, "handdgp.template")
        )
        from baselines.handdgp.src.models.handdgp import HandDGP
        from baselines.handdgp.third_party.HandMesh.utils.preprocessing import (
            generate_patch_image,
        )
        from baselines.handdgp.src.datasets.image_processor import ImageProcessor
        from ultralytics import YOLO

        self.patch_image = generate_patch_image
        self.preprocess = ImageProcessor.preprocess
        self.model = HandDGP(batch_size=self.batch_size, imagenet_pretrain=False)
        checkpoint = weight_path(self.assets, "handdgp")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        saved = {k.removeprefix("model."): v for k, v in saved.items()}
        for key in saved:
            if "up_transform" in key and saved[key].ndim == 3:
                saved[key] = saved[key][0]
        self.model.load_state_dict(saved, strict=True)
        del saved
        self.model.eval().to(self.device)
        detector_path = resource_path(self.assets, "wilor.detector")
        self.detector = YOLO(str(detector_path))
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
            # Scored joints are decoded from the MANO export, so joint_only still fits it.
            if self._mano_model is None:
                self._mano_model = ManoModel(resource_path(self.assets, "mano"))
            mano = native_prediction(raw, camera, self._mano_model)
            return dict(mano=None if joint_only else mano, joint=mano_joint_prediction(mano, self._mano_model))

    @torch.inference_mode()
    def predict(self, record, *, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        if not isinstance(batch_size, int) or not 1 <= batch_size <= self.batch_size:
            raise ValueError("HandDGP batch_size must be within the decoder batch_size chosen at initialization")
        n = record["num_frames"]
        intrinsics = record["intrinsics_per_frame"]
        raw = {
            k: np.full((n, 2, *shape), np.nan, np.float32)
            for k, shape in (("vertices_cam", (778, 3)), ("joints_cam", (21, 3)))
        }
        raw.update(frame_ids=np.arange(n), presence=np.zeros((n, 2), bool))
        pending = []

        def flush():
            prediction = self.model(
                torch.cat([item[2] for item in pending]),
                torch.tensor(np.stack([item[3] for item in pending]), dtype=torch.float32, device=self.device),
                hand_scale=0.2,
            )
            vertices = prediction["mesh_vertices3D_cs"].detach().cpu().numpy().copy()
            joints = prediction["keypoints3D_cs"].detach().cpu().numpy().copy()
            for index, (frame, side, _, _) in enumerate(pending):
                if side == 0:
                    vertices[index, :, 0] *= -1
                    joints[index, :, 0] *= -1
                raw["vertices_cam"][frame, side] = vertices[index]
                raw["joints_cam"][frame, side] = joints[index]
                raw["presence"][frame, side] = True
            pending.clear()

        with VideoReader(record.get("_frames_rgb", record["video"])) as reader:
            for t in range(n):
                ok, rgb = reader.read()
                if not ok or rgb.shape[:2] != (record["height"], record["width"]):
                    raise ValueError("HandDGP RGB timeline mismatch")
                detections = self.detector.predict(
                    rgb[..., ::-1].copy(),
                    conf=0.3,
                    iou=0.3,
                    device=torch.device(self.device),
                    verbose=False,
                )[0].boxes
                boxes = detections.xyxy.detach().cpu().numpy()
                sides = detections.cls.detach().cpu().numpy().astype(int)
                scores = detections.conf.detach().cpu().numpy()
                for side in (0, 1):
                    ix = [
                        i
                        for i in range(len(boxes))
                        if sides[i] == side and valid_crop(boxes[i])
                    ]
                    if not ix:
                        continue
                    i = max(ix, key=lambda i: scores[i])
                    box, k = (boxes[i].copy(), intrinsics[t].copy())
                    image = rgb
                    if side == 0:
                        image = rgb[:, ::-1].copy()
                        box[[0, 2]] = record["width"] - 1 - box[[2, 0]]
                        k[0, 2] = record["width"] - 1 - k[0, 2]
                        k[0, 1] *= -1
                        k[1, 0] *= -1
                    center = (box[:2] + box[2:]) / 2
                    extent = float(np.max(box[2:] - box[:2]))
                    patch_box = [*center - extent / 2, extent, extent]
                    crop, affine, _, _, _ = self.patch_image(
                        image,
                        patch_box,
                        scale=1.3,
                        rot=0.0,
                        shift=(0.0, 0.0),
                        do_flip=False,
                        out_shape=(224, 224),
                        shift_wh=None,
                        mask=None,
                    )
                    transform = np.vstack([affine, [0.0, 0.0, 1.0]])
                    crop_k = transform @ k
                    inp = self.preprocess(crop, 224)[None].to(self.device)
                    pending.append((t, side, inp, crop_k))
                    if len(pending) == batch_size:
                        flush()
            if reader.read()[0]:
                raise ValueError("HandDGP RGB contains extra frames")
        if pending:
            flush()
        return raw


def valid_crop(box):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        return False
    extent = box[2:] - box[:2]
    return bool((extent >= 0).all() and extent.max() > 0)

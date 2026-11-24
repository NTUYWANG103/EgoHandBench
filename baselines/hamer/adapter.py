"""HaMeR inference with native person/hand crops."""

import random
from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, VideoReader, weight_path
from utils.mano import ManoModel, refit_prediction
from pathlib import Path
import numpy as np
import torch


class HaMeRAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        method = "hamer"
        assets = checkpoint_dir(method, checkpoints, version=version)
        device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        device = str(device)
        self.assets = Path(assets)
        self.device = device
        if torch.device(self.device).type == "cuda":
            torch.cuda.set_device(self.device)
        self.repo = Path(__file__).resolve().parent
        self._seed = seed
        self._mano_model = None
        torch.set_num_threads(2)
        # official demo.py (geopavlakos/hamer@3a01849) sets no TF32 flag: keep PyTorch defaults
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        batch_size = 8 if batch_size is None else batch_size
        from baselines.hamer.third_party.detectron2.detectron2.config import LazyConfig
        from baselines.hamer.vitpose_model import ViTPoseModel

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        from baselines.hamer.hamer.configs import get_config
        from baselines.hamer.hamer.models import HAMER as model_class
        from baselines.hamer.hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        assets = Path(assets)
        self.device = torch.device(device)
        official = self.repo
        data = resource_path(assets, "hamer.data")
        checkpoint = weight_path(assets, method)
        self.cfg = get_config(str(resource_path(assets, method + ".config")))
        self.cfg.defrost()
        if self.cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in self.cfg.MODEL:
            assert self.cfg.MODEL.IMAGE_SIZE == 256
            self.cfg.MODEL.BBOX_SHAPE = [192, 256]
        self.cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
        self.cfg.MODEL.pop("PRETRAINED_WEIGHTS_INIT_REGRESSION", None)
        self.cfg.MANO.MODEL_PATH = str(resource_path(assets, "mano"))
        self.cfg.MANO.MEAN_PARAMS = str(data / "data/mano_mean_params.npz")
        self.cfg.freeze()
        self.model = model_class(self.cfg, init_renderer=False)
        weights = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )["state_dict"]
        mismatch = self.model.load_state_dict(weights, strict=False)
        if mismatch.unexpected_keys or any(
            (not k.startswith("mano.") for k in mismatch.missing_keys)
        ):
            raise ValueError(f"Official {method} checkpoint mismatch: {mismatch}")
        del weights
        self.model.eval().to(self.device)
        detection = LazyConfig.load(
            str(official / method / "configs/cascade_mask_rcnn_vitdet_h_75ep.py")
        )
        detection.train.init_checkpoint = str(resource_path(assets, "vitdet"))
        for head in detection.model.roi_heads.box_predictors:
            head.test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(detection)
        self.pose = ViTPoseModel(
            str(
                official
                / "third_party/ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py"
            ),
            str(resource_path(assets, "vitpose")),
            self.device,
        )
        self.keypoint_threshold = 0.5
        self.reset()

    @torch.inference_mode()
    def _predict_raw(self, record, *, batch_size=None):
        from baselines.hamer.hamer.datasets.vitdet_dataset import ViTDetDataset
        from baselines.hamer.hamer.utils import recursive_to
        from baselines.hamer.hamer.utils.renderer import cam_crop_to_full
        n = record["num_frames"]
        result = {
            key: np.full((n, 2, *shape), np.nan, np.float32)
            for key, shape in (
                ("vertices_cam", (778, 3)),
                ("joints_cam", (21, 3)),
                ("global_orient", (1, 3, 3)),
                ("hand_pose", (15, 3, 3)),
                ("betas", (10,)),
                ("cam_trans", (3,)),
            )
        }
        confidence = np.zeros((n, 2), np.float32)
        claims = []
        cap = VideoReader(record.get("_frames_rgb", record["video"]), color="bgr")
        try:
            for t in range(n):
                ok, image = cap.read()
                if not ok or image.shape[:2] != (record["height"], record["width"]):
                    raise ValueError(f"RGB timeline/canvas mismatch at frame {t}")
                det = self.detector(image)["instances"]
                keep = (det.pred_classes == 0) & (det.scores > 0.5)
                boxes = det.pred_boxes.tensor[keep].cpu().numpy()
                scores = det.scores[keep].cpu().numpy()
                candidates = []
                if len(boxes):
                    poses = self.pose.predict_pose(
                        image[:, :, ::-1].copy(),
                        [np.concatenate([boxes, scores[:, None]], axis=1)],
                    )
                    for person, pose in enumerate(poses):
                        score = float(pose["bbox"][4])
                        for side, keypoints in enumerate(
                            (pose["keypoints"][-42:-21], pose["keypoints"][-21:])
                        ):
                            valid = keypoints[:, 2] > self.keypoint_threshold
                            if valid.sum() > 3:
                                xy = keypoints[valid, :2]
                                bbox = [*xy.min(0), *xy.max(0)]
                                candidates.append((side, score, bbox, keypoints.copy()))
                                claims.append([t, person, side, score, *bbox])
                selected = [
                    max(side_rows, key=lambda x: x[1])
                    for side in (0, 1)
                    if (
                        side_rows := [
                            r for r in candidates if r[0] == side and valid_crop(r[2])
                        ]
                    )
                ]
                if selected:
                    dataset = ViTDetDataset(
                        self.cfg,
                        image,
                        np.asarray([r[2] for r in selected]),
                        np.asarray([r[0] for r in selected]),
                        rescale_factor=2.0,
                    )
                    for batch in torch.utils.data.DataLoader(
                        dataset,
                        batch_size=self.batch_size if batch_size is None else batch_size,
                        shuffle=False,
                        num_workers=0,
                    ):
                        batch = recursive_to(batch, self.device)
                        out = self.model(batch)
                        camera = out["pred_cam"].clone()
                        camera[:, 1] *= 2 * batch["right"] - 1
                        size = batch["img_size"].float()
                        focal = (
                            self.cfg.EXTRA.FOCAL_LENGTH
                            / self.cfg.MODEL.IMAGE_SIZE
                            * size.max()
                        )
                        translation = (
                            cam_crop_to_full(
                                camera,
                                batch["box_center"].float(),
                                batch["box_size"].float(),
                                size,
                                focal,
                            )
                            .cpu()
                            .numpy()
                        )
                        for i, index in enumerate(batch["personid"].cpu().tolist()):
                            side, score, _, _ = selected[index]
                            vertices = out["pred_vertices"][i].cpu().numpy().copy()
                            vertices[:, 0] *= 2 * side - 1
                            result["vertices_cam"][t, side] = vertices + translation[i]
                            joints = out["pred_keypoints_3d"][i].cpu().numpy().copy()
                            joints[:, 0] *= 2 * side - 1
                            result["joints_cam"][t, side] = joints + translation[i]
                            result["cam_trans"][t, side] = translation[i]
                            for key in ("global_orient", "hand_pose", "betas"):
                                result[key][t, side] = (
                                    out["pred_mano_params"][key][i].cpu().numpy()
                                )
                            confidence[t, side] = score
                if (t + 1) % 100 == 0:
                    print(f"progress: {record['clip']} {t + 1}/{n}", flush=True)
        finally:
            cap.release()
        focal = (
            self.cfg.EXTRA.FOCAL_LENGTH
            / self.cfg.MODEL.IMAGE_SIZE
            * max(record["width"], record["height"])
        )
        return dict(
            result,
            presence=confidence > 0,
            confidence=confidence,
            detector_claims=np.asarray(claims, np.float32).reshape(-1, 8),
            frame_ids=np.arange(n),
            projection_intrinsics=np.array(
                [focal, focal, record["width"] / 2, record["height"] / 2], np.float64
            ),
        )

    def inference_stream(self, inputs, *, joint_only=False, batch_size=None):
        return self._inference(inputs, "stream", joint_only=joint_only, batch_size=batch_size)

    def inference_offline(self, inputs, *, joint_only=False, batch_size=None):
        return self._inference(inputs, "offline", joint_only=joint_only, batch_size=batch_size)

    def _inference(self, inputs, mode, *, joint_only=False, batch_size=None):
        if torch.device(self.device).type == "cuda":
            torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        with InferenceInput(inputs, mode=mode) as (record, camera):
            raw = self._predict_raw(record, batch_size=batch_size)
        if not joint_only and self._mano_model is None:
            self._mano_model = ManoModel(
                resource_path(self.assets, "mano")
            )
        return refit_prediction(raw, camera, self._mano_model, joint_only=joint_only)

    def reset(self):
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)


def valid_crop(box):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        return False
    extent = box[2:] - box[:2]
    return bool((extent >= 0).all() and extent.max() > 0)

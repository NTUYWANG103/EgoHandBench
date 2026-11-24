"""Official WildHands demo camera with calibrated output localization."""

import random
from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, VideoReader, weight_path
from utils.mano import apply_camera, axis_angle_to_rotmat, ManoModel, mano_joint_prediction
from .mano import native_prediction
from .camera import reference_intrinsics, source_prediction
from pathlib import Path
import cv2
import numpy as np
import torch

OFFICIAL_WEIGHTS_SHA256 = "cac3f9a9334da852f3993e95b4ec088dcc6c69f0337db63dacd83e4642880a7b"  # released wildhands.ckpt


class WildHandsAdapter:

    def _configure_runtime(self):
        """Route pretrained native resources to the selected asset scope."""
        import os

        os.environ["WILDHANDS_MANO_DIR"] = str(resource_path(self.assets, "mano"))
        os.environ["TORCH_HOME"] = str(resource_path(self.assets, "wildhands.torch"))
        os.environ["CACHE_DIR_HAMER"] = str(
            weight_path(self.assets, "wildhands").parent
        )

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        self.batch_size = 8 if batch_size is None else batch_size
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("WildHands batch_size must be a positive integer")
        method = "wildhands"
        assets = checkpoint_dir(method, checkpoints, version=version)
        # official: supplied K drives the network, as upstream demo.py:105-121 builds intrx from
        # --focal_length/--principal -> wildhands/datasets/dataset.py:111-117 (use_gt_k) ->
        # mano_head.py:54-56; egohandbench: the f=1000 centred camera the egohandbench weights were trained with, plus
        # the translation refit under supplied K.
        self.supplied_k = version == "official"
        if self.supplied_k:  # retrained weights expect the f=1000 camera; never serve them with supplied K
            import hashlib
            digest = hashlib.sha256()
            with open(weight_path(assets, "wildhands"), "rb") as stream:
                for block in iter(lambda: stream.read(1 << 24), b""):
                    digest.update(block)
            if digest.hexdigest() != OFFICIAL_WEIGHTS_SHA256:
                raise ValueError("WildHands version 'official' requires the released wildhands.ckpt; "
                                 "use version='egohandbench' for retrained weights")
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
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self._configure_runtime()
        assets = Path(assets)
        from baselines.wildhands.wildhands.configs.parser import construct_args
        from baselines.wildhands.wildhands.models.wrapper import WildHandsWrapper
        from baselines.wildhands.hamer.utils.utils_detectron2 import (
            DefaultPredictor_Lazy,
        )
        from baselines.wildhands.third_party.detectron2.detectron2.config import (
            LazyConfig,
        )
        from baselines.wildhands.vitpose_model import ViTPoseModel

        self.device = torch.device(device)
        self.cfg = construct_args()
        self.model = WildHandsWrapper(self.cfg)
        state = torch.load(
            weight_path(assets, "wildhands"), map_location="cpu", weights_only=False
        )["state_dict"]
        state = {
            k: v
            for k, v in state.items()
            if not any((x in k for x in ("head_o", "arti_head", "grasp_classifier")))
        }
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval().to(self.device)
        official = self.repo
        cfg = LazyConfig.load(
            str(official / "hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py")
        )
        cfg.train.init_checkpoint = str(resource_path(assets, "vitdet"))
        for predictor in cfg.model.roi_heads.box_predictors:
            predictor.test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(cfg)
        self.pose = ViTPoseModel(
            str(
                official
                / "third_party/ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py"
            ),
            str(resource_path(assets, "wildhands.vitpose")),
            self.device,
        )
        self.reset()

    @torch.inference_mode()
    def _predict_raw(self, record, *, batch_size=None, rescale_factor=1.75):
        batch_size = self.batch_size if batch_size is None else batch_size
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("WildHands batch_size must be a positive integer")
        if rescale_factor <= 0:
            raise ValueError("WildHands rescale_factor must be positive")
        from baselines.wildhands.wildhands.datasets.dataset import WildHandsDataset
        from baselines.wildhands.wildhands.common.data_utils import (
            generate_patch_image_clean,
        )
        from baselines.wildhands.wildhands.common.rot import matrix_to_axis_angle
        from baselines.wildhands.hamer.utils import recursive_to

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
                ("source_hand_pose_axis_angle", (45,)),
            )
        }
        result["detector_boxes_original"] = np.full((n, 2, 4), np.nan, np.float64)
        confidence = np.zeros((n, 2), np.float32)
        width, height = (record["width"], record["height"])
        size = max(width, height)
        scale = 840 / size
        intrinsics = reference_intrinsics(width, height)
        transform = np.array([
            [scale, 0, -(width - size) / 2 * scale],
            [0, scale, -(height - size) / 2 * scale],
            [0, 0, 1],
        ])
        cap = VideoReader(record.get("_frames_rgb", record["video"]), color="bgr")
        try:
            for t in range(n):
                ok, bgr = cap.read()
                if not ok or bgr.shape[:2] != (height, width):
                    raise ValueError(f"RGB timeline/canvas mismatch at {t}")
                k = transform @ (record["intrinsics_per_frame"][t] if self.supplied_k else intrinsics)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                patch, _, inverse = generate_patch_image_clean(
                    rgb,
                    [width / 2, height / 2, size, size],
                    1.0,
                    0.0,
                    [840, 840],
                    cv2.INTER_CUBIC,
                )
                image = patch.clip(0, 255).astype(np.uint8)
                image_bgr = image[..., ::-1].copy()
                det = self.detector(image_bgr)["instances"]
                valid = (det.pred_classes == 0) & (det.scores > 0.5)
                boxes = det.pred_boxes.tensor[valid].cpu().numpy()
                scores = det.scores[valid].cpu().numpy()
                rows = []
                if len(boxes):
                    poses = self.pose.predict_pose(
                        image_bgr, [np.concatenate([boxes, scores[:, None]], axis=1)]
                    )
                    for pose in poses:
                        for side, points in enumerate(
                            (pose["keypoints"][-42:-21], pose["keypoints"][-21:])
                        ):
                            visible = points[:, 2] > 0.5
                            if visible.sum() > 3:
                                bbox = [
                                    *points[visible, :2].min(0),
                                    *points[visible, :2].max(0),
                                ]
                                score = float(pose["bbox"][4])
                                rows.append((side, score, bbox))
                selected = [
                    max(r, key=lambda a: a[1])
                    for side in (0, 1)
                    if (r := [x for x in rows if x[0] == side])
                ]
                if selected:
                    for side, _, box in selected:
                        corners = np.asarray(box, dtype=np.float64).reshape(2, 2)
                        homogeneous = np.column_stack((corners, np.ones(2)))
                        result["detector_boxes_original"][t, side] = (
                            np.einsum("ij,qj->qi", inverse.astype(np.float64), homogeneous)
                        ).reshape(4)
                    dataset = WildHandsDataset(
                        self.cfg,
                        image,
                        np.array([x[2] for x in selected]),
                        np.array([x[0] for x in selected]),
                        focal_length=(k[0, 0] + k[1, 1]) / 2,
                        rescale_factor=rescale_factor,
                        intrx=k,
                    )
                    loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, num_workers=0, shuffle=False
                    )
                    offset = 0
                    for batch in loader:
                        batch = recursive_to(batch, self.device)
                        pred = self.model(batch)
                        count = len(batch[0]["img"])
                        for i, (side, score, _) in enumerate(selected[offset:offset + count]):
                            suffix = "r" if side else "l"
                            trans = pred[f"pred.cam_t.{suffix}"][i].cpu().numpy()
                            result["joints_cam"][t, side] = (
                                pred[f"pred.joints3d.{suffix}"][i].cpu().numpy() + trans
                            )
                            confidence[t, side] = score
                            result["cam_trans"][t, side] = trans
                            result["vertices_cam"][t, side] = (
                                pred[f"pred.vertices.{suffix}"][i].cpu().numpy() + trans
                            )
                            result["betas"][t, side] = pred[f"pred.beta.{suffix}"][i].cpu().numpy()
                            aa = matrix_to_axis_angle(
                                pred[f"pred.pose.{suffix}"][i].reshape(16, 3, 3)
                            ).cpu()
                            rotations = axis_angle_to_rotmat(aa.double()).float().numpy()
                            result["global_orient"][t, side] = rotations[:1]
                            result["hand_pose"][t, side] = rotations[1:]
                            result["source_hand_pose_axis_angle"][t, side] = aa[1:].numpy().reshape(45)
                        offset += count
                if (t + 1) % 100 == 0:
                    print(f"progress: {record['clip']} {t + 1}/{n}", flush=True)
        finally:
            cap.release()
        return dict(
            result,
            presence=confidence > 0,
        )

    def inference_stream(self, inputs, *, joint_only=False, batch_size=None, rescale_factor=1.75):
        return self._inference(inputs, "stream", joint_only=joint_only, batch_size=batch_size, rescale_factor=rescale_factor)

    def inference_offline(self, inputs, *, joint_only=False, batch_size=None, rescale_factor=1.75):
        return self._inference(inputs, "offline", joint_only=joint_only, batch_size=batch_size, rescale_factor=rescale_factor)

    def _inference(self, inputs, mode, *, joint_only=False, batch_size=None, rescale_factor=1.75):
        if torch.device(self.device).type == "cuda":
            torch.cuda.set_device(self.device)
        # demo.py leaves PyTorch's TF32 defaults (matmul off, cuDNN on) untouched.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        with InferenceInput(inputs, mode=mode) as (record, camera):
            raw = self._predict_raw(record, batch_size=batch_size, rescale_factor=rescale_factor)
        _, delta = source_prediction(raw, camera, refit=not self.supplied_k)
        # Scored joints are decoded from the MANO export, so joint_only still fits it.
        if self._mano_model is None:
            self._mano_model = ManoModel(
                resource_path(self.assets, "mano")
            )
        mano = apply_camera(
            native_prediction(raw, camera, self._mano_model, resource_path(self.assets, "mano")),
            delta,
        )
        return dict(mano=None if joint_only else mano, joint=mano_joint_prediction(mano, self._mano_model))

    def reset(self):
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)

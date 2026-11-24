"""Official OmniHands temporal model and smoothing with dataset-K camera adaptation."""

import random
from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, VideoReader, weight_path
from utils.mano import axis_angle_to_rotmat, ManoModel, mano_joint_prediction
from .mano import native_prediction
from pathlib import Path
import numpy as np
import torch
from .temporal import gen_sequences, seq_smooth


class OmniHandsAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        method = "omnihands"
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
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.temporal_batch_size = 8
        import os

        os.environ["OMNIHANDS_HUMAN_MODEL_DIR"] = str(
            resource_path(self.assets, "mano").parent
        )
        from baselines.omnihands.third_party.detectron2.detectron2.config import (
            LazyConfig,
        )
        from baselines.omnihands.hands_4d.configs import get_config
        from baselines.omnihands.hands_4d.models import Hands4D
        from baselines.omnihands.hands_4d.utils.utils_detectron2 import (
            DefaultPredictor_Lazy,
        )
        from baselines.omnihands.vitpose_model import ViTPoseModel

        assets = Path(assets).resolve()
        self.device = torch.device(device)
        self.official = self.repo
        self.cfg = get_config(str(resource_path(self.assets, "omnihands.config")))
        self.cfg.defrost()
        self.cfg.MANO.MODEL_PATH = str(resource_path(assets, "mano"))
        self.cfg.MANO.MEAN_PARAMS = str(resource_path(assets, "omnihands.mean_params"))
        self.cfg.freeze()
        self.model = Hands4D(self.cfg)
        weights = torch.load(
            weight_path(assets, "omnihands"),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        mismatch = self.model.load_state_dict(weights, strict=False)
        learned = {k for k, p in self.model.named_parameters() if p.requires_grad}
        unused_discriminator = [
            k for k in mismatch.unexpected_keys if k.startswith("discriminator.")
        ]
        if set(mismatch.unexpected_keys) - set(
            unused_discriminator
        ) or learned.intersection(mismatch.missing_keys):
            raise ValueError(f"Official OmniHands checkpoint mismatch: {mismatch}")
        if any((not k.startswith("mano.") for k in mismatch.missing_keys)):
            raise ValueError(
                f"Unaudited missing OmniHands buffers: {mismatch.missing_keys}"
            )
        del weights
        self.model.eval().to(self.device)
        detection = LazyConfig.load(
            str(self.official / "hands_4d/configs/cascade_mask_rcnn_vitdet_h_75ep.py")
        )
        detection.train.init_checkpoint = str(resource_path(assets, "vitdet"))
        for head in detection.model.roi_heads.box_predictors:
            head.test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(detection)
        pose_path = resource_path(assets, "vitpose")
        self.pose = ViTPoseModel(
            str(
                self.official
                / "third_party/ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py"
            ),
            str(pose_path),
            self.device,
        )
        if batch_size is not None:
            self.temporal_batch_size = batch_size
        self.reset()

    @torch.inference_mode()
    def _predict_raw(self, record, *, temporal_batch_size=None):
        from .camera import known_camera

        with known_camera(self.model.time_head):
            return self._predict(record, temporal_batch_size=temporal_batch_size)

    def _predict(self, record, *, temporal_batch_size=None):
        temporal_batch_size = self.temporal_batch_size if temporal_batch_size is None else temporal_batch_size
        from baselines.omnihands.hands_4d.datasets.vitdet_dataset import (
            ViTDetInterDataset_Batch,
            ViTDetInterDataset_Sequence,
            HandToken_Sequence,
        )
        from baselines.omnihands.hands_4d.utils import recursive_to

        n = record["num_frames"]
        result = {
            k: np.full((n, 2, *shape), np.nan, np.float32)
            for k, shape in (
                ("vertices_cam", (778, 3)),
                ("vertices_cam_unsmoothed", (778, 3)),
                ("joints_cam", (21, 3)),
                ("global_orient", (1, 3, 3)),
                ("hand_pose", (15, 3, 3)),
                ("betas", (10,)),
                ("cam_trans", (3,)),
                ("source_global_orient_axis_angle", (3,)),
                ("source_hand_pose_axis_angle", (45,)),
            )
        }
        result.update(
            frame_ids=np.arange(n),
            presence=np.zeros((n, 2), bool),
            confidence=np.zeros((n, 2), np.float32),
            projection_intrinsics=np.repeat(
                record["intrinsics_per_frame"][:, None, (0, 1, 0, 1), (0, 1, 2, 2)],
                2, axis=1,
            ),
        )

        class FloatFrames(list):

            def __getitem__(self, index):
                return torch.from_numpy(super().__getitem__(index)).to(
                    dtype=torch.float32
                )

        images, boxes = (FloatFrames(), {"left": {}, "right": {}})
        detected = np.zeros((n, 2), bool)
        cap = VideoReader(record.get("_frames_rgb", record["video"]), color="bgr")
        try:
            for t in range(n):
                ok, frame = cap.read()
                if not ok or frame.shape[:2] != (record["height"], record["width"]):
                    raise ValueError(f"RGB timeline/canvas mismatch at {t}")
                images.append(frame)
                instances = self.detector(frame)["instances"]
                keep = (instances.pred_classes == 0) & (instances.scores > 0.5)
                bbox = instances.pred_boxes.tensor[keep].cpu().numpy()
                scores = instances.scores[keep].cpu().numpy()
                poses = (
                    self.pose.predict_pose(
                        frame, [np.concatenate([bbox, scores[:, None]], axis=1)]
                    )
                    if len(bbox)
                    else []
                )
                for pose in poses[:1]:
                    for side, name in enumerate(("left", "right")):
                        kp = (
                            pose["keypoints"][-42:-21]
                            if side == 0
                            else pose["keypoints"][-21:]
                        )
                        valid = kp[:, 2] > 0.5
                        if valid.sum() > 3:
                            candidate = np.r_[
                                kp[valid, :2].min(0), kp[valid, :2].max(0)
                            ].tolist()
                            if valid_crop(candidate):
                                boxes[name][t] = candidate
                                detected[t, side] = True
                            elif t > 0 and t - 1 in boxes[name]:
                                boxes[name][t] = boxes[name][t - 1]
                        elif t > 0 and t - 1 in boxes[name]:
                            boxes[name][t] = boxes[name][t - 1]
                if (t + 1) % 100 == 0:
                    print(f"OmniHands detect: {record['clip']} {t + 1}/{n}", flush=True)
        finally:
            cap.release()
        for t in range(n):
            for name in boxes:
                if t not in boxes[name]:
                    for other in range(t + 1, n):
                        if other in boxes[name]:
                            boxes[name][t] = boxes[name][other]
                            break
                if t not in boxes[name]:
                    for other in range(t - 1, 0, -1):
                        if other in boxes[name]:
                            boxes[name][t] = boxes[name][other]
                            break
        result["detector_observed"] = detected
        result["detector_boxes_filled"] = np.array(
            [
                [boxes[name].get(t, [np.nan] * 4) for name in ("left", "right")]
                for t in range(n)
            ],
            np.float32,
        )
        if any((len(track) != n for track in boxes.values())):
            result["pipeline_input_failure"] = np.array(
                "official_demo_requires_complete_left_and_right_tracks"
            )
            return result
        window = gen_sequences(n, 30, 1)
        smooth_boxes = {
            name: {
                str(t): np.r_[
                    boxes[name][t][:2],
                    np.asarray(
                        [boxes[name][int(i)][2:] for i in window[t]], np.float64
                    ).mean(0),
                ]
                for t in range(n)
            }
            for name in boxes
        }
        sequences = gen_sequences(n, self.cfg.MODEL.SEQ_LEN, 10)
        singles = ViTDetInterDataset_Batch(
            self.cfg, images, smooth_boxes, rescale_factor=2.0
        )
        seq_data = ViTDetInterDataset_Sequence(
            self.cfg, images, smooth_boxes, sequences, rescale_factor=2.0
        )

        def batches(data, size):
            return torch.utils.data.DataLoader(
                data, batch_size=size, shuffle=False, num_workers=0, drop_last=False
            )

        tokens = []
        for batch in batches(singles, 4 * temporal_batch_size):
            token = self.model.inference_token_forward(recursive_to(batch, self.device))
            tokens.append(token.reshape(-1, 2, token.shape[-1]).cpu())
        token_data = HandToken_Sequence(torch.cat(tokens), sequences)
        outputs = []
        batch_start = 0
        for token, batch in zip(
            batches(token_data, temporal_batch_size),
            batches(seq_data, temporal_batch_size),
        ):
            count = token.shape[0]
            selected = np.asarray(sequences[batch_start:batch_start + count], dtype=int)
            batch["camera_K"] = torch.as_tensor(record["intrinsics_per_frame"][selected])
            batch_start += count
            output = self.model.inference_temp_forward(
                token.reshape(-1, token.shape[-1]).to(self.device),
                recursive_to(batch, self.device),
            )
            outputs.append(
                {
                    k: v.detach().cpu().numpy()
                    for k, v in output.items()
                    if k.startswith(
                        (
                            "verts3d_world_",
                            "joints3d_world_",
                            "cam_aligned_",
                            "mano_pose_",
                            "mano_shape_",
                        )
                    )
                }
            )
        output = {k: np.concatenate([row[k] for row in outputs]) for k in outputs[0]}
        world = [output["verts3d_world_" + name] for name in ("left", "right")]
        smooth_right, smooth_left = seq_smooth(world[1].copy(), world[0].copy())
        # Native OpenPose joints (sh_joint_regressor on the mesh), smoothed like it.
        joints_right, joints_left = seq_smooth(
            output["joints3d_world_right"].copy(), output["joints3d_world_left"].copy()
        )
        for side, name in enumerate(("left", "right")):
            camera = output["cam_aligned_" + name].reshape(n, 3)
            aa = output["mano_pose_" + name].reshape(n, 16, 3)
            matrices = (
                axis_angle_to_rotmat(torch.tensor(aa, dtype=torch.float64))
                .float()
                .numpy()
            )
            result["vertices_cam"][:, side] = (
                smooth_left if side == 0 else smooth_right
            ) + camera[:, None]
            result["vertices_cam_unsmoothed"][:, side] = world[side] + camera[:, None]
            result["joints_cam"][:, side] = (
                joints_left if side == 0 else joints_right
            ) + camera[:, None]
            result["cam_trans"][:, side] = camera
            result["global_orient"][:, side] = matrices[:, :1]
            result["hand_pose"][:, side] = matrices[:, 1:]
            result["source_global_orient_axis_angle"][:, side] = aa[:, 0]
            result["source_hand_pose_axis_angle"][:, side] = aa[:, 1:].reshape(n, 45)
            result["betas"][:, side] = output["mano_shape_" + name].reshape(n, 10)
        result["presence"][:] = True
        result["confidence"][:] = 1.0
        result["pipeline_input_failure"] = np.array("")
        return result

    def inference_stream(self, inputs, *, joint_only=False):
        raise NotImplementedError("omnihands requires a complete offline clip")

    def inference_offline(self, inputs, *, joint_only=False, temporal_batch_size=None):
        return self._inference(inputs, "offline", joint_only, temporal_batch_size)

    def _inference(self, inputs, mode, joint_only=False, temporal_batch_size=None):
        if torch.device(self.device).type == "cuda":
            torch.cuda.set_device(self.device)
        # run_demo.py leaves PyTorch's TF32 defaults (matmul off, cuDNN on) untouched.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        with InferenceInput(inputs, mode=mode) as (record, camera):
            if np.any(camera.K[:, 0, 1] != 0) or np.any(camera.K[:, 1, 0] != 0):
                raise ValueError("OmniHands PnP does not support camera skew")
            raw = self._predict_raw(record, temporal_batch_size=temporal_batch_size)
        # Scored joints are decoded from the MANO export, so joint_only still fits it.
        if self._mano_model is None:
            self._mano_model = ManoModel(
                resource_path(self.assets, "mano")
            )
        mano = native_prediction(raw, camera, self._mano_model, resource_path(self.assets, "mano"))
        return {"mano": None if joint_only else mano, "joint": mano_joint_prediction(mano, self._mano_model)}

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

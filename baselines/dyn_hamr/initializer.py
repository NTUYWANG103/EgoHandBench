"""Official Dyn-HaMR HEAD HaMeR stage (YOLO hands, 3-pass bbox cleaning, crop 1.3)."""

import contextlib
import io
import random
from types import SimpleNamespace
from utils.download_data import checkpoint_dir
from utils.io import resource_path, weight_path
from importlib import import_module
from pathlib import Path
import numpy as np
import torch


class HaMeRInitializer:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None):
        method = "hamer"
        assets = checkpoint_dir(method, checkpoints)
        device = str(
            torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        )
        self.assets = Path(assets)
        self.device = torch.device(device)
        self._seed = seed
        torch.set_num_threads(2)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        if batch_size not in (None, 8):
            # run.py's Pass 3 DataLoader is fixed at batch_size=8.
            raise ValueError("The official Dyn-HaMR HaMeR stage uses batch_size=8")
        from ultralytics import YOLO

        package = "baselines.dyn_hamr.hamer.hamer"
        get_config = import_module(package + ".configs").get_config
        HAMER = import_module(package + ".models").HAMER
        self.run = import_module("baselines.dyn_hamr.hamer.run")
        # Mirror the fork's load_hamer(): Dyn-HaMR MANO assets and focal 500.
        cfg = get_config(str(resource_path(assets, method + ".config")), update_cachedir=True)
        cfg.defrost()
        cfg.MANO.MEAN_PARAMS = str(resource_path(assets, "hamer.data") / "data/mano_mean_params.npz")
        cfg.MANO.MODEL_PATH = str(resource_path(assets, "mano"))
        cfg.EXTRA.FOCAL_LENGTH = 500.0
        if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
            assert cfg.MODEL.IMAGE_SIZE == 256
            cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
        cfg.freeze()
        self.cfg = cfg
        self.model = HAMER(cfg)
        weights = torch.load(
            weight_path(assets, method), map_location="cpu", weights_only=False, mmap=True
        )["state_dict"]
        mismatch = self.model.load_state_dict(weights, strict=False)
        if mismatch.unexpected_keys or any(
            (not k.startswith("mano.") for k in mismatch.missing_keys)
        ):
            raise ValueError(f"Official {method} checkpoint mismatch: {mismatch}")
        del weights
        self.model.eval().to(self.device)
        self.detector = YOLO(str(resource_path(assets, "detector")))
        self.detector.to(self.device)
        # Ultralytics 8.3.40's .to() already records this device; keep it a torch.device
        # so select_device() never rewrites CUDA_VISIBLE_DEVICES for SLAM/optimizer children.
        self.detector.overrides["device"] = self.device
        self.model.register_forward_hook(self._capture)
        self.reset()

    def _capture(self, model, args, out):
        # run.py keeps only MANO parameters, camera and 2D joints; the
        # supplied-K translation adaptation also needs the native 3D joints.
        right = args[0]["right"].detach().cpu().numpy().astype(int)
        for i, side in enumerate(right):
            sign = np.array([2 * side - 1, 1, 1], np.float32)
            self.captured.append(
                (
                    side,
                    out["pred_keypoints_3d"][i].detach().cpu().numpy() * sign,
                    out["pred_vertices"][i].detach().cpu().numpy() * sign,
                )
            )

    def _predict_raw(self, record, images):
        """Run run.py Pass 1-3 on the extracted JPEG frames of one clip."""
        # run.py leaves PyTorch's numeric defaults untouched.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        n = record["num_frames"]
        file_type = ["*.jpg", "*.png"]
        img_paths = sorted([img for end in file_type for img in Path(images).glob(end)])
        if len(img_paths) != n:
            raise ValueError("Dyn-HaMR JPEG timeline differs")
        self.captured = []
        # launch_hamer passes no --rescale_factor (run.py default 1.3); rendering is
        # visualization only, and res_folder only names its (absent) output folder.
        args = SimpleNamespace(
            rescale_factor=1.3, render=False, res_folder=str(Path(images).parent / "hamer.pkl")
        )
        # The official passes print per-frame diagnostics only.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw_data = self.run.extract_raw_bboxes(img_paths, self.detector)
            cleaned = self.run.clean_bbox_sequences(raw_data)
            results = self.run.run_hamer_on_cleaned_bboxes(
                cleaned, self.model, self.cfg, None, args
            )
        raw = {
            key: np.full((n, 2, *shape), np.nan, np.float32)
            for key, shape in (
                ("vertices_cam", (778, 3)),
                ("joints_cam", (21, 3)),
                ("global_orient", (1, 3, 3)),
                ("hand_pose", (15, 3, 3)),
                ("betas", (10,)),
            )
        }
        raw["cam_trans"] = np.full((n, 2, 3), np.nan, np.float32)
        anchors = np.full((n, 2, 21, 2), np.nan, np.float64)
        confidence = np.zeros((n, 2), np.float32)
        captured = iter(self.captured)
        for t, path in enumerate(img_paths):
            frame = results[str(path)]
            for k, side in enumerate(frame["tracked_ids"]):
                captured_side, joints, vertices = next(captured)
                if captured_side != side or frame["mano"][k]["is_right"] != side:
                    raise ValueError("HaMeR hand order differs from run.py output")
                translation = np.asarray(frame["cam_trans"][k], np.float32)
                raw["cam_trans"][t, side] = translation
                raw["joints_cam"][t, side] = joints + translation
                raw["vertices_cam"][t, side] = vertices + translation
                for key in ("global_orient", "hand_pose", "betas"):
                    raw[key][t, side] = frame["mano"][k][key]
                anchors[t, side] = np.asarray(frame["extra_data"][k])[:, :2]
                confidence[t, side] = cleaned[t][("left", "right")[side] + "_conf"]
        if next(captured, None) is not None:
            raise ValueError("Unconsumed HaMeR predictions")
        presence = np.zeros((n, 2), bool)
        for t, path in enumerate(img_paths):
            presence[t, results[str(path)]["tracked_ids"]] = True
        focal = (
            self.cfg.EXTRA.FOCAL_LENGTH
            / self.cfg.MODEL.IMAGE_SIZE
            * max(record["width"], record["height"])
        )
        return dict(
            raw,
            presence=presence,
            confidence=confidence,
            frame_ids=np.arange(n),
            projection_intrinsics=np.array(
                [focal, focal, record["width"] / 2, record["height"] / 2], np.float64
            ),
            model_anchors2d_openpose=anchors,
        )

    def reset(self):
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)

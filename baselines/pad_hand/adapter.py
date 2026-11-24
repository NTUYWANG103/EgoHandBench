"""Released PAD-Hand with native WiLoR detector, dual tracks and known K."""

from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, weight_path
from pathlib import Path
import pickle

import numpy as np
import torch
from utils.mano import ManoModel, mano_joint_prediction
from .mano import native_prediction


class PADHandAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        if batch_size is not None:
            raise ValueError("PAD-Hand does not expose a batch_size option")
        self.assets = checkpoint_dir("pad_hand", checkpoints, version=version)
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
        self._mano_model = None
        torch.set_num_threads(2)
        # official demo.py / wilor_inference.py (PAD-Hand-CVPR-2026@ca9ed97) set no TF32 flag: keep PyTorch defaults
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
        self._pad_assets = str(resource_path(self.assets, "pad_hand.assets"))
        self.reset()

    def predict(self, record, *, diff_steps=4, kappa=15 / 180 * np.pi):
        import os

        os.environ["PAD_HAND_ASSETS_DIR"] = self._pad_assets
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        from .pipeline import run

        K = record["intrinsics_per_frame"]
        sides = run(
            dict(
                device=self.device,
                input=record["video"],
                intrinsics=K,
                diff_steps=diff_steps,
                kappa=kappa,
                checkpoint=str(weight_path(self.assets, "pad_hand")),
                assets=str(self.assets),
            )
        )
        n = record["num_frames"]
        # PAD's own 21-joint regressor (MANO16 + tips), as in its MANO.forward.
        with (Path(self._pad_assets) / "processed_MANO_RIGHT.pkl").open("rb") as stream:
            regressor = np.asarray(
                pickle.load(stream, encoding="latin1")["J_regressor_right"], np.float32
            )
        raw = {
            k: np.full(shape, np.nan, np.float32)
            for k, shape in dict(
                vertices_cam=(n, 2, 778, 3),
                joints_cam=(n, 2, 21, 3),
                global_orient=(n, 2, 1, 3, 3),
                hand_pose=(n, 2, 15, 3, 3),
                betas=(n, 2, 10),
                cam_trans=(n, 2, 3),
            ).items()
        }
        raw.update(
            presence=np.zeros((n, 2), bool),
            confidence=np.zeros((n, 2), np.float32),
            was_refined=np.zeros((n, 2), bool),
            frame_ids=np.arange(n),
            projection_intrinsics=np.repeat(
                K[:, None, (0, 1, 0, 1), (0, 1, 2, 2)], 2, axis=1
            ),
        )
        for side in (0, 1):
            if len(sides[str(side)]) != n:
                raise ValueError("Incomplete PAD timeline")
            for t, fd in enumerate(sides[str(side)]):
                if fd is None:
                    continue
                trans = np.asarray(fd["cam_t"])
                raw["vertices_cam"][t, side] = (
                    np.asarray(fd["refined_vertices"]) + trans
                )
                raw["joints_cam"][t, side] = regressor @ raw["vertices_cam"][t, side]
                for key in ("global_orient", "hand_pose", "betas"):
                    raw[key][t, side] = np.asarray(
                        fd.get("initializer_betas", fd[key])
                        if key == "betas"
                        else fd[key]
                    )
                raw["cam_trans"][t, side] = trans
                raw["presence"][t, side] = True
                raw["confidence"][t, side] = fd["confidence"]
                raw["was_refined"][t, side] = fd["was_refined"]
        return raw

    def inference_stream(self, inputs, *, joint_only=False):
        raise NotImplementedError("pad_hand requires a complete clip")

    def inference_offline(self, inputs, *, joint_only=False, diff_steps=4, kappa=15 / 180 * np.pi):
        if str(self.device).startswith("cuda"):
            torch.cuda.set_device(self.device)
        with InferenceInput(
            inputs, mode="offline", requires_video_path=True
        ) as (record, camera):
            return self._canonical(self.predict(record, diff_steps=diff_steps, kappa=kappa), camera, joint_only)

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

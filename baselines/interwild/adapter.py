"""Official InterWild demo camera/overlap policy, retaining source predictions."""

import random
import numpy as np
import torch
from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, VideoReader, weight_path
from utils.mano import axis_angle_to_rotmat, enable_mano_pickle_compatibility, ManoModel
from .mano import native_prediction


class InterWildAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        if batch_size is not None:
            raise ValueError("interwild uses its fixed native batch")
        self.method = "interwild"
        selected_device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if selected_device.type == "cuda" and selected_device.index is None:
            selected_device = torch.device("cuda", torch.cuda.current_device())
        self.device = str(selected_device)
        if not self.device.startswith("cuda"):
            raise ValueError("interwild requires CUDA")
        torch.cuda.set_device(self.device)
        torch.set_num_threads(2)
        # official demo/demo.py (facebookresearch/InterWild@7c01e4a) sets no TF32 flag: keep PyTorch defaults
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # official demo/demo.py:48
        self.seed = seed
        self.assets = checkpoint_dir(self.method, checkpoints, version=version)
        self._mano_model = None
        enable_mano_pickle_compatibility()
        self.reset()
        from baselines.interwild.main.model import get_model
        from baselines.interwild.main.config import cfg

        self.cfg = cfg
        self.model = (
            get_model("test", str(resource_path(self.assets, "interwild.mano").parent))
            .to(self.device)
            .eval()
        )
        checkpoint_path = weight_path(self.assets, "interwild")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = {k.removeprefix("module."): v for k, v in checkpoint["network"].items()}
        result = self.model.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(
            (
                not k.startswith(("mano_layer_left.", "mano_layer_right."))
                for k in result.missing_keys
            )
        ):
            raise ValueError(f"InterWild learned checkpoint mismatch: {result}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        self.reset()

    def inference_stream(self, inputs, *, joint_only=False):
        """Predict canonical MANO from one RGB frame and its Camera."""
        return self._inference(inputs, "stream", joint_only=joint_only)

    def inference_offline(self, inputs, *, joint_only=False):
        return self._inference(inputs, "offline", joint_only=joint_only)

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def _inference(self, inputs, mode, *, joint_only=False):
        torch.cuda.set_device(self.device)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # official demo/demo.py:48
        torch.backends.cudnn.deterministic = False
        with InferenceInput(inputs, mode=mode) as (record, camera):
            raw = self.predict(record)
            # Scored joints are decoded from the MANO export, so joint_only still fits it.
            if self._mano_model is None:
                self._mano_model = ManoModel(resource_path(self.assets, "mano"))
            return native_prediction(
                raw, camera, self._mano_model, resource_path(self.assets, "mano"), joint_only=joint_only
            )

    @torch.inference_mode()
    def predict(self, record):
        from baselines.interwild.common.utils.preprocessing import (
            process_bbox,
            generate_patch_image,
            get_iou,
        )

        n = record["num_frames"]
        vertices = np.full((n, 2, 778, 3), np.nan, np.float32)
        joints = np.full((n, 2, 21, 3), np.nan, np.float32)
        go = np.full((n, 2, 1, 3, 3), np.nan, np.float32)
        hp = np.full((n, 2, 15, 3, 3), np.nan, np.float32)
        beta = np.full((n, 2, 10), np.nan, np.float32)
        source_hp_aa = np.full((n, 2, 45), np.nan, np.float32)
        pair = np.zeros(n, bool)
        projection_intrinsics = np.full((n, 2, 4), np.nan, np.float32)
        cap = VideoReader(record.get("_frames_rgb", record["video"]), color="bgr")
        try:
            for t in range(n):
                ok, image = cap.read()
                if not ok or image.shape[:2] != (record["height"], record["width"]):
                    raise ValueError(
                        f"InterWild RGB timeline/canvas mismatch at frame {t}"
                    )
                image = image[:, :, ::-1].astype(np.float32)
                box = process_bbox(
                    [0, 0, record["width"], record["height"]],
                    record["width"],
                    record["height"],
                )
                crop, _, crop_to_image = generate_patch_image(
                    image, box, 1.0, 0.0, False, self.cfg.input_img_shape
                )
                tensor = (
                    torch.from_numpy(
                        crop.astype(np.float32).transpose(2, 0, 1).copy()
                    ).to(self.device)[None]
                    / 255
                )
                out = self.model({"img": tensor}, {}, {}, "test")
                pair[t] = (
                    get_iou(
                        out["rhand_bbox"][0].cpu().numpy(),
                        out["lhand_bbox"][0].cpu().numpy(),
                        "xyxy",
                    )
                    > 0
                )
                for side, prefix in enumerate(("l", "r")):
                    # Match the native camera to the same overlap/root branch.
                    camera_prefix = "render_" if pair[t] else "render_" + prefix
                    scale = out[camera_prefix + "focal"][0].cpu().numpy()
                    offset = out[camera_prefix + "princpt"][0].cpu().numpy()
                    # Cropping uses zero rotation: map orthographic scale/offset
                    # from the network input canvas back to original RGB pixels.
                    projection_intrinsics[t, side, :2] = (
                        np.diag(crop_to_image[:, :2]) * scale
                    )
                    projection_intrinsics[t, side, 2:] = (
                        crop_to_image[:, :2] @ offset + crop_to_image[:, 2]
                    )
                    root = (
                        out["rroot_cam"] + (out["rel_trans"] if side == 0 else 0)
                        if pair[t]
                        else out[prefix + "root_cam"]
                    )
                    vertices[t, side] = (
                        (out[prefix + "mano_mesh_cam"] + root[:, None])[0].cpu().numpy()
                    )
                    # Native OpenPose joints (sh_joint_regressor), same root as the mesh.
                    joints[t, side] = (
                        (out[prefix + "mano_joint_cam"] + root[:, None])[0].cpu().numpy()
                    )
                    go_aa = out[prefix + "mano_root_pose"]
                    hp_aa = out[prefix + "mano_hand_pose"]
                    b = out[prefix + "mano_shape"]
                    source_hp_aa[t, side] = hp_aa[0].cpu().numpy()
                    go[t, side] = (
                        axis_angle_to_rotmat(go_aa.double().cpu())
                        .numpy()
                        .reshape(1, 3, 3)
                    )
                    hp[t, side] = axis_angle_to_rotmat(
                        hp_aa.reshape(15, 3).double().cpu()
                    ).numpy()
                    beta[t, side] = b[0].cpu().numpy()
        finally:
            cap.release()
        return dict(
            vertices_cam=vertices,
            joints_cam=joints,
            projection_intrinsics=projection_intrinsics,
            global_orient=go,
            hand_pose=hp,
            betas=beta,
            presence=np.ones((n, 2), bool),
            frame_ids=np.arange(n),
            source_hand_pose_axis_angle=source_hp_aa,
        )

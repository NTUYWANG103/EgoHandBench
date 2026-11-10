"""InfiniHand streaming camera-space hands (upstream infinihand.infer) under the supplied per-frame K."""

import json
import time
from contextlib import contextmanager
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, weight_path
from .mano import SLOTS

ANCHOR_FRAMES, MAX_FRAMES = 4, 8192  # infinihand.infer: four anchor frames, at most 8192 frames per invocation


class InfiniHandAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="official"):
        if batch_size is not None:
            raise ValueError("infinihand streams one clip (4 anchor frames, then one frame per step); batch_size is unsupported")
        self.assets = checkpoint_dir("infinihand", checkpoints, version=version)
        self.repo = Path(__file__).resolve().parent
        selected = torch.device("cuda:0" if device in (None, "auto") else device)
        if selected.type != "cuda":
            raise ValueError("infinihand runs on CUDA only (two CUDA streams)")
        if selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
        self.device = selected
        torch.cuda.set_device(self.device)
        self.seed = seed
        self.reset()
        from baselines.infinihand.infinihand.models.hand_model import LingBotMapHand, LingBotMapHandConfig
        from baselines.infinihand.infinihand.weights import load_inference_weights

        # infinihand.infer.build_model, with the bundled code paths and this group's MANO assets
        config = json.loads((self.repo / "configs/infinihand_v1.json").read_text())
        code = self.repo / "infinihand"
        config.update(
            lingbotmap_root=str(code / "lingbot_map"), wilor_root=str(code / "wilor"),
            wilor_cfg=str(code / "wilor/pretrained_models/model_config.yaml"),
            mano_model_path=str(Path(resource_path(self.assets, "mano")).resolve()),
            lingbotmap_ckpt="", wilor_ckpt="", load_wilor_checkpoint=False, step_memory_enabled=True,
            step_memory_scale_frames=4, step_memory_sliding_window=16, step_memory_max_frames=8192,
        )
        self.model = LingBotMapHand(LingBotMapHandConfig(**config))
        state = torch.load(weight_path(self.assets, "infinihand"), map_location="cpu", mmap=True, weights_only=True)
        if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError("Expected a complete exported tensor state_dict")
        load_inference_weights(self.model, state)
        self.model.to(self.device).eval()
        self.mask_threshold = 0.9  # infinihand.infer --mask-threshold default and the evaluation config

    @contextmanager
    def _runtime(self, *, branch_precision, cuda_graphs, fused_rope,
                 compact_kv_cache, mask_box_backend):
        from .infinihand import infer
        from .infinihand.runtime.execution import inference_optimizations

        if branch_precision not in ("fp32", "bf16"):
            raise ValueError("branch_precision must be fp32 or bf16")
        previous = (infer.BENCH_BRANCH_BF16, torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32)
        try:
            infer.BENCH_BRANCH_BF16 = branch_precision == "bf16"
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            with inference_optimizations(self.model, infer, fused_rope=fused_rope,
                                        compact_cache=compact_kv_cache,
                                        mask_backend=mask_box_backend):
                if cuda_graphs:
                    infer.prepare_cuda_graphs(self.model)
                yield infer
        finally:
            (infer.BENCH_BRANCH_BF16, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32) = previous

    def predict(self, record, *, postprocess=False, timestamps=None,
                branch_precision="fp32", cuda_graphs=False, fused_rope=False,
                compact_kv_cache=False, mask_box_backend="torch",
                postprocess_storage="cpu", postprocess_lambda=0.5):
        """Predict a clip without GT boxes/masks/joints.

        Defaults preserve the original raw baseline. Refinement and acceleration
        are explicit options. This stateful, two-stream adapter is not reentrant.
        """
        frames = record["_frames_rgb"]
        n, h, w = len(frames), record["height"], record["width"]
        if n == 0:
            raise ValueError("Require at least one frame")
        if postprocess_storage not in ("cpu", "cpu_async", "cuda"):
            raise ValueError("Unknown postprocess_storage")
        if postprocess:
            timestamps = np.asarray(timestamps, dtype=np.float64)
            if (timestamps.shape != (n,) or not np.isfinite(timestamps).all()
                    or np.any(np.diff(timestamps) <= 0)):
                raise ValueError("Refinement requires increasing timestamps [T]")
        height, width = self.model.cfg.image_height, self.model.cfg.image_width
        k = np.asarray(record["intrinsics_per_frame"], np.float32).copy()
        k[:, 0] *= width / w
        k[:, 1] *= height / h
        parts, reports = [], []
        runtime = dict(branch_precision=branch_precision, cuda_graphs=cuda_graphs,
                       fused_rope=fused_rope, compact_kv_cache=compact_kv_cache,
                       mask_box_backend=mask_box_backend)
        with self._runtime(**runtime) as infer:
            for actual_rows in np.array_split(np.arange(n), -(-n // MAX_FRAMES)):
                # Pad only the network input. Repeats must not affect beta or SG5.
                rows = np.r_[actual_rows, np.full(max(0, ANCHOR_FRAMES-len(actual_rows)), actual_rows[-1])]
                images = np.stack([cv2.resize(frames[i], (width, height)) for i in rows])
                images = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0).to(self.device)
                intrinsics = torch.from_numpy(k[rows]).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    out, stats, timing = infer.run_dual_stream(
                        self.model, images, intrinsics, bbox_source="mask",
                        mask_threshold=self.mask_threshold, capture_postprocess=postprocess,
                        postprocess_storage=postprocess_storage)
                infer.validate_contract(stats, len(rows))
                count = len(actual_rows)
                out = {key: value[:count] for key, value in out.items()}
                post_report = None
                if postprocess:
                    from .infinihand.postprocess import refine_hands, refine_hands_tensors
                    masks = out.pop("postprocess_masks")
                    out["valid"] = out["presence_mask"] > 0.5
                    if postprocess_storage == "cuda":
                        arrays, post_report = refine_hands_tensors(
                            out, intrinsics[0, :count], masks, timestamps[actual_rows],
                            self.model.cfg.mano_model_path, lambda_fp=postprocess_lambda,
                            return_tensors=False)
                    else:
                        arrays = self._numpy(out)
                        arrays, post_report = refine_hands(
                            arrays, k[actual_rows], masks.cpu().numpy(), timestamps[actual_rows],
                            self.model.cfg.mano_model_path, lambda_fp=postprocess_lambda,
                            device=self.device)
                else:
                    arrays = self._numpy(out)
                # Mask/LSP auxiliaries are intentionally absent from benchmark exports.
                keep = ("camera_joints", "camera_vertices", "camera_global_orient",
                        "hand_pose", "betas", "camera_translation", "presence_mask")
                parts.append({key: arrays[key] for key in keep})
                reports.append(dict(first_frame=int(actual_rows[0]), frames=count,
                                    cache=stats, timing=timing, postprocess=post_report))
                del out, images, intrinsics
                torch.cuda.empty_cache()
        arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError("Non-finite model output: " + key)
        self.last_report = dict(runtime=runtime, postprocess=postprocess,
                                postprocess_storage=postprocess_storage,
                                postprocess_lambda=postprocess_lambda, chunks=reports)
        return dict(
            joints_cam=arrays["camera_joints"][:, SLOTS],
            vertices_cam=arrays["camera_vertices"][:, SLOTS],
            global_orient=arrays["camera_global_orient"][:, SLOTS, None],
            hand_pose=arrays["hand_pose"][:, SLOTS],
            betas=arrays["betas"][:, SLOTS],
            cam_trans=arrays["camera_translation"][:, SLOTS],
            presence=arrays["presence_mask"][:, SLOTS] > 0.5,
        )

    @staticmethod
    def _numpy(values):
        return {key: (value.float() if value.is_floating_point() else value).cpu().numpy()
                for key, value in values.items()}

    def inference_stream(self, inputs, *, joint_only=False):
        raise NotImplementedError("infinihand needs at least four frames: use inference_offline")

    def inference_offline(self, inputs, *, joint_only=False, export_device="cpu", **runtime):
        torch.cuda.set_device(self.device)
        started = time.perf_counter()
        with InferenceInput(inputs, mode="offline") as (record, camera):
            # Camera.fps describes the materialized clip; source_frame_ids are export IDs.
            timestamps = np.arange(len(camera), dtype=np.float64) / camera.fps
            raw = self.predict(record, timestamps=timestamps, **runtime)
        from .canonical_export import export_canonical
        prediction = export_canonical(raw, camera, resource_path(self.assets, "mano"),
                                      device=self.device if export_device == "cuda" else export_device)
        self.last_report.update(export_device=export_device,
                                pipeline_seconds=time.perf_counter()-started,
                                frames=len(camera))
        if joint_only:
            prediction["mano"] = None
        return prediction

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

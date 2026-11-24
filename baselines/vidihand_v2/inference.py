"""ViDiHand2 behind the EgoHandBench baseline interface. model.py, mano.py, io.py, config.yaml and archs/ are the model of the ViDiHand
release (https://github.com/NTUYWANG103/ViDiHand); this file adapts the inputs, the outputs and the weights."""

import threading

import numpy as np
import torch

from baselines.vidihand_v2.model import load_model
from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, weight_path

LOCK = threading.Lock()  # one model is built at a time: the build switches the process-wide TF32 flags and redirects stdout


def cuda_device(device):
    """The adapter's CUDA device (None / "auto" = the current one)."""
    device = torch.device("cuda" if device in (None, "auto") else device)
    if device.type != "cuda":
        raise ValueError("vidihand_v2 runs on CUDA only")
    return torch.device("cuda", torch.cuda.current_device()) if device.index is None else device


class VidiHandV2Adapter:
    """Offline only: ViDiHand2 on complete clips, with its weights from the benchmark's vidihand_v2 checkpoint group."""

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="egohandbench"):
        if batch_size is not None:
            raise ValueError("vidihand_v2 uses native latent windows, not a batch_size argument")
        self.device = cuda_device(device)
        self.seed = seed
        torch.cuda.set_device(self.device)
        assets = checkpoint_dir("vidihand_v2", checkpoints, version=version)
        weights = dict(vdm=resource_path(assets, "vidihand_v2.vdm"), projector=weight_path(assets, "vidihand_v2"), mano=resource_path(assets, "mano"))
        with LOCK:
            self.model = load_model(weights, self.device)
        self.reset()

    def inference_stream(self, inputs, *, joint_only=False, mode="autoregressive", decode="clip"):
        """Not supported: ViDiHand2 needs the complete clip."""
        raise NotImplementedError("vidihand_v2 requires a complete clip")

    def inference_offline(self, inputs, *, joint_only=False, mode="autoregressive", decode="clip"):
        """Per-frame joints and (unless `joint_only`) their MANO fit, in the input camera."""
        torch.cuda.set_device(self.device)
        with InferenceInput(inputs, mode="offline") as (record, camera):
            sample = dict(video=record["_frames_rgb"], K=camera.K, fps=float(camera.fps), source_frame_ids=camera.source_frame_ids)
            prediction = self.model(sample, mode=mode, decode=decode, fit_mano=not joint_only)
        frames = np.asarray(camera.source_frame_ids)
        mano = None if joint_only else dict(prediction["mano"], source_frame_ids=frames.copy())
        return dict(mano=mano, joint=dict(prediction["joint"], source_frame_ids=frames.copy()))

    def reset(self):
        """Re-seeds the torch generators with the adapter's seed."""
        if self.seed is not None:
            torch.random.default_generator.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

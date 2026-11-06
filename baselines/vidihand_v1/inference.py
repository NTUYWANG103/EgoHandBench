"""ViDiHand v1 offline inference: Wan2.1 rollout features decoded to per-frame MANO."""

import math

import numpy as np
import torch
from safetensors.torch import load_file

from baselines.vidihand_v1.archs.projector import ControlProjector
from baselines.vidihand_v1.archs.dit import ControlDiT, WanVAE21
from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, weight_path
from utils.mano import ManoModel

WINDOW = 81             # windows start at 0, 81, 162, ...; the last one repeats its final frame
STEPS, SHIFT = 25, 5.0  # Euler ladder of the feature rollout
CAPTURE_STEP = 17       # the forward at this ladder step is tapped
LAYER = 15              # DiT block whose output tokens are the features
CLAIM = 0.5             # presence probability above which a hand is predicted
VERSION = "egohandbench"
MIN_TAIL = 17           # a shorter last window is replaced by the clip's final 81 frames


def cuda_device(device, batch_size):
    if batch_size is not None:
        raise ValueError("vidihand_v1 uses native 81-frame windows, not a batch_size argument")
    device = torch.device("cuda" if device in (None, "auto") else device)
    if device.type != "cuda":
        raise ValueError("vidihand_v1 runs on CUDA only")
    return torch.device("cuda", torch.cuda.current_device()) if device.index is None else device


class VidiHandV1Adapter:
    claim = CLAIM

    def __init__(self, checkpoints=None, device=None, seed=42, batch_size=None, *, version=VERSION):
        self.device = cuda_device(device, batch_size)
        self.seed = seed
        assets = checkpoint_dir("vidihand_v1", checkpoints, version=version)
        vdm = load_file(str(resource_path(assets, "vidihand_v1.vdm")))
        part = lambda prefix: {k[len(prefix):]: v for k, v in vdm.items() if k.startswith(prefix)}
        with torch.device("meta"):
            self.dit = ControlDiT()
        self.dit.load_state_dict(part("dit."), strict=True, assign=True)
        self.dit = self.dit.eval().to(self.device)  # parameters keep requires_grad: kernel choices depend on it
        self.vae = WanVAE21()
        self.vae.load_state_dict(part("vae."), strict=True)
        self.vae = self.vae.eval().to(self.device, torch.bfloat16)
        self.prompt = [vdm["prompt_embedding"].to(self.device)]
        self.projector = ControlProjector(ManoModel(resource_path(assets, "mano"), device=self.device))
        self.projector.load_state_dict(load_file(str(weight_path(assets, "vidihand_v1"))), strict=True)
        self.projector = self.projector.eval().to(self.device)
        train = torch.from_numpy(np.linspace(1, 1000, 1000, dtype=np.float32)[::-1].copy()) / 1000
        train = SHIFT * train / (1 + (SHIFT - 1) * train)
        u = np.linspace(train[0].item() * 1000, train[-1].item() * 1000, STEPS) / 1000
        sigmas = torch.from_numpy(SHIFT * u / (1 + (SHIFT - 1) * u)).to(self.device, torch.float32)
        self.timesteps = sigmas * 1000
        self.sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        self.reset()

    @torch.no_grad()
    def _features(self, frames, generator, vdm=False):
        pixels = (2.0 * (torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0) / 255) - 1.0).to(self.device, torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            control = self.vae.encode(pixels)
            inactive = self.vae.encode(torch.zeros_like(pixels))
        _, _, lf, lh, lw = control.shape
        seq_len = math.ceil(lh * lw / 4 * lf)
        context = [torch.cat([inactive[0], control[0], control.new_ones(64, lf, lh, lw)])]
        x = torch.randn(control.shape, generator=generator, device=self.device, dtype=torch.bfloat16)

        def forward(x, k, tap=None):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.dit(x, self.timesteps[k:k + 1], self.prompt, seq_len, context, tap=tap)

        for k in range(CAPTURE_STEP):
            x = (x.float() + (self.sigmas[k + 1] - self.sigmas[k]) * forward(x, k)).to(torch.bfloat16)
        tokens = forward(x, CAPTURE_STEP, LAYER)
        tokens = tokens[0, : lf * (lh // 2) * (lw // 2)].reshape(lf, lh // 2, lw // 2, -1).to(torch.bfloat16)
        if not vdm:
            return tokens, None
        for k in range(CAPTURE_STEP, STEPS):  # finish the rollout to sigma 0 and decode the generated video
            x = (x.float() + (self.sigmas[k + 1] - self.sigmas[k]) * forward(x, k)).to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = self.vae.decode(x).float().clamp(-1, 1)[0].permute(1, 2, 3, 0)
        return tokens, ((video + 1) * 127.5).round().to(torch.uint8)

    @torch.no_grad()
    def predict(self, video, K, clip, vdm=False):
        n, height, width = video.shape[:3]
        if height % 16 or width % 16:
            raise ValueError("vidihand_v1 needs an RGB canvas whose sides are multiples of 16")
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        generator = None if self.seed is None else torch.Generator(self.device).manual_seed(self.seed)
        starts = list(range(0, n, WINDOW))
        if len(starts) > 1 and n - starts[-1] < MIN_TAIL:
            starts[-1] = n - WINDOW
        rows, covered = [], 0
        for start in starts:
            ids = np.minimum(np.arange(start, start + WINDOW), n - 1)
            feat, generated = self._features(video[ids], generator, vdm)
            out = self.projector(feat, K[ids], (width, height))
            keep = slice(covered - start, min(WINDOW, n - start))
            row = {key: value[keep].cpu().numpy() for key, value in out.items()}
            if vdm:
                row["vdm"] = generated[keep].cpu().numpy()
            rows.append(row)
            covered = start + keep.stop
        raw = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
        raw["joints3d"] = self.projector.mano.mano2joint(dict(
            global_orient=raw["go"][:, :, None], hand_pose=raw["hp"], betas=raw["betas"], cam_trans=raw["cam_trans"],
            valid=raw["presence"] > self.claim))
        return raw

    def inference_stream(self, inputs, *, joint_only=False):
        raise NotImplementedError("vidihand_v1 requires a complete clip")

    def inference_offline(self, inputs, *, joint_only=False, vdm=False):
        metadata = inputs["camera"].metadata
        clip = (f"{metadata['dataset']}/{metadata['clip']}" if "dataset" in metadata and "clip" in metadata
                else inputs.get("name", "clip"))
        with InferenceInput(inputs, mode="offline") as (record, camera):
            raw = self.predict(record["_frames_rgb"], camera.K, clip, vdm)
        valid = raw["presence"] > self.claim
        hide = lambda value: np.where(valid.reshape(valid.shape + (1,) * (value.ndim - 2)), value, np.nan).astype(np.float32)
        frames = np.asarray(camera.source_frame_ids)
        joints = dict(joints3d=hide(raw["joints3d"]), valid=valid, source_frame_ids=frames.copy())
        mano = None if joint_only else dict(
            global_orient=hide(raw["go"][:, :, None]), hand_pose=hide(raw["hp"]), betas=hide(raw["betas"]),
            cam_trans=hide(raw["cam_trans"]), valid=valid.copy(), source_frame_ids=frames.copy())
        return dict(mano=mano, joint=joints, **({"vdm": raw["vdm"]} if vdm else {}))

    def reset(self):
        if self.seed is not None:
            torch.random.default_generator.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

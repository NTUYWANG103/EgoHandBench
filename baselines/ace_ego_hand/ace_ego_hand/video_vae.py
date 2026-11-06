"""Video decode and Wan-VAE encode — the pixel side of inference.

``decode_video`` reads frames with OpenCV, ``load_vae``/``encode`` run the
frozen causal VAE from the Wan2.2-Fun-5B-Control release (see the top-level
README).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
# Directory holding Wan2.2_VAE.pth; the adapter sets ACE_EGO_HAND_VAE_ROOT before import.
MODEL_ROOT = os.environ.get("ACE_EGO_HAND_VAE_ROOT")
VIDEOX_CFG = ROOT / "third_party" / "config" / "wan2.2" / "wan_civitai_5b.yaml"


def decode_video(path: str, n_frames: int, resize_w: int = 0,
                 resize_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Decode up to n_frames -> (F, 3, H, W) float32 in [-1, 1] (RGB).

    ``resize_hw`` rescales to an exact (W, H); ``resize_w`` rescales the width
    only. The caller is responsible for scaling K to match.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n_frames:
        ok, f = cap.read()
        if not ok:
            break
        if resize_hw and (f.shape[1] != resize_hw[0] or f.shape[0] != resize_hw[1]):
            f = cv2.resize(f, resize_hw, interpolation=cv2.INTER_AREA)
        elif resize_w and f.shape[1] != resize_w:
            f = cv2.resize(f, (resize_w, f.shape[0]), interpolation=cv2.INTER_AREA)
        frames.append(f[:, :, ::-1].copy())  # BGR -> RGB
    cap.release()
    if len(frames) < n_frames:
        raise RuntimeError(f"{path}: decoded {len(frames)} < expected {n_frames}")
    a = np.stack(frames).astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


def frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def load_vae(device: torch.device):
    from baselines.ace_ego_hand.third_party.videox_fun.models import AutoencoderKLWan3_8
    if not MODEL_ROOT:
        raise RuntimeError("Set ACE_EGO_HAND_VAE_ROOT to the directory holding Wan2.2_VAE.pth")
    config = OmegaConf.load(str(VIDEOX_CFG))
    vae = AutoencoderKLWan3_8.from_pretrained(
        os.path.join(str(MODEL_ROOT), config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(device, torch.bfloat16)
    vae.eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()


def encode(vae, pixels: torch.Tensor, device) -> torch.Tensor:
    """(F,3,H,W) [-1,1] -> (C_lat, F_lat, H/16, W/16) f16 cpu."""
    x = pixels.permute(1, 0, 2, 3).unsqueeze(0).to(device, torch.bfloat16)  # (1,3,F,H,W)
    enc = vae.encode(x)
    dist = enc[0] if isinstance(enc, (tuple, list)) else getattr(enc, "latent_dist", enc)
    lat = dist.mode() if hasattr(dist, "mode") else dist.sample()
    return lat[0].to(torch.float16).cpu()


# RHD masters store the NATIVE RHD joint order: wrist=0, then per finger
# [tip -> palm] (README "1-4: thumb [tip to palm], 5-8: index, ...").  Our pred
# convention is OpenPose-21 (wrist, then per finger root -> tip), so reorder at
# cache-build time; the map is an involution (RHD->OP == OP->RHD).
_RHD_TO_OPENPOSE21 = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9,
                      16, 15, 14, 13, 20, 19, 18, 17]

"""Clip-level inference: VDM control latents to native hand predictions.

The selected K-given pipeline uses tiled 22-latent windows, covering each
81-frame segment in the released training window.
"""
from __future__ import annotations

import numpy as np
import torch

DUMP_KEYS = ("global_orient", "hand_pose", "betas", "cam_trans", "direct_joints2d",
             "direct_joints_cam", "direct_wrist_cam", "exists_2d", "exists_3d")


@torch.no_grad()
def predict_video(model, tap: int, ctrl: torch.Tensor, intr: dict, mode: str,
                  device, tile_w: int = 22) -> dict[str, np.ndarray]:
    """Return native predictions in the original pixel-frame order."""
    f_lat = ctrl.shape[1]
    n_px = 4 * (f_lat - 1) + 1

    if intr.get("per_frame") is not None and len(intr["per_frame"]) != n_px:
        raise ValueError("ACE camera and decoded video timelines differ")

    def run(lat_slice: torch.Tensor, n_video: int, start: int = 0) -> dict:
        ls = lat_slice.unsqueeze(0).float().to(device)
        segment_intr = dict(intr)
        if intr.get("per_frame") is not None:
            segment_intr["per_frame"] = intr["per_frame"][start:start + n_video]
        out = model.net.forward_emode(ls, n_video, [segment_intr])
        return out["preds"][tap][0]

    if mode == "full":
        p = run(ctrl, n_px)
        return {k: p[k].cpu().numpy() for k in DUMP_KEYS}

    # Tiled windows write outputs to absolute pixel-frame indices 4*a+i.
    w = int(tile_w)
    acc = {k: None for k in DUMP_KEYS}
    n_seg = n_px // 81
    anchors = [(min((81 * i) // 4, f_lat - w) if f_lat >= w else 0, None)
               for i in range(max(n_seg, 1))]
    covered = 4 * anchors[-1][0] + 4 * (min(w, f_lat) - 1) + 1
    if covered < n_px:
        # End-anchor the last window, writing only its previously uncovered gap.
        anchors.append((f_lat - w, covered))
    for a, wlo in anchors:
        ww = min(w, f_lat)
        nv = 4 * (ww - 1) + 1
        p = run(ctrl[:, a:a + ww], nv, 4 * a)
        for k in DUMP_KEYS:
            arr = p[k].cpu().numpy()
            if acc[k] is None:
                shape = (n_px,) + arr.shape[1:]
                acc[k] = np.zeros(shape, arr.dtype)
            lo, hi = 4 * a, min(4 * a + nv, n_px)
            if wlo is not None:
                lo = max(lo, wlo)
            acc[k][lo:hi] = arr[lo - 4 * a: hi - 4 * a]
    return acc

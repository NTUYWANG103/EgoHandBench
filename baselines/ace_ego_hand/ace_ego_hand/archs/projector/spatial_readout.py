"""Spatial-aware joint readout — addresses the pooling bottleneck.

The default projector regresses joint coords from a single *pooled* slot token,
which discards spatial detail. This module instead lets per-joint queries cross-attend to the *pre-pool* spatial feature map
`x` (B*F_lat, H*W, hidden). The attention map over H*W IS a per-joint heatmap, so
soft-argmax gives a sub-pixel 2D location; the attended feature regresses 3D.

Readout runs at latent-frame resolution (F_lat, where `x` lives) and the output
*coordinates* are linearly interpolated to video-frame resolution (coords
interpolate cleanly; heatmaps would not). This sidesteps surgery on the temporal
encoder — the existing slot/temporal pipeline is untouched.

Self-contained and config-gated at the call site; the default model path does
not use it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialJointReadout(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_slots: int,
        num_joints: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.num_joints = int(num_joints)
        self.n_q = self.num_slots * self.num_joints

        self.joint_query = nn.Parameter(
            torch.randn(1, self.n_q, self.hidden_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            self.hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.q_norm = nn.LayerNorm(self.hidden_dim)
        head_hidden = max(128, self.hidden_dim // 2)
        # 3D root-relative joints regressed from the attended per-joint feature.
        self.head_3d = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 3),
        )

    def forward(
        self,
        x: torch.Tensor,        # (B*F_lat, H*W, hidden) — pre-pool spatial features
        *,
        B: int,
        F_lat: int,
        H: int,
        W: int,
        n_video_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (joints2d, joints_rootrel):
        joints2d:      (B, n_video_frames, num_slots, num_joints, 2) in [0,1]
        joints_rootrel:(B, n_video_frames, num_slots, num_joints, 3) root-relative
        """
        BF = x.shape[0]
        assert BF == B * F_lat, f"x batch {BF} != B*F_lat {B*F_lat}"
        q = self.q_norm(self.joint_query).expand(BF, -1, -1)        # (BF, n_q, hid)
        # attn_w: (BF, n_q, H*W), softmax over the spatial positions -> heatmap.
        feat, attn_w = self.cross_attn(
            q, x, x, need_weights=True, average_attn_weights=True
        )

        # --- 2D via soft-argmax over the attention heatmap ---
        hm = attn_w.reshape(BF, self.n_q, H, W)                     # (BF, n_q, H, W)
        # normalized grid coords in [0,1]
        us = torch.linspace(0.0, 1.0, W, device=x.device, dtype=x.dtype)
        vs = torch.linspace(0.0, 1.0, H, device=x.device, dtype=x.dtype)
        wsum = hm.sum(dim=(-1, -2)).clamp(min=1e-6)                 # (BF, n_q)
        u = (hm.sum(dim=-2) * us).sum(dim=-1) / wsum                # (BF, n_q)
        v = (hm.sum(dim=-1) * vs).sum(dim=-1) / wsum                # (BF, n_q)
        joints2d_lat = torch.stack([u, v], dim=-1)                  # (BF, n_q, 2)

        # --- 3D rootrel from the attended feature ---
        rootrel_lat = self.head_3d(feat)                           # (BF, n_q, 3)

        # reshape to (B, F_lat, num_slots, num_joints, C)
        joints2d_lat = joints2d_lat.reshape(B, F_lat, self.num_slots, self.num_joints, 2)
        rootrel_lat = rootrel_lat.reshape(B, F_lat, self.num_slots, self.num_joints, 3)
        # make rootrel actually root-relative (subtract joint-0)
        rootrel_lat = rootrel_lat - rootrel_lat[:, :, :, 0:1, :]

        joints2d = self._interp_t(joints2d_lat, n_video_frames)
        joints_rootrel = self._interp_t(rootrel_lat, n_video_frames)
        return joints2d, joints_rootrel

    @staticmethod
    def _interp_t(coords: torch.Tensor, n_video_frames: int) -> torch.Tensor:
        # coords: (B, F_lat, S, J, C) -> linearly interpolate F_lat -> n_video_frames
        B, F_lat, S, J, C = coords.shape
        if F_lat == n_video_frames:
            return coords
        t = coords.permute(0, 2, 3, 4, 1).reshape(B * S * J * C, 1, F_lat)
        t = F.interpolate(t, size=n_video_frames, mode="linear", align_corners=True)
        return t.reshape(B, S, J, C, n_video_frames).permute(0, 4, 1, 2, 3).contiguous()

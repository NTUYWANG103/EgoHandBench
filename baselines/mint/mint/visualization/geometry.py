#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]


MANO_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),
    (0, 5),  (5, 6),  (6, 7),  (7, 8),
    (0, 9),  (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def rot6d_to_mat(r6: np.ndarray) -> np.ndarray:
    """Internal helper."""
    a0 = r6[..., 0:3]
    a1 = r6[..., 3:6]
    b0 = a0 / np.linalg.norm(a0, axis=-1, keepdims=True)
    a1p = a1 - np.sum(b0 * a1, axis=-1, keepdims=True) * b0
    b1 = a1p / np.linalg.norm(a1p, axis=-1, keepdims=True)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=-2)


def mat_to_6d(m: np.ndarray) -> np.ndarray:
    """Internal helper."""
    return m[..., :2, :].reshape(*m.shape[:-2], 6)


def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """Internal helper."""
    from scipy.spatial.transform import Rotation as _Rot
    out_shape = R.shape[:-2] + (3,)
    flat = R.reshape(-1, 3, 3).astype(np.float64)
    aa = _Rot.from_matrix(flat).as_rotvec().astype(np.float32)
    return aa.reshape(out_shape)


def apply_left_hand_pose_flip_111(hand_pose_aa: np.ndarray) -> np.ndarray:
    """Internal helper."""
    out = np.array(hand_pose_aa, copy=True)
    out[..., 1::3] *= -1.0
    out[..., 2::3] *= -1.0
    return out


def project(points_world: np.ndarray, cam_c2w: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam_w2c = np.linalg.inv(cam_c2w)
    R = cam_w2c[:3, :3]
    t = cam_w2c[:3, 3]
    p_cam = (R @ points_world.T + t[:, None]).T   # (N, 3)
    z = p_cam[:, 2].astype(np.float64)
    eps = 1e-3
    valid = z > eps
    zc = np.where(valid, z, 1.0)
    u = K[0, 0] * p_cam[:, 0] / zc + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / zc + K[1, 2]
    uv = np.stack([u, v], axis=1)
    bad = ~valid | ~np.isfinite(uv).all(axis=1)
    uv[bad] = np.nan
    depth = np.where(bad, -np.inf, z)
    return uv, depth


def hand6d_cam_to_world(hands: dict, cam_c2w: np.ndarray) -> dict:
    c2w = np.asarray(cam_c2w, dtype=np.float32)
    Rcw, tcw = c2w[:, :3, :3], c2w[:, :3, 3]                # (T,3,3),(T,3)
    out = {}
    for side in ("left", "right"):
        h = hands[side]
        o_cam = rot6d_to_mat(np.asarray(h["orient6d"], np.float32))          # (T,3,3)
        o_world = np.einsum("tij,tjk->tik", Rcw, o_cam)
        t_world = np.einsum("tij,tj->ti", Rcw, np.asarray(h["transl_cam"], np.float32)) + tcw
        out[side] = {
            "transl_cam": t_world.astype(np.float32),
            "orient6d": mat_to_6d(o_world).astype(np.float32),
            "pose6d": np.asarray(h["pose6d"], np.float32),
            "betas": np.asarray(h["betas"], np.float32),
        }
    return out


def K_from_fov(fov_rad: tuple, width: int, height: int) -> np.ndarray:
    """Internal helper."""
    fov_x, fov_y = float(fov_rad[0]), float(fov_rad[1])
    fx = (width / 2.0) / np.tan(fov_x / 2.0)
    fy = (height / 2.0) / np.tan(fov_y / 2.0)
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def decode_camera_pose_enc(pose_enc: np.ndarray, height: int, width: int,
                           fov_mean: bool = False) -> tuple[np.ndarray, np.ndarray]:
    import torch
    ling = REPO_DIR / "model_train" / "_vendor"
    from baselines.mint.model_train._vendor.lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    pe = np.asarray(pose_enc, dtype=np.float32).reshape(-1, 9)
    T = pe.shape[0]
    extr, intr = pose_encoding_to_extri_intri(torch.from_numpy(pe)[None], image_size_hw=(height, width))
    extr = extr[0].numpy().astype(np.float64)   # (T,3,4) world->cam (OpenCV)
    intr = intr[0].numpy().astype(np.float64)
    extr44 = np.tile(np.eye(4), (T, 1, 1))
    extr44[:, :3, :4] = extr
    cam_c2w = np.linalg.inv(extr44)             # (T,4,4) cam->world
    K = intr.mean(0) if fov_mean else intr[0]
    return cam_c2w, K

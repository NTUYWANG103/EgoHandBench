#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import threading
import warnings
from pathlib import Path

import numpy as np

from . import geometry as geom

REPO_DIR = Path(__file__).resolve().parents[2]
_MANO_DIR = REPO_DIR / 'assets' / 'mano'
_MANO_RIGHT_DIR = _MANO_DIR / 'mano_right'
_MANO_LEFT_DIR = _MANO_DIR / 'mano_left'
_HAWOR_DATA = REPO_DIR / 'third_party' / 'HaWoR' / '_DATA'

if not (_MANO_RIGHT_DIR / 'MANO_RIGHT.pkl').is_file():
    _MANO_RIGHT_DIR = _HAWOR_DATA / 'data' / 'mano'
if not (_MANO_LEFT_DIR / 'MANO_LEFT.pkl').is_file():
    _MANO_LEFT_DIR = _HAWOR_DATA / 'data_left' / 'mano_left'

_CACHE_LOCK = threading.Lock()
_DEVICE = None
_MANO_CLASS = None
_MANO_MODEL_CACHE: dict = {}


def ensure_mano_weights() -> None:
    """Raise a clear error when the separately licensed MANO files are absent."""
    if not (_MANO_RIGHT_DIR / 'MANO_RIGHT.pkl').exists() or not (_MANO_LEFT_DIR / 'MANO_LEFT.pkl').exists():
        raise FileNotFoundError(
            "MANO model files are required for mesh rendering. Expected:\n"
            f"  {_MANO_RIGHT_DIR / 'MANO_RIGHT.pkl'}\n"
            f"  {_MANO_LEFT_DIR / 'MANO_LEFT.pkl'}\n"
            "Download them from the official MANO website and accept its license."
        )


def _get_device():
    global _DEVICE
    if _DEVICE is None:
        import torch
        with _CACHE_LOCK:
            if _DEVICE is None:
                _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f'[mano] inference device: {_DEVICE}')
    return _DEVICE


def _aa_to_rotmat(theta):
    """Internal helper."""
    import torch
    norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return _quat_to_rotmat(quat)


def _quat_to_rotmat(quat):
    """Internal helper."""
    import torch
    norm_quat = quat / quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                        2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                        2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2],
                       dim=1).view(B, 3, 3)


def _get_mano_class():
    """Internal helper."""
    global _MANO_CLASS
    if _MANO_CLASS is not None:
        return _MANO_CLASS

    import torch
    import smplx
    from smplx.utils import to_tensor
    from smplx.vertex_ids import vertex_ids

    with _CACHE_LOCK:
        if _MANO_CLASS is not None:
            return _MANO_CLASS

        import contextlib
        import io

        class MANO(smplx.MANOLayer):
            def __init__(self, *args, **kwargs):


                with contextlib.redirect_stdout(io.StringIO()):
                    super().__init__(*args, **kwargs)
                mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
                self.register_buffer('extra_joints_idxs',
                                     to_tensor(list(vertex_ids['mano'].values()), dtype=torch.long))
                self.register_buffer('joint_map', torch.tensor(mano_to_openpose, dtype=torch.long))

            def forward(self, *args, **kwargs):
                mano_output = super().forward(*args, **kwargs)
                extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
                joints = torch.cat([mano_output.joints, extra_joints], dim=1)
                joints = joints[:, self.joint_map, :]
                mano_output.joints = joints
                return mano_output

        _MANO_CLASS = MANO
        return MANO


def _mano_cfg(is_right: bool) -> dict:
    if is_right:
        return dict(data_dir=str(_MANO_DIR) + '/', model_path=str(_MANO_RIGHT_DIR),
                    gender='neutral', num_hand_joints=15, create_body_pose=False)
    return dict(data_dir=str(_MANO_DIR) + '/', model_path=str(_MANO_LEFT_DIR),
                gender='neutral', num_hand_joints=15, create_body_pose=False, is_rhand=False)


def _get_mano_model(is_right: bool):
    """Reuse the immutable MANO layer for both J0 and full mesh forward calls."""
    MANO = _get_mano_class()
    device = _get_device()
    selected_path = _MANO_RIGHT_DIR if is_right else _MANO_LEFT_DIR
    key = (str(Path(selected_path).resolve()), bool(is_right), str(device))
    if key not in _MANO_MODEL_CACHE:
        with _CACHE_LOCK:
            if key not in _MANO_MODEL_CACHE:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model = MANO(**_mano_cfg(is_right))
                if not is_right:
                    model.shapedirs[:, 0, :] *= -1
                _MANO_MODEL_CACHE[key] = model.to(device).eval()
    return _MANO_MODEL_CACHE[key]


def build_mano_faces() -> tuple[np.ndarray, np.ndarray]:
    """Internal helper."""
    MANO = _get_mano_class()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = MANO(**_mano_cfg(is_right=True))
    base = m.faces.copy()
    extra = np.array([
        [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
        [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
        [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
        [120, 108, 78], [78, 108, 79],
    ])
    faces_right = np.concatenate([base, extra], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]
    return faces_right, faces_left


def run_mano(trans: np.ndarray, rot: np.ndarray, hand_pose: np.ndarray,
             betas: np.ndarray, is_right: bool) -> tuple[np.ndarray, np.ndarray]:
    import torch
    device = _get_device()

    T = trans.shape[0]
    trans = np.nan_to_num(trans, nan=0.0)
    rot = np.nan_to_num(rot, nan=0.0)
    hand_pose = np.nan_to_num(hand_pose, nan=0.0)
    betas = np.nan_to_num(betas, nan=0.0)

    mano = _get_mano_model(is_right)

    with torch.no_grad():
        g_orient = _aa_to_rotmat(torch.from_numpy(rot).float().to(device).reshape(T, 3)).view(T, 1, 3, 3)
        h_pose = _aa_to_rotmat(torch.from_numpy(hand_pose).float().to(device).reshape(T * 15, 3)).view(T, 15, 3, 3)
        out = mano(global_orient=g_orient, hand_pose=h_pose,
                   betas=torch.from_numpy(betas).float().to(device),
                   transl=torch.from_numpy(trans).float().to(device), pose2rot=False)
        verts = out.vertices.detach().cpu().numpy().astype(np.float32)
        joints = out.joints.detach().cpu().numpy().astype(np.float32)
    return verts, joints


def compute_J0_canon(betas: np.ndarray, is_right: bool) -> np.ndarray:
    import torch
    device = _get_device()
    m = _get_mano_model(is_right)

    b = np.nan_to_num(np.asarray(betas, dtype=np.float32), nan=0.0).reshape(-1, 10)
    N = b.shape[0]
    with torch.no_grad():
        g = torch.eye(3, device=device).view(1, 1, 3, 3).expand(N, 1, 3, 3).contiguous()
        h = torch.eye(3, device=device).view(1, 1, 3, 3).expand(N, 15, 3, 3).contiguous()
        t = torch.zeros(N, 3, device=device)
        out = m(global_orient=g, hand_pose=h,
                betas=torch.from_numpy(b).float().to(device), transl=t, pose2rot=False)
        return out.joints[:, 0, :].detach().cpu().numpy().astype(np.float32)


def decode_hand_6d(transl_world: np.ndarray, orient6d: np.ndarray, pose6d: np.ndarray,
                   betas: np.ndarray, is_right: bool) -> dict:
    T = transl_world.shape[0]
    orient_mat = geom.rot6d_to_mat(orient6d.astype(np.float32).reshape(T, 6))           # (T,3,3)
    pose_mat = geom.rot6d_to_mat(pose6d.astype(np.float32).reshape(T, 15, 6))            # (T,15,3,3)
    rot = geom.rotmat_to_aa(orient_mat).reshape(T, 3)
    hand_pose = geom.rotmat_to_aa(pose_mat).reshape(T, 45)
    if not is_right:
        hand_pose = geom.apply_left_hand_pose_flip_111(hand_pose)
    trans = transl_world.astype(np.float32).reshape(T, 3) - compute_J0_canon(betas, is_right)
    return {'trans': trans, 'rot': rot, 'hand_pose': hand_pose, 'betas': betas.astype(np.float32).reshape(T, 10)}

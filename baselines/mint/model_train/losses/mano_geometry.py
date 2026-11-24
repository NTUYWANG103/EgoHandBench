#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Differentiable MANO-parameter to OpenPose-ordered 21-joint geometry."""
import contextlib
import io
from pathlib import Path

import torch

from baselines.mint.model_train.losses.rotation import rotation_6d_to_matrix


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANO_ROOT = _REPO_ROOT / "weights" / "mano"
_MANO_PATHS = {
    False: _MANO_ROOT / "mano_left" / "MANO_LEFT.pkl",
    True: _MANO_ROOT / "mano_right" / "MANO_RIGHT.pkl",
}
_MANO_CLASS = None
_LAYER_CACHE = {}
PER_HAND = 109
NUM_POSE_JOINTS = 15


def _mano_class():
    global _MANO_CLASS
    if _MANO_CLASS is not None:
        return _MANO_CLASS

    import smplx
    from smplx.utils import to_tensor
    from smplx.vertex_ids import vertex_ids

    class Mano21(smplx.MANOLayer):
        def __init__(self, *args, **kwargs):
            with contextlib.redirect_stdout(io.StringIO()):
                super().__init__(*args, **kwargs)
            mano_to_openpose = [
                0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
                10, 11, 12, 19, 7, 8, 9, 20,
            ]
            self.register_buffer(
                "extra_joints_idxs",
                to_tensor(list(vertex_ids["mano"].values()), dtype=torch.long),
            )
            self.register_buffer("joint_map", torch.tensor(mano_to_openpose, dtype=torch.long))

        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            tips = torch.index_select(output.vertices, 1, self.extra_joints_idxs)
            output.joints = torch.cat([output.joints, tips], dim=1)[:, self.joint_map, :]
            return output

    _MANO_CLASS = Mano21
    return Mano21


def _mano_layer(is_right: bool, device: torch.device):
    model_path = _MANO_PATHS[is_right]
    if not model_path.is_file():
        raise FileNotFoundError(f"MANO model is missing: {model_path}")
    key = (is_right, device.type, device.index)
    if key not in _LAYER_CACHE:
        Mano21 = _mano_class()
        kwargs = {
            "data_dir": str(_MANO_ROOT) + "/",
            "model_path": str(model_path.parent),
            "gender": "neutral",
            "num_hand_joints": 15,
            "create_body_pose": False,
        }
        if not is_right:
            kwargs["is_rhand"] = False
        layer = Mano21(**kwargs)
        if not is_right:
            with torch.no_grad():
                layer.shapedirs[:, 0, :] *= -1
        layer.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
        _LAYER_CACHE[key] = layer
    return _LAYER_CACHE[key]


def _left_pose_mirror(pose: torch.Tensor) -> torch.Tensor:
    """Apply the existing left-hand axis-angle [1,-1,-1] convention in matrix form."""
    diagonal = pose.new_tensor([1.0, -1.0, -1.0])
    transform = torch.diag(diagonal)
    return transform @ pose @ transform


def joints21_from_mano(
    global_orientation: torch.Tensor,
    hand_pose: torch.Tensor,
    betas: torch.Tensor,
    is_right: bool,
) -> torch.Tensor:
    """Return differentiable MANO joints [...,21,3] with zero MANO translation."""
    prefix = betas.shape[:-1]
    count = betas.numel() // betas.shape[-1]
    orientation = global_orientation.reshape(count, 1, 3, 3).float()
    pose = hand_pose.reshape(count, 15, 3, 3).float()
    shape = betas.reshape(count, 10).float()
    if not is_right:
        pose = _left_pose_mirror(pose)
    translation = torch.zeros(count, 3, device=shape.device, dtype=torch.float32)
    layer = _mano_layer(is_right, shape.device)

    autocast = (
        torch.autocast(device_type="cuda", enabled=False)
        if shape.device.type == "cuda"
        else contextlib.nullcontext()
    )
    with autocast:
        output = layer(
            global_orient=orientation,
            hand_pose=pose,
            betas=shape,
            transl=translation,
            pose2rot=False,
        )
    return output.joints.reshape(*prefix, 21, 3)


def _safe_rotation(rotation: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    return torch.where(valid[..., None, None], rotation, identity)


def _safe_vector(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return torch.where(valid[..., None], value, torch.zeros_like(value))


def joints21_from_hand_output(hand: torch.Tensor, valid_mask: torch.Tensor):
    """Decode hand[...,218] into camera-frame absolute and wrist-relative joints."""
    decoded = hand.float().reshape(*hand.shape[:-1], 2, PER_HAND)
    valid_mask = valid_mask.bool()
    orientation = rotation_6d_to_matrix(decoded[..., 3:9])
    pose = rotation_6d_to_matrix(
        decoded[..., 9:99].reshape(*decoded.shape[:-1], NUM_POSE_JOINTS, 6)
    )
    betas = decoded[..., 99:109]
    sides = []
    for side_index, is_right in enumerate((False, True)):
        valid = valid_mask[..., side_index]
        joints = joints21_from_mano(
            _safe_rotation(orientation[..., side_index, :, :], valid),
            _safe_rotation(pose[..., side_index, :, :, :], valid[..., None]),
            _safe_vector(betas[..., side_index, :], valid),
            is_right=is_right,
        )
        sides.append(joints)
    joints = torch.stack(sides, dim=-3)
    root_relative = joints - joints[..., :1, :]
    translation = _safe_vector(decoded[..., 0:3], valid_mask)
    return root_relative + translation[..., None, :], root_relative

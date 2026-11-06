#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared rotation conversions and geodesic distance for loss modules."""
import os
import sys

import torch
import torch.nn.functional as F

_MODEL_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDOR = os.path.join(_MODEL_TRAIN, "_vendor")

from baselines.mint.model_train._vendor.lingbot_map.utils.rotation import quat_to_mat  # noqa: E402

__all__ = [
    "axis_angle_to_matrix",
    "geodesic_angle",
    "matrix_to_rotation_6d",
    "quat_to_mat",
    "rotation_6d_health_metrics",
    "rotation_6d_to_matrix",
]


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert exponential-coordinate rotations [..., 3] to matrices."""
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f"axis_angle last dimension must be 3, got {axis_angle.shape}"
        )

    output_dtype = axis_angle.dtype
    value = (
        axis_angle.float()
        if output_dtype in (torch.float16, torch.bfloat16)
        else axis_angle
    )
    theta_sq = value.square().sum(dim=-1, keepdim=True)
    theta_sq_2 = theta_sq.square()
    small = theta_sq < 1e-6
    safe_theta = theta_sq.clamp_min(1e-12).sqrt()

    # Taylor branches keep both the value and gradient well-defined at omega=0.
    sin_over_theta = torch.where(
        small,
        1.0 - theta_sq / 6.0 + theta_sq_2 / 120.0,
        torch.sin(safe_theta) / safe_theta,
    )
    one_minus_cos_over_theta_sq = torch.where(
        small,
        0.5 - theta_sq / 24.0 + theta_sq_2 / 720.0,
        (1.0 - torch.cos(safe_theta)) / safe_theta.square(),
    )

    x, y, z = value.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*value.shape[:-1], 3, 3)
    identity = torch.eye(3, dtype=value.dtype, device=value.device).expand_as(skew)
    matrix = (
        identity
        + sin_over_theta[..., None] * skew
        + one_minus_cos_over_theta_sq[..., None] * (skew @ skew)
    )
    return matrix.to(dtype=output_dtype)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to the repository's row-major 6D convention."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must end in [3, 3], got {matrix.shape}")
    return matrix[..., :2, :].flatten(start_dim=-2)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert row-major 6D rotations to matrices with Gram-Schmidt."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def rotation_6d_health_metrics(
    d6: torch.Tensor, mask: torch.Tensor, prefix: str
) -> dict[str, torch.Tensor]:
    """Summarize Gram-Schmidt conditioning over valid predicted rotations."""
    with torch.no_grad():
        d6 = d6.detach().float()
        a1, a2 = d6[..., :3], d6[..., 3:]
        a1_norm = a1.norm(dim=-1)
        a2_norm = a2.norm(dim=-1)
        b1 = a1 / a1_norm.clamp_min(1e-12)[..., None]
        residual = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        residual_norm = residual.norm(dim=-1)
        rho = residual_norm / a2_norm.clamp_min(1e-12)

        valid = mask.to(device=d6.device).bool().expand_as(a1_norm)
        if not bool(valid.any()):
            return {}

        base = f"diag/rotation6d/{prefix}"
        metrics = {
            f"{base}/valid_count": valid.sum().float(),
        }
        for name, value in (
            ("a1_norm", a1_norm[valid]),
            ("ortho_residual", residual_norm[valid]),
            ("rho", rho[valid]),
        ):
            metrics[f"{base}/{name}_min"] = value.min()
            metrics[f"{base}/{name}_p01"] = torch.quantile(value, 0.01)
            metrics[f"{base}/{name}_p50"] = torch.quantile(value, 0.50)
            metrics[f"{base}/{name}_warning_frac"] = (value < 1e-2).float().mean()
            metrics[f"{base}/{name}_critical_frac"] = (value < 1e-3).float().mean()
        return metrics


def geodesic_angle(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the stable SO(3) geodesic angle in radians."""
    relative = pred @ target.transpose(-1, -2)
    cosine = (
        relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2] - 1.0
    ) / 2.0
    skew = relative - relative.transpose(-1, -2)
    sine = torch.sqrt((skew ** 2).sum(dim=(-1, -2)) + 1e-8) / (2.0 * (2.0 ** 0.5))
    return torch.atan2(sine, cosine)

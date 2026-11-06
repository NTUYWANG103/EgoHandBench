#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera-head losses and raw physical camera metrics in one place."""
import math

from baselines.mint.model_train.losses.fov_loss import fov_l1, fov_vel_l1
from baselines.mint.model_train.losses.normalization import batch_vector_scale, normalize_error
from baselines.mint.model_train.losses.rotation import geodesic_angle, quat_to_mat


def trans_l1(ctx):
    error = normalize_error(ctx["Tp"] - ctx["Tg"], ctx["position_std"])
    return error.abs().mean()


def rot_geo(ctx):
    return geodesic_angle(ctx["Rp"], ctx["Rg"]).mean()


def _camera_centers(translation, rotation):
    """Convert world-to-camera extrinsics into camera centers in world space."""
    return -(
        rotation.transpose(-1, -2) @ translation.unsqueeze(-1)
    ).squeeze(-1)


def _normalized_centered_trajectory(centers, eps=1.0e-6):
    centered = centers - centers[:, :1]
    rms_displacement = centered.square().sum(dim=-1).mean(dim=1).sqrt()
    return centered / (rms_displacement[:, None, None] + eps)


def trans_scale_aligned_l1(ctx):
    """Compare camera-center trajectory shape after positive scale alignment."""
    if ctx["Tp"].shape[1] < 2:
        return ctx["Tp"].sum() * 0.0
    pred_centers = _camera_centers(ctx["Tp"], ctx["Rp"])
    target_centers = _camera_centers(ctx["Tg"], ctx["Rg"])
    pred_normalized = _normalized_centered_trajectory(pred_centers)
    target_normalized = _normalized_centered_trajectory(target_centers)
    return (pred_normalized - target_normalized).abs().mean()


def _relative_translation(translation, rotation):
    relative_rotation = rotation[:, 1:] @ rotation[:, :-1].transpose(-1, -2)
    return translation[:, 1:] - (
        relative_rotation @ translation[:, :-1].unsqueeze(-1)
    ).squeeze(-1)


def trans_vel_l1(ctx):
    pred, target = ctx["Tp"], ctx["Tg"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    pred_velocity = _relative_translation(pred, ctx["Rp"])
    target_velocity = _relative_translation(target, ctx["Rg"])
    error = pred_velocity - target_velocity
    return normalize_error(error, ctx["velocity_std"]).abs().mean()


def rot_vel_geo(ctx):
    pred, target = ctx["Rp"], ctx["Rg"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    pred_relative = pred[:, 1:] @ pred[:, :-1].transpose(-1, -2)
    target_relative = target[:, 1:] @ target[:, :-1].transpose(-1, -2)
    return geodesic_angle(pred_relative, target_relative).mean()


def trans_acc_l1(ctx):
    pred, target = ctx["Tp"], ctx["Tg"]
    if pred.shape[1] < 3:
        return pred.sum() * 0.0
    pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    target_acc = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    return normalize_error(pred_acc - target_acc, ctx["acceleration_std"]).abs().mean()


_TERMS = {
    "trans_l1": trans_l1,
    "trans_scale_aligned_l1": trans_scale_aligned_l1,
    "rot_geo": rot_geo,
    "fov_l1": fov_l1,
    "fov_vel_l1": fov_vel_l1,
    "trans_vel_l1": trans_vel_l1,
    "rot_vel_geo": rot_vel_geo,
    "trans_acc_l1": trans_acc_l1,
}

_TRANSLATION_TERMS = {"trans_l1", "trans_vel_l1", "trans_acc_l1"}


def _raw_translation_metric(name, ctx):
    pred, target = ctx["Tp"], ctx["Tg"]
    if name == "trans_l1":
        return (pred - target).abs().mean(), "trans_l1_m"
    if name == "trans_vel_l1":
        if pred.shape[1] < 2:
            return pred.sum() * 0.0, "trans_vel_l1_m_per_frame"
        pred_velocity = _relative_translation(pred, ctx["Rp"])
        target_velocity = _relative_translation(target, ctx["Rg"])
        error = pred_velocity - target_velocity
        return error.abs().mean(), "trans_vel_l1_m_per_frame"
    if pred.shape[1] < 3:
        return pred.sum() * 0.0, "trans_acc_l1_m_per_frame2"
    pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    target_acc = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    return (pred_acc - target_acc).abs().mean(), "trans_acc_l1_m_per_frame2"


class CameraLoss:
    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.weight = float(cfg.get("weight", 1.0))
        self.terms = []
        for term_cfg in cfg.get("terms", []):
            name = term_cfg["name"]
            if name not in _TERMS:
                raise KeyError(f"[camera] unknown term {name!r}; available: {sorted(_TERMS)}")
            self.terms.append((name, _TERMS[name], float(term_cfg.get("weight", 1.0))))

    @staticmethod
    def _ctx(pred, batch):
        pose = pred["pose_enc"].float()
        target = batch["gt_pose_enc"].float()
        pred_quat, target_quat = pose[..., 3:7], target[..., 3:7]
        pred_rotation = quat_to_mat(pred_quat.reshape(-1, 4)).reshape(
            *pred_quat.shape[:-1], 3, 3
        )
        target_rotation = quat_to_mat(target_quat.reshape(-1, 4)).reshape(
            *target_quat.shape[:-1], 3, 3
        )
        legacy_std = batch_vector_scale(batch, "camera_trans_std", pose, width=3)
        position_std = batch_vector_scale(
            batch, "camera_trans_position_std", pose, width=3
        )
        velocity_std = batch_vector_scale(
            batch, "camera_trans_velocity_std", pose, width=3
        )
        acceleration_std = batch_vector_scale(
            batch, "camera_trans_acceleration_std", pose, width=3
        )
        position_std = legacy_std if position_std is None else position_std
        velocity_std = legacy_std if velocity_std is None else velocity_std
        acceleration_std = legacy_std if acceleration_std is None else acceleration_std
        return {
            "Tp": pose[..., :3],
            "Tg": target[..., :3],
            "position_std": position_std,
            "velocity_std": velocity_std,
            "acceleration_std": acceleration_std,
            "Rp": pred_rotation,
            "Rg": target_rotation,
            "fp": pose[..., 7:],
            "fg": target[..., 7:],
        }

    def __call__(self, pred, batch):
        ctx = self._ctx(pred, batch)
        total = pred["pose_enc"].sum() * 0.0
        logs = {}
        for name, function, weight in self.terms:
            value = function(ctx)
            total = total + weight * value
            scale = {
                "trans_l1": ctx["position_std"],
                "trans_vel_l1": ctx["velocity_std"],
                "trans_acc_l1": ctx["acceleration_std"],
            }.get(name)
            suffix = "_norm" if name in _TRANSLATION_TERMS and scale is not None else ""
            logs[f"loss/camera/{name}{suffix}"] = value.detach()

            if name in _TRANSLATION_TERMS:
                raw, metric_name = _raw_translation_metric(name, ctx)
                logs[f"metric/camera/{metric_name}"] = raw.detach()
            elif name in {"rot_geo", "rot_vel_geo"}:
                logs[f"metric/camera/{name}_deg"] = (value.detach() * 180.0 / math.pi)
            elif name in {"fov_l1", "fov_vel_l1"}:
                logs[f"metric/camera/{name}_deg"] = (value.detach() * 180.0 / math.pi)
        return self.weight * total, logs

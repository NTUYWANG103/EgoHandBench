#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-head consistency between camera and MANO predictions in world space."""
import math

import torch

from baselines.mint.model_train.losses.mano_geometry import PER_HAND, joints21_from_hand_output
from baselines.mint.model_train.losses.rotation import geodesic_angle, quat_to_mat, rotation_6d_to_matrix


def _masked_mean(value, mask):
    mask = mask.expand_as(value)
    value = torch.where(mask > 0, value, torch.zeros_like(value))
    return (value * mask).sum() / mask.sum().clamp(min=1.0)


def _camera_rotation(pose_enc):
    return quat_to_mat(pose_enc[..., 3:7].reshape(-1, 4)).reshape(
        *pose_enc.shape[:-1], 3, 3
    )


def _to_world(pose_enc, translation_cam, orientation_cam):
    translation = pose_enc[..., :3]
    rotation_t = _camera_rotation(pose_enc).transpose(-1, -2)
    translation_world = torch.einsum(
        "bsij,bshj->bshi", rotation_t, translation_cam - translation[..., None, :]
    )
    return translation_world, rotation_t[:, :, None] @ orientation_cam


def _points_to_world(pose_enc, points_cam):
    translation = pose_enc[..., :3]
    rotation_t = _camera_rotation(pose_enc).transpose(-1, -2)
    return torch.einsum(
        "bsij,bshkj->bshki", rotation_t, points_cam - translation[..., None, None, :]
    )


def world_trans_l1(ctx):
    return _masked_mean(
        (ctx["transl_w_p"] - ctx["transl_w_g"]).abs(), ctx["mano_mask"][..., None]
    )


def world_orient_geo(ctx):
    return _masked_mean(
        geodesic_angle(ctx["orient_w_p"], ctx["orient_w_g"]), ctx["mano_mask"]
    )


def world_trans_vel_l1(ctx):
    pred, target = ctx["transl_w_p"], ctx["transl_w_g"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    error = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()
    return _masked_mean(error, ctx["mano_pair_mask"][..., None])


def world_orient_vel_geo(ctx):
    pred, target = ctx["orient_w_p"], ctx["orient_w_g"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    pred_relative = pred[:, 1:] @ pred[:, :-1].transpose(-1, -2)
    target_relative = target[:, 1:] @ target[:, :-1].transpose(-1, -2)
    return _masked_mean(
        geodesic_angle(pred_relative, target_relative), ctx["mano_pair_mask"]
    )


def world_kp21_l1(ctx):
    error = (ctx["kp21_w_p"] - ctx["kp21_w_g"]).abs()
    return _masked_mean(error, ctx["kp21_mask"][..., None, None])


def world_kp21_vel_l1(ctx):
    pred, target = ctx["kp21_w_p"], ctx["kp21_w_g"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    error = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()
    return _masked_mean(error, ctx["kp21_pair_mask"][..., None, None])


_TERMS = {
    "world_trans_l1": world_trans_l1,
    "world_orient_geo": world_orient_geo,
    "world_trans_vel_l1": world_trans_vel_l1,
    "world_orient_vel_geo": world_orient_vel_geo,
    "world_kp21_l1": world_kp21_l1,
    "world_kp21_vel_l1": world_kp21_vel_l1,
}
_LEGACY_TERMS = {
    "transl_world_l1": "world_trans_l1",
    "orient_world_geo": "world_orient_geo",
    "transl_world_vel_l1": "world_trans_vel_l1",
    "orient_world_vel_geo": "world_orient_vel_geo",
    "kp21_world_l1": "world_kp21_l1",
    "kp21_world_vel_l1": "world_kp21_vel_l1",
}
_KP21_TERMS = {"world_kp21_l1", "world_kp21_vel_l1"}


def _source_mask(batch, key, mask, default):
    valid = batch.get(key)
    if valid is None:
        valid = torch.full((mask.shape[0],), default, dtype=torch.bool, device=mask.device)
    else:
        valid = valid.to(device=mask.device).bool().reshape(mask.shape[0])
    return mask * valid[:, None, None]


def _pair_mask(mask):
    return mask[:, 1:] * mask[:, :-1] if mask.shape[1] >= 2 else mask[:, :0]


class CameraManoConsistencyLoss:
    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.weight = float(cfg.get("weight", 1.0))
        self.terms = []
        for term_cfg in cfg.get("terms", []):
            configured_name = term_cfg["name"]
            name = _LEGACY_TERMS.get(configured_name, configured_name)
            if name not in _TERMS:
                available = sorted(set(_TERMS) | set(_LEGACY_TERMS))
                raise KeyError(
                    f"[camera_mano_consistency] unknown term {configured_name!r}; available: {available}"
                )
            self.terms.append((name, _TERMS[name], float(term_cfg.get("weight", 1.0))))
        self.needs_kp21 = any(name in _KP21_TERMS for name, _fn, _w in self.terms)
        self.needs_mano_gt = any(name not in _KP21_TERMS for name, _fn, _w in self.terms)

    def _ctx(self, pred, batch):
        base_mask = batch["hand_kept"].float()
        mano_mask = _source_mask(batch, "mano_gt_valid", base_mask, "hand_gt" in batch)
        kp21_mask = _source_mask(batch, "kpt21_gt_valid", base_mask, "kpt21_gt" in batch)
        ctx = {
            "mano_mask": mano_mask,
            "mano_pair_mask": _pair_mask(mano_mask),
            "kp21_mask": kp21_mask,
            "kp21_pair_mask": _pair_mask(kp21_mask),
        }

        if self.needs_mano_gt:
            hand = pred["hand"].float().reshape(*pred["hand"].shape[:-1], 2, PER_HAND)
            target = batch["hand_gt"].float().reshape(*batch["hand_gt"].shape[:-1], 2, PER_HAND)
            pred_orientation = rotation_6d_to_matrix(hand[..., 3:9])
            target_orientation = rotation_6d_to_matrix(target[..., 3:9])
            ctx["transl_w_p"], ctx["orient_w_p"] = _to_world(
                pred["pose_enc"].float(), hand[..., 0:3], pred_orientation
            )
            ctx["transl_w_g"], ctx["orient_w_g"] = _to_world(
                batch["gt_pose_enc"].float(), target[..., 0:3], target_orientation
            )

        if self.needs_kp21:
            cached = pred.get("_mano_joints21_cam")
            if cached is None:
                cached = joints21_from_hand_output(pred["hand"], base_mask)
                pred["_mano_joints21_cam"] = cached
            pred_points = cached[0]
            target_points = batch.get("kpt21_gt")
            if target_points is None:
                target_points = torch.zeros_like(pred_points)
            ctx["kp21_w_p"] = _points_to_world(pred["pose_enc"].float(), pred_points)
            ctx["kp21_w_g"] = _points_to_world(
                batch["gt_pose_enc"].float(), target_points.float()
            )
        return ctx

    def __call__(self, pred, batch):
        if "hand" not in pred:
            return pred["pose_enc"].sum() * 0.0, {}
        ctx = self._ctx(pred, batch)
        total = pred["hand"].sum() * 0.0
        logs = {}
        diagnostics_enabled = bool(pred.get("_diagnostics_enabled", False))
        for name, function, weight in self.terms:
            value = function(ctx)
            total = total + weight * value
            if diagnostics_enabled and "orient" in name:
                pred.setdefault("_diagnostic_loss_terms", {})[
                    f"camera_mano_consistency/{name}"
                ] = self.weight * weight * value
            logs[f"loss/camera_mano_consistency/{name}"] = value.detach()
            if "orient" in name:
                logs[f"metric/camera_mano_consistency/{name}_deg"] = (
                    value.detach() * 180.0 / math.pi
                )
            else:
                unit = "m_per_frame" if "vel" in name else "m"
                logs[f"metric/camera_mano_consistency/{name}_{unit}"] = value.detach()
        return self.weight * total, logs

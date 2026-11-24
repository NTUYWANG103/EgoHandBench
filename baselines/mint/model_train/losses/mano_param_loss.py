#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MANO-head parameter losses for translation, rotations, pose, and shape."""
import math

import torch

from baselines.mint.model_train.losses.rotation import (
    geodesic_angle,
    rotation_6d_health_metrics,
    rotation_6d_to_matrix,
)


PER_HAND = 109
NUM_POSE_JOINTS = 15
_TRANSL_ACC_CHARBONNIER_EPS_M = 1.0e-3


def _masked_mean(value, mask):
    mask = mask.expand_as(value)
    value = torch.where(mask > 0, value, torch.zeros_like(value))
    return (value * mask).sum() / mask.sum().clamp(min=1.0)


def transl_l1(ctx):
    error = (ctx["transl_p"] - ctx["transl_g"]).abs()
    return _masked_mean(error, ctx["mask"][..., None])


def orient_geo(ctx):
    return _masked_mean(geodesic_angle(ctx["orient_p"], ctx["orient_g"]), ctx["mask"])


def pose_geo(ctx):
    error = geodesic_angle(ctx["pose_p"], ctx["pose_g"])
    return _masked_mean(error, ctx["mask"][..., None])


def orient_6d_l1(ctx):
    error = (ctx["orient_d6_p"] - ctx["orient_d6_g"]).abs()
    return _masked_mean(error, ctx["mask"][..., None])


def pose_6d_l1(ctx):
    error = (ctx["pose_d6_p"] - ctx["pose_d6_g"]).abs()
    return _masked_mean(error, ctx["mask"][..., None, None])


def initial_orient_6d_l1(ctx):
    if "initial_orient_d6_p" not in ctx:
        raise RuntimeError(
            "initial_orient_6d_l1 requires a hand head that exposes "
            "_hand_refine_initial"
        )
    error = (ctx["initial_orient_d6_p"] - ctx["orient_d6_g"]).abs()
    return _masked_mean(error, ctx["mask"][..., None])


def initial_pose_6d_l1(ctx):
    if "initial_pose_d6_p" not in ctx:
        raise RuntimeError(
            "initial_pose_6d_l1 requires a hand head that exposes "
            "_hand_refine_initial"
        )
    error = (ctx["initial_pose_d6_p"] - ctx["pose_d6_g"]).abs()
    return _masked_mean(error, ctx["mask"][..., None, None])


def betas_l1(ctx):
    error = (ctx["betas_p"] - ctx["betas_g"]).abs()
    return _masked_mean(error, ctx["mask"][..., None])


def betas_vel_l1(ctx):
    pred, target = ctx["betas_p"], ctx["betas_g"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    error = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()
    return _masked_mean(error, ctx["pair_mask"][..., None])


def betas_consistency_l1(ctx):
    pred = ctx["betas_p"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    error = (pred[:, 1:] - pred[:, :-1]).abs()
    return _masked_mean(error, ctx["pair_mask"][..., None])


def transl_vel_l1(ctx):
    pred, target = ctx["transl_p"], ctx["transl_g"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    error = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()
    return _masked_mean(error, ctx["pair_mask"][..., None])


def orient_vel_geo(ctx):
    pred, target = ctx["orient_p"], ctx["orient_g"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    pred_relative = pred[:, 1:] @ pred[:, :-1].transpose(-1, -2)
    target_relative = target[:, 1:] @ target[:, :-1].transpose(-1, -2)
    return _masked_mean(geodesic_angle(pred_relative, target_relative), ctx["pair_mask"])


def pose_vel_geo(ctx):
    pred, target = ctx["pose_p"], ctx["pose_g"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    pred_relative = pred[:, 1:] @ pred[:, :-1].transpose(-1, -2)
    target_relative = target[:, 1:] @ target[:, :-1].transpose(-1, -2)
    error = geodesic_angle(pred_relative, target_relative)
    return _masked_mean(error, ctx["pair_mask"][..., None])


def orient_6d_vel_l1(ctx):
    pred, target = ctx["orient_d6_p"], ctx["orient_d6_g"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    error = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()
    return _masked_mean(error, ctx["pair_mask"][..., None])


def pose_6d_vel_l1(ctx):
    pred, target = ctx["pose_d6_p"], ctx["pose_d6_g"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    error = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()
    return _masked_mean(error, ctx["pair_mask"][..., None, None])


def transl_acc_l1(ctx):
    pred, target = ctx["transl_p"], ctx["transl_g"]
    if pred.shape[1] < 3:
        return pred.sum() * 0.0
    pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    target_acc = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    return _masked_mean((pred_acc - target_acc).abs(), ctx["triplet_mask"][..., None])


def transl_acc_smooth_charbonnier(ctx):
    """Robustly penalize predicted wrist acceleration independent of noisy GT."""
    pred = ctx["transl_p"]
    if pred.shape[1] < 3:
        return pred.sum() * 0.0
    pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    eps = _TRANSL_ACC_CHARBONNIER_EPS_M
    penalty = torch.sqrt(pred_acc.square() + eps * eps) - eps
    return _masked_mean(penalty, ctx["triplet_mask"][..., None])


def _rotation_acceleration(pred, target):
    pred_velocity = pred[:, 1:] @ pred[:, :-1].transpose(-1, -2)
    target_velocity = target[:, 1:] @ target[:, :-1].transpose(-1, -2)
    pred_acc = pred_velocity[:, 1:] @ pred_velocity[:, :-1].transpose(-1, -2)
    target_acc = target_velocity[:, 1:] @ target_velocity[:, :-1].transpose(-1, -2)
    return geodesic_angle(pred_acc, target_acc)


def orient_acc_geo(ctx):
    pred, target = ctx["orient_p"], ctx["orient_g"]
    if pred.shape[1] < 3:
        return pred.sum() * 0.0
    return _masked_mean(
        _rotation_acceleration(pred, target), ctx["triplet_mask"]
    )


def pose_acc_geo(ctx):
    pred, target = ctx["pose_p"], ctx["pose_g"]
    if pred.shape[1] < 3:
        return pred.sum() * 0.0
    return _masked_mean(
        _rotation_acceleration(pred, target), ctx["triplet_mask"][..., None]
    )


def orient_6d_acc_l1(ctx):
    pred, target = ctx["orient_d6_p"], ctx["orient_d6_g"]
    if pred.shape[1] < 3:
        return pred.sum() * 0.0
    pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    target_acc = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    return _masked_mean(
        (pred_acc - target_acc).abs(), ctx["triplet_mask"][..., None]
    )


def pose_6d_acc_l1(ctx):
    pred, target = ctx["pose_d6_p"], ctx["pose_d6_g"]
    if pred.shape[1] < 3:
        return pred.sum() * 0.0
    pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    target_acc = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    return _masked_mean(
        (pred_acc - target_acc).abs(), ctx["triplet_mask"][..., None, None]
    )


_TERMS = {
    "transl_l1": transl_l1,
    "orient_geo": orient_geo,
    "pose_geo": pose_geo,
    "orient_6d_l1": orient_6d_l1,
    "pose_6d_l1": pose_6d_l1,
    "initial_orient_6d_l1": initial_orient_6d_l1,
    "initial_pose_6d_l1": initial_pose_6d_l1,
    "betas_l1": betas_l1,
    "betas_vel_l1": betas_vel_l1,
    "betas_consistency_l1": betas_consistency_l1,
    "transl_vel_l1": transl_vel_l1,
    "orient_vel_geo": orient_vel_geo,
    "pose_vel_geo": pose_vel_geo,
    "orient_6d_vel_l1": orient_6d_vel_l1,
    "pose_6d_vel_l1": pose_6d_vel_l1,
    "transl_acc_l1": transl_acc_l1,
    "transl_acc_smooth_charbonnier": transl_acc_smooth_charbonnier,
    "orient_acc_geo": orient_acc_geo,
    "pose_acc_geo": pose_acc_geo,
    "orient_6d_acc_l1": orient_6d_acc_l1,
    "pose_6d_acc_l1": pose_6d_acc_l1,
}


class ManoParamLoss:
    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.weight = float(cfg.get("weight", 1.0))
        self.terms = []
        for term_cfg in cfg.get("terms", []):
            name = term_cfg["name"]
            if name not in _TERMS:
                raise KeyError(f"[mano_param] unknown term {name!r}; available: {sorted(_TERMS)}")
            self.terms.append((name, _TERMS[name], float(term_cfg.get("weight", 1.0))))

    @staticmethod
    def _ctx(pred, batch):
        hand = pred["hand"].float().reshape(*pred["hand"].shape[:-1], 2, PER_HAND)
        target = batch["hand_gt"].float().reshape(*batch["hand_gt"].shape[:-1], 2, PER_HAND)
        mask = batch["hand_kept"].float()
        valid = batch.get("mano_gt_valid")
        if valid is not None:
            mask = mask * valid.to(device=mask.device).float().reshape(mask.shape[0], 1, 1)
        pair_mask = mask[:, 1:] * mask[:, :-1] if mask.shape[1] >= 2 else mask[:, :0]
        triplet_mask = (
            mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
            if mask.shape[1] >= 3 else mask[:, :0]
        )
        orient_d6_p = hand[..., 3:9]
        orient_d6_g = target[..., 3:9]
        pose_d6_p = hand[..., 9:99].reshape(
            *hand.shape[:-1], NUM_POSE_JOINTS, 6
        )
        pose_d6_g = target[..., 9:99].reshape(
            *target.shape[:-1], NUM_POSE_JOINTS, 6
        )
        ctx = {
            "transl_p": hand[..., 0:3],
            "transl_g": target[..., 0:3],
            "orient_d6_p": orient_d6_p,
            "orient_d6_g": orient_d6_g,
            "pose_d6_p": pose_d6_p,
            "pose_d6_g": pose_d6_g,
            "orient_p": rotation_6d_to_matrix(orient_d6_p),
            "orient_g": rotation_6d_to_matrix(orient_d6_g),
            "pose_p": rotation_6d_to_matrix(pose_d6_p),
            "pose_g": rotation_6d_to_matrix(pose_d6_g),
            "betas_p": hand[..., 99:109],
            "betas_g": target[..., 99:109],
            "mask": mask,
            "pair_mask": pair_mask,
            "triplet_mask": triplet_mask,
        }
        initial = pred.get("_hand_refine_initial")
        if initial is not None:
            initial = initial.float().reshape(
                *initial.shape[:-1], 2, PER_HAND
            )
            ctx["initial_orient_d6_p"] = initial[..., 3:9]
            ctx["initial_pose_d6_p"] = initial[..., 9:99].reshape(
                *initial.shape[:-1], NUM_POSE_JOINTS, 6
            )
        return ctx

    def __call__(self, pred, batch):
        if "hand" not in pred:
            return pred["pose_enc"].sum() * 0.0, {}
        valid = batch.get("mano_gt_valid", batch.get("hand_valid"))
        if valid is not None and not bool(valid.any()):
            zero = pred["hand"].sum() * 0.0
            if bool(pred.get("_diagnostics_enabled", False)):
                diagnostic_terms = pred.setdefault("_diagnostic_loss_terms", {})
                for name, _function, weight in self.terms:
                    if name in {"orient_geo", "pose_geo"}:
                        diagnostic_terms[f"mano_param/{name}"] = (
                            self.weight * weight * zero
                        )
            return zero, {"metric/mano_param/skipped": torch.tensor(1.0)}

        ctx = self._ctx(pred, batch)
        total = pred["hand"].sum() * 0.0
        logs = {}
        diagnostics_enabled = bool(pred.get("_diagnostics_enabled", False))
        if diagnostics_enabled:
            logs.update(rotation_6d_health_metrics(
                ctx["orient_d6_p"], ctx["mask"], "orient"
            ))
            logs.update(rotation_6d_health_metrics(
                ctx["pose_d6_p"], ctx["mask"][..., None], "pose"
            ))
            if "initial_orient_d6_p" in ctx:
                logs.update(rotation_6d_health_metrics(
                    ctx["initial_orient_d6_p"], ctx["mask"], "initial_orient"
                ))
                logs.update(rotation_6d_health_metrics(
                    ctx["initial_pose_d6_p"],
                    ctx["mask"][..., None],
                    "initial_pose",
                ))
        for name, function, weight in self.terms:
            value = function(ctx)
            total = total + weight * value
            if diagnostics_enabled and name in {"orient_geo", "pose_geo"}:
                pred.setdefault("_diagnostic_loss_terms", {})[
                    f"mano_param/{name}"
                ] = self.weight * weight * value
            logs[f"loss/mano_param/{name}"] = value.detach()
            if name.startswith("transl"):
                if "acc" in name:
                    unit = "m_per_frame2"
                elif "vel" in name:
                    unit = "m_per_frame"
                else:
                    unit = "m"
                logs[f"metric/mano_param/{name}_{unit}"] = value.detach()
            elif "geo" in name:
                logs[f"metric/mano_param/{name}_deg"] = value.detach() * 180.0 / math.pi
            else:
                logs[f"metric/mano_param/{name}"] = value.detach()
        return self.weight * total, logs

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""21-joint supervision derived differentiably from the MANO prediction head."""
import torch

from baselines.mint.model_train.losses.mano_geometry import PER_HAND, joints21_from_hand_output


def _masked_mean(value, mask):
    mask = mask.expand_as(value)
    value = torch.where(mask > 0, value, torch.zeros_like(value))
    return (value * mask).sum() / mask.sum().clamp(min=1.0)


def rootrel_mpjpe(ctx):
    error = torch.linalg.vector_norm(ctx["rootrel_p"] - ctx["rootrel_g"], dim=-1)
    return _masked_mean(error, ctx["mano_mask"][..., None])


def abs_mpjpe(ctx):
    error = torch.linalg.vector_norm(ctx["absolute_p"] - ctx["absolute_g"], dim=-1)
    return _masked_mean(error, ctx["mano_mask"][..., None])


def rootrel_vel_mpjpe(ctx):
    pred, target = ctx["rootrel_p"], ctx["rootrel_g"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    error = torch.linalg.vector_norm(
        (pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1]), dim=-1
    )
    return _masked_mean(error, ctx["mano_pair_mask"][..., None])


def kp21_l1(ctx):
    error = (ctx["absolute_p"] - ctx["kp21_g"]).abs()
    return _masked_mean(error, ctx["kp21_mask"][..., None, None])


def kp21_vel_l1(ctx):
    pred, target = ctx["absolute_p"], ctx["kp21_g"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    error = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()
    return _masked_mean(error, ctx["kp21_pair_mask"][..., None, None])


def betas_reg(ctx):
    return _masked_mean(ctx["betas_p"].abs(), ctx["kp21_mask"][..., None])


_TERMS = {
    "rootrel_mpjpe": rootrel_mpjpe,
    "abs_mpjpe": abs_mpjpe,
    "rootrel_vel_mpjpe": rootrel_vel_mpjpe,
    "kp21_l1": kp21_l1,
    "kp21_vel_l1": kp21_vel_l1,
    "betas_reg": betas_reg,
}
_MANO_GT_TERMS = {"rootrel_mpjpe", "abs_mpjpe", "rootrel_vel_mpjpe"}


def _source_mask(batch, key, mask, default):
    valid = batch.get(key)
    if valid is None:
        valid = torch.full(
            (mask.shape[0],), default, dtype=torch.bool, device=mask.device
        )
    else:
        valid = valid.to(device=mask.device).bool().reshape(mask.shape[0])
    return mask * valid[:, None, None]


def _pair_mask(mask):
    return mask[:, 1:] * mask[:, :-1] if mask.shape[1] >= 2 else mask[:, :0]


class ManoJointLoss:
    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.weight = float(cfg.get("weight", 1.0))
        self.terms = []
        for term_cfg in cfg.get("terms", []):
            name = term_cfg["name"]
            if name not in _TERMS:
                raise KeyError(f"[mano_joint] unknown term {name!r}; available: {sorted(_TERMS)}")
            self.terms.append((name, _TERMS[name], float(term_cfg.get("weight", 1.0))))
        self.needs_mano_gt = any(name in _MANO_GT_TERMS for name, _fn, _w in self.terms)

    def _ctx(self, pred, batch):
        hand = pred["hand"].float().reshape(*pred["hand"].shape[:-1], 2, PER_HAND)
        base_mask = batch["hand_kept"].float()
        mano_mask = _source_mask(batch, "mano_gt_valid", base_mask, "hand_gt" in batch)
        kp21_mask = _source_mask(batch, "kpt21_gt_valid", base_mask, "kpt21_gt" in batch)

        cached = pred.get("_mano_joints21_cam")
        if cached is None:
            cached = joints21_from_hand_output(pred["hand"], base_mask)
            pred["_mano_joints21_cam"] = cached
        pred_absolute, pred_rootrel = cached

        target_absolute = torch.zeros_like(pred_absolute)
        target_rootrel = torch.zeros_like(pred_rootrel)
        if self.needs_mano_gt:
            target = batch["hand_gt"].float()
            with torch.no_grad():
                target_absolute, target_rootrel = joints21_from_hand_output(target, mano_mask)

        kp21_target = batch.get("kpt21_gt")
        if kp21_target is None:
            kp21_target = torch.zeros_like(pred_absolute)
        else:
            kp21_target = kp21_target.float()
        return {
            "absolute_p": pred_absolute,
            "absolute_g": target_absolute,
            "rootrel_p": pred_rootrel,
            "rootrel_g": target_rootrel,
            "kp21_g": kp21_target,
            "betas_p": hand[..., 99:109],
            "mano_mask": mano_mask,
            "mano_pair_mask": _pair_mask(mano_mask),
            "kp21_mask": kp21_mask,
            "kp21_pair_mask": _pair_mask(kp21_mask),
        }

    def __call__(self, pred, batch):
        if "hand" not in pred:
            return pred["pose_enc"].sum() * 0.0, {}
        ctx = self._ctx(pred, batch)
        total = pred["hand"].sum() * 0.0
        logs = {}
        for name, function, weight in self.terms:
            value = function(ctx)
            total = total + weight * value
            logs[f"loss/mano_joint/{name}"] = value.detach()
            if name == "betas_reg":
                logs[f"metric/mano_joint/{name}"] = value.detach()
            else:
                logs[f"metric/mano_joint/{name}_mm"] = value.detach() * 1000.0
        return self.weight * total, logs

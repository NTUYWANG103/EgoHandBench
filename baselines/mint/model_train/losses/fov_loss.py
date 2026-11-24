#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent field-of-view head losses and angular metrics."""
import math

import torch.nn.functional as F


def fov_l1(ctx):
    return F.l1_loss(ctx["fp"], ctx["fg"])


def fov_vel_l1(ctx):
    pred, target = ctx["fp"], ctx["fg"]
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    return F.l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


_TERMS = {
    "fov_l1": fov_l1,
    "fov_vel_l1": fov_vel_l1,
}


class FovLoss:
    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.weight = float(cfg.get("weight", 1.0))
        self.terms = []
        for term_cfg in cfg.get("terms", []):
            name = term_cfg["name"]
            if name not in _TERMS:
                raise KeyError(f"[fov] unknown term {name!r}; available: {sorted(_TERMS)}")
            self.terms.append((name, _TERMS[name], float(term_cfg.get("weight", 1.0))))

    @staticmethod
    def _ctx(pred, batch):
        return {
            "fp": pred["pose_enc"].float()[..., 7:9],
            "fg": batch["gt_pose_enc"].float()[..., 7:9],
        }

    def __call__(self, pred, batch):
        ctx = self._ctx(pred, batch)
        total = pred["pose_enc"].sum() * 0.0
        logs = {}
        for name, function, weight in self.terms:
            value = function(ctx)
            total = total + weight * value
            logs[f"loss/fov/{name}"] = value.detach()
            logs[f"metric/fov/{name}_deg"] = value.detach() * 180.0 / math.pi
        return self.weight * total, logs

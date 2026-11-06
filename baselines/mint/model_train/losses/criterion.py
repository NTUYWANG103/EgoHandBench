#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and sum loss modules classified by prediction ownership."""
from baselines.mint.model_train.losses.camera_loss import CameraLoss
from baselines.mint.model_train.losses.camera_mano_consistency_loss import CameraManoConsistencyLoss
from baselines.mint.model_train.losses.fov_loss import FovLoss
from baselines.mint.model_train.losses.hand_presence_loss import HandPresenceLoss
from baselines.mint.model_train.losses.image_hand_loss import ImageHandLoss
from baselines.mint.model_train.losses.mano_diffusion_loss import ManoDiffusionLoss
from baselines.mint.model_train.losses.mano_joint_loss import ManoJointLoss
from baselines.mint.model_train.losses.mano_param_loss import ManoParamLoss


_LOSS_MODULES = {
    "camera": CameraLoss,
    "fov": FovLoss,
    "mano_param": ManoParamLoss,
    "mano_diffusion": ManoDiffusionLoss,
    "mano_joint": ManoJointLoss,
    "camera_mano_consistency": CameraManoConsistencyLoss,
    "hand_presence": HandPresenceLoss,
    "image_hand": ImageHandLoss,
}

_LEGACY_KEYS = {
    "camera": ("camera_loss",),
    "mano_param": ("hand_loss",),
    "mano_joint": ("hand_kp21_loss",),
    "camera_mano_consistency": ("world_loss", "world_kp21_loss"),
}


class Criterion:
    def __init__(self, cfg: dict):
        cfg = cfg or {}
        self.losses = []
        for name, loss_class in _LOSS_MODULES.items():
            aliases = _LEGACY_KEYS.get(name, ())
            present = [key for key in (name, *aliases) if key in cfg]
            if len(present) > 1:
                raise KeyError(f"loss config contains conflicting aliases for {name!r}: {present}")
            if present:
                self.losses.append((name, loss_class(cfg[present[0]])))
        if not self.losses:
            available = list(_LOSS_MODULES) + [
                alias for aliases in _LEGACY_KEYS.values() for alias in aliases
            ]
            raise KeyError(f"loss config must contain at least one of {available}; got {list(cfg)}")

    def __call__(self, pred: dict, batch: dict):
        total = None
        logs = {}
        for name, loss_module in self.losses:
            value, module_logs = loss_module(pred, batch)
            total = value if total is None else total + value
            logs[f"contrib/{name}"] = value.detach()
            logs.update(module_logs)
        logs.update(pred.get("_diagnostic_metrics", {}))
        logs["loss"] = total.detach()
        return total, logs


def build_criterion(cfg: dict) -> Criterion:
    return Criterion(cfg)

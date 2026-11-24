#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from baselines.mint.model_train.models.lingbotmap.hand_head.refine import RefineHandHead
from baselines.mint.model_train.models.lingbotmap.hand_head.mlp import MlpHandHead

_HEADS = {
    "mlp": MlpHandHead,
    "refine": RefineHandHead,
}


def build_hand_head(cfg: dict):
    """Internal helper."""
    cfg = dict(cfg or {})
    name = cfg.pop("name", "mlp")
    if name not in _HEADS:
        raise ValueError(f"[train]  {name!r}; {list(_HEADS)}.")
    return _HEADS[name](**cfg)

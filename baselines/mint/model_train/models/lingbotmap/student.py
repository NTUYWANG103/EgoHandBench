#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict

import torch

from baselines.mint.model_train.core.registry import MODELS
from baselines.mint.model_train.models.base import BaseStudent
from baselines.mint.model_train.models.lingbotmap.hand_head import build_hand_head
from baselines.mint.model_train.utils.weight_loader import load_pretrained
from baselines.mint.model_train.utils.logging import rank0_print


@MODELS.register("lingbotmap")
class LingbotmapStudent(BaseStudent):
    def __init__(self, cfg: dict):
        super().__init__()
        from baselines.mint.model_train._vendor.lingbot_map.models.gct_stream import GCTStream

        bk = dict(cfg.get("backbone", {}))
        bk.setdefault("use_sdpa", True)
        bk.setdefault("embed_dim", 1024)
        self.embed_dim = bk["embed_dim"]
        self.backbone = GCTStream(**bk)

        self.enable_hand = bool(cfg.get("enable_hand", False))
        if self.enable_hand:
            hh = dict(cfg.get("hand_head", {}))
            hh.setdefault("in_dim", 2 * self.embed_dim)
            self.hand_head = build_hand_head(hh)

        self.enable_hand_presence = bool(cfg.get("enable_hand_presence", False))
        if self.enable_hand_presence:
            from baselines.mint.model_train.models.lingbotmap.hand_presence_head import HandPresenceHead
            ph = dict(cfg.get("hand_presence_head", {}))
            ph.setdefault("in_dim", 2 * self.embed_dim)
            self.hand_presence_head = HandPresenceHead(**ph)

        pt = cfg.get("pretrained")
        if pt:



            load_pretrained(self.backbone, pt,
                            exclude=cfg.get("pretrained_exclude"), strict=False)
        elif cfg.get("_ckpt_provided"):


            pass
        elif cfg.get("_inspect_skip_pretrained"):
            rank0_print("[train]")
        else:
            rank0_print("[train]")

    def set_train_diagnostics(self, enabled: bool) -> None:
        if self.enable_hand and hasattr(self.hand_head, "set_diagnostics_enabled"):
            self.hand_head.set_diagnostics_enabled(enabled)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["images"]                 # [B, S, 3, H, W]


        self.backbone.clean_kv_cache()




        images, _ = self.backbone._normalize_input(images)
        agg, patch_start_idx = self.backbone._aggregate_features(images)
        pred = {"pose_enc": self.backbone._predict_camera(agg)["pose_enc"]}

        if self.enable_hand:


            pred["hand"] = self.hand_head(agg[-1], patch_start_idx)
            if hasattr(self.hand_head, "pop_auxiliary_predictions"):
                pred.update(self.hand_head.pop_auxiliary_predictions())
            if hasattr(self.hand_head, "pop_diagnostic_metrics"):
                metrics = self.hand_head.pop_diagnostic_metrics()
                if metrics:
                    pred["_diagnostic_metrics"] = metrics
        if self.enable_hand_presence:

            pred["hand_presence_logits"] = self.hand_presence_head(
                agg[-1], patch_start_idx
            )
        return pred

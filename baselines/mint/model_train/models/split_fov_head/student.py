"""LingBot-Map student with parallel extrinsics and FoV output heads."""

from typing import Dict

import torch

from baselines.mint.model_train.core.registry import MODELS
from baselines.mint.model_train.models.base import BaseStudent
from baselines.mint.model_train.models.lingbotmap.hand_head import build_hand_head
from baselines.mint.model_train.models.split_fov_head.network import SplitFovNetwork
from baselines.mint.model_train.utils.logging import rank0_print
from baselines.mint.model_train.utils.weight_loader import load_pretrained


@MODELS.register("split_fov_head")
class SplitFovHeadStudent(BaseStudent):
    def __init__(self, cfg: dict):
        super().__init__()
        backbone_cfg = dict(cfg.get("backbone", {}))
        backbone_cfg.setdefault("use_sdpa", True)
        backbone_cfg.setdefault("embed_dim", 1024)
        self.embed_dim = int(backbone_cfg["embed_dim"])
        self.backbone = SplitFovNetwork(fov_head=cfg.get("fov_head"), **backbone_cfg)

        self.enable_hand = bool(cfg.get("enable_hand", False))
        if self.enable_hand:
            hand_cfg = dict(cfg.get("hand_head", {}))
            hand_cfg.setdefault("in_dim", 2 * self.embed_dim)
            self.hand_head = build_hand_head(hand_cfg)

        self.enable_hand_presence = bool(cfg.get("enable_hand_presence", False))
        if self.enable_hand_presence:
            from baselines.mint.model_train.models.lingbotmap.hand_presence_head import HandPresenceHead

            presence_cfg = dict(cfg.get("hand_presence_head", {}))
            presence_cfg.setdefault("in_dim", 2 * self.embed_dim)
            self.hand_presence_head = HandPresenceHead(**presence_cfg)

        pretrained = cfg.get("pretrained")
        if pretrained:
            load_pretrained(
                self.backbone,
                pretrained,
                exclude=cfg.get("pretrained_exclude"),
                strict=False,
            )
        elif cfg.get("_ckpt_provided"):
            pass
        elif cfg.get("_inspect_skip_pretrained"):
            rank0_print("[split_fov_head] inspect mode skips pretrained weights")
        else:
            rank0_print("[split_fov_head] no pretrained backbone configured")

    def set_train_diagnostics(self, enabled: bool) -> None:
        if self.enable_hand and hasattr(self.hand_head, "set_diagnostics_enabled"):
            self.hand_head.set_diagnostics_enabled(enabled)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["images"]
        self.backbone.clean_kv_cache()
        images, _ = self.backbone._normalize_input(images)
        aggregated, patch_start_idx = self.backbone._aggregate_features(images)
        prediction = {"pose_enc": self.backbone._predict_camera(aggregated)["pose_enc"]}

        if self.enable_hand:
            prediction["hand"] = self.hand_head(aggregated[-1], patch_start_idx)
            if hasattr(self.hand_head, "pop_auxiliary_predictions"):
                prediction.update(self.hand_head.pop_auxiliary_predictions())
            if hasattr(self.hand_head, "pop_diagnostic_metrics"):
                metrics = self.hand_head.pop_diagnostic_metrics()
                if metrics:
                    prediction["_diagnostic_metrics"] = metrics
        if self.enable_hand_presence:
            prediction["hand_presence_logits"] = self.hand_presence_head(
                aggregated[-1], patch_start_idx
            )
        return prediction

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-side binary supervision for the independent hand-presence head."""

import torch
import torch.nn.functional as F


def presence_bce(ctx):
    return F.binary_cross_entropy_with_logits(
        ctx["logits"], ctx["target"], pos_weight=ctx["pos_weight"]
    )


class HandPresenceLoss:
    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.weight = float(cfg.get("weight", 1.0))
        pos_weight = cfg.get("pos_weight")
        self.pos_weight = None if pos_weight is None else float(pos_weight)
        self.terms = [("bce", presence_bce, 1.0)]

    def _ctx(self, pred, batch):
        logits = pred["hand_presence_logits"].float()
        target = batch["hand_kept"].to(device=logits.device).float()
        if logits.shape != target.shape:
            raise ValueError(
                "hand presence shape mismatch: "
                f"logits={tuple(logits.shape)} target={tuple(target.shape)}"
            )
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = logits.new_tensor(self.pos_weight)
        return {"logits": logits, "target": target, "pos_weight": pos_weight}

    def __call__(self, pred, batch):
        ctx = self._ctx(pred, batch)
        logits, target = ctx["logits"], ctx["target"]
        loss = presence_bce(ctx)

        with torch.no_grad():
            detected = logits >= 0.0
            positive = target.bool()
            tp = (detected & positive).sum().float()
            predicted_positive = detected.sum().float()
            actual_positive = positive.sum().float()
            accuracy = (detected == positive).float().mean()
            precision = tp / predicted_positive.clamp(min=1.0)
            recall = tp / actual_positive.clamp(min=1.0)

        logs = {
            "loss/hand_presence/bce": loss.detach(),
            "metric/hand_presence/accuracy": accuracy,
            "metric/hand_presence/precision": precision,
            "metric/hand_presence/recall": recall,
            "metric/hand_presence/positive_rate": target.mean().detach(),
            "metric/hand_presence/left_accuracy": (
                detected[..., 0] == positive[..., 0]
            ).float().mean(),
            "metric/hand_presence/right_accuracy": (
                detected[..., 1] == positive[..., 1]
            ).float().mean(),
            "metric/hand_presence/left_positive_rate": target[..., 0].mean().detach(),
            "metric/hand_presence/right_positive_rate": target[..., 1].mean().detach(),
        }
        return self.weight * loss, logs

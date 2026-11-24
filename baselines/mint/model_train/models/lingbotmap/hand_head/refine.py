#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn

from baselines.mint.model_train.losses.rotation import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


_COMP = [("transl", 3), ("orient", 6), ("pose", 90), ("betas", 10)]
_AXIS_ANGLE_REFINE_COMP = [
    ("transl", 3),
    ("orient_axis_angle", 3),
    ("pose_axis_angle", 45),
    ("betas", 10),
]
PER_HAND = sum(d for _, d in _COMP)   # 109
OUT_DIM = 2 * PER_HAND                # 218
NUM_POSE_JOINTS = 15


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Internal helper."""
    return x * (1 + scale) + shift


def _record_mean_var(metrics: dict, name: str, value: torch.Tensor) -> None:
    variance, mean = torch.var_mean(value.detach().float(), correction=0)
    metrics[f"diag/hand_head/{name}_mean"] = mean
    metrics[f"diag/hand_head/{name}_var"] = variance


def _clip_vector_norm(value: torch.Tensor, max_norm: float | None) -> torch.Tensor:
    """Cap vector magnitude without changing updates already below the limit."""
    if max_norm is None:
        return value
    norm = value.float().norm(dim=-1, keepdim=True)
    scale = (max_norm / norm.clamp_min(1e-12)).clamp(max=1.0)
    clipped = value * scale.to(dtype=value.dtype)
    # Straight-through gradients let oversized raw deltas learn back toward the cap.
    return clipped.detach() + (value - value.detach())


def _bound_axis_angle(value: torch.Tensor, max_angle: float | None) -> torch.Tensor:
    if max_angle is None:
        return value
    angle = value.float().norm(dim=-1, keepdim=True)
    scale = max_angle * torch.tanh(angle / max_angle) / angle.clamp_min(1e-12)
    scale = torch.where(angle < 1e-6, torch.ones_like(scale), scale)
    return value * scale.to(dtype=value.dtype)


def _record_angle_stats(metrics: dict, name: str, value: torch.Tensor) -> None:
    angle_deg = value.detach().float().norm(dim=-1) * (180.0 / math.pi)
    base = f"diag/hand_head/{name}_angle_deg"
    metrics[f"{base}_p50"] = torch.quantile(angle_deg, 0.50)
    metrics[f"{base}_p95"] = torch.quantile(angle_deg, 0.95)
    metrics[f"{base}_max"] = angle_deg.max()


class RefineHandHead(nn.Module):
    def __init__(self, in_dim: int = 2048, dim: int = 512, num_queries: int = 4,
                 num_iterations: int = 2, trunk_depth: int = 1,
                 num_heads: int | None = None,
                 rotation_refine_mode: str = "compose_axis_angle",
                 rotation_delta_frame: str = "local",
                 rotation_delta_max_angle_deg: float | None = 30.0,
                 translation_delta_max_norm_m: float | None = None):
        super().__init__()
        from baselines.mint.model_train._vendor.lingbot_map.layers.block import Block

        assert num_queries == len(_COMP),\
            f"[train]  {len(_COMP)}."
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if rotation_refine_mode != "compose_axis_angle":
            raise ValueError(
                "rotation_refine_mode only supports 'compose_axis_angle'; "
                "direct 6D-vector addition has been removed, "
                f"got {rotation_refine_mode!r}"
            )
        if rotation_delta_frame not in {"local", "global"}:
            raise ValueError(
                "rotation_delta_frame must be 'local' or 'global', "
                f"got {rotation_delta_frame!r}"
            )
        if rotation_delta_max_angle_deg is not None and rotation_delta_max_angle_deg <= 0:
            raise ValueError("rotation_delta_max_angle_deg must be positive or null")
        if translation_delta_max_norm_m is not None and translation_delta_max_norm_m <= 0:
            raise ValueError("translation_delta_max_norm_m must be positive or null")
        self.dim = dim
        self.num_queries = num_queries
        self.num_iterations = num_iterations
        self.rotation_refine_mode = rotation_refine_mode
        self.rotation_delta_frame = rotation_delta_frame
        self.rotation_delta_max_angle = (
            None
            if rotation_delta_max_angle_deg is None
            else math.radians(float(rotation_delta_max_angle_deg))
        )
        self.translation_delta_max_norm = (
            None
            if translation_delta_max_norm_m is None
            else float(translation_delta_max_norm_m)
        )
        nh = num_heads or max(1, dim // 64)


        self.token_proj = nn.Linear(in_dim, dim)
        self.token_norm = nn.LayerNorm(dim)

        self.query = nn.Parameter(torch.randn(1, 2 * num_queries, dim) * 0.02)


        self.embed_pred = nn.Linear(PER_HAND, dim)
        self.empty_pred = nn.Parameter(torch.zeros(1, 1, PER_HAND))
        self.adaln_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.adaln_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.cross_attn = nn.MultiheadAttention(dim, nh, batch_first=True)
        self.trunk = nn.ModuleList([Block(dim=dim, num_heads=nh, mlp_ratio=4.0)
                                    for _ in range(trunk_depth)])
        self.out_norm = nn.LayerNorm(dim)

        self.comp_heads = nn.ModuleList([nn.Linear(dim, d) for _, d in _COMP])
        self.refine_comp_heads = nn.ModuleList(
            [nn.Linear(dim, d) for _, d in _AXIS_ANGLE_REFINE_COMP]
        )
        # Refinement starts as an exact no-op: zero translation/shape delta and
        # zero axis-angle, whose exponential map is the identity rotation.
        for head in self.refine_comp_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.out_dim = OUT_DIM
        self._diagnostics_enabled = False
        self._diagnostic_metrics = {}
        self._auxiliary_predictions = {}

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ) -> None:
        has_absolute_head = f"{prefix}comp_heads.0.weight" in state_dict
        has_axis_angle_head = f"{prefix}refine_comp_heads.0.weight" in state_dict
        if has_absolute_head and not has_axis_angle_head:
            error_msgs.append(
                f"{prefix or 'RefineHandHead'}: incompatible legacy checkpoint: "
                "found the absolute hand head but no axis-angle refinement head. "
                "Checkpoints trained with direct 6D-vector addition cannot be "
                "loaded by the axis-angle-only RefineHandHead."
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        self._diagnostics_enabled = bool(enabled)
        self._diagnostic_metrics = {}

    def pop_diagnostic_metrics(self) -> dict:
        metrics = self._diagnostic_metrics
        self._diagnostic_metrics = {}
        return metrics

    def pop_auxiliary_predictions(self) -> dict:
        predictions = self._auxiliary_predictions
        self._auxiliary_predictions = {}
        return predictions

    def _component_output(self, qh: torch.Tensor) -> torch.Tensor:
        hands = []
        for hand_index in range(2):
            components = [
                self.comp_heads[index](qh[:, hand_index, index, :])
                for index in range(self.num_queries)
            ]
            hands.append(torch.cat(components, dim=-1))
        return torch.cat(hands, dim=-1)

    def _compose(self, current: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.rotation_delta_frame == "local":
            return current @ delta
        return delta @ current

    def _axis_angle_refinement(
        self, current: torch.Tensor, qh: torch.Tensor, diagnostics: dict | None,
        iteration: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = current.reshape(current.shape[0], 2, PER_HAND)
        deltas = [
            self.refine_comp_heads[index](qh[:, :, index, :])
            for index in range(self.num_queries)
        ]
        translation_delta = _clip_vector_norm(
            deltas[0], self.translation_delta_max_norm
        )
        orient_delta = _bound_axis_angle(deltas[1], self.rotation_delta_max_angle)
        pose_delta = _bound_axis_angle(
            deltas[2].reshape(*deltas[2].shape[:-1], NUM_POSE_JOINTS, 3),
            self.rotation_delta_max_angle,
        )

        orient_current = rotation_6d_to_matrix(current[..., 3:9])
        pose_current = rotation_6d_to_matrix(
            current[..., 9:99].reshape(*current.shape[:-1], NUM_POSE_JOINTS, 6)
        )
        orient_refined = self._compose(
            orient_current, axis_angle_to_matrix(orient_delta)
        )
        pose_refined = self._compose(pose_current, axis_angle_to_matrix(pose_delta))

        refined = torch.cat(
            (
                current[..., 0:3] + translation_delta,
                matrix_to_rotation_6d(orient_refined),
                matrix_to_rotation_6d(pose_refined).flatten(start_dim=-2),
                current[..., 99:109] + deltas[3],
            ),
            dim=-1,
        ).reshape(current.shape[0], OUT_DIM)
        raw_delta = torch.cat(
            (
                translation_delta,
                orient_delta,
                pose_delta.flatten(start_dim=-2),
                deltas[3],
            ),
            dim=-1,
        ).reshape(current.shape[0], -1)

        if diagnostics is not None:
            raw_translation_norm = deltas[0].detach().float().norm(dim=-1)
            translation_norm = translation_delta.detach().float().norm(dim=-1)
            diagnostics[
                f"diag/hand_head/iter{iteration}/translation_delta_raw_norm_m_p95"
            ] = torch.quantile(raw_translation_norm, 0.95)
            diagnostics[
                f"diag/hand_head/iter{iteration}/translation_delta_raw_norm_m_max"
            ] = raw_translation_norm.max()
            if self.translation_delta_max_norm is not None:
                diagnostics[
                    f"diag/hand_head/iter{iteration}/translation_delta_clipped_fraction"
                ] = (
                    raw_translation_norm > self.translation_delta_max_norm
                ).float().mean()
            diagnostics[
                f"diag/hand_head/iter{iteration}/translation_delta_norm_m_p95"
            ] = torch.quantile(translation_norm, 0.95)
            diagnostics[
                f"diag/hand_head/iter{iteration}/translation_delta_norm_m_max"
            ] = translation_norm.max()
            _record_angle_stats(
                diagnostics, f"iter{iteration}/orient_delta", orient_delta
            )
            _record_angle_stats(
                diagnostics, f"iter{iteration}/pose_delta", pose_delta
            )
        return refined, raw_delta

    def forward(self, tokens: torch.Tensor, patch_start_idx: int = 0) -> torch.Tensor:
        """Internal helper."""
        B, S, P, C = tokens.shape
        N = B * S
        nq = self.num_queries
        token_proj = self.token_proj(tokens.reshape(N, P, C))
        self._auxiliary_predictions = {}
        diagnostics = {} if self._diagnostics_enabled else None
        if diagnostics is not None:
            _record_mean_var(diagnostics, "token_proj", token_proj)
        kv = self.token_norm(token_proj)   # [N,P,d]

        q0 = self.query.expand(N, -1, -1)          # [N, 2nq, d]
        pred = None
        for iteration in range(self.num_iterations):

            if pred is None:
                prev = self.empty_pred.expand(N, 2, -1)              # [N,2,109]
            else:
                prev = pred.detach().reshape(N, 2, PER_HAND)
            shift, scale, gate = self.adaln_mod(self.embed_pred(prev)).chunk(3, dim=-1)
            if diagnostics is not None:
                _record_mean_var(diagnostics, f"iter{iteration}/adaln_gate", gate)
            shift = shift.repeat_interleave(nq, dim=1)               # [N,2nq,d]
            scale = scale.repeat_interleave(nq, dim=1)
            gate = gate.repeat_interleave(nq, dim=1)

            q = gate * _modulate(self.adaln_norm(q0), shift, scale) + q0
            cross_attn = self.cross_attn(q, kv, kv, need_weights=False)[0]
            if diagnostics is not None:
                _record_mean_var(
                    diagnostics, f"iter{iteration}/cross_attn", cross_attn
                )
            q = q + cross_attn
            for blk in self.trunk:
                q = blk(q)
            qh = self.out_norm(q).reshape(N, 2, nq, self.dim)          # [N,2,nq,d]

            if pred is None:
                delta = self._component_output(qh)
                pred = delta
                self._auxiliary_predictions["_hand_refine_initial"] = (
                    pred.reshape(B, S, OUT_DIM)
                )
            else:
                pred, delta = self._axis_angle_refinement(
                    pred, qh, diagnostics, iteration
                )
            if diagnostics is not None:
                _record_mean_var(diagnostics, f"iter{iteration}/delta", delta)

        if diagnostics is not None:
            self._diagnostic_metrics = diagnostics
        return pred.reshape(B, S, OUT_DIM)

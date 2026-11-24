#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for validated dataset statistics and per-sample loss normalization."""
import hashlib
import json
from pathlib import Path

import torch


_RESOLVED_SCALE_KEYS = {
    "camera_trans_position_std": "position_std_m",
    "camera_trans_velocity_std": "velocity_std_m_per_frame",
    "camera_trans_acceleration_std": "acceleration_std_m_per_frame2",
}


def resolved_camera_translation_scales(cfg: dict) -> dict[str, torch.Tensor]:
    """Read the global scales injected by train.py's automatic resolver."""
    norm_cfg = cfg.get("camera_translation_normalization", {}) or {}
    if not bool(norm_cfg.get("enabled", False)):
        return {}
    resolved = norm_cfg.get("resolved")
    if not isinstance(resolved, dict):
        raise RuntimeError(
            "[train]"
        )
    scales = {}
    for batch_key, resolved_key in _RESOLVED_SCALE_KEYS.items():
        scale = torch.tensor(resolved.get(resolved_key, []), dtype=torch.float32)
        if scale.shape != (3,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise RuntimeError(f"[train]  {resolved_key}.")
        scales[batch_key] = scale
    return scales


def load_camera_translation_std(
    root, clip_len: int, filename: str = "camera_translation_normalization.json"
) -> torch.Tensor:
    """Load a complete, dataset-specific camera translation scale."""
    root = Path(root)
    path = root / "meta" / filename
    if not path.is_file():
        raise RuntimeError(f"[train]  {path}.")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    normalization = payload.get("normalization", {})
    if normalization.get("type") != "scale_only" or normalization.get("subtract_mean"):
        raise RuntimeError(f"[train]  {path}.")
    if normalization.get("select_stats_by") != "clip_len":
        raise RuntimeError(f"[train]  {path}.")

    target = payload.get("target_definition", {})
    if not target.get("rebase_to_each_clip_first_frame", False):
        raise RuntimeError(f"[train]  {path}.")

    sampling = payload.get("sampling", {})
    if int(sampling.get("temporal_frame_step", -1)) != 1:
        raise RuntimeError(f"[train]  {path}.")
    if sampling.get("complete_dataset_scan") is not True:
        raise RuntimeError(f"[train]  {path}.")
    if sampling.get("max_data_files_per_dataset") is not None:
        raise RuntimeError(f"[train]  {path}.")

    datasets = payload.get("datasets", [])
    if len(datasets) != 1 or datasets[0].get("root_relative_to_input_anchor") != ".":
        raise RuntimeError(f"[train]  {path}.")
    dataset = datasets[0]
    if dataset.get("data_files_read") != dataset.get("data_files_total"):
        raise RuntimeError(f"[train]  {path}.")

    info_path = root / "meta" / "info.json"
    digest = hashlib.sha256()
    with info_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if dataset.get("info_sha256") != digest.hexdigest():
        raise RuntimeError(f"[train]  {path}.")

    clip_stats = payload.get("stats", {}).get(str(int(clip_len)))
    if clip_stats is None:
        raise RuntimeError(f"[train]  {path}; {clip_len}.")
    scale = torch.tensor(clip_stats.get("trans_std_m", []), dtype=torch.float32)
    if scale.shape != (3,) or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise RuntimeError(f"[train]  {path}.")
    return scale


def batch_vector_scale(batch: dict, key: str, reference: torch.Tensor, width: int):
    """Return a batch vector as [B, 1, D] on the reference device, or None."""
    scale = batch.get(key)
    if scale is None:
        return None
    if scale.ndim == 1:
        scale = scale.unsqueeze(0)
    expected = (reference.shape[0], width)
    if tuple(scale.shape) != expected:
        raise RuntimeError(f"batch[{key!r}] shape should be {expected}, got {tuple(scale.shape)}")
    scale = scale.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise RuntimeError(f"batch[{key!r}] must contain finite positive values")
    return scale.view(reference.shape[0], 1, width)


def normalize_error(error: torch.Tensor, scale):
    """Apply a broadcastable scale when present without changing model outputs."""
    return error if scale is None else error / scale

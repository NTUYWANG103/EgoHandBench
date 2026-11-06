#!/usr/bin/env python3
"""Penalized 2D joint reprojection loss for detected and missed hands."""
import math

import torch

from baselines.mint.model_train.losses.mano_geometry import joints21_from_hand_output


def _project_pixels(points, fov_hw, image_hw, min_depth):
    """Project OpenCV camera-frame points with centered intrinsics from FoV."""
    if points.ndim != 5 or points.shape[-3:] != (2, 21, 3):
        raise ValueError(
            "points must have shape [B,S,2,21,3], "
            f"got {tuple(points.shape)}"
        )
    if fov_hw.shape != points.shape[:2] + (2,):
        raise ValueError(
            f"fov_hw must have shape {points.shape[:2] + (2,)}, "
            f"got {tuple(fov_hw.shape)}"
        )

    height, width = image_hw
    points = points.float()
    raw_fov = fov_hw.float()
    fov = raw_fov.clamp(min=math.radians(1.0), max=math.pi - 1.0e-3)
    fy = (height / 2.0) / torch.tan(fov[..., 0] / 2.0)
    fx = (width / 2.0) / torch.tan(fov[..., 1] / 2.0)
    fx = fx[..., None, None]
    fy = fy[..., None, None]

    x, y, z = points.unbind(dim=-1)
    safe_z = z.clamp(min=min_depth)
    u = fx * x / safe_z + width / 2.0
    v = fy * y / safe_z + height / 2.0
    uv = torch.stack((u, v), dim=-1)
    finite = (
        torch.isfinite(points).all(dim=-1)
        & torch.isfinite(raw_fov).all(dim=-1)[..., None, None]
        & torch.isfinite(uv).all(dim=-1)
    )
    return uv, finite & (z > min_depth)


def penalized_reprojection_error(
    pred_joints,
    target_joints,
    presence_logits,
    target_hand_mask,
    fov_hw,
    image_hw,
    *,
    min_depth=0.01,
    presence_threshold=0.0,
):
    """Return EPE-p over GT joints that project inside the input image."""
    if pred_joints.shape != target_joints.shape:
        raise ValueError(
            "predicted and target joint shapes differ: "
            f"{tuple(pred_joints.shape)} != {tuple(target_joints.shape)}"
        )
    expected_hands = pred_joints.shape[:3]
    if presence_logits.shape != expected_hands:
        raise ValueError(
            f"presence_logits must have shape {expected_hands}, "
            f"got {tuple(presence_logits.shape)}"
        )
    if target_hand_mask.shape != expected_hands:
        raise ValueError(
            f"target_hand_mask must have shape {expected_hands}, "
            f"got {tuple(target_hand_mask.shape)}"
        )

    height, width = image_hw
    diagonal_value = math.hypot(height, width)
    diagonal = pred_joints.new_tensor(diagonal_value)
    target_uv, target_projectable = _project_pixels(
        target_joints, fov_hw, image_hw, min_depth
    )
    pred_uv, _ = _project_pixels(pred_joints, fov_hw, image_hw, min_depth)
    on_screen = (
        target_projectable
        & target_hand_mask.bool()[..., None]
        & (target_uv[..., 0] >= 0.0)
        & (target_uv[..., 0] < width)
        & (target_uv[..., 1] >= 0.0)
        & (target_uv[..., 1] < height)
    )

    raw_distance = torch.linalg.vector_norm(pred_uv - target_uv, dim=-1)
    distance = torch.nan_to_num(
        raw_distance,
        nan=diagonal_value,
        posinf=diagonal_value,
        neginf=diagonal_value,
    ).clamp(max=diagonal)
    detected = presence_logits.float() >= presence_threshold
    penalized = torch.where(detected[..., None], distance, diagonal)

    joint_count = on_screen.sum()
    epe = torch.where(on_screen, penalized, torch.zeros_like(penalized)).sum()
    epe = epe / joint_count.clamp(min=1)

    detected_joints = on_screen & detected[..., None]
    detected_count = detected_joints.sum()
    detected_epe = torch.where(
        detected_joints, distance, torch.zeros_like(distance)
    ).sum() / detected_count.clamp(min=1)
    missed_fraction = (joint_count - detected_count).float() / joint_count.clamp(min=1)
    return epe, {
        "detected_epe_px": detected_epe,
        "missed_joint_fraction": missed_fraction,
        "evaluated_joints": joint_count,
        "image_diagonal_px": diagonal,
    }


def image_hand_epe(ctx):
    """Term adapter used by both the training criterion and loss viewer."""
    epe, _ = penalized_reprojection_error(
        ctx["pred_joints"],
        ctx["target_joints"],
        ctx["presence_logits"],
        ctx["target_hand_mask"],
        ctx["fov_hw"],
        ctx["image_hw"],
        min_depth=ctx["min_depth"],
        presence_threshold=ctx["presence_threshold"],
    )
    return epe


def _image_hw_from_batch(batch):
    """Read spatial size without requiring visualization callers to copy images."""
    images = batch.get("images")
    if images is not None:
        if images.ndim < 2:
            raise ValueError(
                f"batch['images'] must have at least two dimensions, got {images.ndim}"
            )
        height, width = int(images.shape[-2]), int(images.shape[-1])
    else:
        image_hw = batch.get("image_hw")
        if image_hw is None:
            raise KeyError("image_hand requires batch['images'] or batch['image_hw']")
        if torch.is_tensor(image_hw):
            image_hw = image_hw.detach().cpu().reshape(-1).tolist()
        if len(image_hw) != 2:
            raise ValueError(
                f"batch['image_hw'] must contain [height, width], got {image_hw!r}"
            )
        height, width = (int(value) for value in image_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"image dimensions must be positive, got {(height, width)}")
    return height, width


def _mano_source_mask(batch, device):
    mask = batch["hand_kept"].to(device=device).bool()
    source_valid = batch.get("mano_gt_valid")
    if source_valid is None:
        source_valid = torch.full(
            (mask.shape[0],), "hand_gt" in batch, dtype=torch.bool, device=device
        )
    else:
        source_valid = source_valid.to(device=device).bool().reshape(mask.shape[0])
    return mask & source_valid[:, None, None]


class ImageHandLoss:
    """Match ViDiHand EPE-p while keeping presence BCE as the detection loss."""

    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.weight = float(cfg.get("weight", 1.0))
        self.min_depth = float(cfg.get("min_depth", 0.01))
        self.presence_threshold = float(cfg.get("presence_threshold", 0.0))
        self.terms = [("epe_penalized_px", image_hand_epe, 1.0)]
        if self.min_depth <= 0.0:
            raise ValueError("[image_hand] min_depth must be greater than zero")

    def _ctx(self, pred, batch):
        if "hand" not in pred:
            raise KeyError("[image_hand] requires pred['hand']")
        if "hand_presence_logits" not in pred:
            raise KeyError("[image_hand] requires pred['hand_presence_logits']")

        valid = _mano_source_mask(batch, pred["hand"].device)
        pred_joints = pred.get("_mano_joints21_cam")
        if pred_joints is None:
            pred_joints, pred_root_relative = joints21_from_hand_output(pred["hand"], valid)
            pred["_mano_joints21_cam"] = (pred_joints, pred_root_relative)
        else:
            pred_joints = pred_joints[0]

        with torch.no_grad():
            target_joints, _ = joints21_from_hand_output(batch["hand_gt"], valid)

        return {
            "pred_joints": pred_joints,
            "target_joints": target_joints,
            "presence_logits": pred["hand_presence_logits"],
            "target_hand_mask": valid,
            "fov_hw": batch["gt_pose_enc"][..., 7:9],
            "image_hw": _image_hw_from_batch(batch),
            "min_depth": self.min_depth,
            "presence_threshold": self.presence_threshold,
        }

    def __call__(self, pred, batch):
        if "hand" not in pred:
            return pred["pose_enc"].sum() * 0.0, {}
        ctx = self._ctx(pred, batch)
        epe, metrics = penalized_reprojection_error(
            ctx["pred_joints"],
            ctx["target_joints"],
            ctx["presence_logits"],
            ctx["target_hand_mask"],
            ctx["fov_hw"],
            ctx["image_hw"],
            min_depth=ctx["min_depth"],
            presence_threshold=ctx["presence_threshold"],
        )
        logs = {"loss/image_hand/epe_penalized_px": epe.detach()}
        logs.update(
            {
                f"metric/image_hand/{key}": value.detach()
                for key, value in metrics.items()
            }
        )
        return self.weight * epe, logs

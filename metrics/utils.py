"""Numerical kernel for canonical MANO metrics.

Camera projection, fixed left/right matching, missed-hand penalties, and
pooling of individual hand, joint, and temporal samples.
The public interface lives in evaluator.py.
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation
import torch


class _CameraMano:
    """Evaluation geometry: FP32 articulation and FP64 camera translation."""

    def __init__(self, model):
        self.model = model
        self.layer = model.layer
        self.device = model.device
        self.layer.parents = self.layer.parents.cpu()

    def mean_pose_rotations(self):
        mean = self.layer.hand_mean.detach().cpu().numpy().reshape(15, 3)
        return torch.from_numpy(
            Rotation.from_rotvec(mean.astype(np.float64)).as_matrix().astype(np.float32)
        )

    def _full_rotation_np(self, global_orient, hand_pose, betas, cam_trans, is_right):
        rotations = np.concatenate(
            [
                np.asarray(global_orient, np.float32).reshape(-1, 1, 3, 3),
                np.asarray(hand_pose, np.float32).reshape(-1, 15, 3, 3),
            ],
            axis=1,
        )
        return self.model.forward(
            rotations, betas, np.asarray(cam_trans, np.float64), int(is_right)
        )

    def mano2joint(self, params):
        values = {key: params[key] for key in PARAM_KEYS}
        values["cam_trans"] = np.asarray(values["cam_trans"], np.float64)
        values["valid"] = np.asarray(params.get("valid", True), dtype=bool) & np.isfinite(values["cam_trans"]).all(-1)
        return self.model.mano2joint(values)


N_HAND_JOINTS, N_JOINTS = 21, 42
MIN_DEPTH = 1e-6
INTRINSIC_KEYS = ("fx", "fy", "cx", "cy")
_SIDES = ("left", "right")
_MEAN_METRICS = {
    "pa_mpjpe_mm": "pa_mpjpe_mm",
    "px_err": "epe2d_px",
    "z_root_mm": "z_root_mm",
    "pred_jitter_mm": "pred_jitter_mm",
    "perfect_frame": "perfect_frame_acc",
    "pen_mpjpe_mm": "mpjpe_penalty_mm",
    "pen_mpjpe_camera_mm": "mpjpe_camera_penalty_mm",
    "pen_pa_mpjpe_mm": "pa_mpjpe_penalty_mm",
    "pen_epe2d_px": "epe2d_penalty_px",
    "pen_wrist_err_mm": "wrist_pos_error_penalty_mm",
    "pen_ct_m": "cam_trans_error_penalty_m",
}
PARAM_KEYS = ("global_orient", "hand_pose", "betas", "cam_trans")
FN_PLACEHOLDER_DEPTH_M = 0.5
CLAIM_THRESHOLD = 0.5
MIN_DEPTH_M = 0.01


def _bbox_from_points(points: np.ndarray) -> np.ndarray:
    """(..., N, 2) points -> (..., 4) xyxy axis-aligned boxes."""
    minimum = points.min(axis=-2)
    maximum = points.max(axis=-2)
    return np.concatenate([minimum, maximum], axis=-1)


def _iou(first_box: np.ndarray, second_box: np.ndarray) -> float:
    intersection_width = max(
        0.0, min(first_box[2], second_box[2]) - max(first_box[0], second_box[0])
    )
    intersection_height = max(
        0.0, min(first_box[3], second_box[3]) - max(first_box[1], second_box[1])
    )
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0
    union_area = (
        (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
        + (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
        - intersection
    )
    return float(intersection / union_area) if union_area > 0 else 0.0


def _dilate(box: np.ndarray, fraction: float) -> np.ndarray:
    padding_width = (box[2] - box[0]) * fraction * 0.5
    padding_height = (box[3] - box[1]) * fraction * 0.5
    return np.array(
        [
            box[0] - padding_width,
            box[1] - padding_height,
            box[2] + padding_width,
            box[3] + padding_height,
        ]
    )


def _iou_match_frames(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    pred_active: np.ndarray,
    pred_on: np.ndarray,
    gt_on: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Match fixed left/right slots using the release IoU rule."""
    num_frames = pred_boxes.shape[0]
    tp_mask = np.zeros((num_frames, 2), bool)
    false_positives = np.zeros((num_frames, 2), bool)
    false_negatives = np.zeros((num_frames, 2), bool)
    for frame in range(num_frames):
        active = [slot for slot in (0, 1) if pred_active[frame, slot]]
        dilated_gt_boxes = [
            _dilate(gt_boxes[frame, gt_slot], 0.10) for gt_slot in (0, 1)
        ]
        matches = {}
        for prediction_slot in active:
            overlaps = [
                _iou(pred_boxes[frame, prediction_slot], dilated_gt_boxes[gt_slot])
                for gt_slot in (0, 1)
            ]
            matching_scores = [
                (overlaps[gt_slot] + 0.5 * (gt_slot == prediction_slot))
                for gt_slot in (0, 1)
            ]
            gt_slot = int(np.argmax(matching_scores))
            if overlaps[gt_slot] <= 0.0:
                false_positives[frame, prediction_slot] = True  # zero overlap anywhere
            elif gt_slot != prediction_slot:
                false_positives[frame, prediction_slot] = True  # label mismatch
            else:
                matches[gt_slot] = (
                    prediction_slot  # same-side; 2 slots -> no tournament
                )
        for gt_slot in (0, 1):
            if gt_slot in matches:
                tp_mask[frame, gt_slot] = True
            else:
                false_negatives[frame, gt_slot] = (
                    True  # GT always present in this contract
                )
    # drop_all re-tally: silence off-screen GT (TP/FN) and off-screen preds
    # (FP); TN where a side has neither an on-screen pred nor on-screen GT
    tp_eff = tp_mask & gt_on
    fn_eff = false_negatives & gt_on
    fp_eff = false_positives & pred_on
    tn_eff = ~(pred_active & pred_on) & ~gt_on
    counts = np.zeros((1, 2, 4), np.int64)
    for slot in (0, 1):
        counts[0, slot] = [
            tp_eff[:, slot].sum(),
            fp_eff[:, slot].sum(),
            fn_eff[:, slot].sum(),
            tn_eff[:, slot].sum(),
        ]
    perfect = (~(fp_eff | fn_eff).any(axis=-1)).astype(np.float32)
    frame_counts = np.stack([tp_eff, fp_eff, fn_eff, tn_eff], axis=-1).astype(np.int64)
    return {
        "tp_mask": tp_eff,
        "counts": counts,
        "perfect": perfect,
        "frame_counts": frame_counts,
    }


def project_camera_points(points, intr, min_depth=MIN_DEPTH):
    """[..., 3] camera-frame metres -> [..., 2] pixels. z clamped to min_depth."""
    camera_points = np.asarray(points, dtype=np.float64)
    depth = np.maximum(camera_points[..., 2], min_depth)
    pixel_x = intr["fx"] * camera_points[..., 0] / depth + intr["cx"]
    pixel_y = intr["fy"] * camera_points[..., 1] / depth + intr["cy"]
    if "skew_xy" in intr:
        pixel_x += intr["skew_xy"] * camera_points[..., 1] / depth
        pixel_y += intr["skew_yx"] * camera_points[..., 0] / depth
    return np.stack([pixel_x, pixel_y], axis=-1)


def _intrinsics_torch(intr, like):
    """fx, fy, cx, cy as tensors broadcastable over ``like[..., 0]`` (``like`` [..., 2 | 3] points): from a mapping (scalars or
    tensors) or a ``[..., 4]`` tensor (fx fy cx cy). Scalars stay scalars. A tensor value is LEADING-aligned with the points:
    it is padded with trailing singleton axes up to the rank of ``like[..., 0]`` (a per-sample [N] value and [N, J, 2] points ->
    [N, 1]; [N] and [N, T, J, 2] -> [N, 1, 1]; an already-shaped [N, 1] value
    is left alone). No length heuristics: a value of length N is per sample because of its position, not its size.
    """
    if isinstance(intr, torch.Tensor):
        values = list(intr.to(like.device, like.dtype).unbind(-1))
    else:
        values = [
            torch.as_tensor(intr[k], device=like.device, dtype=like.dtype)
            for k in INTRINSIC_KEYS
        ]
    target_rank = like.ndim - 1  # rank of like[..., 0]
    components = []
    for value in values:
        if 0 < value.ndim < target_rank:
            value = value.reshape(*value.shape, *([1] * (target_rank - value.ndim)))
        components.append(value)
    return components


def project_torch(points, intr, min_depth=MIN_DEPTH):
    """Differentiable pinhole projection, the one torch owner: [..., 3] camera metres -> [..., 2] px.

    z is floored at ``min_depth`` for the division only (a point behind the camera projects like one at the
    floor); non-positive depth is penalised elsewhere, never declared valid geometry. ``intr``: a mapping or a
    ``[..., 4]`` tensor.
    """
    z = points[..., 2].clamp_min(min_depth)
    fx, fy, cx, cy = _intrinsics_torch(intr, points)
    x, y = fx * points[..., 0] / z + cx, fy * points[..., 1] / z + cy
    if isinstance(intr, dict) and "skew_xy" in intr:
        skew = [torch.as_tensor(intr[key], device=points.device, dtype=points.dtype)
                for key in ("skew_xy", "skew_yx")]
        skew = [value.reshape(*value.shape, *([1] * (points.ndim - 1 - value.ndim)))
                if value.ndim else value for value in skew]
        x = x + skew[0] * points[..., 1] / z
        y = y + skew[1] * points[..., 0] / z
    return torch.stack((x, y), -1)


def visibility_mask(points_px, points_cam, canvas_wh, margin=0.0, margin_frac=0.0):
    """1.0 iff the projection lands on-canvas AND the point is in front of the
    camera. points_px [..., 2], points_cam [..., 3], canvas_wh (W, H).

    ``margin`` is a pixel band around the canvas (both axes); ``margin_frac`` a per-axis
    fractional band (margin_frac * W horizontally, margin_frac * H vertically), the two add.
    Zero margins use the strict canvas boundary."""
    width, height = canvas_wh
    margin_x, margin_y = margin + margin_frac * width, margin + margin_frac * height
    pixel_x, pixel_y = points_px[..., 0], points_px[..., 1]
    on = (
        (pixel_x >= -margin_x)
        & (pixel_x < width + margin_x)
        & (pixel_y >= -margin_y)
        & (pixel_y < height + margin_y)
    )
    return (on & (points_cam[..., 2] > 1e-6)).astype(np.float32)


def hand_support(uv, valid, canvas_wh, margin=0.2, band=0.0):
    """Nearest-joint normalized Euclidean distance to the canvas rectangle.

    ``band``: a joint counts as
    ``inside`` within a per-axis band of ``band`` x canvas outside the frame
    ([-band, 1 + band) in normalized units); 0 = the strict canvas (unchanged).

    uv [...,42,2], valid [...,42]. In-frame hands have unit 3D weight;
    fully off-frame hands decay with cubic smoothstep to zero at margin.
    Invalid annotations are never negatives and never select the nearest joint.
    """
    if margin <= 0:
        raise ValueError("[hand_support] positive normalized margin required")
    normalized_points = uv.reshape(*uv.shape[:-2], 2, 21, 2) / uv.new_tensor(canvas_wh)
    valid_joints = valid.reshape(*valid.shape[:-1], 2, 21) & torch.isfinite(
        normalized_points
    ).all(-1)
    delta = torch.maximum(-normalized_points, normalized_points - 1).clamp_min(0)
    distance = delta.norm(dim=-1).masked_fill(~valid_joints, float("inf")).amin(-1)
    if not (0 <= band < 1):
        raise ValueError("[hand_support] band must be in [0, 1)")
    inside = (
        valid_joints
        & (normalized_points >= -band).all(-1)
        & (normalized_points < 1 + band).all(-1)
    )
    any_in = inside.any(-1)
    transition = (distance / margin).clamp(0, 1)
    weight = 1 - 3 * transition.square() + 2 * transition.pow(3)
    return weight, distance, inside.flatten(-2), any_in


def _block(
    true_positives: float,
    false_positives: float,
    false_negatives: float,
    true_negatives: float,
) -> Dict[str, float]:
    def ratio(numerator, denominator):
        return (
            float(numerator) / float(denominator) if denominator > 0 else float("nan")
        )

    precision = ratio(true_positives, true_positives + false_positives)
    recall = ratio(true_positives, true_positives + false_negatives)
    f1 = ratio(
        2 * true_positives, 2 * true_positives + false_positives + false_negatives
    )
    return {
        "tp": int(true_positives),
        "fp": int(false_positives),
        "fn": int(false_negatives),
        "tn": int(true_negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": ratio(
            true_positives + true_negatives,
            true_positives + false_positives + false_negatives + true_negatives,
        ),
        "miss_rate": 1.0 - recall if recall == recall else float("nan"),
        "fdr": 1.0 - precision if precision == precision else float("nan"),
    }


def _detection_metrics(counts: np.ndarray) -> Dict[str, Dict[str, float]]:
    """(N, 2, 4) per-clip counts -> {left, right, total} metric blocks."""
    summed = np.asarray(counts, np.int64).sum(axis=0)  # (2, 4)
    summary = {side: _block(*summed[slot]) for slot, side in enumerate(_SIDES)}
    summary["total"] = _block(*summed.sum(axis=0))
    return summary


def metrics_from_buffers(buffers: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Reduce flat buffers to the metric dict (NaN where data is absent)."""

    def _mean(key):
        values = buffers.get(key)
        return (
            float(values.mean()) if values is not None and values.size else float("nan")
        )

    summary: Dict[str, Any] = {name: _mean(key) for key, name in _MEAN_METRICS.items()}
    if "mpjpe_camera_raw_mm" in buffers:
        summary["mpjpe_camera_raw_mm"] = _mean("mpjpe_camera_raw_mm")
    if (
        "temporal_samples" in buffers
    ):  # 0 = only one- or two-frame samples: Jitter is undefined (N/A), not a failed measurement
        summary["temporal_samples"] = int(buffers["temporal_samples"].sum())
    joint_errors = buffers.get("joint_err_mm")
    joint_errors = (
        joint_errors
        if joint_errors is not None
        else np.zeros((0, N_HAND_JOINTS), np.float32)
    )
    scored_joint_errors = joint_errors[
        ~np.isnan(joint_errors)
    ]  # Inf = failed prediction, retained
    summary["mpjpe_mm"] = (
        float(scored_joint_errors.mean()) if scored_joint_errors.size else float("nan")
    )
    detection_counts = buffers.get("det_counts")
    summary["hand_detection"] = (
        _detection_metrics(detection_counts)
        if detection_counts is not None and detection_counts.size
        else None
    )
    if "pen_mpjpe_camera_mm" not in buffers:
        summary.pop("mpjpe_camera_penalty_mm", None)
    return summary


def table_row(summary):
    detection = summary["hand_detection"]["total"]
    return {
        "FAcc": summary["perfect_frame_acc"],
        "F1": detection["f1"],
        "MPJPE-p": summary["mpjpe_penalty_mm"],
        "PA-p": summary["pa_mpjpe_penalty_mm"],
        "EPE2D-p": summary["epe2d_penalty_px"],
        "MPJPE-abs-p": summary["mpjpe_camera_penalty_mm"],
        "Jitter": summary["pred_jitter_mm"],
        "fp": detection["fp"],
        "fn": detection["fn"],
        "z_root": summary["z_root_mm"],
    }


def _project_track(points, intr, min_depth=1e-6):
    """Project [F,...,3] with constant or per-frame intrinsics through one owner."""
    points_array = np.asarray(points)
    values = {}
    for key in intr:
        value = np.asarray(intr[key], np.float64)
        if value.ndim:
            if value.shape != (len(points_array),):
                raise ValueError(f"{key} must be scalar or one value per frame")
            value = value.reshape(len(points_array), *([1] * (points_array.ndim - 2)))
        if not np.isfinite(value).all() or (key in ("fx", "fy") and (value <= 0).any()):
            raise ValueError(f"Invalid camera intrinsics: {key}")
        values[key] = value
    return project_camera_points(points_array, values, min_depth=min_depth)


def _procrustes_align_batch(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity-align each ``pred`` sample onto ``gt``: (B, J, 3) -> (B, J, 3).

    One batched SVD, determinant-corrected to proper rotations; the scale
    falls back to 1 for a degenerate (zero-spread) prediction.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mu_pred = pred.mean(axis=1, keepdims=True)
    mu_gt = gt.mean(axis=1, keepdims=True)

    Pc = pred - mu_pred
    Gc = gt - mu_gt
    H = np.einsum("bji,bjk->bik", Pc, Gc)

    U, _, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    UT = U.transpose(0, 2, 1)

    det = np.linalg.det(V @ UT)
    sign = np.where(np.sign(det) == 0, 1.0, np.sign(det))
    D = np.broadcast_to(np.eye(3), (Pc.shape[0], 3, 3)).copy()
    D[:, 2, 2] = sign
    R = V @ D @ UT

    RH = R @ H
    trace_RH = RH[:, 0, 0] + RH[:, 1, 1] + RH[:, 2, 2]
    PtP = np.einsum("bji,bjk->bik", Pc, Pc)
    trace_PtP = PtP[:, 0, 0] + PtP[:, 1, 1] + PtP[:, 2, 2]
    scale = np.where(trace_PtP > 1e-12, trace_RH / np.maximum(trace_PtP, 1e-12), 1.0)
    return scale[:, None, None] * (Pc @ R.transpose(0, 2, 1)) + mu_gt


def _camera_raw_errors(pred3d, gt, valid, intr, canvas_wh):
    """MPJPE-abs TP-population raw errors (mm, float32): every valid joint of a GT hand with an on-canvas joint."""
    prediction = torch.as_tensor(pred3d, dtype=torch.float64).reshape(-1, 42, 3)
    ground_truth = torch.as_tensor(gt, dtype=torch.float64).reshape_as(prediction)
    valid_joints = (
        torch.as_tensor(valid).reshape(-1, 42)
        & torch.isfinite(ground_truth).all(-1)
        & (ground_truth[..., 2] > 1e-6)
    )
    _, _, _, any_in = hand_support(
        project_torch(ground_truth, intr), valid_joints, canvas_wh
    )
    eligible = valid_joints & any_in[..., None].expand(-1, -1, 21).flatten(-2)
    errors = (prediction - ground_truth).norm(dim=-1) * 1000
    errors = torch.nan_to_num(
        errors, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
    )
    return errors[eligible].numpy().astype(np.float32)


def _normalize_joint_observations(prediction, num_frames):
    """Preserve the reference dtypes for decoded MANO joints and projections."""
    joints = np.asarray(prediction["joints3d"])
    dtype = np.float64 if joints.dtype == np.float64 else np.float32
    joints = np.asarray(joints, dtype)
    pixels = np.asarray(prediction["joints2d"], np.float32)
    scores = np.asarray(prediction["vis_hand"], np.float32)
    if joints.shape != (num_frames, N_JOINTS, 3):
        raise ValueError("Decoded MANO joints must have shape (frames, 42, 3)")
    if pixels.shape != (num_frames, N_JOINTS, 2):
        raise ValueError("Projected MANO joints must have shape (frames, 42, 2)")
    if (
        scores.shape != (num_frames, 2)
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError("Hand claims must be finite [frames, 2] probabilities")
    return dict(joints3d=joints, joints2d=pixels, vis_hand=scores)


def _second_diff_mags_mm(track: np.ndarray) -> np.ndarray:
    """(F, 21, 3) metres -> (F-2, 21) accel magnitudes, mm."""
    positions = track.astype(np.float64)
    acceleration = positions[2:] - 2.0 * positions[1:-1] + positions[:-2]
    return np.linalg.norm(acceleration, axis=-1) * 1000.0


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N, J) values, (N, J) bool -> (N,) mean over the masked joints."""
    counts = mask.sum(axis=-1)
    result = np.where(mask, np.nan_to_num(values, nan=np.inf, posinf=np.inf), 0.0).sum(
        axis=-1
    ) / np.maximum(counts, 1)
    return np.where(counts > 0, result, np.nan)


def _masked_procrustes_errors(
    pred_rel: np.ndarray,
    gt_rel: np.ndarray,
    side: np.ndarray,
    mask2: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-sample PA-MPJPE (mm) fitted and averaged over the scored joints of
    each sample's slot (``mask2`` (2, 21)); identical to the plain batch call
    when every joint is scored."""
    masks = mask2[side].copy()
    if valid is not None:
        masks &= valid
    errors = np.full((len(pred_rel),), np.nan, np.float64)
    unique, groups = np.unique(masks, axis=0, return_inverse=True)
    for group, joint_mask in enumerate(unique):
        indices = np.flatnonzero(groups == group)
        if indices.size == 0 or joint_mask.sum() < 3:
            continue
        # A missing prediction for labelled GT is a failure, never an SVD input
        # or a removed sample. Unlabelled joints never enter the alignment.
        finite = np.isfinite(pred_rel[indices][:, joint_mask]).all(axis=(1, 2))
        errors[indices[~finite]] = np.inf
        indices = indices[finite]
        if not len(indices):
            continue
        aligned = _procrustes_align_batch(
            pred_rel[indices][:, joint_mask], gt_rel[indices][:, joint_mask]
        )
        errors[indices] = (
            np.linalg.norm(aligned - gt_rel[indices][:, joint_mask], axis=-1).mean(
                axis=-1
            )
            * 1000.0
        )
    return errors


def _missed_hand_rms_radius_mm(
    ground_truth: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """GT-centred RMS radius over the same scored joints as PA, in mm.

    A least-squares similarity fit can collapse onto the GT centroid, so its
    RMS residual bounds mean joint distance by this radius. No model-dependent
    penalty or new threshold is needed. Fewer than three joints is undefined,
    as in _pa_masked.
    """
    ground_truth = np.asarray(ground_truth, np.float64)
    valid = np.asarray(valid, bool)
    count = valid.sum(-1)
    safe = np.where(valid[..., None], ground_truth, 0.0)
    center = safe.sum(-2) / np.maximum(count[:, None], 1)
    spread = np.where(valid, ((safe - center[:, None]) ** 2).sum(-1), 0.0)
    return np.where(
        count >= 3, np.sqrt(spread.sum(-1) / np.maximum(count, 1)) * 1000.0, np.nan
    )


class MetricKernel:
    """Fixed release protocol: two hand slots, any visible joint, IoU matching.

    Missed hands use the screen-centre placeholder at 0.5 metres; the PA
    penalty uses the GT RMS radius. Individual samples are pooled equally.
    """

    def __init__(self, mano_layers):
        self.mano_layers = mano_layers
        self._mean_hand_joints = None

    def _mean_hand_joints_relative_to_wrist(self) -> np.ndarray:
        """(2, 21, 3) wrist-relative J21 of the MANO mean hand: identity root, the full mean-pose finger rotations
        (``Mano.mean_pose_rotations``), betas 0. The left placeholder is the mirrored right mean hand.
        """
        if self._mean_hand_joints is None:
            identity_orientation = np.eye(3, dtype=np.float32)[None, None]
            mean_hand_pose = self.mano_layers.mean_pose_rotations().cpu().numpy()[None]
            relative_joints = []
            for slot in (0, 1):
                _, joints = self.mano_layers._full_rotation_np(
                    identity_orientation,
                    mean_hand_pose,
                    np.zeros((1, 10), np.float32),
                    np.zeros((1, 3), np.float32),
                    is_right=bool(slot),
                )
                relative_joints.append(joints[0] - joints[0, 0:1])
            self._mean_hand_joints = np.stack(relative_joints)
        return self._mean_hand_joints

    def screen_center_placeholder(self, intr, canvas_wh):
        """Dataset-independent [2,21,3] MANO mean hand, wrist on image-center ray.

        Centre means the image rectangle centre (W/2,H/2), not the principal
        point. Translation positions J21 wrist, not the offset MANO origin.
        """
        width, height = canvas_wh
        depth = float(FN_PLACEHOLDER_DEPTH_M)
        x = (width / 2 - intr["cx"]) / intr["fx"] * depth
        y = (height / 2 - intr["cy"]) / intr["fy"] * depth
        if "skew_xy" in intr:
            determinant = intr["fx"] * intr["fy"] - intr["skew_xy"] * intr["skew_yx"]
            x = ((width / 2 - intr["cx"]) * intr["fy"]
                 - (height / 2 - intr["cy"]) * intr["skew_xy"]) / determinant * depth
            y = ((height / 2 - intr["cy"]) * intr["fx"]
                 - (width / 2 - intr["cx"]) * intr["skew_yx"]) / determinant * depth
        wrist = np.stack(np.broadcast_arrays(x, y, depth), axis=-1)
        if wrist.ndim == 2:
            wrist = wrist[:, None, None, :]
        return self._mean_hand_joints_relative_to_wrist().astype(np.float64) + wrist

    def _screen_placeholder_buffers(
        self,
        buffers,
        pred_j,
        gt_j,
        valid,
        pred_2d,
        gt_2d,
        gt_vis,
        true_positives,
        false_negatives,
        mask2,
        intr,
        canvas_wh,
    ):
        """Add the placeholder-penalty buffers. Missed hands are scored at the screen-centre placeholder."""
        num_frames = len(gt_j)
        side = np.tile(np.array([0, 1]), num_frames)
        placeholder = np.broadcast_to(
            self.screen_center_placeholder(intr, canvas_wh), gt_j.shape
        )
        prediction = np.where(
            false_negatives.reshape(num_frames, 2, 1, 1), placeholder, pred_j
        ).reshape(-1, 21, 3)
        ground_truth = gt_j.reshape(-1, 21, 3).astype(np.float64)
        valid_joints = valid.reshape(-1, 21)
        sample_indices = np.concatenate(
            [np.flatnonzero(true_positives), np.flatnonzero(false_negatives)]
        )
        relative = (prediction - prediction[:, :1]) - (
            ground_truth - ground_truth[:, :1]
        )
        buffers["pen_mpjpe_mm"] = _masked_mean(
            np.linalg.norm(relative[sample_indices], axis=-1) * 1000,
            valid_joints[sample_indices],
        )
        buffers["pen_pa_mpjpe_mm"] = _masked_procrustes_errors(
            prediction[sample_indices],
            ground_truth[sample_indices],
            side[sample_indices],
            mask2,
            valid_joints[sample_indices],
        )
        # True positives come first and keep their aligned errors. Misses
        # receive the GT-only bound, not an aligned synthetic mean hand.
        missed_indices = np.flatnonzero(false_negatives)
        buffers["pen_pa_mpjpe_mm"][int(true_positives.sum()) :] = (
            _missed_hand_rms_radius_mm(
                ground_truth[missed_indices], valid_joints[missed_indices]
            )
        )
        buffers["pen_mpjpe_camera_mm"] = _masked_mean(
            np.linalg.norm(
                prediction[sample_indices] - ground_truth[sample_indices], axis=-1
            )
            * 1000,
            valid_joints[sample_indices],
        )
        buffers["pen_wrist_err_mm"] = (
            np.linalg.norm(
                prediction[sample_indices, 0] - ground_truth[sample_indices, 0], axis=-1
            )
            * 1000
        )
        ph_uv = _project_track(placeholder, intr, MIN_DEPTH_M)
        uv = np.where(false_negatives.reshape(num_frames, 2, 1, 1), ph_uv, pred_2d)
        distance = np.minimum(np.linalg.norm(uv - gt_2d, axis=-1), np.hypot(*canvas_wh))
        buffers["pen_epe2d_px"] = distance[gt_vis]

    def _evaluate_joints(self, prediction, gt_joints, intr, canvas_wh, matching):
        """Compute metric samples from MANO-decoded joints and one shared match."""
        gt_joints = np.asarray(gt_joints, np.float32)
        if gt_joints.ndim != 4 or gt_joints.shape[1:] != (2, N_HAND_JOINTS, 3):
            raise ValueError("GT MANO joints must have shape (frames, 2, 21, 3)")
        num_frames = int(gt_joints.shape[0])
        prediction = _normalize_joint_observations(prediction, num_frames)
        gt_joints_3d = gt_joints
        valid = np.isfinite(gt_joints_3d).all(-1)
        mask2 = np.ones((2, N_HAND_JOINTS), bool)
        valid = valid & mask2[None]
        gt_joints_3d = np.where(valid[..., None], gt_joints_3d, 0.0).astype(np.float32)
        gt_joints_2d = _project_track(gt_joints_3d, intr, MIN_DEPTH_M).astype(
            np.float32
        )
        gt_visible = (
            (visibility_mask(gt_joints_2d, gt_joints_3d, canvas_wh) > 0.5)
            & valid
            & (gt_joints_3d[..., 2] > MIN_DEPTH_M)
        )
        hand_on = gt_visible.sum(-1) >= 1
        pred_joints_3d = prediction["joints3d"].reshape(num_frames, 2, N_HAND_JOINTS, 3)
        pred_joints_2d = prediction["joints2d"].reshape(num_frames, 2, N_HAND_JOINTS, 2)
        match = matching
        if (
            np.shape(match.get("tp_mask")) != (num_frames, 2)
            or np.shape(match.get("counts")) != (1, 2, 4)
            or np.shape(match.get("perfect")) != (num_frames,)
        ):
            raise ValueError("Invalid shared clip matching")

        # TP-only columns: fixed drop_all = on-screen TP hands only
        keep_hands = match["tp_mask"]
        keep = keep_hands.reshape(-1)

        buffers: Dict[str, np.ndarray] = {}
        pred_matched = pred_joints_3d.reshape(-1, N_HAND_JOINTS, 3).astype(np.float64)[
            keep
        ]
        gt_matched = gt_joints_3d.reshape(-1, N_HAND_JOINTS, 3).astype(np.float64)[keep]
        pred_relative = pred_matched - pred_matched[:, 0:1]
        gt_relative = gt_matched - gt_matched[:, 0:1]
        side_all = np.tile(np.array([0, 1]), num_frames)
        valid_matched = valid.reshape(-1, N_HAND_JOINTS)[keep]  # scored joints
        joint_errors = np.linalg.norm(pred_relative - gt_relative, axis=-1) * 1000.0
        # unscored joints are NaN; metrics_from_buffers reduces over the finite entries
        buffers["joint_err_mm"] = np.where(
            valid_matched & valid_matched[:, :1],
            np.nan_to_num(joint_errors, nan=np.inf, posinf=np.inf),
            np.nan,
        ).astype(np.float32)
        depth_errors = np.abs(pred_matched[..., 2] - gt_matched[..., 2]) * 1000.0
        buffers["z_root_mm"] = depth_errors[:, 0][valid_matched[:, 0]].astype(
            np.float32
        )
        buffers["pa_mpjpe_mm"] = _masked_procrustes_errors(
            pred_matched, gt_matched, side_all[keep], mask2, valid_matched
        )

        # temporal: a window counts iff its three consecutive frames are all
        # scored (contiguous TP runs, on-screen gated) for that hand
        def _run_ok(slot):
            hand_matches = keep_hands[:, slot]
            consecutive = (hand_matches[2:] & hand_matches[1:-1] & hand_matches[:-2])[
                :, None
            ]
            return consecutive & valid[2:, slot] & valid[1:-1, slot] & valid[:-2, slot]

        buffers["pred_jitter_mm"] = np.concatenate(
            [
                _second_diff_mags_mm(pred_joints_3d[:, slot])[_run_ok(slot)].ravel()
                for slot in (0, 1)
            ]
        )
        # A clip shorter than three frames contributes no temporal samples.
        buffers["temporal_samples"] = np.array([int(num_frames >= 3)], np.int64)

        diag = float(np.hypot(float(canvas_wh[0]), float(canvas_wh[1])))
        pixel_errors = np.linalg.norm(
            (pred_joints_2d - gt_joints_2d).astype(np.float64), axis=-1
        )  # (F, 2, 21)
        buffers["px_err"] = np.minimum(pixel_errors, diag)[gt_visible].astype(
            np.float32
        )

        # Detection and missed-hand penalties; each slot identifies its hand side.
        true_positives = match["tp_mask"].reshape(-1)
        false_negatives = (hand_on & ~match["tp_mask"]).reshape(-1)
        buffers["det_counts"] = match["counts"]
        buffers["perfect_frame"] = match["perfect"]
        # Hand-frame identities (flat index = frame * 2 + slot) of the detection outcomes, so callers can read which
        # hands the evaluator counted without re-running the matcher. No summary metric reads them.
        self._screen_placeholder_buffers(
            buffers,
            pred_joints_3d,
            gt_joints_3d,
            valid,
            pred_joints_2d,
            gt_joints_2d,
            gt_visible,
            true_positives,
            false_negatives,
            mask2,
            intr,
            canvas_wh,
        )
        buffers["mpjpe_camera_raw_mm"] = _camera_raw_errors(
            pred_joints_3d, gt_joints_3d, valid, intr, canvas_wh
        )
        return buffers

    def evaluate_joint_clip(self, pred_joints, scores, gt_joints, intr, canvas_wh):
        """Collect samples from camera-space joint predictions.

        ``pred_joints`` [F, 2, 21, 3] metres (left, right; MANO16 + the five
        tips, the order ``ManoModel.mano2joint`` decodes the GT in), ``scores``
        [F, 2] hand claims in [0, 1]; a claimed hand needs finite joints.
        Returns the sample buffers and the shared matching.
        """
        num_frames = len(gt_joints)
        if num_frames == 0 or np.shape(pred_joints) != (num_frames, 2, N_HAND_JOINTS, 3):
            raise ValueError("Joint prediction must be [F, 2, 21, 3] on the GT timeline")
        available = np.isfinite(pred_joints).all(axis=(-1, -2))
        scores = np.asarray(scores, np.float32)
        if (
            scores.shape != (num_frames, 2)
            or not np.isfinite(scores).all()
            or (scores < 0).any()
            or (scores > 1).any()
        ):
            raise ValueError("Hand claims must be finite [F,2] probabilities")
        active = scores > CLAIM_THRESHOLD
        if (active & ~available).any():
            raise ValueError("A claimed hand has no finite joints")
        pred_pixels = _project_track(pred_joints, intr, MIN_DEPTH_M)
        gt_pixels = _project_track(gt_joints, intr, MIN_DEPTH_M)
        width, height = canvas_wh
        if not np.isfinite([width, height]).all() or min(width, height) <= 0:
            raise ValueError("Canvas dimensions must be finite and positive")

        def inside(uv, xyz):
            return (
                np.isfinite(xyz).all(-1)
                & (xyz[..., 2] > MIN_DEPTH_M)
                & (uv[..., 0] >= 0)
                & (uv[..., 0] < width)
                & (uv[..., 1] >= 0)
                & (uv[..., 1] < height)
            )

        gt_on = inside(gt_pixels, gt_joints).any(-1)
        pred_on = active & inside(pred_pixels, pred_joints).any(-1)
        pred_boxes = _bbox_from_points(pred_pixels)
        gt_boxes = _bbox_from_points(gt_pixels)
        # Zero boxes only for genuinely unavailable slots; they cannot overlap.
        pred_boxes[~available] = 0
        gt_boxes[~np.isfinite(gt_joints).all(axis=(-1, -2))] = 0
        match = _iou_match_frames(pred_boxes, gt_boxes, active, pred_on, gt_on)
        # Pass binary claims because the matching threshold was applied once above.
        buffers = self._evaluate_joints(
            dict(
                joints3d=pred_joints.reshape(num_frames, 42, 3),
                joints2d=pred_pixels.reshape(num_frames, 42, 2),
                vis_hand=active.astype(np.float32),
            ),
            gt_joints,
            intr,
            canvas_wh,
            match,
        )
        if (
            int(match["tp_mask"].sum() + (gt_on & ~match["tp_mask"]).sum())
            != buffers["pen_mpjpe_mm"].size
        ):
            raise ValueError("Hand and joint metric populations disagree")
        return buffers, dict(match, gt_on=gt_on)

    def evaluate_mano_clip(self, prediction, gt_mano, intr, canvas_wh):
        """Collect samples from a MANO prediction: its decoded joints
        (``evaluate_joint_clip``) plus the camera-translation penalty."""
        pred_params = {key: np.asarray(prediction[key]) for key in PARAM_KEYS}
        if "vis_hand" in prediction:
            pred_params["vis_hand"] = prediction["vis_hand"]
        gt_params = {key: np.asarray(gt_mano[key]) for key in PARAM_KEYS}
        gt_params["valid"] = gt_mano.get("valid", True)
        num_frames = len(gt_params["cam_trans"])
        if num_frames == 0 or len(pred_params["cam_trans"]) != num_frames:
            raise ValueError("MANO prediction and GT must share a nonempty timeline")
        pred_joints = self.mano_layers.mano2joint(pred_params)
        gt_joints = self.mano_layers.mano2joint(gt_params)
        available = np.isfinite(pred_joints).all(axis=(-1, -2))
        scores = pred_params.get("vis_hand")
        scores = (
            available.astype(np.float32)
            if scores is None
            else np.asarray(scores, np.float32)
        )
        buffers, match = self.evaluate_joint_clip(pred_joints, scores, gt_joints, intr, canvas_wh)
        true_positives, false_negatives = match["tp_mask"], match["gt_on"] & ~match["tp_mask"]
        buffers["pen_ct_m"] = np.concatenate(
            [
                np.linalg.norm(
                    pred_params["cam_trans"][true_positives].astype(np.float64)
                    - gt_params["cam_trans"][true_positives],
                    axis=-1,
                ),
                np.linalg.norm(
                    gt_params["cam_trans"][false_negatives].astype(np.float64), axis=-1
                ),
            ]
        )
        return buffers

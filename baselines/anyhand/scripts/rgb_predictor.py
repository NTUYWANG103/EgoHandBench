"""Native AnyHand wilor predictor used by this baseline."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from contextlib import contextmanager
import numpy as np
import torch


@contextmanager
def _torch_load_trusted():
    """
    Temporarily force torch.load(weights_only=False).

    PyTorch 2.6 flipped the default to weights_only=True, which refuses to
    unpickle arbitrary Python objects for safety. This breaks loaders that
    pickle non-tensor objects — notably ultralytics' detector.pt, which
    pickles a full PoseModel instance.

    We trust our own checkpoints (pinned HuggingFace repos). Use this
    context manager around load calls that do NOT expose a weights_only
    kwarg directly (e.g. ultralytics.YOLO()).

    For callers that do expose the kwarg (e.g. Lightning's
    load_from_checkpoint), prefer passing weights_only=False explicitly.
    """
    orig = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = orig


@dataclass
class HandPrediction:
    """One native hand prediction in the original image/camera coordinates."""

    vertices: np.ndarray
    cam_t: np.ndarray
    is_right: bool
    score: float


class AnyHandPredictor:
    """Fixed-backend native hand predictor for this baseline."""

    def __init__(
        self,
        *,
        device,
        wilor_ckpt,
        wilor_mano_dir,
        detector_pt,
        wilor_cfg,
        det_conf=0.3,
        det_iou=0.3,
        rescale_factor=2.0,
        batch_size=16,
    ) -> None:
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.rescale_factor = rescale_factor
        self.batch_size = batch_size
        self.device = torch.device(device)
        self._wilor_ckpt = wilor_ckpt
        self._wilor_cfg = wilor_cfg
        self._wilor_mano_dir = wilor_mano_dir
        self._detector_pt = detector_pt
        self._load_detector()
        self._load_wilor()

    def _load_detector(self) -> None:
        """Load the native YOLO hand detector."""
        from ultralytics import YOLO

        with _torch_load_trusted():
            self._detector = YOLO(self._detector_pt).to(self.device)

    def _load_wilor(self) -> None:
        """Load the AnyHand-WiLoR model from the WiLoR submodule."""
        from baselines.anyhand.WiLoR.wilor.models import load_wilor

        model, cfg = load_wilor(
            checkpoint_path=self._wilor_ckpt,
            cfg_path=self._wilor_cfg,
            mano_dir=self._wilor_mano_dir,
        )
        model = model.to(self.device)
        model.eval()
        self._wilor_model = model
        self._wilor_model_cfg = cfg

    def _detect_hands(
        self, img_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the YOLO hand detector on a single BGR image.

        Returns
        -------
        boxes    : (N, 4) float32  [x1, y1, x2, y2] in pixels
        is_right : (N,)   bool     True = right hand  (YOLO class 1)
        scores   : (N,)   float32  confidence scores
        """
        results = self._detector(
            img_bgr, verbose=False, conf=self.det_conf, iou=self.det_iou
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.array([], dtype=bool),
                np.array([], dtype=np.float32),
            )
        det = results[0]
        boxes = det.boxes.xyxy.cpu().numpy().astype(np.float32)
        cls = det.boxes.cls.cpu().numpy().astype(int)
        scores = det.boxes.conf.cpu().numpy().astype(np.float32)
        is_right = cls == 1
        return (boxes, is_right, scores)

    def _run_wilor(
        self,
        img_bgr: np.ndarray,
        boxes: np.ndarray,
        is_right: np.ndarray,
        scores: np.ndarray,
        batch_size: int | None = None,
    ) -> List[HandPrediction]:
        from baselines.anyhand.WiLoR.wilor.datasets.vitdet_dataset import (
            ViTDetDataset,
        )

        model = self._wilor_model
        model_cfg = self._wilor_model_cfg
        W, H = (img_bgr.shape[1], img_bgr.shape[0])
        img_size = torch.tensor([W, H], dtype=torch.float32, device=self.device)
        scaled_focal = float(
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(W, H)
        )
        dataset = ViTDetDataset(
            model_cfg, img_bgr, boxes, is_right, rescale_factor=self.rescale_factor
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size if batch_size is None else batch_size, shuffle=False, num_workers=0
        )
        return self._collect_predictions(
            loader, model, is_right, scores, img_size, scaled_focal
        )

    def _collect_predictions(
        self,
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        is_right: np.ndarray,
        scores: np.ndarray,
        img_size: torch.Tensor,
        scaled_focal: float,
    ) -> List[HandPrediction]:
        """Run the selected native model on hand crops."""
        preds: List[HandPrediction] = []
        hand_idx = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = model(batch)
                multiplier = (2 * batch["right"] - 1).float()
                pred_cam = out["pred_cam"].clone()
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                pred_verts = out["pred_vertices"]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                cam_t_full = _cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size, scaled_focal
                )
                B = pred_cam.shape[0]
                for i in range(B):
                    is_right_i = float(batch["right"][i].item())
                    flip = float(2 * is_right_i - 1)
                    verts_np = pred_verts[i].cpu().numpy().astype(np.float32)
                    verts_np[:, 0] = flip * verts_np[:, 0]
                    preds.append(
                        HandPrediction(
                            vertices=verts_np,
                            cam_t=cam_t_full[i].cpu().numpy().astype(np.float32),
                            is_right=bool(is_right[hand_idx]),
                            score=float(scores[hand_idx]),
                        )
                    )
                    hand_idx += 1
        return preds

    def predict(self, image: np.ndarray, *, batch_size: int | None = None) -> List[HandPrediction]:
        """Predict hands from one BGR frame."""
        img_bgr = image
        boxes, is_right, scores = self._detect_hands(img_bgr)
        if len(boxes) == 0:
            return []
        results: List[HandPrediction] = []
        results += self._run_wilor(img_bgr, boxes, is_right, scores, batch_size=batch_size)
        return results


def _cam_crop_to_full(
    pred_cam: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    img_size: torch.Tensor,
    focal_length: float,
) -> torch.Tensor:
    """
    Convert weak-perspective camera parameters from crop space to the
    full-image camera translation vector.

    The weak-perspective model: x_img = f * (x_cam / z_cam)
    Given predicted [s, tx, ty] relative to the crop:
        tz = 2 * f / (s * crop_size)
        tx = tx_crop + (cx_box - cx_img) * 2 / (s * crop_size)
        ty = ty_crop + (cy_box - cy_img) * 2 / (s * crop_size)
    """
    s = pred_cam[:, 0]
    tx_crop = pred_cam[:, 1]
    ty_crop = pred_cam[:, 2]
    cx_box = box_center[:, 0]
    cy_box = box_center[:, 1]
    cx_img = img_size[0] * 0.5
    cy_img = img_size[1] * 0.5
    denom = s * box_size + 1e-09
    tz = 2.0 * focal_length / denom
    tx = tx_crop + 2.0 * (cx_box - cx_img) / denom
    ty = ty_crop + 2.0 * (cy_box - cy_img) / denom
    return torch.stack([tx, ty, tz], dim=1)

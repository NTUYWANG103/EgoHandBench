"""Official OmniHands camera solve with supplied per-frame intrinsics."""

from contextlib import contextmanager
from functools import partial

import cv2
import numpy as np
import torch

from .hands_4d.models.heads.time_head import calculate_iou, perspective_projection


@contextmanager
def known_camera(head):
    """Use the current temporal target frame's full calibration in decoding."""
    original = head.solve_cam_t
    head.solve_cam_t = partial(_solve_camera, head)
    try:
        yield
    finally:
        head.solve_cam_t = original


def _solve_camera(head, pred_mano_params, batch, train):
    pred_cam_right = pred_mano_params[f'cam_right']
    pred_cam_left = pred_mano_params[f'cam_left']

    device = pred_mano_params[f'joints3d_right'].device
    dtype = pred_mano_params[f'joints3d_right'].dtype

    batch_size = pred_cam_right.shape[0]
    mean_cam = head.init_cam.expand(batch_size, -1)
    pred_cam_right = pred_cam_right + mean_cam
    pred_cam_left = pred_cam_left + mean_cam
    focal_length = head.cfg.EXTRA.FOCAL_LENGTH * torch.ones(pred_cam_right.shape[0], 2, device=device, dtype=dtype)
    cam_t_right = torch.stack([pred_cam_right[:, 1],
                              pred_cam_right[:, 2],
                              2 * focal_length[:, 0] / (head.cfg.MODEL.IMAGE_SIZE * pred_cam_right[:, 0] + 1e-9)],
                             dim=-1)
    pred_mano_params[f'pred_cam_t_right'] = cam_t_right  # bs x 3

    cam_t_right = cam_t_right.reshape(-1, 3)
    focal_length = focal_length.reshape(-1, 2)
    j2d_right = perspective_projection(pred_mano_params[f'joints3d_right'],
                                               translation=cam_t_right,
                                               focal_length=focal_length / head.cfg.MODEL.IMAGE_SIZE)
    pred_mano_params['joints2d_right'] = j2d_right.clone()


    cam_t_left = torch.stack([pred_cam_left[:, 1],
                              pred_cam_left[:, 2],
                              2 * focal_length[:, 0] / (head.cfg.MODEL.IMAGE_SIZE * pred_cam_left[:, 0] + 1e-9)],
                              dim=-1)
    pred_mano_params[f'pred_cam_t_right'] = cam_t_left  # bs x 3

    cam_t_left = cam_t_left.reshape(-1, 3)
    focal_length = focal_length.reshape(-1, 2)
    j2d_left = perspective_projection(pred_mano_params[f'joints3d_left'],
                                            translation=cam_t_left,
                                            focal_length=focal_length / head.cfg.MODEL.IMAGE_SIZE)
    pred_mano_params['joints2d_left'] = j2d_left.clone()

    j2d_right_glb, j2d_left_glb = head.calc_glb_2d(batch, pred_mano_params[f'joints3d_right'].clone(),
                                                   pred_mano_params[f'joints3d_left'].clone(),
                                                   cam_t_right.clone(), cam_t_left.clone(),
                                                   pred_mano_params['root_rel'].clone())
    pred_mano_params['joints2d_glb_right'] = j2d_right_glb
    pred_mano_params['joints2d_glb_left'] = j2d_left_glb

    if not train:
        j2d_left[..., 0] = -j2d_left[..., 0]

        bbox_right = batch["bbox_right"]
        bbox_left = batch["bbox_left"]
        bbox_full = batch["bbox_full"]
        IoU = calculate_iou(bbox_right, bbox_left)

        j2d_right = (j2d_right + 0.5) * bbox_right[:, None, 2:4]
        j2d_left = (j2d_left + 0.5) * bbox_left[:, None, 2:4]
        j2d_right = j2d_right + bbox_right[:, None, 0:2] - bbox_full[:, None, 0:2]
        j2d_left = j2d_left + bbox_left[:, None, 0:2] - bbox_full[:, None, 0:2]

        pred_mano_params['joints2d_world_right'] = j2d_right.clone()
        pred_mano_params['joints2d_world_left'] = j2d_left.clone()

        j3d_right, j3d_left = pred_mano_params["joints3d_world_right"].clone(), \
                              pred_mano_params["joints3d_world_left"].clone()

        root_dist = torch.linalg.norm(j3d_left[:, 9, :] - j3d_right[:, 9, :], dim=-1)
        j2d_align = torch.cat([j2d_right, j2d_left], dim=1)
        j3d_align = torch.cat([j3d_right, j3d_left], dim=1)

        # Image observations above are relative to bbox_full's origin.
        camera_K = batch["camera_K"].detach().float().cpu().numpy().copy()
        camera_K[:, :2, 2] -= bbox_full[:, :2].detach().float().cpu().numpy()

        j2d_right = j2d_right.float().detach().cpu().numpy()
        j2d_left = j2d_left.float().detach().cpu().numpy()
        j3d_right = j3d_right.float().detach().cpu().numpy()
        j3d_left = j3d_left.float().detach().cpu().numpy()
        j2d_align = j2d_align.float().detach().cpu().numpy()
        j3d_align = j3d_align.float().detach().cpu().numpy()

        cam_left = []
        cam_right = []
        cam_align = []
        for idx in range(j3d_right.shape[0]):
            cam_k = camera_K[idx]
            ret_left, r_left, t_left, _ = cv2.solvePnPRansac(j3d_left[idx], j2d_left[idx], cam_k, np.zeros(4))
            ret_right, r_right, t_right, _ = cv2.solvePnPRansac(j3d_right[idx], j2d_right[idx], cam_k, np.zeros(4))
            ret_align, r_align, t_align, _ = cv2.solvePnPRansac(j3d_align[idx], j2d_align[idx], cam_k, np.zeros(4))
            if not (ret_left and ret_right and ret_align):
                raise RuntimeError(f"OmniHands PnP failed for batch sample {idx}")
            cam_left.append(t_left.squeeze(1))
            cam_right.append(t_right.squeeze(1))
            cam_align.append(t_align.squeeze(1))

        cam_left = np.stack(cam_left, axis=0)
        cam_right = np.stack(cam_right, axis=0)
        cam_align = np.stack(cam_align, axis=0)

        cam_t_right = torch.from_numpy(cam_right).to(device)
        cam_t_left = torch.from_numpy(cam_left).to(device)
        cam_aligned_right = torch.from_numpy(cam_align).to(device)
        cam_aligned_left = torch.from_numpy(cam_align.copy()).to(device)
        align_mask = (IoU < 0.01) + (root_dist > 0.2)
        cam_aligned_right[align_mask] = cam_t_right[align_mask]
        cam_aligned_left[align_mask] = cam_t_left[align_mask]
        pred_mano_params["cam_aligned_right"] = cam_aligned_right
        pred_mano_params["cam_aligned_left"] = cam_aligned_left

    return pred_mano_params

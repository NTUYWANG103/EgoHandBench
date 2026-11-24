"""Crop and camera helpers used by online HaMeR inference."""

from __future__ import annotations

import cv2
import numpy as np
import torch

def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    if target_aspect_ratio is None:
        return input_shape
    w, h = input_shape
    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    return np.array([w_new, h_new])

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height,
                            dst_width, dst_height, scale=1.0, rot=0.0):
    src_w = src_width * scale
    src_h = src_height * scale
    rot_rad = np.pi * rot / 180.0
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_center = np.array([c_x, c_y], dtype=np.float32)
    src_downdir = np.array([0, src_h * 0.5], dtype=np.float32)
    src_downdir = np.array([src_downdir[0]*cs - src_downdir[1]*sn,
                            src_downdir[0]*sn + src_downdir[1]*cs], dtype=np.float32)
    src_rightdir = np.array([src_w * 0.5, 0], dtype=np.float32)
    src_rightdir = np.array([src_rightdir[0]*cs - src_rightdir[1]*sn,
                             src_rightdir[0]*sn + src_rightdir[1]*cs], dtype=np.float32)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(src, dst)
    return trans

def compute_crop_intrinsics(trans, K_orig_4):
    """K_crop = S @ K_orig"""
    fx, fy, cx, cy = K_orig_4
    K_orig = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]], dtype=np.float64)
    S = np.eye(3, dtype=np.float64)
    S[:2, :] = trans.astype(np.float64)
    K_crop = S @ K_orig
    return np.array([K_crop[0, 0], K_crop[1, 1],
                     K_crop[0, 2], K_crop[1, 2]], dtype=np.float32)

def _cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length):
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2.0 * focal_length / bs
    tx = (2.0 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2.0 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)

def _project_full_img(points, cam_trans, focal_length, img_w, img_h):
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array([[focal_length, 0.0, cx],
                  [0.0, focal_length, cy],
                  [0.0,         0.0,  1.0]], dtype=np.float64)
    pts = points.astype(np.float64) + cam_trans
    pts = pts / pts[:, 2:3]
    return (K @ pts.T).T[:, :2]

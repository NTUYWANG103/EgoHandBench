#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def preprocess_frames(frames_uint8, size_hw=(378, 518)) -> torch.Tensor:
    """Internal helper."""
    if not torch.is_tensor(frames_uint8):
        frames_uint8 = torch.as_tensor(frames_uint8)
    x = frames_uint8.permute(0, 3, 1, 2).float() / 255.0   # [N,3,H0,W0]
    x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    return x.contiguous()

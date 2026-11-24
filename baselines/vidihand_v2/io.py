"""The TF32 switch and the model canvas of ViDiHand2 (from utils/io.py of the ViDiHand release)."""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from tqdm import tqdm


@contextmanager
def tf32(enabled):
    """Context manager / decorator that sets TF32 for CUDA matmuls and cuDNN, restoring the previous setting afterwards."""
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def per_frame(K, n):
    """Intrinsics [n, 3, 3] from one 3 x 3 matrix or a per-frame stack."""
    return np.repeat(np.asarray(K, np.float64)[None], n, 0) if np.ndim(K) == 2 else K


def model_size(width, height, short=480):
    """The model canvas (width, height) for a video, every side a multiple of 32 (at least 32): with `short`, the short side is `short`
    rounded to a multiple of 32 and the long side that rounded side times the aspect ratio, rounded; with `short=None`, each input side is
    rounded on its own."""
    if not (width > 0 and height > 0):
        raise ValueError(f"the video size must be positive, got {width} x {height}")
    to32 = lambda v: max(32, round(v / 32) * 32)
    if short is None:
        return to32(width), to32(height)
    if isinstance(short, (bool, np.bool_)) or not isinstance(short, (int, np.integer)) or short <= 0:
        raise ValueError(f"the short side is a positive number of pixels or None, got {short!r}")
    short = to32(int(short))
    long = to32(max(width, height) * (short / min(width, height)))
    return (long, short) if width >= height else (short, long)


def resize(video, K, size):
    """Frames resized to `size` (area filter when shrinking) and the intrinsics mapped to it (pixel centres preserved)."""
    K = per_frame(K, len(video))
    height, width = video.shape[1:3]
    if (width, height) == tuple(size):
        return video, K
    sx, sy = size[0] / width, size[1] / height
    interpolation = cv2.INTER_AREA if sx * sy < 1 else cv2.INTER_LINEAR
    with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool:
        frames = pool.map(lambda frame: cv2.resize(frame, tuple(size), interpolation=interpolation), video)
        video = np.stack(list(tqdm(frames, total=len(video), desc="Resizing to the model canvas", unit="frame", leave=False)))
    K = np.array(K, np.float64)
    K[:, 0] *= sx
    K[:, 1] *= sy
    K[:, 0, 2] += 0.5 * sx - 0.5
    K[:, 1, 2] += 0.5 * sy - 0.5
    return video, K

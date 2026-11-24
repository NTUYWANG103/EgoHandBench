"""Rectify supplied pinhole calibration to HaWoR's native scalar camera."""

import cv2
import numpy as np


def rectify_frames(directory, intrinsics):
    matrices = np.asarray(intrinsics, np.float64)
    virtual = matrices[0].copy()
    virtual[0, 1] = virtual[1, 0] = 0
    virtual[1, 1] = virtual[0, 0]
    images = sorted(directory.glob("*.jpg"), key=lambda path: int(path.stem))  # %04d.jpg passes 9999
    if len(images) != len(matrices):
        raise ValueError("HaWoR camera and JPEG timelines differ")
    if not np.all(matrices == virtual):
        for path, matrix in zip(images, matrices):
            image = cv2.imread(str(path))
            height, width = image.shape[:2]
            transform = virtual @ np.linalg.inv(matrix)
            calibrated = cv2.warpPerspective(image, transform, (width, height))
            if not cv2.imwrite(str(path.with_suffix(".png")), calibrated):
                raise OSError("Cannot write HaWoR calibrated frame")
    return dict(fx=virtual[0, 0], fy=virtual[1, 1], cx=virtual[0, 2], cy=virtual[1, 2])

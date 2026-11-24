"""Map each supplied camera to PAD-Hand's centered scalar-focal camera."""

import cv2
import numpy as np


def rectify_image(image, matrix):
    height, width = image.shape[:2]
    focal = float((matrix[0, 0] + matrix[1, 1]) / 2)
    virtual = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    return cv2.warpPerspective(image, virtual @ np.linalg.inv(matrix), (width, height)), focal

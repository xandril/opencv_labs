from __future__ import annotations

import cv2 as cv
import numpy as np


def shrink_to_fit(image: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
    target_width, target_height = max_size
    assert target_width > 0
    assert target_height > 0

    assert target_width >= target_height
    height, width = image.shape[0], image.shape[1]

    if width >= target_width:
        width, height = target_width, height * target_width // width
        image = cv.resize(image, (width, height))

    height, width = image.shape[0], image.shape[1]

    if height >= target_height:
        width, height = width * target_height // height, target_height
        image = cv.resize(image, (width, height))

    return image

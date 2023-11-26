import cv2 as cv
import numpy as np


def warp_perspective(
        image: np.ndarray,
        a1: int,
        a2: int
) -> np.ndarray:
    a1 = np.clip(a1, 0, 100) - 50
    a2 = np.clip(a2, 0, 100) - 50

    w, h = image.shape[:2]

    in_quad = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype='float32')
    out_quad = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype='float32')

    if a1 > 0:
        out_quad[0, 0] = -a1 * w // 50
        out_quad[1, 0] = (50 + a1) * w // 50
    else:
        out_quad[2, 0] = (50 - a1) * w // 50
        out_quad[3, 0] = a1 * w // 50

    if a2 > 0:
        out_quad[1, 1] = -a2 * h // 50
        out_quad[2, 1] = (50 + a2) * h // 50
    else:
        out_quad[0, 1] = a2 * h // 50
        out_quad[3, 1] = (50 - a2) * h // 50

    matrix = cv.getPerspectiveTransform(in_quad, out_quad)

    return cv.warpPerspective(
        image, matrix, (w, h), cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

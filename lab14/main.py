from dataclasses import dataclass

import cv2 as cv
import numpy as np


def threshold_match(
        match_result: np.ndarray,
        threshold: float,
) -> list[tuple[int, int, float]]:
    locations = np.where(match_result <= threshold)

    boxes = [(x, y, match_result[y, x]) for x, y in zip(locations[1], locations[0])]

    boxes.sort(key=lambda x: x[2])
    return boxes


@dataclass
class MatchingResult:
    x0: int
    y0: int
    w: int
    h: int
    score: float

    @property
    def x1(self):
        return self.x0 + self.w

    @property
    def y1(self):
        return self.y0 + self.h


def match_template(gray_img: np.ndarray,
                   gray_template: np.ndarray,
                   threshold: float) -> list[MatchingResult]:
    match_result = cv.matchTemplate(gray_img, gray_template, method=cv.TM_SQDIFF_NORMED)
    thr_match_result = threshold_match(match_result, threshold)

    template_w = gray_template.shape[1]
    template_h = gray_template.shape[0]

    result = [MatchingResult(x0=x, y0=y, w=template_w, h=template_h, score=score) for x, y, score in thr_match_result]
    return result

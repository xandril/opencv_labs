from __future__ import annotations

from typing import Callable

import cv2 as cv
import numpy as np

from lab9.common import corner_detection


class ChiTomasViewModel:
    on_values_changed: Callable[[ChiTomasViewModel], None]

    maxCorners: int = 10
    qualityLevel: float = 0.01
    minDistance: float = 10.0

    def set_maxCorners(self, value: int) -> None:
        self.maxCorners = max(1, value)
        self.on_values_changed(self)

    def set_qualityLevel(self, value_times_1000: int) -> None:
        self.qualityLevel = max(1e-5, value_times_1000 / 1000)
        self.on_values_changed(self)

    def set_minDistance(self, value: int) -> None:
        self.minDistance = max(1.0, float(value))
        self.on_values_changed(self)


def draw_corners(image_gray: np.ndarray, target: np.ndarray, view: ChiTomasViewModel) -> np.ndarray:
    corners = cv.goodFeaturesToTrack(image_gray, view.maxCorners, view.qualityLevel, view.minDistance)
    if corners is None:
        return target

    for i in corners:
        x, y = i.ravel().astype('int32')
        target = cv.circle(target, (x, y), radius=3, color=(255, 64, 64), thickness=2)

    return target


def init_sliders(sliders_window_name: str, view_model: ChiTomasViewModel) -> None:
    cv.createTrackbar('max_corners',
                      sliders_window_name,
                      view_model.maxCorners,
                      100,
                      view_model.set_maxCorners)

    cv.createTrackbar('quality_level',
                      sliders_window_name,
                      int(1000 * view_model.qualityLevel),
                      1000,
                      view_model.set_qualityLevel)

    cv.createTrackbar('min_dist',
                      sliders_window_name,
                      int(view_model.minDistance),
                      100,
                      view_model.set_minDistance)


if __name__ == '__main__':
    corner_detection(image_path='../imgs/img42.jpg',
                     view_model=ChiTomasViewModel(),
                     init_sliders=init_sliders,
                     draw_corners=draw_corners)

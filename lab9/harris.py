from __future__ import annotations

import cv2 as cv
import numpy as np

from lab9.common import Model, corner_detection


class HarrisModelView(Model):
    blockSize: int = 2
    ksize: int = 3
    k: float = 0.04
    threshold: float = 0.01

    def set_blockSize(self, value: int) -> None:
        self.blockSize = max(1, value)
        self.on_values_changed(self)

    def set_ksize(self, value: int) -> None:
        if value % 2 == 0:
            value += 1

        self.ksize = min(31, max(1, value))
        self.on_values_changed(self)

    def set_k(self, value: int) -> None:
        self.k = max(0.001, value / 1000)
        self.on_values_changed(self)

    def set_threshold(self, value: int) -> None:
        self.threshold = max(0.001, value / 1000)
        self.on_values_changed(self)


def draw_corners(image_gray: np.ndarray, target: np.ndarray, view: HarrisModelView) -> np.ndarray:
    result = cv.cornerHarris(image_gray.astype('float32'), view.blockSize, view.ksize, view.k)

    target[result > view.threshold * result.max()] = 255

    return target


def init_sliders(sliders_window_name: str, view_model: HarrisModelView) -> None:
    cv.createTrackbar('block size', sliders_window_name, view_model.blockSize, 100, view_model.set_blockSize)
    cv.createTrackbar('ksize', sliders_window_name, view_model.ksize, 31, view_model.set_ksize)
    cv.createTrackbar('k * 1000', sliders_window_name, int(view_model.k * 1000), 100, view_model.set_k)
    cv.createTrackbar('threshold * 1000', sliders_window_name, int(view_model.threshold * 1000), 100,
                      view_model.set_threshold)


if __name__ == '__main__':
    corner_detection(image_path='../imgs/img42.jpg',
                     view_model=HarrisModelView(),
                     init_sliders=init_sliders,
                     draw_corners=draw_corners)

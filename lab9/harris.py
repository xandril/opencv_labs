from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2 as cv
import numpy as np

from shrink import shrink_to_fit
from warp import warp_perspective


@dataclass
class Model:
    on_values_changed: Callable[[Model], None]

    blockSize: int = 2
    ksize: int = 3
    k: float = 0.04
    threshold: float = 0.01

    rotation: int = 0

    a1: int = 51
    a2: int = 50

    alpha: float = 1.0
    beta: int = 0

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

    def set_rotation(self, value: int) -> None:
        self.rotation = value
        self.on_values_changed(self)

    def set_a1(self, value: int) -> None:
        self.a1 = value
        self.on_values_changed(self)

    def set_a2(self, value: int) -> None:
        self.a2 = value
        self.on_values_changed(self)

    def set_alpha(self, value: int) -> None:
        self.alpha = 0.01 * max(1, value)
        self.on_values_changed(self)

    def set_beta(self, value: int) -> None:
        self.beta = value
        self.on_values_changed(self)


def draw_corners(image_gray: np.ndarray, target: np.ndarray, model_: Model) -> np.ndarray:
    result = cv.cornerHarris(image_gray.astype('float32'), model_.blockSize, model_.ksize, model_.k)
    result = cv.dilate(result, None)

    target[result > model_.threshold * result.max()] = (32, 255, 32)

    return target


def alpha_beta_correction(image: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    return np.clip(alpha * image.astype('float64') + beta, 0, 255).astype('uint8')


def main(file_path: Path) -> None:
    image = cv.imread(str(file_path))

    image = shrink_to_fit(image, (1200, 600))

    image = shrink_to_fit(image, (1200, 600))
    width = image.shape[1]
    height = image.shape[0]

    window_name = file_path.name
    cv.namedWindow(window_name, cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow(window_name, image.shape[1], image.shape[0])

    def redraw(model_: Model) -> None:
        transform = cv.getRotationMatrix2D((width // 2, height // 2), angle=model_.rotation, scale=1.0)

        image_ab = alpha_beta_correction(image, model_.alpha, model_.beta)

        image_r = warp_perspective(image_ab, a1=model_.a1, a2=model_.a2)
        image_gray_r = warp_perspective(cv.cvtColor(image_ab, cv.COLOR_BGR2GRAY), a1=model_.a1, a2=model_.a2)

        image_r = cv.warpAffine(image_r, transform, (width, height))
        image_gray_r = cv.warpAffine(image_gray_r, transform, (width, height))

        cv.imshow(window_name, draw_corners(image_gray_r, image_r, model_))

    model = Model(redraw)

    sliders_window_name = 'sliders'
    cv.namedWindow(sliders_window_name)
    cv.resizeWindow(sliders_window_name, 500, 400)

    cv.createTrackbar('block size', sliders_window_name, model.blockSize, 100, model.set_blockSize)
    cv.createTrackbar('ksize', sliders_window_name, model.ksize, 31, model.set_ksize)
    cv.createTrackbar('k * 1000', sliders_window_name, int(model.k * 1000), 100, model.set_k)
    cv.createTrackbar('threshold * 1000', sliders_window_name, int(model.threshold * 1000), 100, model.set_threshold)
    cv.createTrackbar('rotation', sliders_window_name, model.rotation, 360, model.set_rotation)
    cv.createTrackbar('a1', sliders_window_name, model.a1, 100, model.set_a1)
    cv.createTrackbar('a2', sliders_window_name, model.a2, 100, model.set_a2)
    cv.createTrackbar('alpha', sliders_window_name, int(100 * model.alpha), 200, model.set_alpha)
    cv.createTrackbar('beta', sliders_window_name, model.beta, 255, model.set_beta)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(file_path=Path('../imgs/SuccessKid.jpg'))

import operator
from functools import reduce
from pathlib import Path
from typing import Final, Iterator, Any

import cv2 as cv
import numpy as np


def resize_image(image: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
    target_width, target_height = max_size
    height, width = image.shape[0], image.shape[1]

    if width >= target_width:
        width, height = target_width, height * target_width // width
        image = cv.resize(image, (width, height))

    height, width = image.shape[0], image.shape[1]

    if height >= target_height:
        width, height = width * target_height // height, target_height
        image = cv.resize(image, (width, height))

    return image


def color_generator() -> Iterator[tuple[int, int, int]]:
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 96, 96),
        (96, 255, 96),
        (96, 0, 255),
    ]
    i = 0
    while True:
        yield colors[i]
        i = (i + 1) % len(colors)


def merge_close(points: np.ndarray, threshold: float) -> np.ndarray:
    original_points = points
    points = sorted(points, key=lambda it: it[0][0])
    result = []
    merged = True
    i = 0
    while merged and i < 3:
        merged = False

        for p1, p2 in zip(points, points[1:]):
            if np.linalg.norm(p1 - p2) > threshold:
                continue

            merged = True
            result.append(np.mean([p1, p2], axis=0, dtype=original_points.dtype))

        points = result
        i += 1

    return np.array(result) if result else original_points


def detect_shape(contour: Any, eps: float) -> tuple[Any, str | None]:
    perimeter = cv.arcLength(contour, True)
    approx_poly = cv.approxPolyDP(contour, eps * perimeter, True)
    approx = merge_close(approx_poly, threshold=10)

    vertices_count = len(approx)

    if vertices_count == 3:
        return vertices_count, 'triangle'

    _, _, w, h = cv.boundingRect(approx)
    aspect_ratio = w / h
    if vertices_count == 4:
        if np.abs(1.0 - aspect_ratio) <= 0.05:
            return vertices_count, 'square'

        return vertices_count, 'rectangle'

    if 5 <= vertices_count <= 7:
        return vertices_count, None

    if np.abs(1.0 - aspect_ratio) <= 0.05:
        return vertices_count, 'circle'

    return vertices_count, 'ellipse'


def approx_size(contour: Any, image: np.ndarray) -> str:
    image_area = reduce(operator.mul, image.shape)
    _, _, w, h = cv.boundingRect(contour)
    contour_area = w * h

    if contour_area > 0.05 * image_area:
        return 'large'

    if contour_area > 0.01 * image_area:
        return 'medium'

    return 'small'


LOWER: Final[str] = 'Lower'
UPPER: Final[str] = 'Upper'


def main(file_path: Path) -> None:
    image = cv.imread(str(file_path))
    image = resize_image(image, (1200, 600))
    window_name = 'image'
    edges_window_name = f'edges'
    cv.namedWindow(window_name, cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow(window_name, image.shape[1], image.shape[0])
    cv.namedWindow(edges_window_name, cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow(edges_window_name, image.shape[1], image.shape[0])

    thresholds = {LOWER: 100, UPPER: 200}
    eps = 0.025
    ksize = 0

    def set_threshold(name: str, value: int) -> None:
        thresholds[name] = value
        update()

    def set_eps(eps_: int) -> None:
        nonlocal eps
        eps = max(1e-2, eps_ / 1000)
        update()

    def set_ksize(ksize_: int) -> None:
        nonlocal ksize
        ksize = ksize_
        update()

    def update() -> None:
        edges = cv.Canny(cv.blur(image, (ksize, ksize)) if ksize > 0 else image, thresholds[LOWER], thresholds[UPPER])
        cv.imshow(edges_window_name, edges)
        contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)

        image_with_contours = image.copy()
        for contour in contours:
            image_with_contours = \
                cv.drawContours(
                    image_with_contours, [contour], 0,
                    color=(96, 96, 255),
                    thickness=3,  # hierarchy=hierarchy
                )
            n_vertices, shape = detect_shape(contour, eps)
            if shape is None:
                continue

            x, y, w, h = cv.boundingRect(contour)
            image_with_contours = cv.putText(
                image_with_contours, f'{approx_size(contour, image)} {shape}', (x + w // 2, y + h // 2),
                fontScale=0.5, fontFace=cv.FONT_HERSHEY_SIMPLEX, color=(96, 96, 255)
            )

        cv.imshow(window_name, image_with_contours)

    update()

    sliders_window_name = 'sliders'

    cv.namedWindow(sliders_window_name)
    cv.createTrackbar(LOWER, sliders_window_name, thresholds[LOWER], 255, lambda value: set_threshold(LOWER, value))
    cv.createTrackbar(UPPER, sliders_window_name, thresholds[UPPER], 255, lambda value: set_threshold(UPPER, value))
    cv.createTrackbar('Approx. Precision', sliders_window_name, int(eps * 1000), 100, lambda value: set_eps(value))
    cv.createTrackbar('Kernel Size', sliders_window_name, ksize, 15, lambda value: set_ksize(value))

    cv.waitKey(0)


if __name__ == '__main__':
    main(Path('../imgs/111.jpg'))

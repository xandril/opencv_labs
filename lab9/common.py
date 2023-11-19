from typing import Callable

import cv2
import cv2 as cv
import numpy as np


class Model:
    on_values_changed: Callable[['Model'], None] = None


def clip_image(image: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
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


def warp_perspective(image: np.ndarray) -> np.ndarray:
    hh, ww = image.shape[:2]

    # specify input coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x,
    in_ = np.float32([[680, 630], [860, 520], [836, 700], [660, 810]])

    # get top and left dimensions and set to output dimensions of red rectangle
    width = round(np.hypot(in_[0, 0] - in_[1, 0], in_[0, 1] - in_[1, 1]))
    height = round(np.hypot(in_[0, 0] - in_[3, 0], in_[0, 1] - in_[3, 1]))
    # print("width:", width, "height:", height)

    # set upper left coordinates for output rectangle
    x = in_[0, 0]
    y = in_[0, 1]

    # specify output coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x,
    output = np.float32([[x, y], [x + width - 1, y], [x + width - 1, y + height - 1], [x, y + height - 1]])

    # compute perspective matrix
    matrix = cv.getPerspectiveTransform(in_, output)
    # print(matrix)

    # do perspective transformation setting area outside input to black
    # Note that output size is the same as the input image size
    return cv.warpPerspective(
        image, matrix, (ww, hh), cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )


def get_input_data(img_path: str) -> tuple[cv.typing.MatLike, cv2.typing.MatLike]:
    image = cv.imread(img_path, 0)
    image = clip_image(image, (600, 600))
    warped_image = warp_perspective(image)
    return image, warped_image


def get_model_update(window_name: str,
                     warped_window_name: str,
                     draw_corners: Callable,
                     image: cv.typing.MatLike,
                     warped_image: cv.typing.MatLike) -> Callable:
    def update(model: Model) -> None:
        cv.imshow(window_name, draw_corners(image, image.copy(), model))
        cv.imshow(warped_window_name, draw_corners(warped_image, warped_image.copy(), model))

    return update


def corner_detection(image_path: str,
                     view_model: Model,
                     init_sliders: Callable[[str, Model], None],
                     draw_corners: Callable[[cv.typing.MatLike, cv.typing.MatLike, Model], np.ndarray]) -> None:
    image, warped_image = get_input_data(image_path)

    window_name = 'img'
    cv.namedWindow(window_name, cv.WINDOW_GUI_EXPANDED)

    warped_window_name = 'warped_img'
    cv.namedWindow(warped_window_name, cv.WINDOW_GUI_EXPANDED)

    view_model.on_values_changed = get_model_update(window_name,
                                                    warped_window_name,
                                                    draw_corners,
                                                    image,
                                                    warped_image)

    sliders_window_name = 'sliders'
    cv.namedWindow(sliders_window_name)

    init_sliders(sliders_window_name, view_model)

    cv.waitKey(0)
    cv.destroyAllWindows()

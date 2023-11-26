from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final, Callable

import cv2 as cv
import numpy as np


def get_perspective_transform(
        size: tuple[int, int],
        a1: int,
        a2: int
) -> np.ndarray:
    a1 = np.clip(a1, 0, 200) - 100
    a2 = np.clip(a2, 0, 200) - 100

    w, h = size

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
        out_quad[0, 0] = -a1 * w // 100
        out_quad[1, 0] = (100 + a1) * w // 100
    else:
        out_quad[2, 0] = (100 - a1) * w // 100
        out_quad[3, 0] = a1 * w // 100

    if a2 > 0:
        out_quad[1, 1] = -a2 * h // 100
        out_quad[2, 1] = (100 + a2) * h // 100
    else:
        out_quad[0, 1] = a2 * h // 100
        out_quad[3, 1] = (100 - a2) * h // 100

    return cv.getPerspectiveTransform(in_quad, out_quad)


@dataclass
class Parameters:
    on_values_changed: Callable[[], None] = lambda: None

    rotation: int = 0
    max_rotation: Final[int] = 359

    scale: int = 100
    max_scale: Final[int] = 200

    a1: int = 100
    max_a1: Final[int] = 200

    a2: int = 100
    max_a2: Final[int] = 200

    alpha: int = 100
    max_alpha: Final[int] = 200

    beta: int = 0
    max_beta: Final[int] = 255

    noise_intensity: int = 0
    max_noise_intensity: Final[int] = 100

    def set_rotation(self, value: int) -> None:
        self.rotation = value
        self.on_values_changed()

    def set_scale(self, value: int) -> None:
        self.scale = min(200, max(1, value))
        self.on_values_changed()

    def set_a1(self, value: int) -> None:
        self.a1 = value
        self.on_values_changed()

    def set_a2(self, value: int) -> None:
        self.a2 = value
        self.on_values_changed()

    def set_alpha(self, value: int) -> None:
        self.alpha = value
        self.on_values_changed()

    def set_beta(self, value: int) -> None:
        self.beta = value
        self.on_values_changed()

    def set_noise_intensity(self, value: int) -> None:
        self.noise_intensity = value
        self.on_values_changed()

    def create_sliders(self) -> Parameters:
        window_name = 'Control Panel'
        cv.namedWindow(window_name, cv.WINDOW_GUI_EXPANDED)

        for field in fields(Parameters):
            if field.name.startswith(('max_', 'set_', 'on_')):
                continue

            cv.createTrackbar(
                field.name.capitalize(),
                window_name,
                int(getattr(self, field.name)),
                int(getattr(self, f'max_{field.name}')),
                getattr(self, f'set_{field.name}'),
            )

        return self


def alpha_beta_correction(image: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    return np.clip(alpha * image.astype('float64') + beta, 0, 255).astype('uint8')


def rand_bin_array(n: int, ones_fraction: float) -> np.ndarray:
    assert 0 <= ones_fraction <= 1
    arr = np.zeros(n)
    arr[:int(n * ones_fraction)] = 1
    np.random.shuffle(arr)
    return arr


def apply_noise(image: np.ndarray, noise_intensity: float) -> np.ndarray:
    h, w, c = image.shape
    return np.clip(
        image.astype('int32')
        + rand_bin_array(h * w * c, noise_intensity).reshape(image.shape)
        * np.random.uniform(-127, 128, size=image.shape),
        0, 255
    ).astype('uint8')


def main(scene_path: Path, template_path: Path) -> None:
    scene: np.ndarray = cv.imread(str(scene_path))
    width = scene.shape[1]
    height = scene.shape[0]

    target: np.ndarray = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
    target_width = target.shape[1]
    target_height = target.shape[0]

    window_name = 'Main'
    cv.namedWindow(window_name, cv.WINDOW_GUI_EXPANDED)

    params = Parameters().create_sliders()

    def redraw() -> None:
        image = scene
        image = apply_noise(image, params.noise_intensity / 100)
        image = alpha_beta_correction(image, params.alpha / 100, params.beta)
        image = cv.warpPerspective(
            image,
            get_perspective_transform((width, height), params.a1, params.a2),
            (width, height), cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        image = cv.warpAffine(
            image,
            cv.getRotationMatrix2D((width // 2, height // 2), params.rotation, params.scale / 100),
            dsize=(width, height)
        )
        markers = cv.matchTemplate(cv.cvtColor(image, cv.COLOR_BGR2GRAY), target, method=cv.TM_SQDIFF_NORMED)
        min_val, _, min_loc, _ = cv.minMaxLoc(markers)
        x, y = min_loc
        image = cv.putText(
            image, f'{min_val:.2}', (x + 10, y + 10),
            fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(23, 255, 64)
        )
        image = cv.rectangle(
            image, (x, y), (x + target_width, y + target_height),
            color=(23, 255, 64)
        )
        cv.imshow(window_name, image)

    params.on_values_changed = redraw
    redraw()

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(scene_path=Path('img.png'), template_path=Path('chinese_template.png'))

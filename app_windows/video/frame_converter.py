"""Safe conversion of backend BGR numpy frames into Qt images."""

from __future__ import annotations

import numpy as np


def decoded_frame_size(frame: np.ndarray) -> tuple[int, int]:
    """Return width/height from the decoded frame, not widget or config size."""

    image = np.asarray(frame)
    if image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("decoded video frame must have positive height and width")
    return int(image.shape[1]), int(image.shape[0])


def frame_to_qimage(frame: np.ndarray):
    """Convert one frame and detach the QImage from numpy-owned memory.

    The import is deliberately lazy so pure provider tests and fake backend tests
    can run without PySide6 installed.
    """

    from PySide6.QtGui import QImage

    image = np.asarray(frame)
    if image.ndim == 2:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)
        result = QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            QImage.Format.Format_Grayscale8,
        )
        return result.copy()

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("video frame must have shape HxW, HxWx3 or HxWx4")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[2] == 3:
        # Qt 6 can consume OpenCV's native BGR order directly. This avoids a
        # full-frame channel swap before the final lifetime-detaching copy.
        converted = np.ascontiguousarray(image)
        format_ = QImage.Format.Format_BGR888
    else:
        converted = np.ascontiguousarray(image[:, :, [2, 1, 0, 3]])
        format_ = QImage.Format.Format_RGBA8888

    result = QImage(
        converted.data,
        converted.shape[1],
        converted.shape[0],
        converted.strides[0],
        format_,
    )
    return result.copy()

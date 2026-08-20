from __future__ import annotations

import numpy as np
import pytest

from app_windows.video.frame_converter import decoded_frame_size


def test_decoded_dimensions_come_from_frame_shape() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert decoded_frame_size(frame) == (1920, 1080)


def test_decoded_dimensions_support_portrait_frames() -> None:
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

    assert decoded_frame_size(frame) == (1080, 1920)


def test_decoded_dimensions_reject_missing_geometry() -> None:
    with pytest.raises(ValueError):
        decoded_frame_size(np.zeros((0, 10, 3), dtype=np.uint8))

from __future__ import annotations

import numpy as np
import pytest

from app.video.motion import MotionDetector


cv2 = pytest.importorskip("cv2")


def test_motion_detector_warmup_static_change_and_reset() -> None:
    detector = MotionDetector(
        pixel_threshold=10,
        min_changed_fraction=0.05,
        resize_width=32,
        warmup_frames=1,
    )
    first = np.zeros((32, 32, 3), dtype=np.uint8)
    changed = first.copy()
    changed[:, :16] = 255

    assert detector.detect(first).warmup is True
    static = detector.detect(first)
    assert static.motion_detected is False
    assert static.changed_fraction == pytest.approx(0.0)

    movement = detector.detect(changed)
    assert movement.motion_detected is True
    assert movement.changed_fraction > 0.05

    detector.reset()
    assert detector.detect(first).warmup is True


def test_motion_detectors_keep_state_independent() -> None:
    first = np.zeros((16, 16, 3), dtype=np.uint8)
    detector_one = MotionDetector(resize_width=16)
    detector_two = MotionDetector(resize_width=16)

    detector_one.detect(first)
    detector_one.detect(first)
    assert detector_two.detect(first).warmup is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pixel_threshold": 0},
        {"pixel_threshold": 256},
        {"min_changed_fraction": -0.1},
        {"min_changed_fraction": 1.1},
        {"resize_width": 0},
        {"warmup_frames": 0},
    ],
)
def test_motion_detector_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MotionDetector(**kwargs)

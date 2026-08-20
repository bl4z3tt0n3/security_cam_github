from __future__ import annotations

import pytest

from app_windows.video.frame_geometry import keep_aspect_ratio_rect


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ((1920, 1080), (700, 500), (0, 53, 700, 394)),
        ((1280, 720), (400, 300), (0, 37, 400, 225)),
        ((1080, 1920), (700, 500), (209, 0, 281, 500)),
        ((1920, 1080), (500, 700), (0, 209, 500, 281)),
        ((1080, 1920), (500, 700), (53, 0, 394, 700)),
    ],
)
def test_keep_aspect_ratio_rect_preserves_source_geometry(source, target, expected) -> None:
    rect = keep_aspect_ratio_rect(*source, *target)

    assert (rect.x, rect.y, rect.width, rect.height) == expected
    assert rect.width <= target[0]
    assert rect.height <= target[1]


def test_keep_aspect_ratio_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        keep_aspect_ratio_rect(1920, 1080, 0, 500)

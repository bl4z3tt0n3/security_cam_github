"""Pure frame-to-widget geometry used to verify non-distorting rendering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderedFrameRect:
    """Centered destination rectangle in target-widget coordinates."""

    x: int
    y: int
    width: int
    height: int


def keep_aspect_ratio_rect(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> RenderedFrameRect:
    """Fit a decoded frame into a widget using letterbox/pillarbox geometry."""

    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source and target dimensions must be positive")

    scale = min(target_width / source_width, target_height / source_height)
    width = max(1, round(source_width * scale))
    height = max(1, round(source_height * scale))
    return RenderedFrameRect(
        x=(target_width - width) // 2,
        y=(target_height - height) // 2,
        width=width,
        height=height,
    )

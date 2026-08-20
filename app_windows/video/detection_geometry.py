"""Pure geometry helpers for drawing detections on transformed video."""

from __future__ import annotations

from dataclasses import dataclass

from app.inference.base import BBox, MaskPolygon
from app_windows.models.camera_display_transform import CameraDisplayTransform
from .frame_geometry import keep_aspect_ratio_rect


@dataclass(frozen=True)
class RenderedDetectionBox:
    """A detection rectangle in target-widget coordinates."""

    x: float
    y: float
    width: float
    height: float


RenderedDetectionPolygon = tuple[tuple[float, float], ...]


def effective_frame_size(
    source_width: int,
    source_height: int,
    transform: CameraDisplayTransform,
) -> tuple[int, int]:
    """Return dimensions after the same visual rotation used by the video."""

    if min(source_width, source_height) <= 0:
        raise ValueError("source dimensions must be positive")
    if transform.rotation_degrees in {90, 270}:
        return source_height, source_width
    return source_width, source_height


def transform_detection_bbox(
    bbox: BBox,
    source_width: int,
    source_height: int,
    transform: CameraDisplayTransform,
) -> BBox:
    """Apply rotation then horizontal mirror to a source-frame bounding box."""

    if min(source_width, source_height) <= 0:
        raise ValueError("source dimensions must be positive")
    x1, y1, x2, y2 = bbox
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox coordinates must be ordered")

    rotation = transform.rotation_degrees
    if rotation == 0:
        rotated_points = ((x1, y1), (x1, y2), (x2, y1), (x2, y2))
        rotated_width, _rotated_height = source_width, source_height
    elif rotation == 90:
        rotated_points = (
            (y1, source_width - x1),
            (y1, source_width - x2),
            (y2, source_width - x1),
            (y2, source_width - x2),
        )
        rotated_width, _rotated_height = source_height, source_width
    elif rotation == 180:
        rotated_points = (
            (source_width - x1, source_height - y1),
            (source_width - x1, source_height - y2),
            (source_width - x2, source_height - y1),
            (source_width - x2, source_height - y2),
        )
        rotated_width, _rotated_height = source_width, source_height
    else:  # 270 degrees counterclockwise
        rotated_points = (
            (source_height - y1, x1),
            (source_height - y1, x2),
            (source_height - y2, x1),
            (source_height - y2, x2),
        )
        rotated_width, _rotated_height = source_height, source_width

    if transform.mirrored:
        rotated_points = tuple(
            (rotated_width - x, y) for x, y in rotated_points
        )

    xs = [point[0] for point in rotated_points]
    ys = [point[1] for point in rotated_points]
    return min(xs), min(ys), max(xs), max(ys)


def transform_detection_polygon(
    polygon: MaskPolygon,
    source_width: int,
    source_height: int,
    transform: CameraDisplayTransform,
) -> MaskPolygon:
    """Apply the same rotation and mirror as the displayed source frame."""

    if min(source_width, source_height) <= 0:
        raise ValueError("source dimensions must be positive")
    rotation = transform.rotation_degrees
    rotated_width = source_height if rotation in {90, 270} else source_width
    output: list[tuple[float, float]] = []
    for x, y in polygon:
        if rotation == 0:
            point = (x, y)
        elif rotation == 90:
            point = (y, source_width - x)
        elif rotation == 180:
            point = (source_width - x, source_height - y)
        else:
            point = (source_height - y, x)
        if transform.mirrored:
            point = (rotated_width - point[0], point[1])
        output.append(point)
    return tuple(output)


def map_detection_bbox_to_widget(
    bbox: BBox,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    transform: CameraDisplayTransform,
) -> RenderedDetectionBox:
    """Map a source detection into centered FIT_CENTER widget coordinates."""

    effective_width, effective_height = effective_frame_size(
        source_width,
        source_height,
        transform,
    )
    transformed = transform_detection_bbox(
        bbox,
        source_width,
        source_height,
        transform,
    )
    rect = keep_aspect_ratio_rect(
        effective_width,
        effective_height,
        target_width,
        target_height,
    )
    scale_x = rect.width / effective_width
    scale_y = rect.height / effective_height
    x1, y1, x2, y2 = transformed
    return RenderedDetectionBox(
        x=rect.x + x1 * scale_x,
        y=rect.y + y1 * scale_y,
        width=max(0.0, (x2 - x1) * scale_x),
        height=max(0.0, (y2 - y1) * scale_y),
    )


def map_detection_polygon_to_widget(
    polygon: MaskPolygon,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    transform: CameraDisplayTransform,
) -> RenderedDetectionPolygon:
    """Map a source mask polygon through rotation, mirror and FIT_CENTER."""

    effective_width, effective_height = effective_frame_size(
        source_width,
        source_height,
        transform,
    )
    rect = keep_aspect_ratio_rect(
        effective_width,
        effective_height,
        target_width,
        target_height,
    )
    scale_x = rect.width / effective_width
    scale_y = rect.height / effective_height
    transformed = transform_detection_polygon(
        polygon,
        source_width,
        source_height,
        transform,
    )
    return tuple(
        (rect.x + x * scale_x, rect.y + y * scale_y)
        for x, y in transformed
    )

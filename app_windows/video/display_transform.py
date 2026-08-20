"""Pixmap transformations shared by the monitor grid and focus view."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QTransform

from app_windows.models.camera_display_transform import CameraDisplayTransform


def rotate_video_pixmap_counterclockwise(pixmap: QPixmap, degrees: int) -> QPixmap:
    """Return a display pixmap rotated by the requested counterclockwise angle."""

    normalized_degrees = degrees % 360
    if pixmap.isNull() or normalized_degrees == 0:
        return pixmap
    return pixmap.transformed(
        QTransform().rotate(-normalized_degrees),
        Qt.TransformationMode.SmoothTransformation,
    )


def mirror_video_pixmap(pixmap: QPixmap, mirrored: bool) -> QPixmap:
    """Return a horizontally mirrored display pixmap when requested."""

    if pixmap.isNull() or not mirrored:
        return pixmap
    return pixmap.transformed(
        QTransform().scale(-1, 1),
        Qt.TransformationMode.FastTransformation,
    )


def transform_video_pixmap(
    pixmap: QPixmap,
    transform: CameraDisplayTransform,
) -> QPixmap:
    """Apply the same rotation-then-mirroring order in every video view."""

    rotated = rotate_video_pixmap_counterclockwise(pixmap, transform.rotation_degrees)
    return mirror_video_pixmap(rotated, transform.mirrored)

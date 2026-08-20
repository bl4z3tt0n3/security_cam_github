"""Shared, visual-only frame transforms for each monitor camera."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal


@dataclass(frozen=True)
class CameraDisplayTransform:
    """The orientation applied to one camera in every monitor view."""

    rotation_degrees: int = 0
    mirrored: bool = False

    def __post_init__(self) -> None:
        normalized_rotation = self.rotation_degrees % 360
        if normalized_rotation % 90 != 0:
            raise ValueError("camera rotation must be a multiple of 90 degrees")
        object.__setattr__(self, "rotation_degrees", normalized_rotation)

    def rotated_counterclockwise(self) -> "CameraDisplayTransform":
        return CameraDisplayTransform(
            rotation_degrees=self.rotation_degrees + 90,
            mirrored=self.mirrored,
        )

    def with_mirrored(self, mirrored: bool) -> "CameraDisplayTransform":
        return CameraDisplayTransform(
            rotation_degrees=self.rotation_degrees,
            mirrored=mirrored,
        )


class CameraDisplayTransformStore(QObject):
    """Keep one display transform per camera and notify every rendered view."""

    transform_changed = Signal(str, int, bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._transforms: dict[str, CameraDisplayTransform] = {}

    def transform_for(self, camera_id: str) -> CameraDisplayTransform:
        return self._transforms.get(camera_id, CameraDisplayTransform())

    def rotate_counterclockwise(self, camera_id: str) -> None:
        self.set_transform(camera_id, self.transform_for(camera_id).rotated_counterclockwise())

    def set_mirrored(self, camera_id: str, mirrored: bool) -> None:
        self.set_transform(camera_id, self.transform_for(camera_id).with_mirrored(mirrored))

    def set_transform(self, camera_id: str, transform: CameraDisplayTransform) -> None:
        current = self.transform_for(camera_id)
        if transform == current:
            return
        self._transforms[camera_id] = transform
        self.transform_changed.emit(
            camera_id,
            transform.rotation_degrees,
            transform.mirrored,
        )

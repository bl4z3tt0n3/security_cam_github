"""Data models used by the Windows monitor UI."""

from .camera_display_transform import CameraDisplayTransform, CameraDisplayTransformStore
from .camera_view_state import (
    CameraSlot,
    CameraViewSnapshot,
    CameraViewStatus,
    camera_slots_from_config,
)
from .face_recognition_state import (
    FaceGalleryState,
    FaceOverlayState,
    FaceRecognitionSettings,
    FaceRecognitionSnapshot,
    FaceRecognitionStatus,
)

__all__ = [
    "CameraDisplayTransform",
    "CameraDisplayTransformStore",
    "CameraSlot",
    "CameraViewSnapshot",
    "CameraViewStatus",
    "camera_slots_from_config",
    "FaceGalleryState",
    "FaceOverlayState",
    "FaceRecognitionSettings",
    "FaceRecognitionSnapshot",
    "FaceRecognitionStatus",
]

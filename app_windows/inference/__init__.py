"""Windows-local inference orchestration."""

from .person_detection_controller import (
    InferenceFrameSource,
    PersonDetectionController,
)
from .face_recognition_controller import FaceRecognitionController

__all__ = ["FaceRecognitionController", "InferenceFrameSource", "PersonDetectionController"]

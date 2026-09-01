"""Windows-local inference orchestration."""

from .person_detection_controller import (
    InferenceFrameSource,
    PersonDetectionController,
)
from .face_recognition_controller import FaceRecognitionController
from .face_recognition_fleet_controller import FleetFaceRecognitionController
from .person_detection_fleet_controller import FleetPersonDetectionController

__all__ = [
    "FaceRecognitionController",
    "FleetFaceRecognitionController",
    "FleetPersonDetectionController",
    "InferenceFrameSource",
    "PersonDetectionController",
]

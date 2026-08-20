"""Replaceable person-detection adapters and result types."""

from .base import (
    DisabledPersonDetector,
    PersonDetection,
    PersonDetectionError,
    PersonDetector,
)
from .fake import FakePersonDetector
from .factory import create_person_detector
from .prompts import normalize_prompts
from .onnx import OnnxPersonDetector
from .openvino import OpenVINOPersonDetector
from .yoloe import YoloEPersonDetector
from .synchronization import InferenceGate, SynchronizedPersonDetector
from .yoloe import YoloESegmentationDetector

__all__ = [
    "DisabledPersonDetector",
    "FakePersonDetector",
    "OnnxPersonDetector",
    "OpenVINOPersonDetector",
    "YoloEPersonDetector",
    "YoloESegmentationDetector",
    "PersonDetection",
    "PersonDetectionError",
    "PersonDetector",
    "InferenceGate",
    "SynchronizedPersonDetector",
    "create_person_detector",
    "normalize_prompts",
]

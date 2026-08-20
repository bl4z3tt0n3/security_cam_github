"""Immutable Windows-facing DTOs for the optional face pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from typing import Any, Literal

from app.config import AppConfig
from app.face import FaceCapability


FaceBackend = Literal["auto", "onnxruntime", "openvino", "opencv_dnn", "fake"]
FaceDevice = Literal["auto", "cpu", "gpu", "cuda", "npu"]


def percentage_to_fraction(value: int | float, *, minimum: int = 0) -> float:
    """Convert a UI percentage to the normalized face-stage value."""

    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum or numeric > 100:
        raise ValueError("face percentage must be between the configured bounds and 100")
    return numeric / 100.0


def fraction_to_percentage(value: float, *, minimum: int = 0) -> int:
    """Render a normalized face-stage value as the nearest UI percentage."""

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("face fraction must be finite")
    return max(minimum, min(100, int(math.floor(numeric * 100 + 0.5))))


class FaceRecognitionStatus(str, Enum):
    DISABLED = "disabled"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    UNSUPPORTED = "unsupported"
    MODEL_MISSING = "model_missing"

    @property
    def label(self) -> str:
        return {
            self.DISABLED: "DISABILITATO",
            self.LOADING: "CARICAMENTO",
            self.READY: "PRONTO",
            self.RUNNING: "IN ESECUZIONE",
            self.ERROR: "ERRORE",
            self.UNSUPPORTED: "NON SUPPORTATO",
            self.MODEL_MISSING: "MODELLO MANCANTE",
        }[self]


@dataclass(frozen=True)
class FaceRecognitionSettings:
    """Settings for detection, landmarks and recognition as separate stages."""

    face_detection_enabled: bool = False
    detector_id: str | None = None
    detector_backend: FaceBackend = "auto"
    detector_model: str | None = None
    detector_device: FaceDevice = "auto"
    detector_confidence_threshold: float = 0.5
    detector_inference_fps: float = 2.0
    landmarks_enabled: bool = False
    landmarker_id: str = "landmarks-regression-retail-0009"
    landmarker_backend: Literal["openvino"] = "openvino"
    landmarker_model: str | None = None
    landmarker_device: Literal["auto", "cpu", "gpu", "npu"] = "auto"
    recognition_enabled: bool = False
    recognizer_id: str | None = None
    recognizer_backend: Literal["auto", "onnxruntime", "openvino", "fake"] = "auto"
    recognizer_model: str | None = None
    recognizer_device: FaceDevice = "auto"
    recognition_threshold: float | None = None
    recognition_inference_fps: float = 1.0
    min_confirmations: int = 2
    confirmation_window_seconds: float = 10.0
    show_face_boxes: bool = True
    show_landmarks: bool = True

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "FaceRecognitionSettings":
        face = config.face_detection
        landmarks = config.face_landmarks
        recognition = config.recognition
        return cls(
            face_detection_enabled=face.enabled,
            detector_id=face.detector_id,
            detector_backend=face.backend,
            detector_model=face.model,
            detector_device=face.device,
            detector_confidence_threshold=face.confidence_threshold,
            detector_inference_fps=face.inference_fps,
            landmarks_enabled=landmarks.enabled,
            landmarker_id=landmarks.landmarker_id,
            landmarker_backend=landmarks.backend,
            landmarker_model=landmarks.model,
            landmarker_device=landmarks.device,
            recognition_enabled=recognition.enabled,
            recognizer_id=recognition.recognizer_id,
            recognizer_backend=recognition.backend,
            recognizer_model=recognition.model,
            recognizer_device=recognition.device,
            recognition_threshold=recognition.threshold,
            recognition_inference_fps=recognition.inference_fps,
            min_confirmations=recognition.min_confirmations,
            confirmation_window_seconds=recognition.confirmation_window_seconds,
            show_face_boxes=True,
            show_landmarks=landmarks.enabled,
        )

    @property
    def enabled(self) -> bool:
        return self.face_detection_enabled

    def __post_init__(self) -> None:
        if self.detector_backend not in {"auto", "onnxruntime", "openvino", "opencv_dnn", "fake"}:
            raise ValueError("unsupported face detector backend")
        if self.detector_device not in {"auto", "cpu", "gpu", "cuda", "npu"}:
            raise ValueError("unsupported face detector device")
        if self.recognizer_backend not in {"auto", "onnxruntime", "openvino", "fake"}:
            raise ValueError("unsupported face recognizer backend")
        if self.recognizer_device not in {"auto", "cpu", "gpu", "cuda", "npu"}:
            raise ValueError("unsupported face recognizer device")
        if self.landmarker_device not in {"auto", "cpu", "gpu", "npu"}:
            raise ValueError("unsupported face landmarker device")
        if not 0 <= float(self.detector_confidence_threshold) <= 1:
            raise ValueError("detector_confidence_threshold must be between 0 and 1")
        if self.recognition_threshold is not None and (
            not math.isfinite(float(self.recognition_threshold))
            or self.recognition_threshold < 0
        ):
            raise ValueError("recognition_threshold must be non-negative")
        if (
            not math.isfinite(float(self.detector_inference_fps))
            or not math.isfinite(float(self.recognition_inference_fps))
            or self.detector_inference_fps <= 0
            or self.recognition_inference_fps <= 0
        ):
            raise ValueError("face inference FPS values must be positive")
        if self.min_confirmations < 1 or self.confirmation_window_seconds <= 0:
            raise ValueError("recognition confirmation settings are invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "face_detection_enabled": self.face_detection_enabled,
            "detector_id": self.detector_id,
            "detector_backend": self.detector_backend,
            "detector_model": self.detector_model,
            "detector_device": self.detector_device,
            "detector_confidence_threshold": self.detector_confidence_threshold,
            "detector_inference_fps": self.detector_inference_fps,
            "landmarks_enabled": self.landmarks_enabled,
            "landmarker_id": self.landmarker_id,
            "landmarker_backend": self.landmarker_backend,
            "landmarker_model": self.landmarker_model,
            "landmarker_device": self.landmarker_device,
            "recognition_enabled": self.recognition_enabled,
            "recognizer_id": self.recognizer_id,
            "recognizer_backend": self.recognizer_backend,
            "recognizer_model": self.recognizer_model,
            "recognizer_device": self.recognizer_device,
            "recognition_threshold": self.recognition_threshold,
            "recognition_inference_fps": self.recognition_inference_fps,
            "min_confirmations": self.min_confirmations,
            "confirmation_window_seconds": self.confirmation_window_seconds,
            "show_face_boxes": self.show_face_boxes,
            "show_landmarks": self.show_landmarks,
        }


@dataclass(frozen=True)
class FaceOverlayState:
    camera_id: str
    track_id: int
    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...] = ()
    recognition_status: str = "unknown"
    person_id: str | None = None
    person_name: str | None = None
    score: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "landmarks": [list(point) for point in self.landmarks],
            "recognition_status": self.recognition_status,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "score": self.score,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class FaceRecognitionSnapshot:
    camera_id: str | None = None
    status: FaceRecognitionStatus = FaceRecognitionStatus.DISABLED
    message: str = "Riconoscimento facciale disabilitato"
    error: str | None = None
    detection_status: FaceRecognitionStatus | None = None
    detection_message: str | None = None
    detection_error: str | None = None
    recognition_status: FaceRecognitionStatus | None = None
    recognition_message: str | None = None
    recognition_error: str | None = None
    requested_detector_device: str | None = None
    actual_detector_device: str | None = None
    requested_recognizer_device: str | None = None
    actual_recognizer_device: str | None = None
    detector_id: str | None = None
    recognizer_id: str | None = None
    effective_recognizer_id: str | None = None
    detector_backend: str | None = None
    detector_model: str | None = None
    recognizer_backend: str | None = None
    recognizer_model: str | None = None
    frame_sequence: int | None = None
    frame_timestamp: datetime | None = None
    face_count: int = 0
    recognized_count: int = 0
    unknown_count: int = 0
    gallery_count: int = 0
    overlays: tuple[FaceOverlayState, ...] = ()
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "status": self.status.value,
            "message": self.message,
            "error": self.error,
            "detection_status": (
                self.detection_status.value if self.detection_status is not None else None
            ),
            "detection_message": self.detection_message,
            "detection_error": self.detection_error,
            "recognition_status": (
                self.recognition_status.value if self.recognition_status is not None else None
            ),
            "recognition_message": self.recognition_message,
            "recognition_error": self.recognition_error,
            "requested_detector_device": self.requested_detector_device,
            "actual_detector_device": self.actual_detector_device,
            "requested_recognizer_device": self.requested_recognizer_device,
            "actual_recognizer_device": self.actual_recognizer_device,
            "detector_id": self.detector_id,
            "recognizer_id": self.recognizer_id,
            "effective_recognizer_id": self.effective_recognizer_id,
            "detector_backend": self.detector_backend,
            "detector_model": self.detector_model,
            "recognizer_backend": self.recognizer_backend,
            "recognizer_model": self.recognizer_model,
            "frame_sequence": self.frame_sequence,
            "frame_timestamp": (
                self.frame_timestamp.isoformat() if self.frame_timestamp is not None else None
            ),
            "face_count": self.face_count,
            "recognized_count": self.recognized_count,
            "unknown_count": self.unknown_count,
            "gallery_count": self.gallery_count,
            "overlays": [overlay.to_dict() for overlay in self.overlays],
            "telemetry": dict(self.telemetry),
        }


@dataclass(frozen=True)
class FaceGalleryState:
    recognizer_id: str | None = None
    fingerprint: str | None = None
    persons: tuple[dict[str, Any], ...] = ()
    enrollment_people: tuple[dict[str, Any], ...] = ()
    enrollment_root: str | None = None
    enrollment_root_present: bool = True
    status: str = "ready"
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recognizer_id": self.recognizer_id,
            "fingerprint": self.fingerprint,
            "persons": [dict(person) for person in self.persons],
            "enrollment_people": [dict(person) for person in self.enrollment_people],
            "enrollment_root": self.enrollment_root,
            "enrollment_root_present": self.enrollment_root_present,
            "status": self.status,
            "message": self.message,
            "error": self.error,
        }


def capability_options(capabilities: tuple[FaceCapability, ...]) -> dict[str, tuple[str, ...]]:
    """Return stable IDs for UI combo boxes without duplicating capability logic."""

    return {
        "detectors": tuple(
            sorted({row.model_id for row in capabilities if row.component == "face_detection" and row.available})
        ),
        "recognizers": tuple(
            sorted({row.model_id for row in capabilities if row.component == "recognition" and row.available})
        ),
        "landmarkers": tuple(
            sorted({row.model_id for row in capabilities if row.component == "face_landmarks" and row.available})
        ),
    }


__all__ = [
    "FaceGalleryState",
    "FaceOverlayState",
    "FaceRecognitionSettings",
    "FaceRecognitionSnapshot",
    "FaceRecognitionStatus",
    "capability_options",
    "fraction_to_percentage",
    "percentage_to_fraction",
]

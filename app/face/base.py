"""Contracts for face detection, crop/alignment and quality filtering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Literal, Protocol

import cv2
import numpy as np


BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class FaceLandmark5:
    """Five ordered face landmarks in the same coordinate space as a face box."""

    points: tuple[Point, Point, Point, Point, Point]

    def __post_init__(self) -> None:
        if len(self.points) != 5:
            raise ValueError("face landmarks must contain exactly five points")
        normalized: list[Point] = []
        for point in self.points:
            if len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
                raise ValueError("face landmarks must contain finite x,y points")
            normalized.append((float(point[0]), float(point[1])))
        object.__setattr__(self, "points", tuple(normalized))

    def translated(self, dx: float, dy: float) -> "FaceLandmark5":
        """Return landmarks translated by one ROI origin."""

        return FaceLandmark5(
            tuple((x + float(dx), y + float(dy)) for x, y in self.points)  # type: ignore[arg-type]
        )


class FaceDetectorError(RuntimeError):
    """Raised when a face detector cannot execute."""


@dataclass(frozen=True)
class FaceDetection:
    bbox: BBox
    confidence: float
    landmarks: FaceLandmark5 | None = None
    coordinate_space: Literal["frame"] = "frame"
    detector_id: str | None = None
    backend: str | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if len(self.bbox) != 4 or not all(math.isfinite(value) for value in self.bbox):
            raise ValueError("face bbox must contain four finite values")
        x1, y1, x2, y2 = self.bbox
        if x2 < x1 or y2 < y1:
            raise ValueError("face bbox coordinates must be ordered as x1,y1,x2,y2")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("face confidence must be a finite value between 0 and 1")
        if self.coordinate_space != "frame":
            raise ValueError("face detection coordinate_space must be 'frame'")
        for name in ("detector_id", "backend", "device"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when present")

    def with_frame_bbox(self, bbox: BBox) -> "FaceDetection":
        """Return this detection with a translated frame-space bounding box."""

        return FaceDetection(
            bbox=bbox,
            confidence=self.confidence,
            landmarks=self.landmarks,
            coordinate_space=self.coordinate_space,
            detector_id=self.detector_id,
            backend=self.backend,
            device=self.device,
        )


class FaceDetector(ABC):
    """Replaceable detector contract. Bboxes are relative to the input frame."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect faces in one BGR frame."""


class FaceLandmarker(Protocol):
    """Produce five landmarks for a detection that did not expose them."""

    def landmark(
        self,
        image: np.ndarray,
        detection: FaceDetection,
    ) -> FaceLandmark5 | None:
        """Return landmarks in the detection's frame coordinate space."""


class FaceAligner(Protocol):
    def align(self, crop: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """Return an aligned face image, or the original crop when unsupported."""


class NoOpFaceAligner:
    """Alignment hook used until reliable landmarks are available."""

    def align(self, crop: np.ndarray, detection: FaceDetection) -> np.ndarray:
        del detection
        return crop.copy()


@dataclass(frozen=True)
class FaceCrop:
    image: np.ndarray
    bbox: BBox
    was_partial: bool


class FaceCropper:
    """Crop a bbox while retaining whether the source bbox crossed the frame."""

    def crop(self, frame: np.ndarray, bbox: BBox) -> FaceCrop | None:
        return crop_frame(frame, bbox)


def crop_frame(frame: np.ndarray, bbox: BBox) -> FaceCrop | None:
    if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame must have non-zero height and width")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(value) for value in bbox) or x2 <= x1 or y2 <= y1:
        return None
    clipped = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
    cx1, cy1, cx2, cy2 = clipped
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    left, top = int(math.floor(cx1)), int(math.floor(cy1))
    right, bottom = int(math.ceil(cx2)), int(math.ceil(cy2))
    image = frame[top:bottom, left:right].copy()
    return FaceCrop(
        image=image,
        bbox=(float(left), float(top), float(right), float(bottom)),
        was_partial=x1 < 0 or y1 < 0 or x2 > width or y2 > height,
    )


class FaceQualityReason(StrEnum):
    TOO_SMALL = "too_small"
    BLURRY = "blurry"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    PARTIAL_BBOX = "partial_bbox"
    LOW_CONFIDENCE = "low_confidence"
    LANDMARKS_MISSING = "landmarks_missing"
    ALIGNMENT_FAILED = "alignment_failed"


@dataclass(frozen=True)
class FaceQualityResult:
    accepted: bool
    reasons: tuple[FaceQualityReason, ...]
    width: int
    height: int
    blur_score: float
    brightness: float


class FaceQualityEvaluator:
    def __init__(
        self,
        *,
        min_width: int = 80,
        min_height: int = 80,
        blur_threshold: float = 40.0,
        min_brightness: float = 30.0,
        max_brightness: float = 225.0,
        min_confidence: float = 0.5,
    ) -> None:
        if min_width <= 0 or min_height <= 0:
            raise ValueError("minimum face dimensions must be positive")
        if blur_threshold < 0 or not math.isfinite(blur_threshold):
            raise ValueError("blur_threshold must be finite and non-negative")
        if not 0 <= min_brightness <= max_brightness <= 255:
            raise ValueError("brightness thresholds must satisfy 0 <= min <= max <= 255")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self.min_width = min_width
        self.min_height = min_height
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_confidence = min_confidence

    def evaluate(
        self,
        image: np.ndarray,
        detection: FaceDetection,
        *,
        partial_bbox: bool = False,
    ) -> FaceQualityResult:
        x1, y1, x2, y2 = detection.bbox
        width = max(0, int(math.ceil(x2) - math.floor(x1)))
        height = max(0, int(math.ceil(y2) - math.floor(y1)))
        face = image[max(0, int(math.floor(y1))):max(0, int(math.ceil(y2))),
                     max(0, int(math.floor(x1))):max(0, int(math.ceil(x2)))]
        if face.size == 0:
            blur_score = 0.0
            brightness = 0.0
        else:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(np.mean(gray))
        reasons: list[FaceQualityReason] = []
        if width < self.min_width or height < self.min_height:
            reasons.append(FaceQualityReason.TOO_SMALL)
        if blur_score < self.blur_threshold:
            reasons.append(FaceQualityReason.BLURRY)
        if brightness < self.min_brightness:
            reasons.append(FaceQualityReason.TOO_DARK)
        if brightness > self.max_brightness:
            reasons.append(FaceQualityReason.TOO_BRIGHT)
        if partial_bbox:
            reasons.append(FaceQualityReason.PARTIAL_BBOX)
        if detection.confidence < self.min_confidence:
            reasons.append(FaceQualityReason.LOW_CONFIDENCE)
        return FaceQualityResult(not reasons, tuple(reasons), width, height, blur_score, brightness)


@dataclass(frozen=True)
class FaceQualityDecision:
    detection: FaceDetection
    frame_bbox: BBox
    aligned_face: np.ndarray | None
    quality: FaceQualityResult
    recognition: object | None = None

"""Landmark-based face alignment for recognizer-specific input templates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .base import FaceAligner, FaceDetection, FaceLandmark5, FaceDetectorError


class FaceAlignmentError(RuntimeError):
    """Raised when a face cannot be aligned from its landmark contract."""


@dataclass(frozen=True)
class AlignmentTemplate:
    """Canonical five-point template for one embedding model."""

    width: int
    height: int
    points: FaceLandmark5

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("alignment template dimensions must be positive")
        for x, y in self.points.points:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("alignment template points must be finite")

    @classmethod
    def from_normalized(
        cls,
        width: int,
        height: int,
        points: tuple[tuple[float, float], ...],
    ) -> "AlignmentTemplate":
        if len(points) != 5:
            raise ValueError("normalized alignment template must contain five points")
        return cls(
            width,
            height,
            FaceLandmark5(tuple((x * width, y * height) for x, y in points)),  # type: ignore[arg-type]
        )


RETAIL_0095_TEMPLATE = AlignmentTemplate.from_normalized(
    128,
    128,
    (
        (0.31556875, 0.46157411),
        (0.68262292, 0.46157411),
        (0.50026250, 0.64050536),
        (0.34947188, 0.82469196),
        (0.65343646, 0.82469196),
    ),
)


ARC_FACE_TEMPLATE = AlignmentTemplate.from_normalized(
    112,
    112,
    (
        (0.34191607, 0.46157411),
        (0.65653393, 0.46157411),
        (0.50000000, 0.64050536),
        (0.37097590, 0.82469196),
        (0.62902410, 0.82469196),
    ),
)


FACENET_TEMPLATE = AlignmentTemplate.from_normalized(
    160,
    160,
    (
        (0.31556875, 0.46157411),
        (0.68262292, 0.46157411),
        (0.50026250, 0.64050536),
        (0.34947188, 0.82469196),
        (0.65343646, 0.82469196),
    ),
)


class SimilarityFaceAligner:
    """Align a face with all five landmarks using a similarity transform."""

    def __init__(self, template: AlignmentTemplate) -> None:
        self.template = template

    def align(self, crop: np.ndarray, detection: FaceDetection) -> np.ndarray:
        if detection.landmarks is None:
            raise FaceAlignmentError("face landmarks are required for alignment")
        image = np.asarray(crop)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise FaceAlignmentError("alignment input must be a non-empty BGR image")
        source = np.asarray(detection.landmarks.points, dtype=np.float32)
        target = np.asarray(self.template.points.points, dtype=np.float32)
        matrix, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.LMEDS,
        )
        if matrix is None or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise FaceAlignmentError("could not estimate a finite face alignment transform")
        if inliers is not None and int(np.count_nonzero(inliers)) < 3:
            raise FaceAlignmentError("face landmark alignment has fewer than three inliers")
        return cv2.warpAffine(
            image,
            matrix,
            (self.template.width, self.template.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )


def localize_detection(detection: FaceDetection, origin_x: float, origin_y: float) -> FaceDetection:
    """Translate a frame-space detection to coordinates relative to a crop."""

    x1, y1, x2, y2 = detection.bbox
    landmarks = detection.landmarks
    if landmarks is not None:
        landmarks = landmarks.translated(-origin_x, -origin_y)
    return FaceDetection(
        bbox=(x1 - origin_x, y1 - origin_y, x2 - origin_x, y2 - origin_y),
        confidence=detection.confidence,
        landmarks=landmarks,
        detector_id=detection.detector_id,
        backend=detection.backend,
        device=detection.device,
    )


__all__ = [
    "AlignmentTemplate",
    "ARC_FACE_TEMPLATE",
    "FACENET_TEMPLATE",
    "FaceAlignmentError",
    "RETAIL_0095_TEMPLATE",
    "SimilarityFaceAligner",
    "localize_detection",
]

"""Selective face detection, quality and recognition for tracked people."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math
import time
from typing import Any

import numpy as np

from app.metrics import CameraMetrics
from app.inference.synchronization import InferenceGate
from app.tracking.models import CameraState, Track

from .alignment import FaceAlignmentError, localize_detection
from .base import (
    FaceAligner,
    FaceCropper,
    FaceDetection,
    FaceDetector,
    FaceDetectorError,
    FaceLandmarker,
    FaceQualityDecision,
    FaceQualityEvaluator,
    FaceQualityReason,
    FaceQualityResult,
)
from .matcher import FaceMatcher, RecognitionResult


@dataclass(frozen=True)
class TrackedFaceResult:
    track_id: int
    decisions: tuple[FaceQualityDecision, ...]
    recognitions: tuple[RecognitionResult, ...] = ()

    @property
    def accepted(self) -> tuple[FaceQualityDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.quality.accepted)

    @property
    def best_recognition(self) -> RecognitionResult | None:
        if not self.recognitions:
            return None
        return max(
            self.recognitions,
            key=lambda result: result.score if result.score is not None else float("-inf"),
        )


@dataclass(frozen=True)
class FaceAnalysisResult:
    camera_id: str
    results: tuple[TrackedFaceResult, ...]
    skipped: bool = False
    recognition_error: str | None = None

    @property
    def recognitions(self) -> tuple[RecognitionResult, ...]:
        return tuple(
            recognition
            for result in self.results
            for recognition in result.recognitions
        )


class FaceAnalysisService:
    """Run face analysis only on active person tracks for one camera."""

    def __init__(
        self,
        camera_id: str,
        detector: FaceDetector,
        *,
        cropper: FaceCropper | None = None,
        aligner: FaceAligner | None = None,
        landmarker: FaceLandmarker | None = None,
        matcher: FaceMatcher | None = None,
        evaluator: FaceQualityEvaluator | None = None,
        metrics: CameraMetrics | None = None,
        inference_gate: InferenceGate | None = None,
    ) -> None:
        normalized = camera_id.strip()
        if not normalized:
            raise ValueError("camera_id cannot be empty")
        self.camera_id = normalized
        self.detector = detector
        self.cropper = cropper or FaceCropper()
        if matcher is not None and aligner is None:
            raise ValueError("a real FaceAligner is required when recognition is enabled")
        self.aligner = aligner
        self.landmarker = landmarker
        self.matcher = matcher
        self.evaluator = evaluator or FaceQualityEvaluator()
        self.metrics = metrics
        self.inference_gate = inference_gate
        self._last_recognition_error: str | None = None

    @property
    def last_recognition_error(self) -> str | None:
        return self._last_recognition_error

    def _detect(self, image: np.ndarray) -> list[FaceDetection]:
        if self.inference_gate is not None:
            return self.inference_gate.run(self.detector.detect, image)
        return self.detector.detect(image)

    def _match(self, image: np.ndarray) -> RecognitionResult:
        if self.matcher is None:
            raise RuntimeError("face matcher is not configured")
        # The matcher/gallery/embedder may be shared by the entire fleet.  Pass
        # this service's metrics as request-scoped context so accounting stays
        # per-camera without mutating shared matcher state.
        return self.matcher.match(image, metrics=self.metrics)

    @staticmethod
    def _face_belongs_to_track(
        detection: FaceDetection,
        *,
        person_bbox: tuple[float, float, float, float],
        origin_x: float,
        origin_y: float,
    ) -> bool:
        """Associate an ROI face with its source person using three guards."""

        x1, y1, x2, y2 = detection.bbox
        face = (x1 + origin_x, y1 + origin_y, x2 + origin_x, y2 + origin_y)
        px1, py1, px2, py2 = person_bbox
        center_x = (face[0] + face[2]) / 2.0
        center_y = (face[1] + face[3]) / 2.0
        contained = px1 <= center_x <= px2 and py1 <= center_y <= py2
        intersection = max(0.0, min(face[2], px2) - max(face[0], px1)) * max(
            0.0, min(face[3], py2) - max(face[1], py1)
        )
        face_area = max(0.0, face[2] - face[0]) * max(0.0, face[3] - face[1])
        person_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
        union = face_area + person_area - intersection
        iou = intersection / union if union > 0 else 0.0
        if contained or iou >= 0.01:
            return True
        person_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
        distance = math.hypot(center_x - person_center[0], center_y - person_center[1])
        diagonal = math.hypot(max(0.0, px2 - px1), max(0.0, py2 - py1))
        return diagonal > 0 and distance <= diagonal * 0.75

    def process(
        self,
        frame: np.ndarray,
        *,
        state: CameraState,
        tracks: Sequence[Track],
    ) -> FaceAnalysisResult:
        if state not in {
            CameraState.TRACKING,
            CameraState.FACE_ANALYSIS,
            CameraState.KNOWN,
            CameraState.UNKNOWN,
        } or not tracks:
            return FaceAnalysisResult(self.camera_id, (), skipped=True)
        results = tuple(self.analyze_track(frame, track) for track in tracks)
        return FaceAnalysisResult(self.camera_id, results)

    def analyze_track(
        self,
        frame: np.ndarray,
        track: Track,
        *,
        recognize: bool = True,
    ) -> TrackedFaceResult:
        """Analyze one person ROI and return frame-space decisions."""

        person_crop = self.cropper.crop(frame, track.bbox)
        if person_crop is None:
            return TrackedFaceResult(track.track_id, ())
        decisions: list[FaceQualityDecision] = []
        recognitions: list[RecognitionResult] = []
        detections: list[FaceDetection] = []
        detection_started = time.perf_counter()
        try:
            detections = self._detect(person_crop.image)
        finally:
            if self.metrics is not None:
                self.metrics.record_face_detection(
                    (time.perf_counter() - detection_started) * 1000.0,
                    len(detections),
                )
        origin_x, origin_y, _, _ = person_crop.bbox
        frame_height, frame_width = frame.shape[:2]
        for detection in detections:
            if not self._face_belongs_to_track(
                detection,
                person_bbox=track.bbox,
                origin_x=origin_x,
                origin_y=origin_y,
            ):
                continue
            enriched = detection
            if enriched.landmarks is None and self.landmarker is not None:
                landmark_started = time.perf_counter()
                try:
                    landmarks = self.landmarker.landmark(person_crop.image, enriched)
                except FaceDetectorError:
                    landmarks = None
                finally:
                    if self.metrics is not None:
                        self.metrics.record_face_landmark(
                            (time.perf_counter() - landmark_started) * 1000.0
                        )
                if landmarks is not None:
                    enriched = FaceDetection(
                        enriched.bbox,
                        enriched.confidence,
                        landmarks=landmarks,
                        detector_id=enriched.detector_id,
                        backend=enriched.backend,
                        device=enriched.device,
                    )
            frame_detection = localize_detection(enriched, -origin_x, -origin_y)
            frame_bbox = frame_detection.bbox
            face_partial = (
                frame_bbox[0] < 0
                or frame_bbox[1] < 0
                or frame_bbox[2] > frame_width
                or frame_bbox[3] > frame_height
            )
            quality = self.evaluator.evaluate(
                person_crop.image,
                enriched,
                partial_bbox=face_partial or person_crop.was_partial,
            )
            if self.matcher is not None and enriched.landmarks is None:
                quality = FaceQualityResult(
                    accepted=False,
                    reasons=quality.reasons + (FaceQualityReason.LANDMARKS_MISSING,),
                    width=quality.width,
                    height=quality.height,
                    blur_score=quality.blur_score,
                    brightness=quality.brightness,
                )
            if self.metrics is not None and not quality.accepted:
                self.metrics.record_face_quality_reject()
            decisions.append(
                FaceQualityDecision(
                    detection=frame_detection,
                    frame_bbox=frame_bbox,
                    aligned_face=None,
                    quality=quality,
                    recognition=None,
                )
            )
        result = TrackedFaceResult(track.track_id, tuple(decisions), tuple(recognitions))
        return self.recognize_track(frame, track, result) if recognize else result

    def recognize_track(
        self,
        frame: np.ndarray,
        track: Track,
        result: TrackedFaceResult,
    ) -> TrackedFaceResult:
        """Recognize cached detections without invoking the detector again."""

        if self.matcher is None or not result.decisions:
            return result
        person_crop = self.cropper.crop(frame, track.bbox)
        if person_crop is None:
            return result
        recognitions: list[RecognitionResult] = []
        decisions: list[FaceQualityDecision] = []
        self._last_recognition_error = None
        origin_x, origin_y, _, _ = person_crop.bbox
        for decision in result.decisions:
            aligned: np.ndarray | None = None
            recognition: RecognitionResult | None = None
            quality = decision.quality
            if quality.accepted:
                alignment_started = time.perf_counter()
                try:
                    if self.aligner is None:
                        raise FaceAlignmentError("a face aligner is required for recognition")
                    local_detection = localize_detection(
                        decision.detection,
                        origin_x,
                        origin_y,
                    )
                    aligned = self.aligner.align(person_crop.image, local_detection)
                except FaceAlignmentError:
                    quality = FaceQualityResult(
                        accepted=False,
                        reasons=quality.reasons + (FaceQualityReason.ALIGNMENT_FAILED,),
                        width=quality.width,
                        height=quality.height,
                        blur_score=quality.blur_score,
                        brightness=quality.brightness,
                    )
                finally:
                    if self.metrics is not None:
                        self.metrics.record_alignment(
                            (time.perf_counter() - alignment_started) * 1000.0
                        )
                if aligned is not None:
                    try:
                        recognition = self._match(aligned)
                        recognitions.append(recognition)
                    except Exception as exc:
                        # Recognition is an optional consumer of detector
                        # output.  Keep boxes/quality decisions alive when a
                        # model or gallery fails during one frame.
                        self._last_recognition_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
            decisions.append(
                FaceQualityDecision(
                    detection=decision.detection,
                    frame_bbox=decision.frame_bbox,
                    aligned_face=aligned,
                    quality=quality,
                    recognition=recognition,
                )
            )
        return TrackedFaceResult(track.track_id, tuple(decisions), tuple(recognitions))


__all__ = ["FaceAnalysisResult", "FaceAnalysisService", "TrackedFaceResult"]

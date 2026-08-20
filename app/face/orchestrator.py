"""Shared person-track to face-recognition orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import threading
import time
from typing import Any

import numpy as np

from app.tracking import CameraTrackingPipeline, CameraTrackingUpdate

from .service import FaceAnalysisResult, FaceAnalysisService, TrackedFaceResult
from .matcher import RecognitionResult


@dataclass(frozen=True)
class FaceOrchestrationResult:
    camera_id: str
    analysis: FaceAnalysisResult
    confirmations: tuple[Any, ...] = ()
    skipped: bool = False
    final_recognitions: tuple[tuple[int, RecognitionResult], ...] = ()

    def final_for_track(self, track_id: int) -> RecognitionResult | None:
        """Return the recognition state allowed to leave the temporal gate."""

        for candidate_track_id, result in self.final_recognitions:
            if candidate_track_id == track_id:
                return result
        return None


class FaceRecognitionOrchestrator:
    """Run one shared face service after person tracking, with latest-frame gating."""

    def __init__(
        self,
        service: FaceAnalysisService,
        *,
        face_fps: float = 1.0,
        recognition_fps: float | None = None,
        recognition_error: str | None = None,
        enabled: bool = True,
        clock: Any = time.monotonic,
    ) -> None:
        if face_fps <= 0 or not np.isfinite(float(face_fps)):
            raise ValueError("face_fps must be finite and positive")
        if recognition_fps is not None and (
            recognition_fps <= 0 or not np.isfinite(float(recognition_fps))
        ):
            raise ValueError("recognition_fps must be finite and positive")
        self.service = service
        self.face_fps = float(face_fps)
        self.recognition_fps = (
            float(recognition_fps if recognition_fps is not None else face_fps)
            if service.matcher is not None
            else None
        )
        self.recognition_error = recognition_error
        self.enabled = bool(enabled)
        self._clock = clock
        self._lock = threading.RLock()
        self._next_by_track: dict[int, float] = {}
        self._next_recognition_by_track: dict[int, float] = {}
        self._latest_by_track: dict[int, TrackedFaceResult] = {}
        self._last_final_by_track: dict[int, RecognitionResult] = {}

    @property
    def camera_id(self) -> str:
        return self.service.camera_id

    def reset(self) -> None:
        with self._lock:
            self._next_by_track.clear()
            self._next_recognition_by_track.clear()
            self._latest_by_track.clear()
            self._last_final_by_track.clear()

    def _due(self, track_id: int, now: float) -> bool:
        with self._lock:
            next_at = self._next_by_track.get(track_id, 0.0)
            if now < next_at:
                return False
            self._next_by_track[track_id] = now + 1.0 / self.face_fps
            return True

    def _recognition_due(self, track_id: int, now: float) -> bool:
        if self.recognition_fps is None:
            return False
        with self._lock:
            next_at = self._next_recognition_by_track.get(track_id, 0.0)
            if now < next_at:
                return False
            self._next_recognition_by_track[track_id] = now + 1.0 / self.recognition_fps
            return True

    def process(
        self,
        frame: np.ndarray,
        update: CameraTrackingUpdate,
        pipeline: CameraTrackingPipeline,
        *,
        timestamp: datetime | None = None,
    ) -> FaceOrchestrationResult:
        if not self.enabled or not update.active_tracks:
            return FaceOrchestrationResult(
                self.camera_id,
                FaceAnalysisResult(self.camera_id, (), skipped=True),
                skipped=True,
            )
        now = float(self._clock())
        active_ids = {track.track_id for track in update.active_tracks}
        with self._lock:
            for track_id in tuple(self._next_by_track):
                if track_id not in active_ids:
                    self._next_by_track.pop(track_id, None)
                    self._next_recognition_by_track.pop(track_id, None)
                    self._latest_by_track.pop(track_id, None)
                    self._last_final_by_track.pop(track_id, None)
        results: list[TrackedFaceResult] = []
        confirmations: list[Any] = []
        final_recognitions: list[tuple[int, RecognitionResult]] = []
        for track in update.active_tracks:
            detector_due = self._due(track.track_id, now)
            if detector_due:
                base = self.service.analyze_track(frame, track, recognize=False)
                with self._lock:
                    self._latest_by_track[track.track_id] = base
            else:
                with self._lock:
                    base = self._latest_by_track.get(track.track_id)
            if base is None:
                continue
            result = base
            recognition_updated = False
            if self.recognition_fps is not None and self._recognition_due(track.track_id, now):
                result = self.service.recognize_track(frame, track, base)
                recognition_updated = True
                with self._lock:
                    self._latest_by_track[track.track_id] = result
            results.append(result)
            recognition = result.best_recognition
            if not recognition_updated:
                with self._lock:
                    cached_final = self._last_final_by_track.get(track.track_id)
                if cached_final is not None:
                    final_recognitions.append((track.track_id, cached_final))
                continue
            if recognition is None:
                if recognition_updated:
                    with self._lock:
                        self._last_final_by_track.pop(track.track_id, None)
                continue
            confirmer = pipeline.recognition_confirmer
            if confirmer is None:
                final = _unknown_result(recognition)
                with self._lock:
                    self._last_final_by_track[track.track_id] = final
                final_recognitions.append((track.track_id, final))
                continue
            pipeline.begin_face_analysis(track.track_id)
            confirmation = pipeline.observe_recognition(
                track.track_id,
                recognition,
                frame=frame,
                timestamp=timestamp,
            )
            confirmations.append(confirmation)
            final = (
                confirmation.stable_result
                if confirmation.confirmed and confirmation.stable_result is not None
                else _unknown_result(recognition)
            )
            with self._lock:
                self._last_final_by_track[track.track_id] = final
            final_recognitions.append((track.track_id, final))
        if not results:
            return FaceOrchestrationResult(
                self.camera_id,
                FaceAnalysisResult(self.camera_id, (), skipped=True),
                skipped=True,
            )
        return FaceOrchestrationResult(
            self.camera_id,
            FaceAnalysisResult(
                self.camera_id,
                tuple(results),
                recognition_error=self.service.last_recognition_error,
            ),
            tuple(confirmations),
            final_recognitions=tuple(final_recognitions),
        )


def _unknown_result(result: RecognitionResult) -> RecognitionResult:
    """Hide an unconfirmed candidate identity while retaining calibration data."""

    if result.status == "unknown":
        return result
    return RecognitionResult(
        status="unknown",
        person_id=None,
        person_name=None,
        score=result.score,
        threshold=result.threshold,
        recognizer_id=result.recognizer_id,
        requested_device=result.requested_device,
        actual_device=result.actual_device,
    )


__all__ = ["FaceOrchestrationResult", "FaceRecognitionOrchestrator"]

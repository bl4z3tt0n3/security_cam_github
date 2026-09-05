"""Shared person-track to face-recognition orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
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
    """Run face work after person tracking with session and load guards.

    The default two-track work budget bounds a single scheduler pass.  One slot
    is preferentially given to a new/unconfirmed track and remaining slots use
    round-robin order across all active tracks, so confirmed tracks still
    receive periodic rechecks and no track can permanently monopolize work.
    """

    def __init__(
        self,
        service: FaceAnalysisService,
        *,
        face_fps: float = 1.0,
        recognition_fps: float | None = None,
        recognition_error: str | None = None,
        enabled: bool = True,
        max_tracks_per_call: int | None = 2,
        clock: Any = time.monotonic,
    ) -> None:
        if face_fps <= 0 or not np.isfinite(float(face_fps)):
            raise ValueError("face_fps must be finite and positive")
        if recognition_fps is not None and (
            recognition_fps <= 0 or not np.isfinite(float(recognition_fps))
        ):
            raise ValueError("recognition_fps must be finite and positive")
        if max_tracks_per_call is not None and (
            isinstance(max_tracks_per_call, bool)
            or not isinstance(max_tracks_per_call, int)
            or max_tracks_per_call < 1
        ):
            raise ValueError("max_tracks_per_call must be a positive integer or None")
        self.service = service
        self.face_fps = float(face_fps)
        self.recognition_fps = (
            float(recognition_fps if recognition_fps is not None else face_fps)
            if service.matcher is not None
            else None
        )
        self.recognition_error = recognition_error
        self.enabled = bool(enabled)
        self.max_tracks_per_call = max_tracks_per_call
        self._clock = clock
        self._lock = threading.RLock()
        self._next_by_track: dict[int, float] = {}
        self._next_recognition_by_track: dict[int, float] = {}
        self._latest_by_track: dict[int, TrackedFaceResult] = {}
        self._last_final_by_track: dict[int, RecognitionResult] = {}
        self._pipeline_session_generation: int | None = None
        self._round_robin_cursor = 0
        self._uncertain_cursor = 0

    @property
    def camera_id(self) -> str:
        return self.service.camera_id

    def reset(self) -> None:
        with self._lock:
            self._next_by_track.clear()
            self._next_recognition_by_track.clear()
            self._latest_by_track.clear()
            self._last_final_by_track.clear()
            self._pipeline_session_generation = None
            self._round_robin_cursor = 0
            self._uncertain_cursor = 0

    @staticmethod
    def _advance_deadline(previous_due: float, now: float, fps: float) -> float:
        interval = 1.0 / fps
        if previous_due <= 0:
            return now + interval
        deadline = previous_due + interval
        if deadline <= now:
            missed = math.floor((now - deadline) / interval) + 1
            deadline += missed * interval
        return deadline

    def _due(self, track_id: int, now: float) -> bool:
        with self._lock:
            next_at = self._next_by_track.get(track_id, 0.0)
            if now < next_at:
                return False
            self._next_by_track[track_id] = self._advance_deadline(
                next_at,
                now,
                self.face_fps,
            )
            return True

    def _recognition_due(self, track_id: int, now: float) -> bool:
        if self.recognition_fps is None:
            return False
        with self._lock:
            next_at = self._next_recognition_by_track.get(track_id, 0.0)
            if now < next_at:
                return False
            self._next_recognition_by_track[track_id] = self._advance_deadline(
                next_at,
                now,
                self.recognition_fps,
            )
            return True

    def _select_tracks(self, tracks: tuple[Any, ...]) -> tuple[Any, ...]:
        limit = self.max_tracks_per_call
        if limit is None or len(tracks) <= limit:
            return tracks
        with self._lock:
            final_by_track = dict(self._last_final_by_track)
            rr_start = self._round_robin_cursor % len(tracks)
            round_robin = tracks[rr_start:] + tracks[:rr_start]
            self._round_robin_cursor = (rr_start + limit) % len(tracks)

            uncertain = tuple(
                track
                for track in tracks
                if (
                    track.track_id not in final_by_track
                    or final_by_track[track.track_id].status == "unknown"
                )
            )
            selected: list[Any] = []
            if uncertain:
                uncertain_start = self._uncertain_cursor % len(uncertain)
                first = uncertain[uncertain_start]
                self._uncertain_cursor = (uncertain_start + 1) % len(uncertain)
                selected.append(first)
            for track in round_robin:
                if len(selected) >= limit:
                    break
                if all(candidate.track_id != track.track_id for candidate in selected):
                    selected.append(track)
            return tuple(selected)

    def _session_is_current(
        self,
        pipeline: CameraTrackingPipeline,
        update: CameraTrackingUpdate,
        session_generation: int,
    ) -> bool:
        return (
            pipeline.session_generation == session_generation
            and pipeline.latest_update is update
        )

    def _skipped_result(self) -> FaceOrchestrationResult:
        return FaceOrchestrationResult(
            self.camera_id,
            FaceAnalysisResult(self.camera_id, (), skipped=True),
            skipped=True,
        )

    def process(
        self,
        frame: np.ndarray,
        update: CameraTrackingUpdate,
        pipeline: CameraTrackingPipeline,
        *,
        timestamp: datetime | None = None,
    ) -> FaceOrchestrationResult:
        if not self.enabled or not update.active_tracks:
            return self._skipped_result()

        session_generation = pipeline.session_generation
        # A reset may occur after the caller snapshots latest_update but before
        # face work starts.  Refuse such an orphaned update immediately.
        if pipeline.latest_update is not update:
            self.reset()
            with self._lock:
                self._pipeline_session_generation = session_generation
            return self._skipped_result()

        with self._lock:
            previous_session = self._pipeline_session_generation
        if previous_session is None:
            with self._lock:
                self._pipeline_session_generation = session_generation
        elif previous_session != session_generation:
            self.reset()
            with self._lock:
                self._pipeline_session_generation = session_generation

        now = float(self._clock())
        active_ids = {track.track_id for track in update.active_tracks}
        with self._lock:
            for track_id in tuple(self._next_by_track):
                if track_id not in active_ids:
                    self._next_by_track.pop(track_id, None)
                    self._next_recognition_by_track.pop(track_id, None)
                    self._latest_by_track.pop(track_id, None)
                    self._last_final_by_track.pop(track_id, None)

        selected_tracks = self._select_tracks(tuple(update.active_tracks))
        results: list[TrackedFaceResult] = []
        confirmations: list[Any] = []
        final_recognitions: list[tuple[int, RecognitionResult]] = []
        for track in selected_tracks:
            if not self._session_is_current(pipeline, update, session_generation):
                self.reset()
                return self._skipped_result()

            detector_due = self._due(track.track_id, now)
            if detector_due:
                base = self.service.analyze_track(frame, track, recognize=False)
                if not self._session_is_current(pipeline, update, session_generation):
                    self.reset()
                    return self._skipped_result()
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
                if not self._session_is_current(pipeline, update, session_generation):
                    self.reset()
                    return self._skipped_result()
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
                with self._lock:
                    self._last_final_by_track.pop(track.track_id, None)
                continue

            if not self._session_is_current(pipeline, update, session_generation):
                self.reset()
                return self._skipped_result()
            confirmer = pipeline.recognition_confirmer
            if confirmer is None:
                final = _unknown_result(recognition)
                with self._lock:
                    self._last_final_by_track[track.track_id] = final
                final_recognitions.append((track.track_id, final))
                continue
            try:
                pipeline.begin_face_analysis(track.track_id)
                if not self._session_is_current(pipeline, update, session_generation):
                    self.reset()
                    return self._skipped_result()
                confirmation = pipeline.observe_recognition(
                    track.track_id,
                    recognition,
                    frame=frame,
                    timestamp=timestamp,
                )
            except ValueError:
                # A concurrent tracker/session transition can invalidate a
                # track between the guards and the serialized pipeline call.
                self.reset()
                return self._skipped_result()
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
            return self._skipped_result()
        if not self._session_is_current(pipeline, update, session_generation):
            self.reset()
            return self._skipped_result()
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

"""Independent camera pipeline combining detection tracking and camera state."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from functools import wraps
import threading
from typing import Any

from app.inference.base import PersonDetection
from app.metrics import CameraMetrics

from .models import (
    CameraState,
    CameraTrackingUpdate,
    FaceAnalysisHook,
    FaceAnalysisOutcome,
    EventPublisherLike,
    RecognitionConfirmationLike,
    RecognitionResultLike,
    Track,
    TrackRecognitionConfirmerLike,
)
from .state_machine import CameraStateMachine
from .tracker import IoUGreedyTracker, PersonTracker


def _synchronized(method: Any) -> Any:
    """Serialize state-machine mutations while face inference runs elsewhere."""

    @wraps(method)
    def wrapped(self: "CameraTrackingPipeline", *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


class CameraTrackingPipeline:
    """Own one tracker and one state machine for exactly one camera."""

    def __init__(
        self,
        camera_id: str,
        *,
        tracker: PersonTracker | None = None,
        state_machine: CameraStateMachine | None = None,
        recognition_confirmer: TrackRecognitionConfirmerLike | None = None,
        event_publisher: EventPublisherLike | None = None,
        metrics: CameraMetrics | None = None,
    ) -> None:
        normalized_id = camera_id.strip()
        if not normalized_id:
            raise ValueError("camera_id cannot be empty")
        self._camera_id = normalized_id
        self._tracker = tracker or IoUGreedyTracker()
        self._state_machine = state_machine or CameraStateMachine()
        self._recognition_confirmer = recognition_confirmer
        self._event_publisher = event_publisher
        self._metrics = metrics
        self._lock = threading.RLock()
        self._latest_update: CameraTrackingUpdate | None = None
        # Monotonically increasing camera-session token.  Face orchestration
        # observes this value so track IDs and cached identities can never be
        # reused across a provider replacement/reconnect or other reset.
        self._session_generation = 0

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def tracker(self) -> PersonTracker:
        return self._tracker

    @property
    def state_machine(self) -> CameraStateMachine:
        return self._state_machine

    @property
    def recognition_confirmer(self) -> TrackRecognitionConfirmerLike | None:
        return self._recognition_confirmer

    @property
    def metrics(self) -> CameraMetrics | None:
        return self._metrics

    @property
    def session_generation(self) -> int:
        """Return the current logical camera/tracking session token."""

        with self._lock:
            return self._session_generation

    @property
    def latest_update(self) -> CameraTrackingUpdate | None:
        """Return the most recent immutable tracking snapshot for this camera."""

        with self._lock:
            return self._latest_update

    @_synchronized
    def set_recognition_confirmer(
        self,
        confirmer: TrackRecognitionConfirmerLike | None,
    ) -> None:
        """Attach the face confirmation policy without creating another tracker."""

        self._recognition_confirmer = confirmer

    @property
    def event_publisher(self) -> EventPublisherLike | None:
        return self._event_publisher

    @property
    def state(self) -> CameraState:
        return self._state_machine.state

    @_synchronized
    def reset(self, *, reason: str = "camera session reset") -> None:
        """Reset tracker, state and per-track recognition confirmation together."""

        active_tracks = self._state_machine.active_tracks
        if self._recognition_confirmer is not None:
            for track in active_tracks:
                self._recognition_confirmer.forget(track.track_id)
        reset_tracker = getattr(self._tracker, "reset", None)
        if callable(reset_tracker):
            reset_tracker()
        self._latest_update = None
        self._state_machine.reset(reason=reason)
        self._session_generation += 1
        if self._metrics is not None:
            self._metrics.set_active_tracks(0)

    @_synchronized
    def update(self, detections: Sequence[PersonDetection]) -> CameraTrackingUpdate:
        """Process one sampled detection result for this camera only."""

        tracking = self._tracker.update(detections)
        if self._metrics is not None:
            self._metrics.set_active_tracks(len(tracking.active_tracks))
        if self._recognition_confirmer is not None:
            for lost_track in tracking.lost_tracks:
                self._recognition_confirmer.forget(lost_track.track_id)
        transitions = self._state_machine.observe(tracking)
        result = CameraTrackingUpdate(
            camera_id=self._camera_id,
            state=self._state_machine.state,
            tracking=tracking,
            transitions=transitions,
        )
        self._latest_update = result
        return result

    @_synchronized
    def begin_face_analysis(self, track_id: int) -> Track:
        return self._state_machine.begin_face_analysis(track_id)

    @_synchronized
    def analyze_current_track(self, hook: FaceAnalysisHook) -> FaceAnalysisOutcome | None:
        return self._state_machine.analyze_current_track(hook)

    @_synchronized
    def observe_recognition(
        self,
        track_id: int,
        result: RecognitionResultLike,
        *,
        frame: Any = None,
        timestamp: datetime | None = None,
    ) -> RecognitionConfirmationLike:
        """Feed one face-match result through this camera's temporal gate.

        The confirmer owns per-track evidence.  A confirmed result changes the
        aggregate camera state only when that track is the one currently in
        ``FACE_ANALYSIS``; other active tracks keep their own independent
        confirmation history and can be selected afterward.
        """

        if self._recognition_confirmer is None:
            raise RuntimeError("no recognition confirmer configured for this camera")
        if track_id not in {track.track_id for track in self._state_machine.active_tracks}:
            raise ValueError(f"track {track_id} is not active")
        confirmation = self._recognition_confirmer.observe(track_id, result)
        if (
            confirmation.confirmed
            and self.state is CameraState.FACE_ANALYSIS
            and self._state_machine.analysis_track_id == track_id
        ):
            self._state_machine.apply_confirmed_face_analysis(
                track_id,
                confirmation.result.status,
                confirmed=True,
            )
        if confirmation.confirmed and self._event_publisher is not None:
            event = self._event_publisher.publish_recognition(
                camera_id=self._camera_id,
                track_id=track_id,
                result=confirmation.result,
                frame=frame,
                timestamp=timestamp,
            )
            if event is not None and self._metrics is not None:
                self._metrics.record_event()
        return confirmation

    @_synchronized
    def start_cooldown(self, track_id: int | None = None) -> None:
        self._state_machine.start_cooldown(track_id)

    @_synchronized
    def finish_cooldown(self) -> None:
        self._state_machine.finish_cooldown()

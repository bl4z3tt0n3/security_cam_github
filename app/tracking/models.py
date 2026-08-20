"""Public data contracts for camera tracking and state transitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.inference.base import PersonDetection


class CameraState(str, Enum):
    """Aggregated processing state for one camera."""

    PERSON_SCAN = "person_scan"
    PERSON_DETECTED = "person_detected"
    TRACKING = "tracking"
    FACE_ANALYSIS = "face_analysis"
    KNOWN = "known"
    UNKNOWN = "unknown"
    COOLDOWN = "cooldown"


class FaceAnalysisOutcome(str, Enum):
    """Outcome accepted by the future face-analysis hook."""

    KNOWN = "known"
    UNKNOWN = "unknown"


class InvalidStateTransitionError(RuntimeError):
    """Raised when a caller requests a transition not valid for the state."""


@dataclass(frozen=True)
class Track:
    """Immutable snapshot of one temporary person track."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    first_seen_at: datetime
    last_seen_at: datetime
    missed_samples: int = 0


@dataclass(frozen=True)
class TrackingUpdate:
    """Result of associating one detection sample with existing tracks."""

    active_tracks: tuple[Track, ...]
    new_tracks: tuple[Track, ...]
    updated_tracks: tuple[Track, ...]
    lost_tracks: tuple[Track, ...]


@dataclass(frozen=True)
class StateTransition:
    """One observable state transition made by the camera state machine."""

    previous: CameraState
    current: CameraState
    reason: str


@dataclass(frozen=True)
class CameraTrackingUpdate:
    """Combined tracker and state-machine result for one camera sample."""

    camera_id: str
    state: CameraState
    tracking: TrackingUpdate
    transitions: tuple[StateTransition, ...]

    @property
    def active_tracks(self) -> tuple[Track, ...]:
        return self.tracking.active_tracks

    @property
    def new_tracks(self) -> tuple[Track, ...]:
        return self.tracking.new_tracks

    @property
    def lost_tracks(self) -> tuple[Track, ...]:
        return self.tracking.lost_tracks


class FaceAnalysisHook(Protocol):
    """Future face-analysis adapter; returning ``None`` keeps analysis pending."""

    def analyze(self, track: Track) -> FaceAnalysisOutcome | None:
        ...


class RecognitionResultLike(Protocol):
    """Minimal face-recognition contract needed by the tracking layer."""

    status: str
    person_id: str | None
    person_name: str | None
    score: float | None


class RecognitionConfirmationLike(Protocol):
    """Minimal temporal-confirmation contract needed by one camera pipeline."""

    result: RecognitionResultLike
    confirmed: bool


class TrackRecognitionConfirmerLike(Protocol):
    """Per-camera confirmer contract, kept independent from face adapters."""

    def observe(
        self,
        track_id: int,
        result: RecognitionResultLike,
    ) -> RecognitionConfirmationLike:
        ...

    def forget(self, track_id: int) -> None:
        ...


class EventPublisherLike(Protocol):
    """Optional boundary for publishing a confirmed recognition event."""

    def publish_recognition(
        self,
        *,
        camera_id: str,
        track_id: int,
        result: RecognitionResultLike,
        frame: Any = None,
        timestamp: datetime | None = None,
    ) -> Any:
        ...


def detection_timestamp(detection: PersonDetection) -> datetime:
    """Return the detection timestamp while keeping model imports lazy."""

    return detection.timestamp


def as_detection_sequence(
    detections: Sequence[PersonDetection],
) -> tuple[PersonDetection, ...]:
    """Materialize a detection sequence once at the tracker boundary."""

    return tuple(detections)

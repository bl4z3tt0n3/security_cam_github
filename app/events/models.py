"""Data contracts for local recognition events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import math


class EventType(StrEnum):
    """Recognition outcomes that are meaningful as local events."""

    KNOWN_PERSON = "known_person"
    UNKNOWN_PERSON = "unknown_person"


@dataclass(frozen=True)
class Event:
    """One deduplicated recognition event persisted by :class:`EventManager`."""

    id: str
    timestamp: datetime
    camera_id: str
    track_id: int
    type: EventType
    person_id: str | None = None
    person_name: str | None = None
    recognition_score: float | None = None
    snapshot_path: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("event id cannot be empty")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("event timestamp must be timezone-aware")

        camera_id = self.camera_id.strip()
        if not camera_id:
            raise ValueError("event camera_id cannot be empty")
        object.__setattr__(self, "camera_id", camera_id)

        if (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, int)
            or self.track_id < 1
        ):
            raise ValueError("event track_id must be a positive integer")

        try:
            event_type = EventType(self.type)
        except ValueError as exc:
            raise ValueError("event type must be known_person or unknown_person") from exc
        object.__setattr__(self, "type", event_type)

        if self.recognition_score is not None and not math.isfinite(self.recognition_score):
            raise ValueError("recognition_score must be finite when present")

        if event_type is EventType.KNOWN_PERSON:
            if not self.person_id or not self.person_name:
                raise ValueError("known_person events require person_id and person_name")
        elif self.person_id is not None or self.person_name is not None:
            raise ValueError("unknown_person events cannot contain a person identity")

        if self.snapshot_path is not None and not self.snapshot_path.strip():
            raise ValueError("snapshot_path cannot be blank when present")

    def to_metadata(self) -> dict[str, object]:
        """Return the portable JSON representation stored on disk."""

        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "type": self.type.value,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "recognition_score": self.recognition_score,
            "snapshot_path": self.snapshot_path,
        }

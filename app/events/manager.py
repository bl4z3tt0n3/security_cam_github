"""Deduplicated local event publication."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from .models import Event, EventType
from .storage import EventStorage, SnapshotWriter


class EventManager:
    """Publish known/unknown recognition events independently per camera."""

    def __init__(
        self,
        events_dir: Path | str,
        *,
        save_snapshot: bool = True,
        known_person_cooldown_seconds: float = 30.0,
        unknown_person_cooldown_seconds: float = 15.0,
        storage: EventStorage | None = None,
        snapshot_writer: SnapshotWriter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._validate_cooldown(
            known_person_cooldown_seconds,
            "known_person_cooldown_seconds",
        )
        self._validate_cooldown(
            unknown_person_cooldown_seconds,
            "unknown_person_cooldown_seconds",
        )
        self.save_snapshot = bool(save_snapshot)
        self.known_person_cooldown_seconds = float(known_person_cooldown_seconds)
        self.unknown_person_cooldown_seconds = float(unknown_person_cooldown_seconds)
        self._storage = storage or EventStorage(events_dir)
        self._snapshot_writer = snapshot_writer or SnapshotWriter(logger=logger)
        self._logger = logger or logging.getLogger(__name__)
        self._last_events: dict[tuple[str, int, EventType], datetime] = {}
        self._lock = threading.RLock()
        self._closed = False

    @staticmethod
    def _validate_cooldown(value: float, name: str) -> None:
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    @staticmethod
    def _timestamp(value: datetime | None) -> datetime:
        actual = value or datetime.now(timezone.utc)
        if actual.tzinfo is None or actual.utcoffset() is None:
            raise ValueError("event timestamp must be timezone-aware")
        return actual.astimezone(timezone.utc)

    def emit(
        self,
        *,
        camera_id: str,
        track_id: int,
        event_type: EventType | str,
        timestamp: datetime | None = None,
        person_id: str | None = None,
        person_name: str | None = None,
        recognition_score: float | None = None,
        frame: Any = None,
    ) -> Event | None:
        """Persist one event, returning ``None`` when deduplicated or unavailable."""

        try:
            normalized_type = EventType(event_type)
        except ValueError as exc:
            raise ValueError("event_type must be known_person or unknown_person") from exc

        event_timestamp = self._timestamp(timestamp)
        event_id = uuid4().hex
        event = Event(
            id=event_id,
            timestamp=event_timestamp,
            camera_id=camera_id,
            track_id=track_id,
            type=normalized_type,
            person_id=person_id,
            person_name=person_name,
            recognition_score=recognition_score,
        )
        key = (event.camera_id, event.track_id, event.type)
        cooldown = self._cooldown_for(event.type)

        with self._lock:
            if self._closed:
                self._logger.warning("event publication ignored after manager close")
                return None
            previous = self._last_events.get(key)
            if previous is not None:
                elapsed = (event.timestamp - previous).total_seconds()
                if elapsed < cooldown:
                    return None

            try:
                directory = self._storage.prepare_event_directory(event)
                self._storage.write_metadata(event, directory)
            except OSError:
                self._logger.exception("event metadata write failed")
                return None

            snapshot_path: str | None = None
            if self.save_snapshot and frame is not None:
                destination = directory / "snapshot.jpg"
                if self._snapshot_writer.submit(destination, frame):
                    snapshot_path = self._storage.relative_path(destination)
                    event = replace(event, snapshot_path=snapshot_path)
                    try:
                        self._storage.write_metadata(event, directory)
                    except OSError:
                        self._logger.exception("event snapshot metadata update failed")
                        event = replace(event, snapshot_path=None)

            self._last_events[key] = event.timestamp
            return event

    def publish_recognition(
        self,
        *,
        camera_id: str,
        track_id: int,
        result: Any,
        frame: Any = None,
        timestamp: datetime | None = None,
    ) -> Event | None:
        """Map a confirmed recognition result to a known/unknown event."""

        status = getattr(result, "status", None)
        if status == "known":
            event_type = EventType.KNOWN_PERSON
        elif status == "unknown":
            event_type = EventType.UNKNOWN_PERSON
        else:
            raise ValueError("recognition result status must be known or unknown")
        return self.emit(
            camera_id=camera_id,
            track_id=track_id,
            event_type=event_type,
            timestamp=timestamp,
            person_id=getattr(result, "person_id", None),
            person_name=getattr(result, "person_name", None),
            recognition_score=getattr(result, "score", None),
            frame=frame,
        )

    def flush(self, timeout_s: float | None = None) -> bool:
        return self._snapshot_writer.flush(timeout_s)

    def close(self, timeout_s: float | None = 5.0) -> bool:
        with self._lock:
            self._closed = True
        return self._snapshot_writer.close(timeout_s)

    def __enter__(self) -> "EventManager":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _cooldown_for(self, event_type: EventType) -> float:
        if event_type is EventType.KNOWN_PERSON:
            return self.known_person_cooldown_seconds
        return self.unknown_person_cooldown_seconds

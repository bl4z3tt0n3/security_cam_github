"""Optional recording boundary reserved for a future circular-buffer backend."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from app.config import RecordingConfig


class RecordingBackend(Protocol):
    """Backend boundary for future pre/post-event or continuous recording."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def append_frame(self, frame: Any, timestamp: datetime) -> None:
        ...

    def publish_event(self, event: Any) -> None:
        ...


class RecordingNotConfiguredError(RuntimeError):
    """Raised when recording is enabled without a concrete backend."""


class RecordingController:
    """Gate recording startup without creating components when disabled."""

    def __init__(self, *, enabled: bool = False, backend: RecordingBackend | None = None) -> None:
        self.enabled = bool(enabled)
        self._backend = backend if self.enabled else None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> bool:
        if not self.enabled:
            return False
        if self._backend is None:
            raise RecordingNotConfiguredError(
                "recording is enabled but no recording backend is configured"
            )
        if not self._started:
            self._backend.start()
            self._started = True
        return True

    def append_frame(self, frame: Any, timestamp: datetime) -> None:
        if self._started and self._backend is not None:
            self._backend.append_frame(frame, timestamp)

    def publish_event(self, event: Any) -> None:
        if self._started and self._backend is not None:
            self._backend.publish_event(event)

    def stop(self) -> bool:
        if not self._started or self._backend is None:
            return False
        self._backend.stop()
        self._started = False
        return True


def create_recording_controller(
    config: RecordingConfig,
    *,
    backend: RecordingBackend | None = None,
) -> RecordingController:
    """Create the recording gate from the application configuration."""

    return RecordingController(enabled=config.enabled, backend=backend)

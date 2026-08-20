from __future__ import annotations

from datetime import datetime, timezone

from app.config import RecordingConfig
from app.recording import RecordingController, create_recording_controller


class SpyBackend:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.frame_calls = 0
        self.event_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def append_frame(self, frame: object, timestamp: datetime) -> None:
        del frame, timestamp
        self.frame_calls += 1

    def publish_event(self, event: object) -> None:
        del event
        self.event_calls += 1


def test_recording_disabled_does_not_start_backend() -> None:
    backend = SpyBackend()
    controller = RecordingController(enabled=False, backend=backend)

    assert controller.start() is False
    controller.append_frame(object(), datetime.now(timezone.utc))
    controller.publish_event(object())
    assert controller.stop() is False
    assert controller.started is False
    assert backend.start_calls == 0
    assert backend.stop_calls == 0
    assert backend.frame_calls == 0
    assert backend.event_calls == 0


def test_recording_config_creates_disabled_gate() -> None:
    backend = SpyBackend()
    controller = create_recording_controller(
        RecordingConfig(enabled=False),
        backend=backend,
    )

    assert controller.start() is False
    assert backend.start_calls == 0

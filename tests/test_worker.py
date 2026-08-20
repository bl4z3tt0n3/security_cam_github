from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pytest

from app.video.base import (
    FramePacket,
    ReadResult,
    ReadStatus,
    StreamInfo,
    VideoSource,
    VideoSourceError,
    utc_now,
)
from app.video.fake_source import FakeVideoSource
from app.video.opencv_source import OpenCVVideoSource, cv2
from app.video.worker import CameraWorker, WorkerState
from app.video.worker import reconnect_delay_for_attempt


def wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class AlwaysFailSource(VideoSource):
    def __init__(self) -> None:
        self.reconnect_calls = 0
        self.closed = False

    def open(self) -> StreamInfo:
        raise VideoSourceError("source is offline", code="offline")

    def read(self, timeout_s: float) -> ReadResult:
        return ReadResult.status_result(ReadStatus.DISCONNECTED, "source is offline")

    def reconnect(self) -> StreamInfo:
        self.reconnect_calls += 1
        raise VideoSourceError("source is still offline", code="offline")

    def close(self) -> None:
        self.closed = True


class OfflineThenLiveSource(VideoSource):
    """Fail opening repeatedly, then expose a live session."""

    def __init__(self, *, failures_before_recovery: int) -> None:
        self.failures_before_recovery = failures_before_recovery
        self.open_calls = 0
        self.reconnect_calls = 0
        self.closed_sessions = 0
        self._active = False
        self._sequence = 0

    def open(self) -> StreamInfo:
        self.open_calls += 1
        if self.open_calls <= self.failures_before_recovery:
            raise VideoSourceError("source is offline", code="offline")
        self._active = True
        self._sequence = 0
        return StreamInfo(
            url=f"fake://offline-then-live/session-{self.open_calls}",
            backend="fake",
            width=2,
            height=2,
            declared_fps=10.0,
            codec="FAKE",
            opened_at_utc=utc_now(),
        )

    def read(self, timeout_s: float) -> ReadResult:
        del timeout_s
        if not self._active:
            return ReadResult.status_result(ReadStatus.DISCONNECTED, "source is offline")
        self._sequence += 1
        return ReadResult.frame_result(
            FramePacket(
                frame=np.full((2, 2, 3), 7, dtype=np.uint8),
                sequence=self._sequence,
                received_at_utc=utc_now(),
                received_monotonic=time.monotonic(),
                read_duration_ms=0.0,
            )
        )

    def reconnect(self) -> StreamInfo:
        self.reconnect_calls += 1
        self.close()
        return self.open()

    def close(self) -> None:
        if self._active:
            self.closed_sessions += 1
        self._active = False


class RotatingSessionSource(VideoSource):
    """Source fake that exposes a distinct session after recovery."""

    def __init__(self, *, failures_before_recovery: int = 0) -> None:
        self.failures_before_recovery = failures_before_recovery
        self.sessions: list[dict[str, object]] = []
        self.current_session: dict[str, object] | None = None
        self.open_calls = 0
        self.reconnect_calls = 0
        self.close_calls = 0
        self._frames_in_session = 0

    def open(self) -> StreamInfo:
        self.open_calls += 1
        session = {"id": self.open_calls, "closed": False}
        self.sessions.append(session)
        self.current_session = session
        self._frames_in_session = 0
        return StreamInfo(
            url=f"fake://camera/session-{self.open_calls}",
            backend="fake",
            width=2,
            height=2,
            declared_fps=10.0,
            codec="FAKE",
            opened_at_utc=utc_now(),
        )

    def read(self, timeout_s: float) -> ReadResult:
        del timeout_s
        if self.current_session is None:
            return ReadResult.status_result(ReadStatus.DISCONNECTED, "session is closed")
        if self.reconnect_calls == 0 and self._frames_in_session >= 1:
            return ReadResult.status_result(ReadStatus.DISCONNECTED, "session failed")

        self._frames_in_session += 1
        packet = FramePacket(
            frame=np.zeros((2, 2, 3), dtype=np.uint8),
            sequence=self._frames_in_session,
            received_at_utc=utc_now(),
            received_monotonic=time.monotonic(),
            read_duration_ms=0.0,
        )
        return ReadResult.frame_result(packet)

    def reconnect(self) -> StreamInfo:
        self.reconnect_calls += 1
        self.close()
        if self.reconnect_calls <= self.failures_before_recovery:
            raise VideoSourceError("session is still offline", code="offline")
        return self.open()

    def close(self) -> None:
        self.close_calls += 1
        if self.current_session is not None:
            self.current_session["closed"] = True
            self.current_session = None


class SecretFailSource(AlwaysFailSource):
    def open(self) -> StreamInfo:
        raise VideoSourceError(
            "cannot open rtsp://user:secret@camera.local:8554/live",
            code="offline",
        )

    def reconnect(self) -> StreamInfo:
        self.reconnect_calls += 1
        raise VideoSourceError(
            "still unavailable rtsp://user:secret@camera.local:8554/live",
            code="offline",
        )


class StartupDelaySource(VideoSource):
    def __init__(self, *, temporary_misses: int = 2) -> None:
        self.temporary_misses = temporary_misses
        self.open_calls = 0
        self.reconnect_calls = 0
        self.close_calls = 0
        self.read_calls = 0

    def open(self) -> StreamInfo:
        self.open_calls += 1
        self.read_calls = 0
        return StreamInfo(
            url="fake://startup-delay",
            backend="fake",
            width=2,
            height=2,
            declared_fps=10.0,
            codec="FAKE",
            opened_at_utc=utc_now(),
        )

    def read(self, timeout_s: float) -> ReadResult:
        del timeout_s
        self.read_calls += 1
        if self.read_calls <= self.temporary_misses:
            return ReadResult.status_result(ReadStatus.TIMEOUT, "temporary startup no frame")
        return ReadResult.frame_result(
            FramePacket(
                frame=np.zeros((2, 2, 3), dtype=np.uint8),
                sequence=self.read_calls,
                received_at_utc=utc_now(),
                received_monotonic=time.monotonic(),
                read_duration_ms=0.0,
            )
        )

    def reconnect(self) -> StreamInfo:
        self.reconnect_calls += 1
        return self.open()

    def close(self) -> None:
        self.close_calls += 1


class EndToEndSessionSource(VideoSource):
    def __init__(self) -> None:
        self.sessions: list[dict[str, object]] = []
        self.current_session: dict[str, object] | None = None
        self.open_calls = 0
        self.reconnect_calls = 0
        self.close_calls = 0
        self._read_index = 0

    def open(self) -> StreamInfo:
        self.open_calls += 1
        session = {"id": self.open_calls, "closed": False}
        self.sessions.append(session)
        self.current_session = session
        self._read_index = 0
        return StreamInfo(
            url=f"fake://session/{self.open_calls}",
            backend="fake",
            width=2,
            height=2,
            declared_fps=10.0,
            codec="FAKE",
            opened_at_utc=utc_now(),
        )

    def read(self, timeout_s: float) -> ReadResult:
        del timeout_s
        if self.current_session is None:
            return ReadResult.status_result(ReadStatus.DISCONNECTED, "session is closed")
        self._read_index += 1
        session_id = int(self.current_session["id"])
        if self._read_index <= 2:
            return ReadResult.status_result(ReadStatus.TIMEOUT, "waiting for startup keyframe")
        if self._read_index <= 4 or session_id > 1:
            return ReadResult.frame_result(
                FramePacket(
                    frame=np.full((2, 2, 3), session_id, dtype=np.uint8),
                    sequence=self._read_index,
                    received_at_utc=utc_now(),
                    received_monotonic=time.monotonic(),
                    read_duration_ms=0.0,
                )
            )
        return ReadResult.status_result(ReadStatus.DISCONNECTED, "simulated stream loss")

    def reconnect(self) -> StreamInfo:
        self.reconnect_calls += 1
        self.close()
        if self.reconnect_calls <= 2:
            raise VideoSourceError("simulated reconnect failure", code="offline")
        return self.open()

    def close(self) -> None:
        self.close_calls += 1
        if self.current_session is not None:
            self.current_session["closed"] = True
            self.current_session = None


class DelayedOpenCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.released:
            return False, None
        return True, self.frame

    def get(self, property_id: int) -> float:
        if cv2 is None:
            return 0.0
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: float(self.frame.shape[1]),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self.frame.shape[0]),
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FOURCC: float(
                ord("F")
                | (ord("A") << 8)
                | (ord("K") << 16)
                | (ord("E") << 24)
            ),
        }
        return values.get(property_id, 0.0)

    def set(self, property_id: int, value: float) -> bool:
        del property_id, value
        return True

    def release(self) -> None:
        self.released = True


def test_worker_starts_stops_and_collects_frame_metrics() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    source = FakeVideoSource([frame], read_delay_s=0.001)
    worker = CameraWorker(
        "camera-a",
        source,
        read_timeout_s=0.05,
        reconnect_delay_s=0,
        max_buffer_frames=1,
    )

    worker.start()
    try:
        assert worker.wait_for_state(WorkerState.RUNNING, 1.0) is WorkerState.RUNNING
        assert wait_until(lambda: worker.snapshot().frames_received >= 3)
        packet = worker.get_latest(timeout_s=0.1)
        snapshot = worker.snapshot()
        assert packet is not None
        assert snapshot.last_received_at_utc is not None
        assert snapshot.actual_fps > 0
        assert snapshot.decoded_fps == snapshot.actual_fps
        assert snapshot.stream_fps == 10.0
        assert snapshot.queue_size == 0
    finally:
        worker.stop()

    assert worker.state is WorkerState.STOPPED
    assert not worker.is_alive
    assert source.read(0.01).status is ReadStatus.DISCONNECTED


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV worker test")
def test_worker_reuses_late_opencv_capture_and_returns_to_running() -> None:
    frame = np.full((2, 2, 3), 6, dtype=np.uint8)
    open_started = threading.Event()
    allow_open = threading.Event()
    captures: list[DelayedOpenCapture] = []
    factory_calls = 0

    def delayed_factory(_url: str, _backend: str) -> DelayedOpenCapture:
        nonlocal factory_calls
        factory_calls += 1
        open_started.set()
        if not allow_open.wait(timeout=2.0):
            raise RuntimeError("test open gate was not released")
        capture = DelayedOpenCapture(frame)
        captures.append(capture)
        return capture

    source = OpenCVVideoSource(
        "rtsp://camera.local:8554/live",
        open_timeout_s=0.05,
        read_timeout_s=0.02,
        capture_factory=delayed_factory,
    )
    worker = CameraWorker(
        "late-open-camera",
        source,
        read_timeout_s=0.02,
        reconnect_delay_s=0.01,
        max_reconnect_attempts=0,
        stop_timeout_s=1.0,
    )

    worker.start()
    try:
        assert open_started.wait(timeout=1.0)
        assert (
            worker.wait_for_state(WorkerState.RECONNECTING, timeout_s=1.0)
            is WorkerState.RECONNECTING
        )
        time.sleep(0.02)
        assert factory_calls == 1

        allow_open.set()
        assert wait_until(lambda: worker.snapshot().frames_received >= 1, timeout_s=2.0)

        snapshot = worker.snapshot()
        assert snapshot.state is WorkerState.RUNNING
        assert snapshot.successful_reconnects >= 1
        assert snapshot.reconnect_count >= 1
        assert factory_calls == 1
        packet = worker.get_latest(timeout_s=0.2)
        assert packet is not None
        assert int(packet.frame[0, 0, 0]) == 6
    finally:
        allow_open.set()
        worker.stop(timeout_s=1.0)

    assert len(captures) == 1
    assert captures[0].released is True


def test_worker_waits_for_first_frame_without_reconnecting() -> None:
    source = StartupDelaySource(temporary_misses=2)
    worker = CameraWorker(
        "startup-camera",
        source,
        read_timeout_s=0.01,
        reconnect_delay_s=0.01,
        max_reconnect_attempts=0,
    )

    worker.start()
    try:
        assert wait_until(lambda: worker.snapshot().frames_received >= 2, timeout_s=1.0)
        snapshot = worker.snapshot()
        assert snapshot.state is WorkerState.RUNNING
        assert snapshot.reconnect_count == 0
        assert source.open_calls == 1
        assert source.reconnect_calls == 0
        assert snapshot.last_error is None
    finally:
        worker.stop(timeout_s=1.0)


def test_worker_stop_interrupts_startup_frame_wait() -> None:
    source = StartupDelaySource(temporary_misses=10_000)
    worker = CameraWorker(
        "startup-stop-camera",
        source,
        read_timeout_s=0.01,
        reconnect_delay_s=1.0,
        max_reconnect_attempts=0,
    )

    worker.start()
    assert wait_until(lambda: source.read_calls > 0)
    started = time.perf_counter()
    worker.stop(timeout_s=0.5)

    assert time.perf_counter() - started < 1.0
    assert worker.state is WorkerState.STOPPED
    assert source.reconnect_calls == 0


def test_worker_end_to_end_recovers_after_startup_and_failed_reconnects() -> None:
    source = EndToEndSessionSource()
    worker = CameraWorker(
        "end-to-end-camera",
        source,
        read_timeout_s=0.01,
        reconnect_delay_s=0.01,
        max_reconnect_attempts=0,
    )

    worker.start()
    try:
        assert wait_until(
            lambda: worker.snapshot().successful_reconnects >= 1,
            timeout_s=3.0,
        )
        snapshot = worker.snapshot()
        packet = worker.get_latest(timeout_s=0.2)
        assert snapshot.state is WorkerState.RUNNING
        assert snapshot.reconnect_count >= 3
        assert snapshot.failed_reconnects >= 2
        assert len(source.sessions) == 2
        assert source.sessions[0]["closed"] is True
        assert packet is not None
        assert int(packet.frame[0, 0, 0]) == 2
    finally:
        worker.stop(timeout_s=1.0)

    assert all(session["closed"] is True for session in source.sessions)


def test_worker_reconnects_after_fake_source_disconnects() -> None:
    frame = np.ones((3, 3, 3), dtype=np.uint8)
    source = FakeVideoSource([frame], read_delay_s=0.001, fail_after_frames=1)
    worker = CameraWorker(
        "camera-reconnect",
        source,
        read_timeout_s=0.05,
        reconnect_delay_s=0,
        max_reconnect_attempts=3,
    )

    worker.start()
    try:
        assert wait_until(lambda: worker.snapshot().successful_reconnects >= 1)
        snapshot = worker.snapshot()
        assert snapshot.reconnect_count >= 1
        assert snapshot.successful_reconnects >= 1
        assert snapshot.frames_received >= 1
    finally:
        worker.stop()


def test_worker_closes_dead_session_and_recovers_with_a_new_session() -> None:
    source = RotatingSessionSource(failures_before_recovery=3)
    worker = CameraWorker(
        "rotating-camera",
        source,
        read_timeout_s=0.05,
        reconnect_delay_s=0.01,
        max_reconnect_attempts=0,
    )

    worker.start()
    try:
        assert wait_until(
            lambda: worker.snapshot().successful_reconnects >= 1,
            timeout_s=2.0,
        )
        snapshot = worker.snapshot()
        assert worker.state is WorkerState.RUNNING
        assert source.reconnect_calls >= 4
        assert snapshot.failed_reconnects >= 3
        assert len(source.sessions) == 2
        assert source.sessions[0] is not source.sessions[1]
        assert source.sessions[0]["closed"] is True
        assert source.close_calls >= 1
    finally:
        worker.stop(timeout_s=1.0)

    assert all(session["closed"] is True for session in source.sessions)


def test_worker_retries_indefinitely_without_busy_loop_when_limit_is_zero() -> None:
    source = AlwaysFailSource()
    worker = CameraWorker(
        "persistent-offline-camera",
        source,
        read_timeout_s=0.05,
        reconnect_delay_s=0,
        max_reconnect_attempts=0,
    )

    worker.start()
    try:
        assert wait_until(lambda: source.reconnect_calls >= 10, timeout_s=2.0)
        time.sleep(0.12)
        snapshot = worker.snapshot()
        assert snapshot.state is WorkerState.RECONNECTING
        assert snapshot.thread_alive is True
        assert source.reconnect_calls < 20
    finally:
        worker.stop(timeout_s=1.0)


def test_reconnect_delay_uses_bounded_progressive_backoff() -> None:
    assert [reconnect_delay_for_attempt(1.0, attempt) for attempt in range(1, 7)] == [
        1.0,
        2.0,
        3.0,
        5.0,
        5.0,
        5.0,
    ]
    assert reconnect_delay_for_attempt(2.0, 3) == 5.0
    assert reconnect_delay_for_attempt(0.0, 1) == pytest.approx(0.05)


def test_worker_recovers_after_ten_failed_initial_opens_with_unlimited_retries() -> None:
    source = OfflineThenLiveSource(failures_before_recovery=10)
    worker = CameraWorker(
        "offline-at-start-camera",
        source,
        read_timeout_s=0.01,
        reconnect_delay_s=0,
        max_reconnect_attempts=0,
    )

    worker.start()
    try:
        assert wait_until(
            lambda: worker.snapshot().frames_received >= 1,
            timeout_s=3.0,
        )
        snapshot = worker.snapshot()
        assert snapshot.state is WorkerState.RUNNING
        assert snapshot.thread_alive is True
        assert snapshot.successful_reconnects == 1
        assert snapshot.reconnect_count >= 10
        assert source.open_calls >= 11
        assert source.reconnect_calls >= 10
        packet = worker.get_latest(timeout_s=0.2)
        assert packet is not None
        assert int(packet.frame[0, 0, 0]) == 7
    finally:
        worker.stop(timeout_s=1.0)


def test_worker_logs_reconnect_configuration_and_survival(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    source = AlwaysFailSource()
    worker = CameraWorker(
        "diagnostic-camera",
        source,
        read_timeout_s=0.05,
        reconnect_delay_s=0,
        max_reconnect_attempts=0,
    )

    worker.start()
    try:
        assert wait_until(lambda: source.reconnect_calls >= 2, timeout_s=1.0)
    finally:
        worker.stop(timeout_s=1.0)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=camera_worker_created camera=diagnostic-camera" in messages
    assert "max_reconnect_attempts=0" in messages
    assert "event=camera_worker_started camera=diagnostic-camera" in messages
    assert "event=reconnect_callback_entered camera=diagnostic-camera" in messages
    assert "event=worker_still_alive_after_disconnect camera=diagnostic-camera" in messages
    assert "thread_alive=True" in messages


def test_worker_logging_redacts_reconnect_credentials(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    worker = CameraWorker(
        "redacted-camera",
        SecretFailSource(),
        read_timeout_s=0.05,
        reconnect_delay_s=0.01,
        max_reconnect_attempts=1,
    )

    worker.start()
    assert worker.wait_for_state(WorkerState.FAILED, 1.0) is WorkerState.FAILED
    worker.stop(timeout_s=1.0)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "secret" not in messages
    assert "user:***@camera.local:8554/live" in messages


def test_worker_enters_failed_after_reconnect_attempts_are_exhausted() -> None:
    source = AlwaysFailSource()
    worker = CameraWorker(
        "offline-camera",
        source,
        read_timeout_s=0.05,
        reconnect_delay_s=0,
        max_reconnect_attempts=3,
    )

    worker.start()
    assert worker.wait_for_state(WorkerState.FAILED, 1.0) is WorkerState.FAILED
    snapshot = worker.snapshot()
    worker.stop()

    assert snapshot.reconnect_count == 3
    assert snapshot.failed_reconnects == 3
    assert snapshot.last_error == "source is still offline"
    assert source.reconnect_calls == 3
    assert source.closed is True


def test_one_failed_camera_does_not_stop_another_worker() -> None:
    good_source = FakeVideoSource(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        read_delay_s=0.001,
    )
    bad_source = AlwaysFailSource()
    good_worker = CameraWorker(
        "good-camera",
        good_source,
        read_timeout_s=0.05,
        reconnect_delay_s=0,
        max_reconnect_attempts=1,
    )
    bad_worker = CameraWorker(
        "bad-camera",
        bad_source,
        read_timeout_s=0.05,
        reconnect_delay_s=0,
        max_reconnect_attempts=1,
    )

    good_worker.start()
    bad_worker.start()
    try:
        assert bad_worker.wait_for_state(WorkerState.FAILED, 1.0) is WorkerState.FAILED
        assert wait_until(lambda: good_worker.snapshot().frames_received > 0)
        assert good_worker.state is WorkerState.RUNNING
        assert bad_worker.state is WorkerState.FAILED
    finally:
        good_worker.stop()
        bad_worker.stop()


def test_worker_shutdown_interrupts_reconnect_delay() -> None:
    source = FakeVideoSource(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        read_delay_s=0.001,
        fail_after_frames=1,
    )
    worker = CameraWorker(
        "slow-reconnect",
        source,
        read_timeout_s=0.05,
        reconnect_delay_s=2.0,
        max_reconnect_attempts=3,
    )

    worker.start()
    assert wait_until(lambda: worker.state is WorkerState.RECONNECTING)
    started = time.perf_counter()
    worker.stop(timeout_s=0.5)

    assert time.perf_counter() - started < 1.0
    assert worker.state is WorkerState.STOPPED

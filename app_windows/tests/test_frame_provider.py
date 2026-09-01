from __future__ import annotations

import time

import numpy as np

from app.video.base import FramePacket, ReadResult, ReadStatus, StreamInfo, VideoSource, utc_now
from app.video.fake_source import FakeVideoSource
from app.video.worker import WorkerState

from app_windows.models.camera_view_state import CameraSlot
from app_windows.video.fake_provider import FakeFrameProvider
from app_windows.video.frame_provider import BackendFrameProvider


def wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def make_slot(camera_id: str = "cam_1") -> CameraSlot:
    return CameraSlot(
        slot_index=1,
        camera_id=camera_id,
        name="Test camera",
        enabled=True,
        configured=True,
        stream_url=f"fake://{camera_id}/live",
    )


class StartupFrameSource(VideoSource):
    def __init__(self) -> None:
        self.open_calls = 0
        self.reconnect_calls = 0
        self.read_calls = 0

    def open(self) -> StreamInfo:
        self.open_calls += 1
        self.read_calls = 0
        return StreamInfo(
            url="fake://startup-frame",
            backend="fake",
            width=4,
            height=4,
            declared_fps=10.0,
            codec="FAKE",
            opened_at_utc=utc_now(),
        )

    def read(self, timeout_s: float) -> ReadResult:
        del timeout_s
        self.read_calls += 1
        if self.read_calls <= 2:
            return ReadResult.status_result(ReadStatus.TIMEOUT, "waiting for first frame")
        return ReadResult.frame_result(
            FramePacket(
                frame=np.full((4, 4, 3), 17, dtype=np.uint8),
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
        return None


def test_fake_provider_delivers_recent_frames_and_stops_cleanly() -> None:
    provider = FakeFrameProvider(make_slot(), camera_index=0, fps=40.0)
    provider.start()
    try:
        assert wait_until(lambda: provider.snapshot().worker.frames_received > 2)
        packet = provider.latest_frame()
        snapshot = provider.snapshot()
        assert packet is not None
        assert snapshot.worker is not None
        assert snapshot.worker.state is WorkerState.RUNNING
        assert snapshot.worker.max_buffer_frames == 1
        status = provider.status()
        assert status.worker is not None
        assert status.worker.camera_id == "cam_1"
        assert status.worker.state is WorkerState.RUNNING
    finally:
        provider.stop(timeout_s=1.0)

    assert provider.snapshot().worker is not None
    assert provider.snapshot().worker.thread_alive is False


def test_two_backend_providers_keep_failure_and_live_state_independent() -> None:
    healthy = FakeFrameProvider(make_slot("healthy"), camera_index=0, fps=25.0)
    offline = FakeFrameProvider(
        make_slot("offline"),
        camera_index=1,
        fps=25.0,
        offline=True,
    )
    healthy.start()
    offline.start()
    try:
        assert wait_until(lambda: healthy.snapshot().worker.frames_received > 0)
        assert wait_until(
            lambda: offline.snapshot().worker.state is WorkerState.FAILED,
            timeout_s=3.0,
        )
        assert healthy.snapshot().worker.state is WorkerState.RUNNING
        assert healthy.latest_frame() is not None
    finally:
        healthy.stop(timeout_s=1.0)
        offline.stop(timeout_s=1.0)


def test_backend_provider_can_use_the_existing_worker_contract() -> None:
    frame = np.full((12, 16, 3), 7, dtype=np.uint8)
    source = FakeVideoSource([frame], fps=30.0, read_delay_s=0.001)
    provider = BackendFrameProvider(
        "worker-contract",
        source,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
        max_reconnect_attempts=1,
        max_buffer_frames=1,
        stop_timeout_s=1.0,
    )
    provider.start()
    try:
        assert wait_until(lambda: provider.latest_frame() is not None)
    finally:
        provider.stop(timeout_s=1.0)

    assert provider.start_calls == 1


def test_backend_provider_delivers_frame_after_temporary_startup_misses() -> None:
    source = StartupFrameSource()
    provider = BackendFrameProvider(
        "startup-provider",
        source,
        read_timeout_s=0.01,
        reconnect_delay_s=0.01,
        max_reconnect_attempts=0,
        max_buffer_frames=1,
        stop_timeout_s=1.0,
    )

    provider.start()
    try:
        assert wait_until(lambda: provider.latest_frame() is not None)
        snapshot = provider.snapshot()
        assert snapshot.worker is not None
        assert snapshot.worker.state is WorkerState.RUNNING
        assert source.open_calls == 1
        assert source.reconnect_calls == 0
    finally:
        provider.stop(timeout_s=1.0)


def test_fake_provider_can_simulate_reconnect_without_losing_worker_contract() -> None:
    provider = FakeFrameProvider(
        make_slot("reconnecting"),
        camera_index=0,
        fps=40.0,
        fail_after_frames=1,
    )
    provider.start()
    try:
        assert wait_until(lambda: provider.snapshot().worker.reconnect_count > 0, timeout_s=3.0)
        assert provider.snapshot().worker.thread_alive is True
    finally:
        provider.stop(timeout_s=1.0)

def test_backend_provider_shares_cached_latest_frame_between_consumers(monkeypatch) -> None:
    source = FakeVideoSource([], fps=30.0)
    provider = BackendFrameProvider(
        "shared-latest",
        source,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
        max_reconnect_attempts=1,
        max_buffer_frames=1,
        stop_timeout_s=1.0,
    )
    packet = FramePacket(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        sequence=7,
        received_at_utc=utc_now(),
        received_monotonic=time.monotonic(),
        read_duration_ms=0.1,
    )
    values = iter((packet, None))
    monkeypatch.setattr(
        provider.worker,
        "get_latest",
        lambda timeout_s=0.0: next(values),
    )

    first_consumer = provider.latest_frame()
    second_consumer = provider.latest_frame()

    assert first_consumer is packet
    assert second_consumer is packet


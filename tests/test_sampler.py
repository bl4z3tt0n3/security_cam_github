from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from app.video.base import FramePacket
from app.video.buffer import LatestFrameBuffer
from app.video.fake_source import FakeVideoSource
from app.video.sampler import FrameSampler
from app.video.worker import CameraWorker, WorkerState


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def make_packet(
    sequence: int,
    *,
    received_monotonic: float,
    source_timestamp_s: float | None = None,
) -> FramePacket:
    return FramePacket(
        frame=np.full((2, 2, 3), sequence % 255, dtype=np.uint8),
        sequence=sequence,
        received_at_utc=datetime.now(timezone.utc),
        received_monotonic=received_monotonic,
        read_duration_ms=1.0,
        source_timestamp_s=source_timestamp_s,
    )


@pytest.mark.parametrize("source_fps", [20.0, 25.0, 30.0])
def test_sampler_targets_about_two_fps_for_common_source_rates(source_fps: float) -> None:
    clock = FakeClock()
    sampler = FrameSampler(LatestFrameBuffer(), target_fps=2.0, clock=clock)

    for sequence in range(int(source_fps * 5.0)):
        consumed_at = sequence / source_fps
        clock.value = consumed_at
        sampler.accept(
            make_packet(
                sequence,
                received_monotonic=consumed_at - 0.01,
                source_timestamp_s=consumed_at,
            ),
            consumed_monotonic=consumed_at,
        )

    clock.value = 5.0
    snapshot = sampler.snapshot()

    assert 9 <= snapshot.frames_sampled <= 11
    assert snapshot.sampled_fps == pytest.approx(2.0, abs=0.2)
    assert snapshot.frames_sampled < snapshot.frames_received
    assert snapshot.mean_latency_ms == pytest.approx(10.0)


def test_sampler_forwards_all_frames_when_target_exceeds_source_rate() -> None:
    clock = FakeClock()
    sampler = FrameSampler(LatestFrameBuffer(), target_fps=2.0, clock=clock)

    for sequence in range(5):
        clock.value = float(sequence)
        assert sampler.accept(
            make_packet(sequence, received_monotonic=float(sequence)),
            consumed_monotonic=float(sequence),
        )

    snapshot = sampler.snapshot()
    assert snapshot.frames_received == 5
    assert snapshot.frames_sampled == 5
    assert snapshot.skipped_frames == 0


def test_sampler_reconfiguration_takes_effect_without_restart() -> None:
    clock = FakeClock()
    sampler = FrameSampler(LatestFrameBuffer(), target_fps=2.0, clock=clock)

    for sequence in range(4):
        consumed_at = sequence * 0.5
        clock.value = consumed_at
        sampler.accept(
            make_packet(sequence, received_monotonic=consumed_at),
            consumed_monotonic=consumed_at,
        )
    assert sampler.snapshot().frames_sampled == 4

    sampler.set_target_fps(5.0)
    for sequence in range(4, 9):
        consumed_at = 2.0 + (sequence - 4) * 0.2
        clock.value = consumed_at
        sampler.accept(
            make_packet(sequence, received_monotonic=consumed_at),
            consumed_monotonic=consumed_at,
        )

    assert sampler.snapshot().frames_sampled == 9


def test_source_timestamp_jitter_does_not_change_sampling_cadence() -> None:
    clock = FakeClock()
    sampler = FrameSampler(LatestFrameBuffer(), target_fps=2.0, clock=clock)
    jittered_timestamps = [0.0, 0.08, 0.01, 0.23, 0.16, 0.41, 0.29, 0.7, 0.51, 0.95]

    for sequence in range(100):
        consumed_at = sequence / 20.0
        clock.value = consumed_at
        sampler.accept(
            make_packet(
                sequence,
                received_monotonic=consumed_at,
                source_timestamp_s=jittered_timestamps[sequence % len(jittered_timestamps)],
            ),
            consumed_monotonic=consumed_at,
        )

    clock.value = 5.0
    snapshot = sampler.snapshot()
    assert snapshot.frames_sampled == 10
    assert snapshot.sampled_fps == pytest.approx(2.0, abs=0.2)


def test_sampler_output_does_not_accumulate_when_consumer_is_slow() -> None:
    clock = FakeClock()
    sampler = FrameSampler(LatestFrameBuffer(), target_fps=2.0, clock=clock)

    for sequence in range(5):
        consumed_at = sequence * 0.5
        clock.value = consumed_at
        sampler.accept(
            make_packet(sequence, received_monotonic=consumed_at),
            consumed_monotonic=consumed_at,
        )

    snapshot = sampler.snapshot()
    assert snapshot.queue_size == 1
    assert snapshot.max_buffer_frames == 1
    assert snapshot.dropped_frames == 4
    assert sampler.get_latest() is not None
    assert sampler.get_latest() is None


def test_disabled_sampler_is_transparent() -> None:
    clock = FakeClock()
    sampler = FrameSampler(LatestFrameBuffer(), target_fps=None, enabled=False, clock=clock)

    for sequence in range(5):
        clock.value = sequence * 0.05
        assert sampler.accept(
            make_packet(sequence, received_monotonic=clock.value),
            consumed_monotonic=clock.value,
        )

    snapshot = sampler.snapshot()
    assert snapshot.frames_received == 5
    assert snapshot.frames_sampled == 5
    assert snapshot.skipped_frames == 0


@pytest.mark.parametrize("target_fps", [None, 0, -1, float("nan"), float("inf"), "invalid"])
def test_invalid_sampler_frequency_is_rejected(target_fps: object) -> None:
    with pytest.raises(ValueError):
        FrameSampler(LatestFrameBuffer(), target_fps=target_fps)  # type: ignore[arg-type]


def wait_until(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_fake_source_worker_and_sampler_integrate_without_a_camera() -> None:
    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    source = FakeVideoSource([frame], fps=20.0, read_delay_s=0.05)
    worker = CameraWorker(
        "sampled-camera",
        source,
        read_timeout_s=0.1,
        reconnect_delay_s=0,
        max_buffer_frames=1,
    )
    sampler = FrameSampler(worker, target_fps=2.0, input_wait_timeout_s=0.02)

    worker.start()
    assert worker.wait_for_state(WorkerState.RUNNING, 1.0) is WorkerState.RUNNING
    sampler.start()
    try:
        assert wait_until(lambda: sampler.snapshot().frames_sampled >= 3)
        worker_snapshot = worker.snapshot()
        sampler_snapshot = sampler.snapshot()
        assert worker_snapshot.stream_fps == 20.0
        assert sampler_snapshot.frames_received > sampler_snapshot.frames_sampled
        assert worker_snapshot.frames_received > sampler_snapshot.frames_sampled
        assert sampler_snapshot.thread_alive is True
    finally:
        sampler.stop(timeout_s=0.5)
        worker.stop(timeout_s=0.5)

    assert sampler.snapshot().thread_alive is False
    assert worker.state is WorkerState.STOPPED


def test_two_samplers_keep_independent_metrics() -> None:
    first_clock = FakeClock()
    second_clock = FakeClock()
    first = FrameSampler(LatestFrameBuffer(), target_fps=2.0, clock=first_clock)
    second = FrameSampler(LatestFrameBuffer(), target_fps=5.0, clock=second_clock)

    for sequence in range(10):
        first_clock.value = sequence * 0.1
        first.accept(
            make_packet(sequence, received_monotonic=first_clock.value),
            consumed_monotonic=first_clock.value,
        )
    for sequence in range(10):
        second_clock.value = sequence * 0.1
        second.accept(
            make_packet(sequence, received_monotonic=second_clock.value),
            consumed_monotonic=second_clock.value,
        )

    assert first.snapshot().frames_sampled == 2
    assert second.snapshot().frames_sampled == 5
    assert first.snapshot().frames_sampled != second.snapshot().frames_sampled

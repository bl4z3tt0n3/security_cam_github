from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.video.base import FramePacket
from app.video.buffer import LatestFrameBuffer


def make_packet(sequence: int) -> FramePacket:
    now = datetime.now(timezone.utc)
    return FramePacket(
        frame=np.full((2, 2, 3), sequence, dtype=np.uint8),
        sequence=sequence,
        received_at_utc=now,
        received_monotonic=float(sequence),
        read_duration_ms=1.0,
    )


def test_latest_frame_buffer_discards_oldest_when_full() -> None:
    buffer = LatestFrameBuffer(max_frames=2)
    buffer.put(make_packet(1))
    buffer.put(make_packet(2))
    buffer.put(make_packet(3))

    assert buffer.size == 2
    assert buffer.dropped_frames == 1
    packet = buffer.get_latest()

    assert packet is not None
    assert packet.sequence == 3
    assert buffer.size == 0
    assert buffer.dropped_frames == 2


def test_latest_frame_buffer_waits_and_wakes_for_a_frame() -> None:
    buffer = LatestFrameBuffer()
    buffer.put(make_packet(7))

    packet = buffer.get_latest(timeout_s=0.05)

    assert packet is not None
    assert packet.sequence == 7


def test_closed_latest_frame_buffer_rejects_new_packets() -> None:
    buffer = LatestFrameBuffer()
    buffer.close()

    assert buffer.put(make_packet(1)) == 0
    assert buffer.get_latest(timeout_s=0.01) is None

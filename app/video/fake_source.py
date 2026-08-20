"""Deterministic source used by tests and future pipeline simulations."""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import datetime

import numpy as np

from .base import (
    FramePacket,
    ReadResult,
    ReadStatus,
    StreamInfo,
    VideoSource,
    VideoSourceError,
    utc_now,
)


class FakeVideoSource(VideoSource):
    """A finite or repeating in-memory source with controllable failures."""

    def __init__(
        self,
        frames: Iterable[np.ndarray] | None = None,
        *,
        url: str = "fake://camera",
        width: int = 64,
        height: int = 48,
        fps: float = 10.0,
        codec: str = "FAKE",
        read_delay_s: float = 0.0,
        fail_after_frames: int | None = None,
        repeat: bool = True,
    ) -> None:
        self._frames = [np.asarray(frame) for frame in (frames or [])]
        self._url = url
        self._width = width
        self._height = height
        self._fps = fps
        self._codec = codec
        self._read_delay_s = max(0.0, read_delay_s)
        self._fail_after_frames = fail_after_frames
        self._repeat = repeat
        self._opened = False
        self._index = 0
        self._sequence = 0
        self._reconnect_count = 0

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    def open(self) -> StreamInfo:
        self._opened = True
        self._index = 0
        self._sequence = 0
        return StreamInfo(
            url=self._url,
            backend="fake",
            width=self._width,
            height=self._height,
            declared_fps=self._fps,
            codec=self._codec,
            opened_at_utc=utc_now(),
        )

    def read(self, timeout_s: float) -> ReadResult:
        if not self._opened:
            return ReadResult.status_result(ReadStatus.DISCONNECTED, "fake source is closed")

        if self._read_delay_s > timeout_s:
            if timeout_s > 0:
                time.sleep(timeout_s)
            return ReadResult.status_result(ReadStatus.TIMEOUT, "fake read delay exceeded timeout")
        if self._read_delay_s:
            time.sleep(self._read_delay_s)

        if self._fail_after_frames is not None and self._sequence >= self._fail_after_frames:
            self._opened = False
            return ReadResult.status_result(ReadStatus.DISCONNECTED, "configured fake failure")

        if not self._frames:
            return ReadResult.status_result(ReadStatus.TIMEOUT, "fake source has no frames")

        if self._index >= len(self._frames):
            if not self._repeat:
                return ReadResult.status_result(ReadStatus.TIMEOUT, "fake source reached end of frames")
            self._index = 0

        frame = self._frames[self._index]
        self._index += 1
        self._sequence += 1
        packet = FramePacket(
            frame=frame.copy(),
            sequence=self._sequence,
            received_at_utc=utc_now(),
            received_monotonic=time.monotonic(),
            read_duration_ms=self._read_delay_s * 1000,
        )
        return ReadResult.frame_result(packet)

    def reconnect(self) -> StreamInfo:
        self._reconnect_count += 1
        return self.open()

    def close(self) -> None:
        self._opened = False

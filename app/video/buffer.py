"""Thread-safe bounded buffers that prefer the most recent video frame."""

from __future__ import annotations

import threading
import time
from collections import deque

from .base import FramePacket


class LatestFrameBuffer:
    """A bounded frame buffer that never lets consumer lag grow without a limit."""

    def __init__(self, max_frames: int = 1) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be at least one")

        self._max_frames = max_frames
        self._items: deque[FramePacket] = deque()
        self._condition = threading.Condition()
        self._dropped_frames = 0
        self._closed = False

    @property
    def max_frames(self) -> int:
        return self._max_frames

    @property
    def size(self) -> int:
        with self._condition:
            return len(self._items)

    @property
    def dropped_frames(self) -> int:
        with self._condition:
            return self._dropped_frames

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def put(self, packet: FramePacket) -> int:
        """Store a packet and return how many old packets were discarded."""

        dropped = 0
        with self._condition:
            if self._closed:
                return 0
            while len(self._items) >= self._max_frames:
                self._items.popleft()
                dropped += 1
            self._items.append(packet)
            self._dropped_frames += dropped
            self._condition.notify()
        return dropped

    def get_latest(self, timeout_s: float = 0.0) -> FramePacket | None:
        """Return the newest packet and discard any older packets still queued."""

        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")

        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._items and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

            if not self._items:
                return None

            newest = self._items[-1]
            stale_count = len(self._items) - 1
            self._items.clear()
            self._dropped_frames += stale_count
            return newest

    def clear(self) -> int:
        """Discard queued packets and return the number discarded."""

        with self._condition:
            discarded = len(self._items)
            self._items.clear()
            self._dropped_frames += discarded
            return discarded

    def close(self) -> None:
        """Wake blocked consumers and reject packets published afterwards."""

        with self._condition:
            self._closed = True
            self._condition.notify_all()

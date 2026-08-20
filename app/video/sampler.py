"""Asynchronous, bounded frame sampling for future inference stages."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .base import FramePacket
from .buffer import LatestFrameBuffer


class FrameReader(Protocol):
    """Minimal input contract implemented by ``CameraWorker`` and buffers."""

    def get_latest(self, timeout_s: float = 0.0) -> FramePacket | None:
        ...


@dataclass(frozen=True)
class FrameSamplerSnapshot:
    """Point-in-time metrics for one sampler instance."""

    enabled: bool
    target_fps: float | None
    frames_received: int
    frames_sampled: int
    sampled_fps: float
    skipped_frames: int
    dropped_frames: int
    queue_size: int
    max_buffer_frames: int
    mean_latency_ms: float | None
    last_latency_ms: float | None
    thread_alive: bool
    last_error: str | None

    @property
    def frames_seen(self) -> int:
        """Compatibility-friendly name for packets consumed from the input."""

        return self.frames_received


class FrameSampler:
    """Consume a live frame reader and publish at most the configured rate.

    Sampling is driven by a monotonic clock at the point where the sampler
    consumes a packet. Declared stream FPS and source timestamps never control
    the rate gate. The output is a latest-frame buffer, so a slow future
    inference consumer cannot create an unbounded backlog.
    """

    MAX_TARGET_FPS = 60.0
    CLOCK_EPSILON_S = 1e-9

    def __init__(
        self,
        input_reader: FrameReader,
        target_fps: float | None,
        *,
        enabled: bool = True,
        input_wait_timeout_s: float = 0.1,
        stop_timeout_s: float | None = None,
        thread_name: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        if input_reader is None:
            raise ValueError("input_reader is required")
        if input_wait_timeout_s <= 0:
            raise ValueError("input_wait_timeout_s must be greater than zero")
        if stop_timeout_s is not None and stop_timeout_s <= 0:
            raise ValueError("stop_timeout_s must be greater than zero")
        normalized_thread_name = (thread_name or "frame-sampler").strip()
        if not normalized_thread_name:
            raise ValueError("thread_name cannot be empty")

        normalized_target = self._validate_target_fps(target_fps)
        if enabled and normalized_target is None:
            raise ValueError("target_fps is required when sampler is enabled")

        self._input_reader = input_reader
        self._enabled = enabled
        self._target_fps = normalized_target
        self._input_wait_timeout_s = input_wait_timeout_s
        self._stop_timeout_s = stop_timeout_s or max(1.0, input_wait_timeout_s + 1.0)
        self._thread_name = normalized_thread_name
        self._clock = clock
        self._logger = logger or logging.getLogger(__name__)

        self._output = LatestFrameBuffer(max_frames=1)
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_monotonic: float | None = None
        self._next_sample_monotonic: float | None = None
        self._frames_received = 0
        self._frames_sampled = 0
        self._skipped_frames = 0
        self._latency_sum_ms = 0.0
        self._last_latency_ms: float | None = None
        self._last_error: str | None = None

    @staticmethod
    def _validate_target_fps(value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("target_fps must be a finite number")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_fps must be a finite number") from exc
        if not math.isfinite(normalized):
            raise ValueError("target_fps must be finite")
        if normalized <= 0:
            raise ValueError("target_fps must be greater than zero")
        if normalized > FrameSampler.MAX_TARGET_FPS:
            raise ValueError(
                f"target_fps cannot be greater than {FrameSampler.MAX_TARGET_FPS:g}"
            )
        return normalized

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def target_fps(self) -> float | None:
        with self._condition:
            return self._target_fps

    @property
    def thread_name(self) -> str:
        return self._thread_name

    def start(self) -> None:
        """Start the sampler thread; repeated starts while active are no-ops."""

        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return

            self._output = LatestFrameBuffer(max_frames=1)
            self._stop_event.clear()
            self._started_monotonic = self._clock()
            self._next_sample_monotonic = None
            self._frames_received = 0
            self._frames_sampled = 0
            self._skipped_frames = 0
            self._latency_sum_ms = 0.0
            self._last_latency_ms = None
            self._last_error = None
            self._thread = threading.Thread(
                target=self._run,
                name=self._thread_name,
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def stop(self, timeout_s: float | None = None) -> None:
        """Request shutdown and wait a bounded time for the sampler thread."""

        with self._condition:
            thread = self._thread
            self._stop_event.set()

        self._output.close()
        if thread is not None and thread is not threading.current_thread():
            join_timeout = timeout_s if timeout_s is not None else self._stop_timeout_s
            if join_timeout <= 0:
                raise ValueError("timeout_s must be greater than zero")
            thread.join(timeout=join_timeout)

    def get_latest(self, timeout_s: float = 0.0) -> FramePacket | None:
        """Return the newest sampled packet and discard older output packets."""

        return self._output.get_latest(timeout_s)

    def set_target_fps(self, target_fps: float) -> None:
        """Change the sampling rate without restarting the sampler thread."""

        normalized = self._validate_target_fps(target_fps)
        assert normalized is not None
        with self._condition:
            self._target_fps = normalized
            self._next_sample_monotonic = None

    def accept(self, packet: FramePacket, *, consumed_monotonic: float | None = None) -> bool:
        """Process one packet and return whether it was forwarded.

        The optional timestamp makes the rate gate deterministic in unit tests;
        the background thread omits it and uses the injected monotonic clock.
        """

        now = self._clock() if consumed_monotonic is None else float(consumed_monotonic)
        if not math.isfinite(now):
            raise ValueError("consumed_monotonic must be finite")

        with self._condition:
            if self._stop_event.is_set():
                return False
            if self._started_monotonic is None:
                self._started_monotonic = now
            self._frames_received += 1

            if self._enabled:
                assert self._target_fps is not None
                if (
                    self._next_sample_monotonic is not None
                    and now + self.CLOCK_EPSILON_S < self._next_sample_monotonic
                ):
                    self._skipped_frames += 1
                    return False
                self._next_sample_monotonic = now + (1.0 / self._target_fps)
            else:
                self._next_sample_monotonic = None

            latency_ms = max(0.0, (now - packet.received_monotonic) * 1000.0)
            self._frames_sampled += 1
            self._latency_sum_ms += latency_ms
            self._last_latency_ms = latency_ms

        self._output.put(packet)
        return True

    def snapshot(self) -> FrameSamplerSnapshot:
        """Return sampler metrics without blocking the sampling loop."""

        with self._condition:
            started = self._started_monotonic
            elapsed = max(0.0, self._clock() - started) if started is not None else 0.0
            frames_received = self._frames_received
            frames_sampled = self._frames_sampled
            skipped_frames = self._skipped_frames
            latency_sum_ms = self._latency_sum_ms
            last_latency_ms = self._last_latency_ms
            thread = self._thread
            last_error = self._last_error
            enabled = self._enabled
            target_fps = self._target_fps

        return FrameSamplerSnapshot(
            enabled=enabled,
            target_fps=target_fps,
            frames_received=frames_received,
            frames_sampled=frames_sampled,
            sampled_fps=frames_sampled / elapsed if elapsed > 0 else 0.0,
            skipped_frames=skipped_frames,
            dropped_frames=self._output.dropped_frames,
            queue_size=self._output.size,
            max_buffer_frames=self._output.max_frames,
            mean_latency_ms=latency_sum_ms / frames_sampled if frames_sampled else None,
            last_latency_ms=last_latency_ms,
            thread_alive=thread is not None and thread.is_alive(),
            last_error=last_error,
        )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                packet = self._input_reader.get_latest(timeout_s=self._input_wait_timeout_s)
                if packet is not None:
                    self.accept(packet)
        except Exception as exc:  # pragma: no cover - defensive pipeline isolation
            with self._condition:
                self._last_error = str(exc) or type(exc).__name__
            self._logger.exception("frame sampler failed")
        finally:
            self._output.close()

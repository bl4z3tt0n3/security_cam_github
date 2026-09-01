"""Independent camera acquisition worker with bounded live-frame delivery."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from app.logging_setup import log_event, redact_log_text

from .base import FramePacket, ReadStatus, StreamInfo, VideoSource, utc_now
from .buffer import LatestFrameBuffer


_MIN_RECONNECT_DELAY_S = 0.05
_STARTUP_GRACE_MIN_S = 2.0
_STARTUP_RETRY_WAIT_S = 0.05
_RECONNECT_BACKOFF_FACTORS = (1.0, 2.0, 3.0, 5.0, 5.0)
_MAX_RECONNECT_DELAY_S = 5.0


def reconnect_delay_for_attempt(base_delay_s: float, attempt: int) -> float:
    """Return a bounded, deterministic delay for one outage attempt."""

    if base_delay_s < 0 or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("base_delay_s must be non-negative and attempt must be positive")
    if base_delay_s == 0:
        # An explicit zero remains a useful deterministic fast-fake setting,
        # but still yields to the scheduler and cannot become a busy loop.
        return _MIN_RECONNECT_DELAY_S
    factor = _RECONNECT_BACKOFF_FACTORS[min(attempt, len(_RECONNECT_BACKOFF_FACTORS)) - 1]
    return min(_MAX_RECONNECT_DELAY_S, max(_MIN_RECONNECT_DELAY_S, base_delay_s) * factor)


class WorkerState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    STOPPING = "stopping"


@dataclass(frozen=True)
class CameraWorkerSnapshot:
    """Point-in-time metrics for one camera worker."""

    camera_id: str
    state: WorkerState
    frames_received: int
    actual_fps: float
    dropped_frames: int
    reconnect_count: int
    successful_reconnects: int
    failed_reconnects: int
    queue_size: int
    max_buffer_frames: int
    last_received_at_utc: datetime | None
    started_at_utc: datetime | None
    last_error: str | None
    thread_alive: bool
    stream_fps: float | None = None

    @property
    def decoded_fps(self) -> float:
        """Observed FPS produced by the decoder; kept separate from stream metadata."""

        return self.actual_fps


class CameraWorker:
    """Own one source, one control thread and one bounded latest-frame buffer.

    A source must not be shared between workers. ``VideoSource.reconnect()`` is
    used for recovery so concrete sources remain responsible for their own
    capture lifecycle and source-specific timeouts.
    """

    def __init__(
        self,
        camera_id: str,
        source: VideoSource,
        *,
        read_timeout_s: float = 3.0,
        reconnect_delay_s: float = 2.0,
        max_reconnect_attempts: int = 0,
        max_buffer_frames: int = 1,
        stop_timeout_s: float | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        normalized_id = camera_id.strip()
        if not normalized_id:
            raise ValueError("camera_id cannot be empty")
        if read_timeout_s <= 0:
            raise ValueError("read_timeout_s must be greater than zero")
        if reconnect_delay_s < 0:
            raise ValueError("reconnect_delay_s cannot be negative")
        if max_reconnect_attempts < 0:
            raise ValueError("max_reconnect_attempts cannot be negative")
        if stop_timeout_s is not None and stop_timeout_s <= 0:
            raise ValueError("stop_timeout_s must be greater than zero")

        self._camera_id = normalized_id
        self._source = source
        self._read_timeout_s = read_timeout_s
        self._reconnect_delay_s = float(reconnect_delay_s)
        self._startup_grace_s = max(_STARTUP_GRACE_MIN_S, read_timeout_s * 2.0)
        self._max_reconnect_attempts = max_reconnect_attempts
        self._stop_timeout_s = stop_timeout_s or max(1.0, read_timeout_s + 1.0)
        self._buffer = LatestFrameBuffer(max_buffer_frames)
        self._logger = logger or logging.getLogger(__name__)

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = WorkerState.STOPPED
        self._started_at_utc: datetime | None = None
        self._started_monotonic: float | None = None
        self._last_received_at_utc: datetime | None = None
        self._stream_info: StreamInfo | None = None
        self._last_error: str | None = None
        self._frames_received = 0
        self._reconnect_count = 0
        self._successful_reconnects = 0
        self._failed_reconnects = 0
        log_event(
            self._logger,
            logging.DEBUG,
            "camera_worker_created",
            camera=self._camera_id,
            max_reconnect_attempts=self._max_reconnect_attempts,
            reconnect_delay=f"{self._reconnect_delay_s:.2f}s",
            read_timeout=f"{self._read_timeout_s:.2f}s",
        )

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def state(self) -> WorkerState:
        with self._condition:
            return self._state

    @property
    def is_alive(self) -> bool:
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the worker thread; repeated starts while active are no-ops."""

        with self._condition:
            if self._state is WorkerState.FAILED:
                raise RuntimeError("a failed CameraWorker cannot be restarted")
            if self._thread is not None and self._thread.is_alive():
                return
            if self._state is WorkerState.STOPPING:
                raise RuntimeError("CameraWorker is still stopping")

            self._buffer = LatestFrameBuffer(self._buffer.max_frames)
            self._stop_event.clear()
            self._started_at_utc = utc_now()
            self._started_monotonic = time.monotonic()
            self._last_received_at_utc = None
            self._stream_info = None
            self._last_error = None
            self._frames_received = 0
            self._reconnect_count = 0
            self._successful_reconnects = 0
            self._failed_reconnects = 0
            self._state = WorkerState.STARTING
            self._condition.notify_all()
            self._thread = threading.Thread(
                target=self._run,
                name=f"camera-worker-{self._camera_id}",
                daemon=True,
            )
            thread = self._thread
        thread.start()
        log_event(
            self._logger,
            logging.DEBUG,
            "camera_worker_started",
            camera=self._camera_id,
            max_reconnect_attempts=self._max_reconnect_attempts,
            thread_name=thread.name,
            thread_alive=thread.is_alive(),
        )

    def stop(self, timeout_s: float | None = None) -> None:
        """Request shutdown and wait a bounded time for the worker thread."""

        with self._condition:
            thread = self._thread
            failed = self._state is WorkerState.FAILED
            if thread is None:
                self._buffer.close()
                if not failed:
                    self._state = WorkerState.STOPPED
                    self._condition.notify_all()
                return
            if not failed:
                self._state = WorkerState.STOPPING
                self._condition.notify_all()
            self._stop_event.set()

        log_event(
            self._logger,
            logging.DEBUG,
            "camera_worker_stop_requested",
            camera=self._camera_id,
            stop_reason="explicit_stop",
            thread_alive=thread is not None and thread.is_alive(),
        )
        self._close_source(reason="worker_stop")

        if thread is not threading.current_thread():
            join_timeout = (
                timeout_s
                if timeout_s is not None
                else self._stop_timeout_s
            )
            thread.join(timeout=join_timeout)

        with self._condition:
            if not thread.is_alive() and self._state is not WorkerState.FAILED:
                self._state = WorkerState.STOPPED
                self._condition.notify_all()

    def get_latest(self, timeout_s: float = 0.0) -> FramePacket | None:
        """Return the newest available packet, discarding older queued packets."""

        return self._buffer.get_latest(timeout_s)

    def wait_for_state(
        self,
        states: WorkerState | Iterable[WorkerState],
        timeout_s: float,
    ) -> WorkerState | None:
        """Wait until the worker reaches one of ``states`` or the timeout expires."""

        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        wanted = {states} if isinstance(states, WorkerState) else set(states)
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._state not in wanted:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            return self._state

    def snapshot(self) -> CameraWorkerSnapshot:
        """Return metrics without blocking the acquisition loop."""

        with self._condition:
            started_monotonic = self._started_monotonic
            elapsed = (
                max(0.0, time.monotonic() - started_monotonic)
                if started_monotonic is not None
                else 0.0
            )
            frames_received = self._frames_received
            state = self._state
            thread = self._thread
            last_received_at_utc = self._last_received_at_utc
            started_at_utc = self._started_at_utc
            last_error = self._last_error
            stream_info = self._stream_info
            reconnect_count = self._reconnect_count
            successful_reconnects = self._successful_reconnects
            failed_reconnects = self._failed_reconnects

        source_dropped = 0
        try:
            source_dropped = max(0, int(getattr(self._source, "dropped_frames", 0)))
        except (TypeError, ValueError):
            source_dropped = 0

        return CameraWorkerSnapshot(
            camera_id=self._camera_id,
            state=state,
            frames_received=frames_received,
            actual_fps=frames_received / elapsed if elapsed > 0 else 0.0,
            dropped_frames=source_dropped + self._buffer.dropped_frames,
            reconnect_count=reconnect_count,
            successful_reconnects=successful_reconnects,
            failed_reconnects=failed_reconnects,
            queue_size=self._buffer.size,
            max_buffer_frames=self._buffer.max_frames,
            last_received_at_utc=last_received_at_utc,
            started_at_utc=started_at_utc,
            last_error=last_error,
            thread_alive=thread is not None and thread.is_alive(),
            stream_fps=stream_info.declared_fps if stream_info is not None else None,
        )

    def _run(self) -> None:
        connected = False
        initial_open = True
        awaiting_first_frame = False
        reconnecting_session = False
        startup_deadline = 0.0
        startup_failures = 0
        last_frame_monotonic: float | None = None
        attempts_for_outage = 0

        try:
            while not self._stop_event.is_set():
                if not connected:
                    if initial_open:
                        self._set_state(WorkerState.STARTING)
                        log_event(
                            self._logger,
                            logging.DEBUG,
                            "source_open_start",
                            camera=self._camera_id,
                            phase="initial",
                        )
                        try:
                            stream_info = self._source.open()
                        except Exception as exc:
                            initial_open = False
                            self._mark_disconnected(exc, event="connect_failed")
                            continue
                        with self._condition:
                            self._stream_info = stream_info
                        connected = True
                        awaiting_first_frame = True
                        reconnecting_session = False
                        startup_deadline = time.monotonic() + self._startup_grace_s
                        startup_failures = 0
                        attempts_for_outage = 0
                        log_event(
                            self._logger,
                            logging.DEBUG,
                            "source_open_success",
                            camera=self._camera_id,
                            phase="initial",
                            startup_grace=f"{self._startup_grace_s:.2f}s",
                        )
                        continue

                    if (
                        self._max_reconnect_attempts > 0
                        and attempts_for_outage >= self._max_reconnect_attempts
                    ):
                        message = self._last_error or "reconnect attempts exhausted"
                        self._fail(message)
                        log_event(
                            self._logger,
                            logging.ERROR,
                            "reconnect_exhausted",
                            camera=self._camera_id,
                            attempts=attempts_for_outage,
                            reason=message,
                        )
                        break

                    self._set_state(WorkerState.RECONNECTING)
                    next_attempt = attempts_for_outage + 1
                    delay_s = reconnect_delay_for_attempt(
                        self._reconnect_delay_s,
                        next_attempt,
                    )
                    log_event(
                        self._logger,
                        logging.INFO,
                        "reconnect_scheduled",
                        camera=self._camera_id,
                        delay=f"{delay_s:.2f}s",
                        attempt=next_attempt,
                    )
                    if self._stop_event.wait(timeout=delay_s):
                        break

                    if self._stop_event.is_set():
                        break
                    attempts_for_outage += 1
                    with self._condition:
                        self._reconnect_count += 1
                    log_event(
                        self._logger,
                        logging.INFO,
                        "reconnect_attempt",
                        camera=self._camera_id,
                        attempt=attempts_for_outage,
                    )
                    log_event(
                        self._logger,
                        logging.DEBUG,
                        "reconnect_callback_entered",
                        camera=self._camera_id,
                        attempt=attempts_for_outage,
                        thread_alive=True,
                    )
                    try:
                        stream_info = self._source.reconnect()
                    except Exception as exc:
                        with self._condition:
                            self._failed_reconnects += 1
                        self._mark_disconnected(exc, event="reconnect_failed")
                        continue

                    with self._condition:
                        self._stream_info = stream_info
                    connected = True
                    awaiting_first_frame = True
                    reconnecting_session = True
                    startup_deadline = time.monotonic() + self._startup_grace_s
                    startup_failures = 0
                    log_event(
                        self._logger,
                        logging.DEBUG,
                        "source_open_success",
                        camera=self._camera_id,
                        phase="reconnect",
                        attempt=attempts_for_outage,
                        startup_grace=f"{self._startup_grace_s:.2f}s",
                    )
                    continue

                try:
                    result = self._source.read(self._read_timeout_s)
                except Exception as exc:
                    if self._stop_event.is_set():
                        break
                    if awaiting_first_frame and time.monotonic() < startup_deadline:
                        startup_failures += 1
                        log_event(
                            self._logger,
                            logging.DEBUG,
                            "startup_frame_wait",
                            camera=self._camera_id,
                            status="exception",
                            reason=redact_log_text(exc),
                            consecutive_failures=startup_failures,
                            last_frame_age=(
                                f"{time.monotonic() - last_frame_monotonic:.3f}s"
                                if last_frame_monotonic is not None
                                else "none"
                            ),
                        )
                        if self._stop_event.wait(timeout=_STARTUP_RETRY_WAIT_S):
                            break
                        continue
                    connected = False
                    initial_open = False
                    self._mark_disconnected(exc)
                    continue

                if self._stop_event.is_set():
                    break

                if result.status is ReadStatus.FRAME and result.packet is not None:
                    now = time.monotonic()
                    with self._condition:
                        self._frames_received += 1
                        self._last_received_at_utc = result.packet.received_at_utc
                    dropped = self._buffer.put(result.packet)
                    if self._logger.isEnabledFor(logging.DEBUG):
                        log_event(
                            self._logger,
                            logging.DEBUG,
                            "worker_frame_published",
                            camera=self._camera_id,
                            status=result.status.value,
                            sequence=result.packet.sequence,
                            buffer_size=self._buffer.size,
                            dropped=dropped,
                            last_frame_age=(
                                f"{max(0.0, now - result.packet.received_monotonic):.3f}s"
                            ),
                        )
                    last_frame_monotonic = result.packet.received_monotonic
                    if awaiting_first_frame:
                        awaiting_first_frame = False
                        startup_deadline = 0.0
                        startup_failures = 0
                        if reconnecting_session:
                            attempts_for_outage = 0
                            with self._condition:
                                self._successful_reconnects += 1
                            self._set_state(WorkerState.RUNNING)
                            log_event(
                                self._logger,
                                logging.INFO,
                                "rtsp_reconnected",
                                camera=self._camera_id,
                            )
                        else:
                            self._set_state(WorkerState.RUNNING)
                            log_event(
                                self._logger,
                                logging.INFO,
                                "rtsp_connected",
                                camera=self._camera_id,
                            )
                    continue

                if awaiting_first_frame and time.monotonic() < startup_deadline:
                    startup_failures += 1
                    log_event(
                        self._logger,
                        logging.DEBUG,
                        "worker_read_result",
                        camera=self._camera_id,
                        status=result.status.value,
                        reason=result.message or "no frame yet",
                        consecutive_failures=startup_failures,
                        buffer_size=self._buffer.size,
                        last_frame_age=(
                            f"{time.monotonic() - last_frame_monotonic:.3f}s"
                            if last_frame_monotonic is not None
                            else "none"
                        ),
                    )
                    if self._stop_event.wait(timeout=_STARTUP_RETRY_WAIT_S):
                        break
                    continue

                connected = False
                initial_open = False
                log_event(
                    self._logger,
                    logging.DEBUG,
                    "worker_read_result",
                    camera=self._camera_id,
                    status=result.status.value,
                    reason=result.message or "no frame",
                    consecutive_failures=startup_failures,
                    buffer_size=self._buffer.size,
                    last_frame_age=(
                        f"{time.monotonic() - last_frame_monotonic:.3f}s"
                        if last_frame_monotonic is not None
                        else "none"
                    ),
                )
                self._mark_disconnected(
                    result.message or f"source returned {result.status.value}"
                )
        except Exception as exc:  # pragma: no cover - final isolation guard
            if not self._stop_event.is_set():
                self._fail(f"worker loop failed: {exc}")
        finally:
            self._close_source(reason="worker_finally")
            self._buffer.close()
            with self._condition:
                if self._state is not WorkerState.FAILED:
                    self._state = WorkerState.STOPPED
                self._condition.notify_all()

    def _set_state(self, state: WorkerState) -> None:
        with self._condition:
            if self._state is WorkerState.FAILED and state is not WorkerState.FAILED:
                return
            if self._stop_event.is_set() and state not in {
                WorkerState.STOPPING,
                WorkerState.STOPPED,
                WorkerState.FAILED,
            }:
                return
            self._state = state
            self._condition.notify_all()

    def _set_error(self, error: BaseException | str) -> None:
        message = redact_log_text(error) or type(error).__name__
        with self._condition:
            self._last_error = message
            self._condition.notify_all()

    def _fail(self, message: str) -> None:
        message = redact_log_text(message)
        with self._condition:
            self._last_error = message
            self._state = WorkerState.FAILED
            self._condition.notify_all()

    def _mark_disconnected(
        self,
        error: BaseException | str,
        *,
        event: str = "stream_lost",
    ) -> None:
        """Invalidate the current stream before waiting for a new session."""

        self._set_state(WorkerState.RECONNECTING)
        self._set_error(error)
        with self._condition:
            reason = self._last_error or type(error).__name__
            self._stream_info = None
            self._condition.notify_all()
        log_event(
            self._logger,
            logging.WARNING,
            event,
            camera=self._camera_id,
            reason=reason,
        )
        log_event(
            self._logger,
            logging.DEBUG,
            "disconnect_decision",
            camera=self._camera_id,
            action="close_and_reconnect",
            reason=reason,
        )
        self._close_source(reason=event, for_reconnect=True)
        with self._condition:
            state = self._state
            thread = self._thread
        log_event(
            self._logger,
            logging.DEBUG,
            "worker_still_alive_after_disconnect",
            camera=self._camera_id,
            state=state.value,
            thread_alive=thread is not None and thread.is_alive(),
            stop_requested=self._stop_event.is_set(),
        )

    def _close_source(self, *, reason: str, for_reconnect: bool = False) -> None:
        log_event(
            self._logger,
            logging.DEBUG,
            "source_close_requested",
            camera=self._camera_id,
            reason=reason,
            for_reconnect=for_reconnect,
        )
        try:
            if for_reconnect:
                self._source.close_for_reconnect()
            else:
                self._source.close()
        except Exception as exc:  # pragma: no cover - defensive source isolation
            self._logger.debug(
                "camera=%s source close failed: %s",
                self._camera_id,
                redact_log_text(exc),
            )

"""OpenCV-backed source with a bounded latest-frame buffer."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import logging
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from app.logging_setup import log_event, redact_log_text

from .base import (
    FramePacket,
    ReadResult,
    ReadStatus,
    StreamInfo,
    VideoSource,
    VideoSourceError,
    redact_url,
    utc_now,
)


_STARTUP_GRACE_MIN_S = 2.0
_STARTUP_RETRY_WAIT_S = 0.05

try:  # Keep health-check imports useful even before OpenCV is installed.
    import cv2
except ImportError:  # pragma: no cover - exercised by environment check, not unit tests.
    cv2 = None  # type: ignore[assignment]


CaptureFactory = Callable[[str, str], Any]
_FFMPEG_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_FFMPEG_OPTIONS_LOCK = threading.Lock()


@dataclass
class _CaptureOpenAttempt:
    """One capture construction that may outlive its public timeout."""

    completed: threading.Event = field(default_factory=threading.Event)
    capture: Any | None = None
    error: BaseException | None = None
    cancelled: bool = False
    claimed: bool = False


def _is_rtsp_url(url: str) -> bool:
    return urlsplit(url).scheme.lower() in {"rtsp", "rtsps"}


def _replace_rtsp_transport(options: str | None, transport: str) -> str:
    """Set one FFmpeg option while preserving unrelated existing options."""

    entries: list[str] = []
    for entry in (options or "").split("|"):
        if not entry:
            continue
        key, separator, _value = entry.partition(";")
        if separator and key.strip().lower() == "rtsp_transport":
            continue
        entries.append(entry)
    entries.append(f"rtsp_transport;{transport}")
    return "|".join(entries)


@contextlib.contextmanager
def _scoped_rtsp_transport(url: str, backend: str, transport: str):
    """Apply RTSP transport only while an FFmpeg-backed capture is created.

    OpenCV reads this setting from a process-level environment variable. The lock
    prevents two camera openings from seeing each other's temporary value, and the
    original environment is restored even when capture creation raises.
    """

    should_apply = (
        transport != "auto"
        and backend in {"auto", "ffmpeg"}
        and _is_rtsp_url(url)
    )
    if not should_apply:
        yield
        return

    with _FFMPEG_OPTIONS_LOCK:
        previous = os.environ.get(_FFMPEG_OPTIONS_ENV)
        applied = _replace_rtsp_transport(previous, transport)
        os.environ[_FFMPEG_OPTIONS_ENV] = applied
        try:
            yield
        finally:
            # A timed-out daemon capture can finish after its caller returned.
            # Do not overwrite a newer value installed by another owner while
            # this scope was waiting for the backend.
            if os.environ.get(_FFMPEG_OPTIONS_ENV) == applied:
                if previous is None:
                    os.environ.pop(_FFMPEG_OPTIONS_ENV, None)
                else:
                    os.environ[_FFMPEG_OPTIONS_ENV] = previous


def _fourcc_to_text(value: float) -> str | None:
    if not value:
        return None
    integer = int(value)
    chars = [(integer >> (8 * index)) & 0xFF for index in range(4)]
    text = "".join(chr(char) if 32 <= char <= 126 else "?" for char in chars).strip("?")
    return text or None


def _safe_capture_get(capture: Any, property_id: int, default: float | None = None) -> float | None:
    try:
        value = float(capture.get(property_id))
    except (AttributeError, TypeError, ValueError, cv2.error if cv2 is not None else Exception):
        return default
    return value if np.isfinite(value) else default


def _default_capture_factory(url: str, backend: str) -> Any:
    if cv2 is None:
        raise VideoSourceError(
            "OpenCV is not installed; run 'python -m pip install -e .[dev]'",
            code="opencv_missing",
        )

    if backend in {"auto", "ffmpeg"} and hasattr(cv2, "CAP_FFMPEG"):
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if capture.isOpened() or backend == "ffmpeg":
            return capture
        capture.release()

    return cv2.VideoCapture(url)


class OpenCVVideoSource(VideoSource):
    """Decode a stream on one reader thread and expose only recent frames."""

    def __init__(
        self,
        url: str,
        *,
        backend: str = "auto",
        rtsp_transport: str = "tcp",
        open_timeout_s: float = 5.0,
        read_timeout_s: float = 3.0,
        max_buffer_frames: int = 1,
        capture_factory: CaptureFactory | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not url or not url.strip():
            raise ValueError("video source URL cannot be empty")
        if backend not in {"auto", "opencv", "ffmpeg"}:
            raise ValueError(f"unsupported OpenCV backend: {backend}")
        if rtsp_transport not in {"auto", "tcp", "udp"}:
            raise ValueError(f"unsupported RTSP transport: {rtsp_transport}")
        if open_timeout_s <= 0 or read_timeout_s <= 0:
            raise ValueError("source timeouts must be greater than zero")
        if max_buffer_frames < 1:
            raise ValueError("max_buffer_frames must be at least one")

        self._url = url.strip()
        self._backend = backend
        self._rtsp_transport = rtsp_transport
        self._open_timeout_s = open_timeout_s
        self._read_timeout_s = read_timeout_s
        self._max_buffer_frames = max_buffer_frames
        self._capture_factory = capture_factory or _default_capture_factory
        self._logger = logger or logging.getLogger(__name__)

        self._capture: Any | None = None
        self._info: StreamInfo | None = None
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=max_buffer_frames)
        self._pending_events: deque[ReadResult] = deque(maxlen=32)
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._reader_thread: threading.Thread | None = None
        self._session_generation = 0
        self._startup_deadline_monotonic: float | None = None
        self._startup_failures = 0
        self._first_frame_received = False
        self._connected = False
        self._sequence = 0
        self._dropped_frames = 0
        self._reconnect_count = 0
        self._pending_open: _CaptureOpenAttempt | None = None

    @property
    def stream_info(self) -> StreamInfo | None:
        with self._state_lock:
            return self._info

    @property
    def dropped_frames(self) -> int:
        with self._state_lock:
            return self._dropped_frames

    @property
    def reconnect_count(self) -> int:
        with self._state_lock:
            return self._reconnect_count

    @property
    def is_connected(self) -> bool:
        with self._state_lock:
            return self._connected

    def open(self) -> StreamInfo:
        if self.is_connected:
            assert self._info is not None
            return self._info

        self.close_for_reconnect()
        with self._state_lock:
            generation = self._session_generation
            self._stop_event.clear()
        log_event(
            self._logger,
            logging.DEBUG,
            "source_open_start",
            generation=generation,
        )
        capture = self._create_capture_with_timeout()

        try:
            if capture is None or not bool(capture.isOpened()):
                if capture is not None:
                    self._safe_release(capture)
                raise VideoSourceError(
                    f"OpenCV could not open {redact_url(self._url)}",
                    code="unreachable",
                )
            self._configure_capture(capture)
            info = self._read_stream_info(capture)
        except VideoSourceError:
            self._safe_release(capture)
            log_event(
                self._logger,
                logging.DEBUG,
                "source_open_failure",
                generation=generation,
            )
            raise
        except Exception as exc:
            self._safe_release(capture)
            log_event(
                self._logger,
                logging.DEBUG,
                "source_open_failure",
                generation=generation,
                reason=redact_log_text(exc),
            )
            raise VideoSourceError(
                f"OpenCV failed while opening {redact_url(self._url)}: {redact_log_text(exc)}",
                code="open_failed",
            ) from exc

        with self._state_lock:
            if generation != self._session_generation or self._stop_event.is_set():
                stale = True
                reader = None
            else:
                stale = False
                self._capture = capture
                self._info = info
                self._connected = True
                self._startup_deadline_monotonic = time.monotonic() + max(
                    _STARTUP_GRACE_MIN_S,
                    self._read_timeout_s * 2.0,
                )
                self._startup_failures = 0
                self._first_frame_received = False
                self._pending_events.clear()
                self._reader_thread = threading.Thread(
                    target=self._reader_loop,
                    args=(capture, generation),
                    name="opencv-video-reader",
                    daemon=True,
                )
                reader = self._reader_thread
                reader.start()

        if stale:
            self._safe_release(capture)
            log_event(
                self._logger,
                logging.DEBUG,
                "source_open_discarded",
                generation=generation,
                reason="stale_generation_or_stop",
            )
            raise VideoSourceError(
                "OpenCV capture opening was cancelled",
                code="open_cancelled",
            )
        assert reader is not None
        log_event(
            self._logger,
            logging.DEBUG,
            "source_open_success",
            generation=generation,
            startup_grace=f"{max(_STARTUP_GRACE_MIN_S, self._read_timeout_s * 2.0):.2f}s",
        )
        return info

    def _create_capture_with_timeout(self) -> Any:
        """Run at most one capture construction and reuse a late result.

        OpenCV backends differ in whether their open timeout property is honored.
        The public call therefore remains bounded, but a backend call that outlives
        that timeout stays as the source's single pending attempt. A later reconnect
        waits for and claims that result instead of creating a second capture and
        discarding the connection that eventually became live.
        """

        with self._state_lock:
            attempt = self._pending_open
            start_attempt = attempt is None
            if start_attempt:
                attempt = _CaptureOpenAttempt()
                self._pending_open = attempt

        assert attempt is not None
        if start_attempt:
            self._start_capture_open_attempt(attempt)

        if not attempt.completed.wait(timeout=self._open_timeout_s):
            raise VideoSourceError(
                f"timed out after {self._open_timeout_s:.1f}s while opening "
                f"{redact_url(self._url)}",
                code="open_timeout",
            )

        with self._state_lock:
            if attempt.claimed:
                raise VideoSourceError(
                    "OpenCV capture opening was claimed by another session",
                    code="open_cancelled",
                )
            attempt.claimed = True
            if self._pending_open is attempt:
                self._pending_open = None
            capture = attempt.capture
            error = attempt.error
            cancelled = attempt.cancelled

        if cancelled:
            self._safe_release(capture)
            raise VideoSourceError(
                "OpenCV capture opening was cancelled",
                code="open_cancelled",
            )

        if error is not None:
            if isinstance(error, VideoSourceError):
                raise error
            raise VideoSourceError(
                f"could not create OpenCV capture for {redact_url(self._url)}: "
                f"{redact_log_text(error)}",
                code="open_failed",
            ) from error
        return capture

    def _start_capture_open_attempt(self, attempt: _CaptureOpenAttempt) -> None:
        def create() -> None:
            capture: Any | None = None
            error: BaseException | None = None
            try:
                with _scoped_rtsp_transport(self._url, self._backend, self._rtsp_transport):
                    capture = self._capture_factory(self._url, self._backend)
            except Exception as exc:  # forwarded to the caller below
                error = exc

            release_capture: Any | None = None
            with self._state_lock:
                attempt.capture = capture
                attempt.error = error
                attempt.completed.set()
                if attempt.cancelled and not attempt.claimed:
                    attempt.claimed = True
                    if self._pending_open is attempt:
                        self._pending_open = None
                    release_capture = capture
            if release_capture is not None:
                self._safe_release(release_capture)

        worker = threading.Thread(target=create, name="opencv-capture-open", daemon=True)
        worker.start()

    def read(self, timeout_s: float) -> ReadResult:
        timeout_s = max(0.0, timeout_s)
        deadline = time.monotonic() + timeout_s

        while True:
            with self._state_lock:
                if self._pending_events:
                    return self._pending_events.popleft()
                connected = self._connected
                opened = self._capture is not None

            remaining = max(0.0, deadline - time.monotonic())
            try:
                packet = self._queue.get(timeout=remaining)
                return ReadResult.frame_result(packet)
            except queue.Empty:
                with self._state_lock:
                    if self._pending_events:
                        return self._pending_events.popleft()
                    if not opened or not connected:
                        return ReadResult.status_result(
                            ReadStatus.DISCONNECTED,
                            "video source is not connected",
                        )
                return ReadResult.status_result(ReadStatus.TIMEOUT, "no recent frame before timeout")

    def reconnect(self) -> StreamInfo:
        with self._state_lock:
            self._reconnect_count += 1
        self.close_for_reconnect()
        return self.open()

    def close(self) -> None:
        self._close(preserve_pending_open=False)

    def close_for_reconnect(self) -> None:
        self._close(preserve_pending_open=True)

    def _close(self, *, preserve_pending_open: bool) -> None:
        pending_capture: Any | None = None
        with self._state_lock:
            previous_generation = self._session_generation
            self._session_generation += 1
            self._stop_event.set()
            capture = self._capture
            reader = self._reader_thread
            self._capture = None
            self._reader_thread = None
            self._info = None
            self._connected = False
            self._startup_deadline_monotonic = None
            self._startup_failures = 0
            self._first_frame_received = False
            if not preserve_pending_open and self._pending_open is not None:
                attempt = self._pending_open
                attempt.cancelled = True
                if attempt.completed.is_set() and not attempt.claimed:
                    attempt.claimed = True
                    self._pending_open = None
                    pending_capture = attempt.capture

        log_event(
            self._logger,
            logging.DEBUG,
            "source_generation_invalidated",
            generation=previous_generation,
            next_generation=previous_generation + 1,
            reader_thread_id=reader.ident if reader is not None else "none",
        )
        log_event(
            self._logger,
            logging.DEBUG,
            "source_close_called",
            generation=previous_generation + 1,
            had_capture=capture is not None,
        )

        if capture is not None:
            self._safe_release(capture)
        if pending_capture is not None:
            self._safe_release(pending_capture)
        if (
            reader is not None
            and reader is not threading.current_thread()
            and reader.ident is not None
        ):
            reader.join(timeout=max(0.25, self._read_timeout_s + 0.25))
            log_event(
                self._logger,
                logging.DEBUG,
                "source_reader_joined",
                generation=previous_generation,
                reader_thread_id=reader.ident,
                alive=reader.is_alive(),
            )

        self._drain_queue()
        with self._state_lock:
            self._pending_events.clear()

    def _configure_capture(self, capture: Any) -> None:
        if cv2 is None:
            return
        properties = (
            ("CAP_PROP_OPEN_TIMEOUT_MSEC", self._open_timeout_s * 1000),
            ("CAP_PROP_READ_TIMEOUT_MSEC", self._read_timeout_s * 1000),
        )
        for property_name, value in properties:
            property_id = getattr(cv2, property_name, None)
            if property_id is None:
                continue
            try:
                capture.set(property_id, value)
            except Exception:
                self._logger.debug("capture property unsupported: %s", property_name)

    def _read_stream_info(self, capture: Any) -> StreamInfo:
        if cv2 is None:
            raise VideoSourceError("OpenCV is not available", code="opencv_missing")
        width = _safe_capture_get(capture, cv2.CAP_PROP_FRAME_WIDTH)
        height = _safe_capture_get(capture, cv2.CAP_PROP_FRAME_HEIGHT)
        fps = _safe_capture_get(capture, cv2.CAP_PROP_FPS)
        fourcc = _safe_capture_get(capture, cv2.CAP_PROP_FOURCC)
        return StreamInfo(
            url=redact_url(self._url),
            backend=self._backend,
            width=int(width) if width and width > 0 else None,
            height=int(height) if height and height > 0 else None,
            declared_fps=fps if fps and fps > 0 else None,
            codec=_fourcc_to_text(fourcc or 0),
            opened_at_utc=utc_now(),
        )

    def _reader_loop(self, capture: Any, generation: int) -> None:
        reader_thread_id = threading.get_ident()
        debug_frames = self._logger.isEnabledFor(logging.DEBUG)
        exit_reason = "stop_event"
        log_event(
            self._logger,
            logging.DEBUG,
            "source_reader_started",
            generation=generation,
            reader_thread_id=reader_thread_id,
        )
        try:
            while not self._stop_event.is_set():
                with self._state_lock:
                    if not self._session_is_active_locked(capture, generation):
                        exit_reason = "stale_generation"
                        break

                started = time.perf_counter()
                if debug_frames:
                    log_event(
                        self._logger,
                        logging.DEBUG,
                        "source_capture_read_start",
                        generation=generation,
                        reader_thread_id=reader_thread_id,
                    )
                try:
                    ok, frame = capture.read()
                except Exception as exc:
                    read_duration_ms = (time.perf_counter() - started) * 1000
                    log_event(
                        self._logger,
                        logging.DEBUG,
                        "source_capture_read_result",
                        generation=generation,
                        reader_thread_id=reader_thread_id,
                        result="exception",
                        read_duration_ms=f"{read_duration_ms:.1f}",
                        reason=redact_log_text(exc),
                    )
                    if self._wait_for_first_frame(
                        capture,
                        generation,
                        reader_thread_id=reader_thread_id,
                        reason=f"capture_read_error:{redact_log_text(exc)}",
                        read_duration_ms=read_duration_ms,
                    ):
                        continue
                    if self._stop_event.is_set():
                        exit_reason = "stop_event"
                    else:
                        with self._state_lock:
                            active = self._session_is_active_locked(capture, generation)
                        if not active:
                            exit_reason = "stale_generation"
                        else:
                            self._publish_status(
                                ReadStatus.CORRUPT,
                                f"capture read raised: {redact_log_text(exc)}",
                                capture=capture,
                                generation=generation,
                            )
                            exit_reason = "capture_read_error"
                    break
                read_duration_ms = (time.perf_counter() - started) * 1000
                frame_valid = bool(
                    ok
                    and frame is not None
                    and isinstance(frame, np.ndarray)
                    and frame.size > 0
                )
                if debug_frames:
                    log_event(
                        self._logger,
                        logging.DEBUG,
                        "source_capture_read_result",
                        generation=generation,
                        reader_thread_id=reader_thread_id,
                        result="frame" if frame_valid else "no_frame",
                        ok=ok,
                        frame_valid=frame_valid,
                        read_duration_ms=f"{read_duration_ms:.1f}",
                    )

                if self._stop_event.is_set():
                    exit_reason = "stop_event"
                    break
                with self._state_lock:
                    if not self._session_is_active_locked(capture, generation):
                        exit_reason = "stale_generation"
                        break
                if not frame_valid:
                    if self._wait_for_first_frame(
                        capture,
                        generation,
                        reader_thread_id=reader_thread_id,
                        reason="capture returned an invalid frame",
                        read_duration_ms=read_duration_ms,
                    ):
                        continue
                    if self._stop_event.is_set():
                        exit_reason = "stop_event"
                    else:
                        with self._state_lock:
                            active = self._session_is_active_locked(capture, generation)
                        if not active:
                            exit_reason = "stale_generation"
                        else:
                            self._publish_status(
                                ReadStatus.CORRUPT,
                                "capture returned an invalid frame",
                                capture=capture,
                                generation=generation,
                            )
                            exit_reason = "invalid_frame"
                    break

                with self._state_lock:
                    if not self._session_is_active_locked(capture, generation):
                        exit_reason = "stale_generation"
                        break
                    self._first_frame_received = True
                    self._startup_deadline_monotonic = None
                    self._startup_failures = 0
                    self._sequence += 1
                    sequence = self._sequence
                    packet = FramePacket(
                        frame=frame,
                        sequence=sequence,
                        received_at_utc=utc_now(),
                        received_monotonic=time.monotonic(),
                        read_duration_ms=read_duration_ms,
                    )
                    dropped = 0
                    try:
                        self._queue.put_nowait(packet)
                    except queue.Full:
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._dropped_frames += 1
                        dropped += 1
                        try:
                            self._queue.put_nowait(packet)
                        except queue.Full:
                            self._dropped_frames += 1
                            dropped += 1
                    buffer_size = self._queue.qsize()

                if debug_frames:
                    log_event(
                        self._logger,
                        logging.DEBUG,
                        "source_frame_received",
                        generation=generation,
                        reader_thread_id=reader_thread_id,
                        sequence=sequence,
                    )
                    log_event(
                        self._logger,
                        logging.DEBUG,
                        "source_frame_published",
                        generation=generation,
                        reader_thread_id=reader_thread_id,
                        sequence=sequence,
                        buffer_size=buffer_size,
                        dropped=dropped,
                        frame_age="0.000s",
                    )
        finally:
            with self._state_lock:
                if (
                    generation == self._session_generation
                    and self._capture is capture
                    and not self._stop_event.is_set()
                ):
                    self._connected = False
            log_event(
                self._logger,
                logging.DEBUG,
                "source_reader_exit",
                generation=generation,
                reader_thread_id=reader_thread_id,
                reason=exit_reason,
            )

    def _wait_for_first_frame(
        self,
        capture: Any,
        generation: int,
        *,
        reader_thread_id: int,
        reason: str,
        read_duration_ms: float,
    ) -> bool:
        with self._state_lock:
            if not self._session_is_active_locked(capture, generation):
                return False
            deadline = self._startup_deadline_monotonic
            if self._first_frame_received or deadline is None:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._startup_failures += 1
            consecutive_failures = self._startup_failures

        log_event(
            self._logger,
            logging.DEBUG,
            "source_frame_discarded",
            generation=generation,
            reader_thread_id=reader_thread_id,
            reason=reason,
            read_duration_ms=f"{read_duration_ms:.1f}",
            consecutive_failures=consecutive_failures,
            remaining_startup=f"{remaining:.3f}s",
        )
        if self._stop_event.wait(timeout=min(_STARTUP_RETRY_WAIT_S, remaining)):
            return False
        return True

    def _publish_status(
        self,
        status: ReadStatus,
        message: str,
        *,
        capture: Any,
        generation: int,
    ) -> None:
        with self._state_lock:
            if not self._session_is_active_locked(capture, generation):
                return
            self._pending_events.append(ReadResult.status_result(status, message))
            self._connected = False
        log_event(
            self._logger,
            logging.DEBUG,
            "source_frame_discarded",
            generation=generation,
            reader_thread_id=threading.get_ident(),
            reason=f"terminal_{status.value}:{message}",
        )

    def _session_is_active_locked(self, capture: Any, generation: int) -> bool:
        return (
            generation == self._session_generation
            and self._capture is capture
            and not self._stop_event.is_set()
        )

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _safe_release(capture: Any) -> None:
        try:
            capture.release()
        except Exception:
            pass

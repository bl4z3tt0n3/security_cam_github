"""Filesystem persistence for event metadata and snapshots."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timezone
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any
from uuid import uuid4

import numpy as np

from .models import Event


class EventStorage:
    """Store one event per date/ID directory below a configured root."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @staticmethod
    def event_relative_directory(event: Event) -> Path:
        timestamp = event.timestamp.astimezone(timezone.utc)
        return Path(
            f"{timestamp.year:04d}",
            f"{timestamp.month:02d}",
            f"{timestamp.day:02d}",
            event.id,
        )

    def prepare_event_directory(self, event: Event) -> Path:
        directory = self.root / self.event_relative_directory(event)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def relative_path(self, path: Path) -> str:
        """Return a portable path relative to the configured events root."""

        return path.relative_to(self.root).as_posix()

    def write_metadata(self, event: Event, directory: Path) -> Path:
        """Atomically replace ``metadata.json`` for one event."""

        metadata_path = directory / "metadata.json"
        temporary_path = directory / f".metadata.{uuid4().hex}.tmp"
        payload = json.dumps(
            event.to_metadata(),
            ensure_ascii=False,
            indent=2,
        )
        try:
            temporary_path.write_text(payload + "\n", encoding="utf-8")
            os.replace(temporary_path, metadata_path)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        return metadata_path


SnapshotTask = tuple[Path, np.ndarray]


class SnapshotWriter:
    """Write JPEG snapshots on a bounded background queue.

    The frame is copied before it enters the queue, so callers may safely reuse
    the decoded frame buffer immediately after publishing an event.
    """

    def __init__(
        self,
        *,
        max_queue_size: int = 32,
        logger: logging.Logger | None = None,
        encoder: Callable[[np.ndarray], bytes] | None = None,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self._queue: queue.Queue[SnapshotTask | None] = queue.Queue(maxsize=max_queue_size)
        self._logger = logger or logging.getLogger(__name__)
        self._encoder = encoder or self._encode_jpeg
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._stopped = False

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> bytes:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("OpenCV could not encode the snapshot as JPEG")
        return encoded.tobytes()

    def submit(self, destination: Path, frame: Any) -> bool:
        """Queue a snapshot and return ``False`` when it cannot be accepted."""

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return False
        try:
            copied_frame = np.array(frame, copy=True)
        except (MemoryError, TypeError, ValueError):
            return False

        with self._lock:
            if self._closed:
                return False
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="event-snapshot-writer",
                    daemon=True,
                )
                self._thread.start()
            try:
                self._queue.put_nowait((destination, copied_frame))
            except queue.Full:
                return False
        return True

    def flush(self, timeout_s: float | None = None) -> bool:
        """Wait until all currently queued snapshots have been processed."""

        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            if self._queue.unfinished_tasks == 0:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.005)

    def close(self, timeout_s: float | None = 5.0) -> bool:
        """Flush and stop the writer; repeated calls are harmless."""

        with self._lock:
            if self._closed:
                thread = self._thread
            else:
                self._closed = True
                thread = self._thread

            if self._stopped:
                return True

        if thread is None:
            return True
        if not self.flush(timeout_s):
            return False
        self._queue.put(None)
        thread.join(timeout_s)
        stopped = not thread.is_alive()
        if stopped:
            with self._lock:
                self._stopped = True
        return stopped

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                destination, frame = task
                self._write(destination, frame)
            except Exception:
                self._logger.exception("event snapshot write failed")
            finally:
                self._queue.task_done()

    def _write(self, destination: Path, frame: np.ndarray) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            temporary_path.write_bytes(self._encoder(frame))
            os.replace(temporary_path, destination)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

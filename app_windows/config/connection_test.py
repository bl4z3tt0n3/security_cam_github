"""Asynchronous bounded connection tests for the existing video backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from app.logging_setup import redact_log_text
from app.video.base import ReadStatus, VideoSource, VideoSourceError, redact_url


SourceFactory = Callable[[str, str], VideoSource]
ExistingProbe = Callable[[str], tuple[bool, bool, str]]


@dataclass(frozen=True)
class ConnectionTestResult:
    camera_id: str
    success: bool
    message: str
    url: str


class _ConnectionTestWorker(QThread):
    completed = Signal(object)

    def __init__(
        self,
        camera_id: str,
        url: str,
        transport: str,
        source_factory: SourceFactory,
        read_timeout_s: float,
        logger: logging.Logger,
    ) -> None:
        super().__init__()
        self._camera_id = camera_id
        self._url = url
        self._transport = transport
        self._source_factory = source_factory
        self._read_timeout_s = read_timeout_s
        self._logger = logger

    def run(self) -> None:
        source: VideoSource | None = None
        result: ConnectionTestResult
        try:
            source = self._source_factory(self._url, self._transport)
            source.open()
            read_result = source.read(self._read_timeout_s)
            if read_result.status is ReadStatus.FRAME and read_result.packet is not None:
                result = ConnectionTestResult(
                    self._camera_id,
                    True,
                    "Connessione riuscita",
                    redact_url(self._url),
                )
            else:
                detail = redact_log_text(read_result.message or read_result.status.value)
                prefix = "Connessione scaduta" if read_result.status is ReadStatus.TIMEOUT else "Stream non disponibile"
                result = ConnectionTestResult(
                    self._camera_id,
                    False,
                    f"{prefix}: {detail}",
                    redact_url(self._url),
                )
        except (VideoSourceError, OSError, ValueError) as exc:
            result = ConnectionTestResult(
                self._camera_id,
                False,
                f"Errore connessione: {redact_log_text(exc)}",
                redact_url(self._url),
            )
        except Exception as exc:  # keep unexpected backend errors readable
            self._logger.error("camera connection test failed: %s", redact_log_text(exc))
            result = ConnectionTestResult(
                self._camera_id,
                False,
                f"Errore connessione: {redact_log_text(exc)}",
                redact_url(self._url),
            )
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as exc:  # cleanup must not hide the test result
                    self._logger.debug("connection test source close failed: %s", redact_log_text(exc))
        self.completed.emit(result)


class AsyncConnectionTester(QObject):
    """Run one connection test without occupying the Qt GUI thread."""

    started = Signal(str)
    finished = Signal(object)

    def __init__(
        self,
        source_factory: SourceFactory,
        *,
        read_timeout_s: float,
        existing_probe: ExistingProbe | None = None,
        logger: logging.Logger | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if read_timeout_s <= 0:
            raise ValueError("read_timeout_s must be greater than zero")
        self._source_factory = source_factory
        self._read_timeout_s = read_timeout_s
        self._existing_probe = existing_probe
        self._logger = logger or logging.getLogger(__name__)
        self._thread: QThread | None = None
        self._pending_result: object | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, camera_id: str, url: str, transport: str) -> None:
        if self.running:
            raise RuntimeError("a connection test is already running")

        if self._existing_probe is not None:
            has_provider, success, message = self._existing_probe(camera_id)
            if has_provider:
                result = ConnectionTestResult(
                    camera_id,
                    success,
                    message or ("Connessione riuscita" if success else "Stream non disponibile"),
                    redact_url(url),
                )
                self.started.emit(camera_id)
                QTimer.singleShot(0, lambda: self.finished.emit(result))
                return

        self.started.emit(camera_id)
        worker = _ConnectionTestWorker(
            camera_id,
            url,
            transport,
            self._source_factory,
            self._read_timeout_s,
            self._logger,
        )
        worker.setParent(self)
        self._thread = worker
        worker.completed.connect(self._on_completed)
        worker.finished.connect(self._on_thread_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    @Slot(object)
    def _on_completed(self, result: object) -> None:
        self._pending_result = result

    @Slot()
    def _on_thread_finished(self) -> None:
        result = self._pending_result
        self._pending_result = None
        self._thread = None
        if result is not None:
            self.finished.emit(result)

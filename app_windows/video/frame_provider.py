"""Small adapter between the existing camera worker and the Qt monitor."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Protocol

from app.video.base import FramePacket, StreamInfo, VideoSource
from app.video.worker import CameraWorker, CameraWorkerSnapshot


@dataclass(frozen=True)
class ProviderSnapshot:
    """Non-blocking provider state returned to the Qt polling timer."""

    worker: CameraWorkerSnapshot | None
    stream_info: StreamInfo | None
    last_error: str | None = None
    hardware_acceleration: str | None = None


class FrameProvider(Protocol):
    """Minimal UI-facing frame contract."""

    camera_id: str

    def start(self) -> None:
        ...

    def stop(self, timeout_s: float | None = None) -> None:
        ...

    def latest_frame(self) -> FramePacket | None:
        ...

    def status(self) -> ProviderSnapshot:
        ...

    def snapshot(self) -> ProviderSnapshot:
        ...


class BackendFrameProvider:
    """Own one existing ``CameraWorker`` and never run inference in the UI."""

    def __init__(
        self,
        camera_id: str,
        source: VideoSource,
        *,
        read_timeout_s: float,
        reconnect_delay_s: float,
        max_reconnect_attempts: int,
        max_buffer_frames: int,
        stop_timeout_s: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self.camera_id = camera_id.strip()
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        self._source = source
        self._logger = logger or logging.getLogger(__name__)
        self._worker = CameraWorker(
            self.camera_id,
            source,
            read_timeout_s=read_timeout_s,
            reconnect_delay_s=reconnect_delay_s,
            max_reconnect_attempts=max_reconnect_attempts,
            max_buffer_frames=max_buffer_frames,
            stop_timeout_s=stop_timeout_s,
            logger=self._logger,
        )
        self._start_calls = 0

    @property
    def worker(self) -> CameraWorker:
        return self._worker

    @property
    def source(self) -> VideoSource:
        return self._source

    @property
    def start_calls(self) -> int:
        return self._start_calls

    def start(self) -> None:
        self._start_calls += 1
        self._worker.start()

    def stop(self, timeout_s: float | None = None) -> None:
        self._worker.stop(timeout_s=timeout_s)

    def latest_frame(self) -> FramePacket | None:
        return self._worker.get_latest(timeout_s=0.0)

    def status(self) -> ProviderSnapshot:
        """Return the non-blocking acquisition status for this camera."""

        return self.snapshot()

    def snapshot(self) -> ProviderSnapshot:
        worker_snapshot = self._worker.snapshot()
        stream_info = getattr(self._source, "stream_info", None)
        return ProviderSnapshot(
            worker=worker_snapshot,
            stream_info=stream_info,
            last_error=worker_snapshot.last_error,
            hardware_acceleration=getattr(
                self._source,
                "hardware_acceleration_used",
                None,
            ),
        )

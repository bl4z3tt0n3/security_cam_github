"""Synthetic providers for six-camera UI development without hardware."""

from __future__ import annotations

from collections.abc import Callable
import logging

import numpy as np

from app.video.base import ReadResult, ReadStatus, StreamInfo, VideoSource, VideoSourceError, utc_now
from app.video.fake_source import FakeVideoSource

from app_windows.models.camera_view_state import CameraSlot

from .frame_provider import BackendFrameProvider


class OfflineFakeVideoSource(VideoSource):
    """A source that stays offline while retaining the real worker contract."""

    def __init__(self, url: str) -> None:
        self.url = url

    def open(self) -> StreamInfo:
        raise VideoSourceError("synthetic camera is offline", code="offline")

    def read(self, timeout_s: float) -> ReadResult:
        del timeout_s
        return ReadResult.status_result(ReadStatus.DISCONNECTED, "synthetic camera is offline")

    def reconnect(self) -> StreamInfo:
        raise VideoSourceError("synthetic camera is still offline", code="offline")

    def close(self) -> None:
        return None


def fake_connection_source_factory(url: str, transport: str) -> VideoSource:
    """Build a fake ``VideoSource`` for the GUI connection-test path."""

    del transport
    if url.rstrip("/").endswith("/offline"):
        return OfflineFakeVideoSource(url)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    return FakeVideoSource(
        [frame],
        url=url,
        width=32,
        height=24,
        fps=30.0,
        read_delay_s=0.001,
        repeat=True,
    )


def _pattern_frame(camera_index: int, phase: int, *, width: int, height: int) -> np.ndarray:
    """Create a distinct BGR pattern without depending on OpenCV drawing APIs."""

    y, x = np.indices((height, width), dtype=np.uint16)
    base = np.empty((height, width, 3), dtype=np.uint8)
    colors = (
        (42, 178, 235),
        (210, 112, 52),
        (96, 202, 116),
        (180, 84, 205),
        (50, 190, 190),
        (220, 150, 58),
    )
    blue, green, red = colors[camera_index % len(colors)]
    base[:, :, 0] = (blue + x // 8 + phase * 3) % 256
    base[:, :, 1] = (green + y // 6 + phase * 5) % 256
    base[:, :, 2] = (red + (x + y) // 12 + phase * 7) % 256

    # A large moving band makes stale-frame behavior obvious in the preview.
    band_start = ((phase * 24) + camera_index * 31) % max(1, width)
    band_width = max(8, width // 12)
    base[:, band_start : min(width, band_start + band_width), :] = 245
    return base


class FakeFrameProvider(BackendFrameProvider):
    """BackendFrameProvider backed by a bounded, repeatable synthetic source."""

    def __init__(
        self,
        slot: CameraSlot,
        *,
        camera_index: int,
        fps: float = 12.0,
        offline: bool = False,
        fail_after_frames: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fake FPS must be greater than zero")
        width, height = 640, 360
        if offline:
            source: VideoSource = OfflineFakeVideoSource(
                f"fake://{slot.camera_id}/offline"
            )
        else:
            frames = tuple(
                _pattern_frame(camera_index, phase, width=width, height=height)
                for phase in range(4)
            )
            source = FakeVideoSource(
                frames,
                url=f"fake://{slot.camera_id}/live",
                width=width,
                height=height,
                fps=fps,
                read_delay_s=1.0 / fps,
                fail_after_frames=fail_after_frames,
                repeat=True,
            )

        super().__init__(
            slot.camera_id,
            source,
            read_timeout_s=0.25,
            reconnect_delay_s=0.5,
            max_reconnect_attempts=3,
            max_buffer_frames=1,
            stop_timeout_s=1.0,
            logger=logger,
        )


def fake_camera_factory(
    *,
    offline_camera_id: str | None = None,
    reconnect_camera_id: str | None = None,
    logger: logging.Logger | None = None,
) -> Callable[[CameraSlot], FakeFrameProvider]:
    """Return a six-camera provider factory for the command-line fake mode."""

    def create(slot: CameraSlot) -> FakeFrameProvider:
        index = slot.slot_index - 1
        fps = 8.0 + (index % 4) * 3.0
        return FakeFrameProvider(
            slot,
            camera_index=index,
            fps=fps,
            offline=slot.camera_id == offline_camera_id,
            fail_after_frames=1 if slot.camera_id == reconnect_camera_id else None,
            logger=logger,
        )

    return create

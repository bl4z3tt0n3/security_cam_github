"""Shared construction helpers for configured OpenCV video sources."""

from __future__ import annotations

import logging

from app.config import VideoConfig

from .opencv_source import OpenCVVideoSource


def create_opencv_source(
    url: str,
    *,
    video: VideoConfig | None = None,
    backend: str | None = None,
    rtsp_transport: str | None = None,
    open_timeout_s: float | None = None,
    read_timeout_s: float | None = None,
    max_buffer_frames: int | None = None,
    logger: logging.Logger | None = None,
) -> OpenCVVideoSource:
    """Create the same bounded source used by CLI and Windows UI callers."""

    settings = video or VideoConfig()
    return OpenCVVideoSource(
        url,
        backend=backend if backend is not None else settings.backend,
        rtsp_transport=(
            rtsp_transport if rtsp_transport is not None else settings.rtsp_transport
        ),
        open_timeout_s=(
            open_timeout_s if open_timeout_s is not None else settings.open_timeout_seconds
        ),
        read_timeout_s=(
            read_timeout_s if read_timeout_s is not None else settings.read_timeout_seconds
        ),
        max_buffer_frames=(
            max_buffer_frames if max_buffer_frames is not None else settings.max_buffer_frames
        ),
        logger=logger,
    )

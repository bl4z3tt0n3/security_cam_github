"""Stable interfaces shared by real and fake video sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import numpy as np


class ReadStatus(str, Enum):
    FRAME = "frame"
    TIMEOUT = "timeout"
    DISCONNECTED = "disconnected"
    CORRUPT = "corrupt"


class VideoSourceError(RuntimeError):
    """A recoverable or fatal error raised by a video source."""

    def __init__(self, message: str, *, code: str = "source_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StreamInfo:
    """Metadata observed when a source is opened."""

    url: str
    backend: str
    width: int | None
    height: int | None
    declared_fps: float | None
    codec: str | None
    opened_at_utc: datetime


@dataclass(frozen=True)
class FramePacket:
    """One decoded frame and the timestamps needed for diagnostics."""

    frame: np.ndarray
    sequence: int
    received_at_utc: datetime
    received_monotonic: float
    read_duration_ms: float
    source_timestamp_s: float | None = None


@dataclass(frozen=True)
class ReadResult:
    """Result of a bounded read operation."""

    status: ReadStatus
    packet: FramePacket | None = None
    message: str | None = None

    @classmethod
    def frame_result(cls, packet: FramePacket) -> "ReadResult":
        return cls(status=ReadStatus.FRAME, packet=packet)

    @classmethod
    def status_result(cls, status: ReadStatus, message: str | None = None) -> "ReadResult":
        return cls(status=status, message=message)


class VideoSource(ABC):
    """Source contract independent of phone, RTSP camera or local webcam."""

    @abstractmethod
    def open(self) -> StreamInfo:
        raise NotImplementedError

    @abstractmethod
    def read(self, timeout_s: float) -> ReadResult:
        raise NotImplementedError

    @abstractmethod
    def reconnect(self) -> StreamInfo:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def close_for_reconnect(self) -> None:
        """Close the current session while allowing reconnect handoff state."""

        self.close()

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def redact_url(url: str) -> str:
    """Hide URL credentials before a URL is written to logs or reports."""

    try:
        parsed = urlsplit(url)
        if parsed.username is None and parsed.password is None:
            return url

        hostname = parsed.hostname or ""
        host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        netloc = f"{parsed.username or 'user'}:***@{host}"
        safe = SplitResult(parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
        return urlunsplit(safe)
    except ValueError:
        return "<invalid-url>"

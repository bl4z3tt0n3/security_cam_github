"""ffprobe integration and bounded stream-run measurements."""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from .base import ReadStatus, StreamInfo, VideoSource, VideoSourceError, redact_url


@dataclass(frozen=True)
class FfprobeResult:
    available: bool
    success: bool
    executable: str | None
    streams: tuple[dict[str, Any], ...] = ()
    format_name: str | None = None
    stderr: str | None = None
    error: str | None = None


@dataclass
class StreamRunReport:
    url: str
    requested_duration_s: float
    elapsed_duration_s: float = 0.0
    stream_info: StreamInfo | None = None
    frames_received: int = 0
    timeouts: int = 0
    corrupt_frames: int = 0
    disconnections: int = 0
    reconnect_attempts: int = 0
    successful_reconnects: int = 0
    failed_reconnects: int = 0
    reconnect_times_ms: list[float] = field(default_factory=list)
    read_latencies_ms: list[float] = field(default_factory=list)
    dropped_frames: int = 0
    error: str | None = None
    stop_reason: str | None = None

    @property
    def actual_fps(self) -> float:
        if self.elapsed_duration_s <= 0:
            return 0.0
        return self.frames_received / self.elapsed_duration_s

    @property
    def mean_read_latency_ms(self) -> float | None:
        if not self.read_latencies_ms:
            return None
        return sum(self.read_latencies_ms) / len(self.read_latencies_ms)

    @property
    def p95_read_latency_ms(self) -> float | None:
        if not self.read_latencies_ms:
            return None
        values = sorted(self.read_latencies_ms)
        index = min(len(values) - 1, max(0, math.ceil(len(values) * 0.95) - 1))
        return values[index]

    @property
    def mean_reconnect_time_ms(self) -> float | None:
        if not self.reconnect_times_ms:
            return None
        return sum(self.reconnect_times_ms) / len(self.reconnect_times_ms)


def parse_fps(value: Any) -> float | None:
    """Parse ffprobe values such as ``30000/1001`` or ``25``."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    text = str(value).strip()
    if not text or text in {"0/0", "N/A"}:
        return None
    try:
        result = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def run_ffprobe(url: str, *, executable: str = "ffprobe", timeout_s: float = 5.0) -> FfprobeResult:
    """Run ffprobe without invoking a shell and parse its JSON response."""

    path = shutil.which(executable)
    if path is None:
        return FfprobeResult(
            available=False,
            success=False,
            executable=None,
            error="ffprobe was not found on PATH",
        )

    command = [
        path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        url,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return FfprobeResult(
            available=True,
            success=False,
            executable=path,
            error=f"ffprobe timed out after {timeout_s:.1f}s",
        )
    except OSError as exc:
        return FfprobeResult(
            available=True,
            success=False,
            executable=path,
            error=f"could not start ffprobe: {exc}",
        )

    stderr = completed.stderr.strip() or None
    if completed.returncode != 0:
        return FfprobeResult(
            available=True,
            success=False,
            executable=path,
            stderr=stderr,
            error=f"ffprobe exited with code {completed.returncode}",
        )

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return FfprobeResult(
            available=True,
            success=False,
            executable=path,
            stderr=stderr,
            error=f"ffprobe returned invalid JSON: {exc}",
        )

    streams = tuple(item for item in payload.get("streams", []) if isinstance(item, dict))
    format_data = payload.get("format")
    format_name = format_data.get("format_name") if isinstance(format_data, dict) else None
    return FfprobeResult(
        available=True,
        success=True,
        executable=path,
        streams=streams,
        format_name=format_name,
        stderr=stderr,
    )


def video_stream_from_probe(result: FfprobeResult) -> dict[str, Any] | None:
    for stream in result.streams:
        if stream.get("codec_type") == "video":
            return stream
    return None


def run_stream_test(
    source: VideoSource,
    *,
    url: str,
    duration_s: float,
    read_timeout_s: float,
    reconnect_attempts: int,
    reconnect_delay_s: float,
    logger: logging.Logger | None = None,
) -> StreamRunReport:
    """Open a source and collect bounded receive/reconnect statistics."""

    if duration_s <= 0:
        raise ValueError("duration_s must be greater than zero")
    if read_timeout_s <= 0:
        raise ValueError("read_timeout_s must be greater than zero")
    if reconnect_attempts < 0:
        raise ValueError("reconnect_attempts cannot be negative")

    logger = logger or logging.getLogger(__name__)
    report = StreamRunReport(url=redact_url(url), requested_duration_s=duration_s)
    started = time.monotonic()
    try:
        report.stream_info = source.open()
    except VideoSourceError as exc:
        report.error = str(exc)
        report.stop_reason = "open_failed"
        report.elapsed_duration_s = time.monotonic() - started
        return report

    reconnects_used = 0
    try:
        while time.monotonic() - started < duration_s:
            remaining = duration_s - (time.monotonic() - started)
            result = source.read(min(read_timeout_s, max(0.05, remaining)))
            if result.status is ReadStatus.FRAME and result.packet is not None:
                report.frames_received += 1
                report.read_latencies_ms.append(result.packet.read_duration_ms)
                continue
            if result.status is ReadStatus.TIMEOUT:
                report.timeouts += 1
                continue
            if result.status is ReadStatus.CORRUPT:
                report.corrupt_frames += 1
                logger.warning("event=corrupt_frame message=%s", result.message or "unknown")
            elif result.status is ReadStatus.DISCONNECTED:
                report.disconnections += 1

            if reconnects_used >= reconnect_attempts:
                report.error = result.message or "stream disconnected and reconnect attempts are exhausted"
                report.stop_reason = "reconnect_exhausted"
                break

            reconnects_used += 1
            report.reconnect_attempts += 1
            if reconnect_delay_s:
                time.sleep(reconnect_delay_s)
            reconnect_started = time.perf_counter()
            try:
                report.stream_info = source.reconnect()
                report.successful_reconnects += 1
                report.reconnect_times_ms.append((time.perf_counter() - reconnect_started) * 1000)
            except VideoSourceError as exc:
                report.failed_reconnects += 1
                report.error = str(exc)
                logger.warning("event=reconnect_failed error=%s", exc)
    finally:
        report.elapsed_duration_s = time.monotonic() - started
        report.dropped_frames = int(getattr(source, "dropped_frames", 0))
        source.close()

    if report.error is None:
        if report.frames_received == 0:
            report.error = "no valid frames were received during the test"
            report.stop_reason = "no_frames"
        else:
            report.stop_reason = "duration_complete"
    return report


def connection_guidance(url: str, error: str | None) -> str:
    """Return actionable diagnostics without exposing credentials."""

    lower_error = (error or "").lower()
    if "timeout" in lower_error:
        lead = "La connessione o la lettura dello stream e' andata in timeout."
    elif "opencv" in lower_error or "open" in lower_error:
        lead = "OpenCV non e' riuscito ad aprire lo stream."
    else:
        lead = "Impossibile ricevere uno stream video valido."

    lines = [
        lead,
        f"URL verificato: {redact_url(url)}",
        "Cause possibili:",
        "- Huawei non raggiungibile o non collegato alla stessa rete del computer",
        "- IP o porta errati, oppure IP cambiato senza DHCP reservation",
        "- stream non avviato nell'app Android",
        "- client isolation/AP isolation attiva sul router",
        "- firewall del computer o porta non esposta dalla app",
        "- protocollo, codec o URL non supportati dall'app scelta",
        "- telefono entrato in sospensione o app terminata in background",
    ]
    return "\n".join(lines)


def format_number(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/d"
    return f"{value:.2f}{suffix}"

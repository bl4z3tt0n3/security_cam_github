from __future__ import annotations

import json
from pathlib import Path

from app.video.diagnostics import (
    connection_guidance,
    parse_fps,
    run_ffprobe,
    run_stream_test,
)
from app.video.fake_source import FakeVideoSource


def test_parse_ffprobe_frame_rates() -> None:
    assert parse_fps("30000/1001") == 30000 / 1001
    assert parse_fps("25") == 25.0
    assert parse_fps("0/0") is None
    assert parse_fps("N/A") is None


def test_ffprobe_json_is_parsed(monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = json.dumps(
            {
                "streams": [{"codec_type": "video", "codec_name": "h264"}],
                "format": {"format_name": "rtsp"},
            }
        )
        stderr = ""

    monkeypatch.setattr("app.video.diagnostics.shutil.which", lambda _name: "ffprobe.exe")
    monkeypatch.setattr("app.video.diagnostics.subprocess.run", lambda *args, **kwargs: Completed())
    result = run_ffprobe("rtsp://user:secret@camera.local/live")
    assert result.success is True
    assert result.streams[0]["codec_name"] == "h264"
    assert result.format_name == "rtsp"


def test_stream_report_collects_frames_without_unbounded_work() -> None:
    source = FakeVideoSource([__import__("numpy").zeros((2, 2, 3), dtype="uint8")], read_delay_s=0.005)
    report = run_stream_test(
        source,
        url="fake://camera",
        duration_s=0.05,
        read_timeout_s=0.02,
        reconnect_attempts=0,
        reconnect_delay_s=0,
    )
    assert report.error is None
    assert report.frames_received > 0
    assert report.actual_fps > 0
    assert report.mean_read_latency_ms is not None


def test_connection_guidance_does_not_leak_credentials() -> None:
    text = connection_guidance("rtsp://user:secret@camera.local/live", "connection timeout")
    assert "secret" not in text
    assert "stessa rete" in text

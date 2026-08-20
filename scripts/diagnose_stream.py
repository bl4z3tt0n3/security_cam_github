"""Short, detailed stream diagnostic including optional ffprobe metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from app.config import ConfigurationError
from app.logging_setup import configure_logging
from app.video.diagnostics import (
    connection_guidance,
    format_number,
    parse_fps,
    run_ffprobe,
    run_stream_test,
    video_stream_from_probe,
)
from app.video.base import redact_url
from scripts._common import add_runtime_arguments, add_target_arguments, build_source, resolve_target, print_stream_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a local RTSP/HTTP stream.")
    add_target_arguments(parser)
    add_runtime_arguments(parser, default_duration=10.0)
    parser.add_argument("--ffprobe-timeout", type=float, default=5.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        target = resolve_target(args)
        source = build_source(target, args)
    except ConfigurationError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2

    print("=== STREAM DIAGNOSTIC ===")
    print(f"target: {redact_url(target.url)}")
    print("ffprobe:")
    probe = run_ffprobe(target.url, timeout_s=args.ffprobe_timeout)
    if probe.success:
        video = video_stream_from_probe(probe)
        if video is None:
            print("  WARN: ffprobe did not report a video stream")
        else:
            print(f"  codec: {video.get('codec_name') or 'n/d'}")
            print(f"  resolution: {video.get('width') or 'n/d'}x{video.get('height') or 'n/d'}")
            fps = parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate"))
            print(f"  declared FPS: {fps:.2f}" if fps else "  declared FPS: n/d")
            print(f"  container: {probe.format_name or 'n/d'}")
    elif probe.available:
        print(f"  WARN: {probe.error or 'ffprobe failed'}")
        if probe.stderr:
            print(f"  detail: {probe.stderr}")
    else:
        print(f"  INFO: {probe.error}")

    print("OpenCV receive test:")
    report = run_stream_test(
        source,
        url=target.url,
        duration_s=args.duration,
        read_timeout_s=args.read_timeout or (target.config.video.read_timeout_seconds if target.config else 3.0),
        reconnect_attempts=args.reconnect_attempts,
        reconnect_delay_s=args.reconnect_delay,
    )
    if report.stream_info:
        print_stream_info(report.stream_info)
    print(f"received frames: {report.frames_received}")
    print(f"actual FPS: {report.actual_fps:.2f}")
    print(f"timeouts: {report.timeouts}")
    print(f"corrupt frames: {report.corrupt_frames}")
    print(f"disconnections: {report.disconnections}")
    print(f"reconnects: {report.successful_reconnects}/{report.reconnect_attempts}")
    print(f"dropped frames: {report.dropped_frames}")
    print(f"mean read/decode latency: {format_number(report.mean_read_latency_ms, ' ms')}")
    print(f"p95 read/decode latency: {format_number(report.p95_read_latency_ms, ' ms')}")
    print(f"mean reconnect time: {format_number(report.mean_reconnect_time_ms, ' ms')}")
    print("latency note: this is local capture/read latency, not end-to-end camera latency")

    if report.error:
        print("\nRESULT: FAIL")
        print(connection_guidance(target.url, report.error))
        return 1
    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

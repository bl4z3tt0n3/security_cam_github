"""Sustained stream receive test with bounded reconnect handling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from app.config import ConfigurationError
from app.logging_setup import configure_logging
from app.video.diagnostics import connection_guidance, format_number, run_stream_test
from app.video.base import redact_url
from scripts._common import add_runtime_arguments, add_target_arguments, build_source, resolve_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sustained local stream receive test.")
    add_target_arguments(parser)
    add_runtime_arguments(parser, default_duration=60.0)
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

    report = run_stream_test(
        source,
        url=target.url,
        duration_s=args.duration,
        read_timeout_s=args.read_timeout or (target.config.video.read_timeout_seconds if target.config else 3.0),
        reconnect_attempts=args.reconnect_attempts,
        reconnect_delay_s=args.reconnect_delay,
    )
    print("=== STREAM TEST ===")
    print(f"target: {redact_url(target.url)}")
    print(f"duration: {report.elapsed_duration_s:.2f}s")
    print(f"frames: {report.frames_received}")
    print(f"actual FPS: {report.actual_fps:.2f}")
    print(f"timeouts: {report.timeouts}")
    print(f"corrupt frames: {report.corrupt_frames}")
    print(f"disconnections: {report.disconnections}")
    print(f"successful reconnects: {report.successful_reconnects}")
    print(f"dropped frames: {report.dropped_frames}")
    print(f"mean read/decode latency: {format_number(report.mean_read_latency_ms, ' ms')}")
    print(f"p95 read/decode latency: {format_number(report.p95_read_latency_ms, ' ms')}")

    if report.error:
        print("\nRESULT: FAIL")
        print(connection_guidance(target.url, report.error))
        return 1
    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

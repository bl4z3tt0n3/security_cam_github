"""Run one independent camera worker and print live acquisition metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from app.config import ConfigurationError
from app.logging_setup import configure_logging
from app.video.base import redact_url
from app.video.sampler import FrameSampler
from app.video.worker import CameraWorker, WorkerState
from scripts._common import add_target_arguments, build_source, print_stream_info, resolve_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one independent local camera acquisition worker."
    )
    add_target_arguments(parser)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds; default is until Ctrl+C.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between metric lines (default: 1).",
    )
    parser.add_argument("--read-timeout", type=float, default=None)
    parser.add_argument("--open-timeout", type=float, default=None)
    parser.add_argument("--reconnect-attempts", type=int, default=None)
    parser.add_argument("--reconnect-delay", type=float, default=None)
    parser.add_argument("--backend", choices=("auto", "opencv", "ffmpeg"), default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def format_metric(value: float | None, suffix: str = "") -> str:
    return "n/d" if value is None else f"{value:.2f}{suffix}"


def main() -> int:
    args = parse_args()
    if args.duration is not None and args.duration <= 0:
        print("ARGUMENT ERROR: --duration must be greater than zero", file=sys.stderr)
        return 2
    if args.interval <= 0:
        print("ARGUMENT ERROR: --interval must be greater than zero", file=sys.stderr)
        return 2
    if args.reconnect_attempts is not None and args.reconnect_attempts < 0:
        print("ARGUMENT ERROR: --reconnect-attempts cannot be negative", file=sys.stderr)
        return 2
    if args.reconnect_delay is not None and args.reconnect_delay < 0:
        print("ARGUMENT ERROR: --reconnect-delay cannot be negative", file=sys.stderr)
        return 2

    configure_logging(args.log_level)
    try:
        target = resolve_target(args)
        source = build_source(target, args)
    except ConfigurationError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2

    video = target.config.video if target.config is not None else None
    read_timeout_s = args.read_timeout or (video.read_timeout_seconds if video else 3.0)
    open_timeout_s = args.open_timeout or (video.open_timeout_seconds if video else 5.0)
    worker = CameraWorker(
        target.camera.id if target.camera is not None else "camera",
        source,
        read_timeout_s=read_timeout_s,
        reconnect_delay_s=args.reconnect_delay
        if args.reconnect_delay is not None
        else (video.reconnect_delay_seconds if video else 2.0),
        max_reconnect_attempts=args.reconnect_attempts
        if args.reconnect_attempts is not None
        else (video.max_reconnect_attempts if video else 0),
        max_buffer_frames=video.max_buffer_frames if video else 1,
        stop_timeout_s=max(1.0, open_timeout_s + 1.0, read_timeout_s + 1.0),
    )
    target_fps = target.config.inference.person_detection_fps if target.config else 2.0
    sampler = FrameSampler(
        worker,
        target_fps=target_fps,
        input_wait_timeout_s=min(0.1, read_timeout_s),
        stop_timeout_s=max(1.0, read_timeout_s + 1.0),
    )

    print("=== CAMERA WORKER ===")
    print(f"target: {redact_url(target.url)}")
    print(f"sampling target FPS: {target_fps:.2f}")
    print("Press Ctrl+C to stop cleanly.")
    worker.start()
    sampler.start()

    started = time.monotonic()
    last_state: WorkerState | None = None
    exit_code = 0
    try:
        while True:
            sampler.get_latest(0.0)
            snapshot = worker.snapshot()
            sampler_snapshot = sampler.snapshot()
            if snapshot.state is not last_state:
                print(f"state: {snapshot.state.value}")
                if snapshot.state is WorkerState.RUNNING:
                    print("connection established")
                    info = getattr(source, "stream_info", None)
                    if info is not None:
                        print_stream_info(info)
                last_state = snapshot.state

            print(
                f"stream fps: {format_metric(snapshot.stream_fps)} | "
                f"decoded fps: {snapshot.decoded_fps:.2f} | "
                f"sampled fps: {sampler_snapshot.sampled_fps:.2f} | "
                f"reconnect count: {snapshot.reconnect_count} "
                f"(ok {snapshot.successful_reconnects}, failed {snapshot.failed_reconnects}) | "
                f"acquisition dropped: {snapshot.dropped_frames} | "
                f"sampling skipped: {sampler_snapshot.skipped_frames} | "
                f"sample output dropped: {sampler_snapshot.dropped_frames} | "
                f"input queue: {snapshot.queue_size}/{snapshot.max_buffer_frames} | "
                f"sample queue: {sampler_snapshot.queue_size}/{sampler_snapshot.max_buffer_frames} | "
                f"sampling latency: {format_metric(sampler_snapshot.mean_latency_ms, ' ms')}"
            )

            if snapshot.state is WorkerState.FAILED:
                if snapshot.last_error:
                    print(f"worker error: {snapshot.last_error}", file=sys.stderr)
                exit_code = 1
                break
            if args.duration is not None and time.monotonic() - started >= args.duration:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("shutdown requested")
    finally:
        sampler.stop(timeout_s=max(1.0, read_timeout_s + 1.0))
        worker.stop(timeout_s=max(1.0, open_timeout_s + 1.0, read_timeout_s + 1.0))

    final = worker.snapshot()
    final_sampler = sampler.snapshot()
    print(
        f"final state: {final.state.value} | frames: {final.frames_received} | "
        f"stream fps: {format_metric(final.stream_fps)} | "
        f"decoded fps: {final.decoded_fps:.2f} | "
        f"sampled fps: {final_sampler.sampled_fps:.2f} | "
        f"reconnect count: {final.reconnect_count} | "
        f"acquisition dropped: {final.dropped_frames} | "
        f"sampling skipped: {final_sampler.skipped_frames} | "
        f"sample output dropped: {final_sampler.dropped_frames} | "
        f"sample queue size: {final_sampler.queue_size} | "
        f"sampling latency: {format_metric(final_sampler.mean_latency_ms, ' ms')}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

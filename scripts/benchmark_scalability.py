"""Measure ordered 1--6 camera scalability levels locally or with real streams."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from app.benchmark import (
    ScalabilityCameraReport,
    ScalabilityLevelReport,
    ScalabilityReport,
    aggregate_scalability_resources,
)
from app.camera import CameraRuntime, MultiCameraRuntime
from app.config import ConfigurationError, MotionDetectionConfig, load_config, validate_stream_url
from app.inference import InferenceGate, PersonDetection, FakePersonDetector, create_person_detector
from app.metrics import ResourceSnapshot, read_resource_snapshot
from app.video.fake_source import FakeVideoSource
from app.video.motion import MotionDetector
from app.video.worker import WorkerState
from scripts._common import build_source


SCENARIOS = ("none", "one_person", "two_persons")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure ordered 1--6 camera scalability levels."
    )
    parser.add_argument("--mode", choices=("fake", "real"), default="fake")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-cameras", type=int, default=6)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--scenario", choices=SCENARIOS, default="none")
    parser.add_argument(
        "--parallel-inference",
        type=int,
        default=0,
        help="0=production auto policy; positive values force an inference gate width for benchmarking",
    )
    parser.add_argument(
        "--fake-inference-ms",
        type=float,
        default=0.0,
        help="synthetic detector latency used only in --mode fake",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.max_cameras <= 6:
        raise ValueError("--max-cameras must be between 1 and 6")
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.mode == "real" and args.config is None:
        raise ValueError("--mode real requires --config")
    if args.parallel_inference < 0:
        raise ValueError("--parallel-inference cannot be negative")
    if args.fake_inference_ms < 0:
        raise ValueError("--fake-inference-ms cannot be negative")


def _person_detection(value: int) -> list[PersonDetection]:
    if value == 0:
        return []
    return [
        PersonDetection(
            bbox=(4.0, 4.0, 28.0, 28.0),
            confidence=0.95,
            timestamp=datetime.now(timezone.utc),
        )
    ]


def _fake_runtimes(
    count: int,
    scenario: str,
    inference_ms: float = 0.0,
) -> tuple[list[CameraRuntime], FakePersonDetector]:
    def detect(frame: np.ndarray) -> list[PersonDetection]:
        if inference_ms:
            time.sleep(inference_ms / 1000.0)
        return _person_detection(int(frame[0, 0, 0]))

    detector = FakePersonDetector(callback=detect)
    runtimes: list[CameraRuntime] = []
    for index in range(count):
        active = scenario == "one_person" and index == 0
        active = active or (scenario == "two_persons" and index < 2)
        value = 1 if active else 0
        source = FakeVideoSource(
            [np.full((32, 32, 3), value, dtype=np.uint8)],
            url=f"fake://camera-{index + 1}",
            width=32,
            height=32,
            fps=20.0,
            read_delay_s=0.001,
        )
        runtimes.append(
            CameraRuntime(
                f"fake_{index + 1}",
                source,
                target_fps=10.0,
                read_timeout_s=0.05,
                reconnect_delay_s=0.0,
                max_reconnect_attempts=0,
            )
        )
    return runtimes, detector


def _motion_from_config(config: object) -> MotionDetector | None:
    motion = getattr(config, "motion_detection", MotionDetectionConfig())
    if not motion.enabled:
        return None
    return MotionDetector(
        pixel_threshold=motion.pixel_threshold,
        min_changed_fraction=motion.min_changed_fraction,
        resize_width=motion.resize_width,
        warmup_frames=motion.warmup_frames,
    )


def _real_runtimes(config_path: Path, count: int) -> tuple[list[CameraRuntime], object, str | None]:
    config = load_config(config_path)
    enabled = [camera for camera in config.cameras if camera.enabled]
    if len(enabled) < count:
        return [], None, f"only {len(enabled)} enabled cameras are configured"

    try:
        for camera in enabled[:count]:
            validate_stream_url(camera.stream_url)
    except ConfigurationError as exc:
        return [], None, str(exc)

    try:
        detection_config = config.person_detection.model_copy(update={"enabled": True})
        detector = create_person_detector(detection_config, model_root=SCRIPT_ROOT)
    except Exception as exc:
        return [], None, f"person detector unavailable: {type(exc).__name__}: {exc}"

    runtimes: list[CameraRuntime] = []
    args = SimpleNamespace(
        backend=None,
        open_timeout=None,
        read_timeout=None,
    )
    for camera in enabled[:count]:
        target = SimpleNamespace(
            url=validate_stream_url(camera.stream_url),
            config=config,
            camera=camera,
        )
        source = build_source(target, args)
        runtimes.append(
            CameraRuntime(
                camera.id,
                source,
                target_fps=config.inference.person_detection_fps,
                read_timeout_s=config.video.read_timeout_seconds,
                reconnect_delay_s=config.video.reconnect_delay_seconds,
                max_reconnect_attempts=config.video.max_reconnect_attempts,
                max_buffer_frames=config.video.max_buffer_frames,
                stop_timeout_s=max(
                    1.0,
                    config.video.open_timeout_seconds + 1.0,
                    config.video.read_timeout_seconds + 1.0,
                ),
                tracking_config=config.tracking,
                motion_detector=_motion_from_config(config),
            )
        )
    return runtimes, detector, None


def _unavailable_level(
    count: int,
    scenario: str,
    duration: float,
    warmup: float,
    reason: str,
) -> ScalabilityLevelReport:
    return ScalabilityLevelReport(
        camera_count=count,
        status="unavailable",
        scenario=scenario,
        duration_seconds=duration,
        warmup_seconds=warmup,
        resources=None,
        cameras=(),
        reason=reason,
    )


def _measure_level(
    runtimes: list[CameraRuntime],
    detector: object,
    *,
    count: int,
    mode: str,
    scenario: str,
    duration: float,
    warmup: float,
    parallel_inference: int = 0,
) -> ScalabilityLevelReport:
    gate = InferenceGate(max_parallel=parallel_inference) if parallel_inference > 0 else None
    fleet = MultiCameraRuntime(  # type: ignore[arg-type]
        runtimes,
        detector=detector,
        inference_gate=gate,
    )
    resource_samples: list[ResourceSnapshot] = []
    queue_max = {runtime.camera_id: 0 for runtime in runtimes}
    fleet.start()
    try:
        if warmup:
            time.sleep(warmup)
        started = time.monotonic()
        while time.monotonic() - started < duration:
            snapshots = fleet.snapshot()
            resource_samples.append(read_resource_snapshot())
            for camera_id, snapshot in snapshots.items():
                queue_max[camera_id] = max(
                    queue_max[camera_id],
                    snapshot.sampler.queue_size,
                    snapshot.worker.queue_size,
                )
            time.sleep(0.05)
        final = fleet.snapshot()
    finally:
        fleet.stop(timeout_s=2.0)

    reports: list[ScalabilityCameraReport] = []
    failed = False
    for runtime in runtimes:
        snapshot = final[runtime.camera_id]
        camera_failed = snapshot.worker.state is WorkerState.FAILED or snapshot.processed_samples == 0
        failed = failed or camera_failed
        reports.append(
            ScalabilityCameraReport(
                camera_id=runtime.camera_id,
                status="unavailable" if camera_failed else "measured",
                metrics=snapshot.metrics.to_dict(),
                queue_max=queue_max[runtime.camera_id],
                face_pipeline_concurrency=None,
                reason=snapshot.last_error if camera_failed else None,
            )
        )

    return ScalabilityLevelReport(
        camera_count=count,
        status="unavailable" if failed else "measured",
        scenario=scenario,
        duration_seconds=duration,
        warmup_seconds=warmup,
        resources=aggregate_scalability_resources(tuple(resource_samples)),
        cameras=tuple(reports),
        reason="one or more cameras failed or produced no samples" if failed else None,
    )


def run_scalability(
    *,
    mode: str,
    max_cameras: int,
    duration: float,
    warmup: float,
    scenario: str,
    config: Path | None = None,
    parallel_inference: int = 0,
    fake_inference_ms: float = 0.0,
) -> ScalabilityReport:
    if mode not in {"fake", "real"}:
        raise ValueError("mode must be fake or real")
    if not 1 <= max_cameras <= 6:
        raise ValueError("max_cameras must be between 1 and 6")
    if parallel_inference < 0:
        raise ValueError("parallel_inference cannot be negative")
    if fake_inference_ms < 0:
        raise ValueError("fake_inference_ms cannot be negative")
    levels: list[ScalabilityLevelReport] = []
    for count in range(1, max_cameras + 1):
        if mode == "fake":
            runtimes, detector = _fake_runtimes(count, scenario, fake_inference_ms)
            levels.append(
                _measure_level(
                    runtimes,
                    detector,
                    count=count,
                    mode=mode,
                    scenario=scenario,
                    duration=duration,
                    warmup=warmup,
                    parallel_inference=parallel_inference,
                )
            )
            continue

        assert config is not None
        runtimes, detector, reason = _real_runtimes(config, count)
        if reason is not None or detector is None:
            levels.append(
                _unavailable_level(count, scenario, duration, warmup, reason or "unknown prerequisite failure")
            )
            continue
        levels.append(
            _measure_level(
                runtimes,
                detector,
                count=count,
                mode=mode,
                scenario=scenario,
                duration=duration,
                warmup=warmup,
                parallel_inference=parallel_inference,
            )
        )

    return ScalabilityReport(
        execution="simulated" if mode == "fake" else "real",
        max_cameras_requested=max_cameras,
        scenario=scenario,
        levels=tuple(levels),
    )


def _print_report(report: ScalabilityReport) -> None:
    print(f"SCALABILITY execution: {report.execution}")
    print(f"scenario: {report.scenario}")
    for level in report.levels:
        print(
            f"LEVEL {level.camera_count}: {level.status} | "
            f"duration={level.duration_seconds:.1f}s"
        )
        if level.reason:
            print(f"  reason: {level.reason}")
        if level.resources is not None:
            print(
                f"  host: CPU mean={level.resources.cpu_mean_percent!s} peak="
                f"{level.resources.cpu_peak_percent!s} | RAM={level.resources.ram_percent!s}% | "
                f"GPU={level.resources.gpu_status}"
            )
        for camera in level.cameras:
            metrics = camera.metrics or {}
            print(
                f"  [{camera.camera_id}] {camera.status} | "
                f"sampled={metrics.get('sampled_fps', 'n/d')} | "
                f"person={metrics.get('person_detection_fps', 'n/d')} | "
                f"dropped={metrics.get('dropped_frames', 'n/d')} | "
                f"queue_max={camera.queue_max} | face_pipelines=n/d"
            )


def main() -> int:
    args = parse_args()
    try:
        _validate_args(args)
        report = run_scalability(
            mode=args.mode,
            max_cameras=args.max_cameras,
            duration=args.duration,
            warmup=args.warmup,
            scenario=args.scenario,
            config=args.config,
            parallel_inference=args.parallel_inference,
            fake_inference_ms=args.fake_inference_ms,
        )
    except (ConfigurationError, OSError, RuntimeError, ValueError) as exc:
        print(f"SCALABILITY ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

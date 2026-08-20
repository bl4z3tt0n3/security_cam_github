"""Benchmark local pipeline components with fake or explicitly configured adapters."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from app.benchmark import (
    BenchmarkExecution,
    BenchmarkMeasurement,
    BenchmarkReport,
    benchmark_operations,
    run_fake_benchmark,
)
from app.config import (
    AppConfig,
    ConfigurationError,
    PersonDetectionConfig,
    get_camera,
    load_config,
    validate_stream_url,
)
from app.face import (
    FaceDetector,
    FaceEmbedder,
    FaceMatcher,
    PersonStore,
    create_face_detector,
    create_face_embedder,
)
from app.inference import PersonDetector, create_person_detector
from app.tracking import CameraTrackingPipeline
from app.video.base import ReadStatus
from app.video.opencv_source import OpenCVVideoSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark person/face inference and local matching without "
            "claiming camera scalability."
        )
    )
    parser.add_argument("--mode", choices=("fake", "real"), default="fake")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--camera-id", default="benchmark-camera")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Local image for --mode real.")
    source.add_argument("--url", help="Concrete local stream URL for --mode real.")
    source.add_argument("--config", type=Path, help="Config containing a camera for --mode real.")
    parser.add_argument("--json", action="store_true", help="Print the complete report as JSON.")
    return parser.parse_args()


def _unavailable(name: str, detail: str) -> BenchmarkMeasurement:
    return BenchmarkMeasurement(
        name=name,
        status="unavailable",
        execution="unavailable",
        iterations=0,
        mean_ms=None,
        minimum_ms=None,
        maximum_ms=None,
        detail=detail,
    )


def _storage_path(config: AppConfig, value: Path) -> Path:
    return value if value.is_absolute() else SCRIPT_ROOT / value


def _real_frame(
    args: argparse.Namespace,
) -> tuple[np.ndarray, BenchmarkExecution, AppConfig | None]:
    import cv2

    if args.image is not None:
        frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"cannot read benchmark image: {args.image}")
        return frame, "real", None

    config: AppConfig | None = None
    if args.config is not None:
        config = load_config(args.config)
        camera = get_camera(config, args.camera_id)
        if not camera.enabled:
            raise ConfigurationError(f"camera '{camera.id}' is disabled")
        url = validate_stream_url(camera.stream_url)
    elif args.url is not None:
        url = validate_stream_url(args.url)
    else:
        raise ValueError("--mode real requires --image, --url, or --config")

    video = config.video if config is not None else None
    source = OpenCVVideoSource(
        url,
        backend=video.backend if video is not None else "auto",
        rtsp_transport=video.rtsp_transport if video is not None else "tcp",
        open_timeout_s=video.open_timeout_seconds if video is not None else 5.0,
        read_timeout_s=video.read_timeout_seconds if video is not None else 3.0,
        max_buffer_frames=video.max_buffer_frames if video is not None else 1,
    )
    try:
        source.open()
        result = source.read(video.read_timeout_seconds if video is not None else 3.0)
        if result.status is not ReadStatus.FRAME or result.packet is None:
            raise RuntimeError(result.message or f"stream returned {result.status.value}")
        return result.packet.frame.copy(), "real_stream", config
    finally:
        source.close()


def _real_benchmark(args: argparse.Namespace) -> BenchmarkReport:
    frame, execution, config = _real_frame(args)
    person_config = (
        config.person_detection if config is not None else PersonDetectionConfig(enabled=True)
    )
    camera_id = args.camera_id
    errors: dict[str, str] = {}
    person_detector: PersonDetector | None = None
    face_detector: FaceDetector | None = None
    embedder: FaceEmbedder | None = None
    matcher: FaceMatcher | None = None

    try:
        if person_config.enabled:
            person_detector = create_person_detector(person_config, model_root=SCRIPT_ROOT)
        else:
            errors["person_detection"] = "person_detection.enabled is false"
    except Exception as exc:
        errors["person_detection"] = f"{type(exc).__name__}: {exc}"

    if config is not None and config.face_detection.enabled:
        try:
            face_detector = create_face_detector(config.face_detection, model_root=SCRIPT_ROOT)
        except Exception as exc:
            errors["face_detection"] = f"{type(exc).__name__}: {exc}"
    else:
        errors["face_detection"] = "face_detection is not enabled in the configuration"

    if config is not None and config.recognition.enabled:
        try:
            embedder = create_face_embedder(config.recognition, model_root=SCRIPT_ROOT)
            persons_root = _storage_path(config, config.storage.persons_dir)
            matcher = FaceMatcher(
                embedder,
                PersonStore(persons_root),
                threshold=config.recognition.threshold,
            )
        except Exception as exc:
            errors["embedding"] = f"{type(exc).__name__}: {exc}"
            errors["matching"] = errors["embedding"]
    else:
        errors["embedding"] = "recognition is not enabled in the configuration"
        errors["matching"] = errors["embedding"]

    operations: dict[str, Callable[[], object]] = {}
    if person_detector is not None:
        operations["person_detection"] = lambda: person_detector.detect(frame)
    if face_detector is not None:
        operations["face_detection"] = lambda: face_detector.detect(frame)
    if embedder is not None:
        operations["embedding"] = lambda: embedder.embed(frame)
    if matcher is not None:
        operations["matching"] = lambda: matcher.match(frame)

    if (
        person_detector is not None
        and face_detector is not None
        and embedder is not None
        and matcher is not None
    ):
        pipeline = CameraTrackingPipeline(camera_id)

        def pipeline_operation() -> object:
            detections = person_detector.detect(frame)
            pipeline.update(detections)
            face_detector.detect(frame)
            embedder.embed(frame)
            return matcher.match(frame)

        operations["pipeline"] = pipeline_operation
    else:
        errors["pipeline"] = "one or more configured pipeline components are unavailable"

    report = benchmark_operations(
        operations,
        iterations=args.iterations,
        execution=execution,
        camera_id=camera_id,
    )
    measurements = list(report.results)
    present = {measurement.name for measurement in measurements}
    for name in ("person_detection", "face_detection", "embedding", "matching", "pipeline"):
        if name not in present:
            measurements.append(_unavailable(name, errors.get(name, "component unavailable")))
    return replace(report, results=tuple(measurements))


def _format(value: float | None, suffix: str = "") -> str:
    return "n/d" if value is None else f"{value:.3f}{suffix}"


def _gigabytes(value: int | None) -> float | None:
    return None if value is None else value / (1024**3)


def _print_human(report: BenchmarkReport) -> None:
    print(f"BENCHMARK execution: {report.execution}")
    print(f"iterations requested: {report.iterations_requested}")
    print("RESULTS")
    for result in report.results:
        if result.status == "measured":
            print(
                f"{result.name}: measured ({result.execution}) | "
                f"iterations={result.iterations} | mean={_format(result.mean_ms, ' ms')} | "
                f"min={_format(result.minimum_ms, ' ms')} | max={_format(result.maximum_ms, ' ms')}"
            )
        else:
            print(f"{result.name}: unavailable | {result.detail or 'no detail'}")

    resources = report.resources
    print("RESOURCES")
    print(f"CPU: {_format(resources.cpu_percent, ' %')}")
    ram = (
        f"{_format(_gigabytes(resources.ram_used_bytes), ' GB')} "
        f"/ {_format(_gigabytes(resources.ram_total_bytes), ' GB')}"
    )
    print(f"RAM: {ram} ({_format(resources.ram_percent, ' %')})")
    if resources.gpu_status == "available":
        print(
            f"GPU: {_format(resources.gpu_percent, ' %')} | VRAM: "
            f"{_format(_gigabytes(resources.vram_used_bytes), ' GB')} "
            f"/ {_format(_gigabytes(resources.vram_total_bytes), ' GB')}"
        )
    else:
        print(f"GPU: unavailable ({resources.gpu_detail})")

    if report.camera_metrics is not None:
        metrics = report.camera_metrics
        print("CAMERA METRICS")
        print(
            f"[{metrics.camera_id}] stream={_format(metrics.stream_fps, ' FPS')} | "
            f"decoded={_format(metrics.decoded_fps, ' FPS')} | "
            f"sampled={_format(metrics.sampled_fps, ' FPS')} | "
            f"person={_format(metrics.person_detection_fps, ' FPS')} | "
            f"face={_format(metrics.face_detection_fps, ' FPS')} | "
            f"queue={metrics.queue_size} | dropped={metrics.dropped_frames} | "
            f"latency={_format(metrics.processing_latency_ms, ' ms')}"
        )


def main() -> int:
    args = parse_args()
    if args.iterations < 1:
        print("ARGUMENT ERROR: --iterations must be a positive integer", file=sys.stderr)
        return 2
    try:
        report = (
            run_fake_benchmark(iterations=args.iterations, camera_id=args.camera_id)
            if args.mode == "fake"
            else _real_benchmark(args)
        )
    except (ConfigurationError, ValueError, OSError, RuntimeError) as exc:
        print(f"BENCHMARK ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

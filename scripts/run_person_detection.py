"""Run person detection on one or all configured local camera streams."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from app.camera import CameraRuntime, CameraRuntimeSnapshot, MultiCameraRuntime
from app.config import (
    ConfigurationError,
    MotionDetectionConfig,
    PersonDetectionConfig,
    TrackingConfig,
)
from app.inference import PersonDetectionError, create_person_detector
from app.logging_setup import configure_logging
from app.metrics import CameraMetricsSnapshot
from app.video.base import redact_url
from app.video.motion import MotionDetector
from app.video.worker import WorkerState
from scripts._common import (
    add_target_arguments,
    build_source,
    print_stream_info,
    resolve_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run replaceable person detection on sampled local camera frames."
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
    parser.add_argument(
        "--model",
        default=None,
        help="Override the configured checkpoint or model identifier.",
    )
    parser.add_argument(
        "--confidence", type=float, default=None, help="Override confidence threshold."
    )
    parser.add_argument(
        "--detector-backend",
        choices=("auto", "openvino", "yoloe", "onnx", "fake"),
        default=None,
        help="Override person-detector backend; --backend remains the video backend.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "cuda"), default=None)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default=None)
    parser.add_argument("--fallback-device", choices=("none", "cpu"), default=None)
    parser.add_argument("--imgsz", type=int, default=None, help="Override detector input size.")
    parser.add_argument(
        "--preview", action="store_true", help="Show detections in OpenCV windows."
    )
    parser.add_argument("--backend", choices=("auto", "opencv", "ffmpeg"), default=None)
    parser.add_argument("--read-timeout", type=float, default=None)
    parser.add_argument("--open-timeout", type=float, default=None)
    parser.add_argument("--reconnect-attempts", type=int, default=None)
    parser.add_argument("--reconnect-delay", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def format_metric(value: float | None, suffix: str = "") -> str:
    return "n/d" if value is None else f"{value:.2f}{suffix}"


def print_camera_metrics(snapshot: CameraMetricsSnapshot) -> None:
    """Print the stable per-camera metric contract used by this live runner."""

    print(
        f"[{snapshot.camera_id}] stream: {format_metric(snapshot.stream_fps, ' FPS')} | "
        f"decoded: {format_metric(snapshot.decoded_fps, ' FPS')} | "
        f"sampled: {format_metric(snapshot.sampled_fps, ' FPS')} | "
        f"person detector: {format_metric(snapshot.person_detection_fps, ' FPS')} | "
        f"inference: {format_metric(snapshot.person_detection_inference_ms, ' ms')} | "
        f"persons: {snapshot.last_person_count} last/{snapshot.persons_detected} total | "
        f"face detector: {format_metric(snapshot.face_detection_fps, ' FPS')} | "
        f"queue: {snapshot.queue_size} | dropped: {snapshot.dropped_frames} | "
        f"reconnects: {snapshot.reconnect_count} | "
        f"latency: {format_metric(snapshot.processing_latency_ms, ' ms')} | "
        f"active tracks: {snapshot.active_tracks} | "
        f"face rejects: {snapshot.face_quality_reject_count} | "
        f"recognition attempts: {snapshot.recognition_attempts} | "
        f"events: {snapshot.events_generated}"
    )


def _configured_detection(args: argparse.Namespace, target: object) -> PersonDetectionConfig:
    configured = getattr(target, "config", None)
    base = (
        configured.person_detection
        if configured is not None
        else PersonDetectionConfig(enabled=True)
    )
    overrides = base.model_dump()
    overrides["enabled"] = True if configured is None else base.enabled
    if args.model is not None:
        overrides["model"] = args.model
    if args.confidence is not None:
        overrides["confidence_threshold"] = args.confidence
    if args.device is not None:
        overrides["device"] = args.device
    if getattr(args, "detector_backend", None) is not None:
        overrides["backend"] = args.detector_backend
    if getattr(args, "precision", None) is not None:
        overrides["precision"] = args.precision
    if getattr(args, "fallback_device", None) is not None:
        overrides["fallback_device"] = args.fallback_device
    if getattr(args, "imgsz", None) is not None:
        overrides["image_size"] = args.imgsz
    return PersonDetectionConfig.model_validate(overrides)


def _draw_preview(cv2: object, frame: object, detections: object) -> object:
    """Draw detections without coupling preview behavior to the detector."""

    image = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.putText(
            image,
            f"{getattr(detection, 'label', 'person')} {detection.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )
    return image


def _camera_id(target: object) -> str:
    camera = getattr(target, "camera", None)
    return camera.id if camera is not None else "camera"


def _build_runtime(
    target: object,
    args: argparse.Namespace,
    detection_config: PersonDetectionConfig,
) -> tuple[CameraRuntime, object]:
    source = build_source(target, args)
    config = getattr(target, "config", None)
    video = config.video if config is not None else None
    tracking_config = config.tracking if config is not None else TrackingConfig()
    motion_detector = None
    motion_config = (
        config.motion_detection if config is not None else MotionDetectionConfig()
    )
    if motion_config.enabled:
        motion_detector = MotionDetector(
            pixel_threshold=motion_config.pixel_threshold,
            min_changed_fraction=motion_config.min_changed_fraction,
            resize_width=motion_config.resize_width,
            warmup_frames=motion_config.warmup_frames,
        )
    camera_id = _camera_id(target)
    read_timeout_s = (
        args.read_timeout
        if args.read_timeout is not None
        else (video.read_timeout_seconds if video else 3.0)
    )
    open_timeout_s = (
        args.open_timeout
        if args.open_timeout is not None
        else (video.open_timeout_seconds if video else 5.0)
    )
    reconnect_delay_s = (
        args.reconnect_delay
        if args.reconnect_delay is not None
        else (video.reconnect_delay_seconds if video else 2.0)
    )
    reconnect_attempts = (
        args.reconnect_attempts
        if args.reconnect_attempts is not None
        else (video.max_reconnect_attempts if video else 0)
    )
    target_fps = (
        config.inference.person_detection_fps
        if config is not None
        else detection_config.inference_fps
    )
    runtime = CameraRuntime(
        camera_id,
        source,
        target_fps=target_fps,
        read_timeout_s=read_timeout_s,
        reconnect_delay_s=reconnect_delay_s,
        max_reconnect_attempts=reconnect_attempts,
        max_buffer_frames=video.max_buffer_frames if video else 1,
        stop_timeout_s=max(1.0, open_timeout_s + 1.0, read_timeout_s + 1.0),
        tracking_config=tracking_config,
        motion_detector=motion_detector,
    )
    return runtime, source


def _print_runtime_snapshot(snapshot: CameraRuntimeSnapshot) -> None:
    print(
        f"[{snapshot.camera_id}] worker: {snapshot.worker.state.value} | "
        f"camera: {snapshot.state.value} | "
        f"processed: {snapshot.processed_samples} | "
        f"detector errors: {snapshot.detector_failures}"
    )
    if snapshot.last_error:
        print(f"[{snapshot.camera_id}] last error: {snapshot.last_error}", file=sys.stderr)
    print_camera_metrics(snapshot.metrics)


def _show_previews(cv2: object, runtimes: dict[str, CameraRuntime]) -> bool:
    """Render the latest result from each camera on the main/UI thread."""

    for camera_id, runtime in runtimes.items():
        frame, detections = runtime.latest_result()
        if frame is not None:
            cv2.imshow(
                f"Person detection [{camera_id}]",
                _draw_preview(cv2, frame, detections),
            )
    key = cv2.waitKey(1) & 0xFF
    return key not in (27, ord("q"))


def main() -> int:
    args = parse_args()
    if args.duration is not None and args.duration <= 0:
        print("ARGUMENT ERROR: --duration must be greater than zero", file=sys.stderr)
        return 2
    if args.interval <= 0:
        print("ARGUMENT ERROR: --interval must be greater than zero", file=sys.stderr)
        return 2
    if args.confidence is not None and not 0 <= args.confidence <= 1:
        print("ARGUMENT ERROR: --confidence must be between 0 and 1", file=sys.stderr)
        return 2
    if args.imgsz is not None and not 1 <= args.imgsz <= 2048:
        print("ARGUMENT ERROR: --imgsz must be between 1 and 2048", file=sys.stderr)
        return 2
    if args.reconnect_attempts is not None and args.reconnect_attempts < 0:
        print("ARGUMENT ERROR: --reconnect-attempts cannot be negative", file=sys.stderr)
        return 2
    if args.reconnect_delay is not None and args.reconnect_delay < 0:
        print("ARGUMENT ERROR: --reconnect-delay cannot be negative", file=sys.stderr)
        return 2

    configure_logging(args.log_level)
    try:
        targets = resolve_targets(args)
    except (ConfigurationError, ValueError) as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        detection_config = _configured_detection(args, targets[0])
        detector = create_person_detector(detection_config, model_root=SCRIPT_ROOT)
    except (PersonDetectionError, ValueError) as exc:
        print(f"MODEL CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2

    if detection_config.enabled is False:
        print("person detection disabled; no model loaded")
        return 0

    runtimes: list[CameraRuntime] = []
    sources: dict[str, object] = {}
    try:
        for target in targets:
            runtime, source = _build_runtime(target, args, detection_config)
            runtimes.append(runtime)
            sources[runtime.camera_id] = source
    except ConfigurationError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        for runtime in runtimes:
            runtime.stop()
        return 2

    cv2 = None
    if args.preview:
        try:
            import cv2 as cv2_module

            cv2 = cv2_module
        except ImportError as exc:
            print(f"PREVIEW ERROR: OpenCV is required for --preview: {exc}", file=sys.stderr)
            for runtime in runtimes:
                runtime.stop()
            return 2

    config = getattr(targets[0], "config", None)
    tracking_config = config.tracking if config is not None else TrackingConfig()
    target_fps = (
        config.inference.person_detection_fps
        if config is not None
        else detection_config.inference_fps
    )
    print("=== PERSON DETECTION ===")
    print(f"cameras: {', '.join(runtime.camera_id for runtime in runtimes)}")
    for target in targets:
        print(f"[{_camera_id(target)}] target: {redact_url(target.url)}")
    print(f"model: {detection_config.model}")
    print(f"backend: {getattr(detector, 'backend', detection_config.backend)}")
    print(f"prompt: {', '.join(getattr(detector, 'prompts', tuple(detection_config.prompts)))}")
    print(f"image size: {getattr(detector, 'image_size', detection_config.image_size)}")
    print(
        f"precision: {getattr(detector, 'precision', detection_config.precision)} | "
        f"fallback: {detection_config.fallback_device}"
    )
    print(f"confidence threshold: {detection_config.confidence_threshold:.2f}")
    print(
        f"device: {detector.device_used} ({detector.provider_used}) "
        f"verified={getattr(detector, 'device_verified', False)}"
    )
    print(f"sampling target FPS: {target_fps:.2f}")
    motion_config = config.motion_detection if config is not None else None
    print(
        "motion detection: "
        + ("enabled" if motion_config is not None and motion_config.enabled else "disabled")
    )
    print(
        "tracking: "
        f"IoU >= {tracking_config.iou_threshold:.2f}, "
        f"center distance <= {tracking_config.max_center_distance_px:.0f} px, "
        f"missing samples <= {tracking_config.max_missed_samples}"
    )
    print("shared detector: serialized per-camera access")
    print("Press Ctrl+C to stop cleanly.")

    fleet = MultiCameraRuntime(runtimes, detector=detector)
    fleet.start()
    started = time.monotonic()
    next_report = started
    last_worker_states: dict[str, WorkerState | None] = {
        runtime.camera_id: None for runtime in runtimes
    }
    exit_code = 0
    try:
        while True:
            now = time.monotonic()
            snapshots = fleet.snapshot()
            for camera_id, snapshot in snapshots.items():
                previous_state = last_worker_states[camera_id]
                if snapshot.worker.state is not previous_state:
                    print(f"[{camera_id}] state: {snapshot.worker.state.value}")
                    if snapshot.worker.state is WorkerState.RUNNING:
                        print(f"[{camera_id}] connection established")
                        info = getattr(sources[camera_id], "stream_info", None)
                        if info is not None:
                            print(f"[{camera_id}]", end=" ")
                            print_stream_info(info)
                    last_worker_states[camera_id] = snapshot.worker.state

            if args.preview and cv2 is not None and not _show_previews(cv2, fleet.cameras):
                break

            if now >= next_report:
                for snapshot in snapshots.values():
                    _print_runtime_snapshot(snapshot)
                next_report = now + args.interval

            if args.duration is not None and now - started >= args.duration:
                break
            if snapshots and all(
                snapshot.worker.state is WorkerState.FAILED
                and not snapshot.thread_alive
                for snapshot in snapshots.values()
            ):
                break
            time.sleep(min(0.05, args.interval))
    except KeyboardInterrupt:
        print("shutdown requested")
    finally:
        fleet.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()

    final_snapshots = fleet.snapshot()
    print(
        f"device final: {fleet.detector.device_used} ({fleet.detector.provider_used}) "
        f"verified={getattr(fleet.detector, 'device_verified', False)}"
    )
    for snapshot in final_snapshots.values():
        print(f"[{snapshot.camera_id}] final state: {snapshot.state.value}")
        _print_runtime_snapshot(snapshot)
        if snapshot.worker.state is WorkerState.FAILED or snapshot.detector_failures:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

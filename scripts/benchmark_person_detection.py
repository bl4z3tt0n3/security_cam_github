"""Benchmark YOLO26s CPU PyTorch and OpenVINO backends.

The benchmark is intentionally explicit about failed or unverified cases.  It
does not turn a missing Intel telemetry tool into a made-up utilization value,
and a GPU run is successful only when the detector reports a matching
``EXECUTION_DEVICES`` value after real inference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from app.inference import OpenVINOPersonDetector


@dataclass
class BenchmarkCase:
    name: str
    backend: str
    requested_device: str
    status: str = "error"
    error: str | None = None
    image: str | None = None
    iterations: int = 0
    warmup_iterations: int = 0
    load_export_ms: float | None = None
    warmup_ms: float | None = None
    mean_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    fps: float | None = None
    cpu_percent: float | None = None
    ram_used_bytes: int | None = None
    ram_total_bytes: int | None = None
    requested_precision: str | None = None
    effective_precision: str | None = None
    device: str | None = None
    execution_devices: list[str] | None = None
    device_verified: bool = False
    gpu_utilization_percent: float | None = None
    gpu_memory_used_bytes: int | None = None
    gpu_telemetry_reason: str | None = None


def _resource_sample() -> tuple[float | None, int | None, int | None]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return float(psutil.cpu_percent(interval=0.05)), int(memory.used), int(memory.total)
    except (ImportError, OSError, ValueError, AttributeError):
        return None, None, None


def _intel_gpu_telemetry() -> tuple[float | None, int | None, str]:
    """Return only values from a known Intel telemetry command, otherwise nulls."""

    for command in ("xpu-smi", "intel_gpu_top"):
        executable = shutil.which(command)
        if executable is None:
            continue
        # Command output differs across driver/tool versions; avoid parsing a
        # guessed format and report the tool as available for follow-up.
        return None, None, f"{command} detected but no stable parser is configured"
    return None, None, "No supported Intel GPU telemetry tool detected"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("at least one timing is required")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _resolve_image(path: Path | None) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"benchmark image not found: {resolved}")
        return resolved
    package_assets = Path(__import__("ultralytics").__file__).resolve().parent / "assets"
    default = package_assets / "bus.jpg"
    if default.is_file():
        return default
    raise FileNotFoundError("Ultralytics local asset bus.jpg was not found")


def _load_image(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the benchmark image loader") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read benchmark image: {path}")
    return image


def _timed_case(
    case: BenchmarkCase,
    *,
    image: np.ndarray,
    load: Callable[[], Any],
    predict: Callable[[Any, np.ndarray], Any],
    iterations: int,
    warmup: int,
    precision: str,
) -> Any:
    started = time.perf_counter()
    detector = load()
    case.load_export_ms = (time.perf_counter() - started) * 1000.0
    case.requested_precision = precision

    warmup_started = time.perf_counter()
    for _ in range(warmup):
        predict(detector, image)
    case.warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        predict(detector, image)
        timings.append((time.perf_counter() - started) * 1000.0)
    case.iterations = iterations
    case.warmup_iterations = warmup
    case.mean_ms = statistics.fmean(timings)
    case.p50_ms = _percentile(timings, 0.50)
    case.p95_ms = _percentile(timings, 0.95)
    case.fps = 1000.0 / case.mean_ms if case.mean_ms > 0 else None
    case.cpu_percent, case.ram_used_bytes, case.ram_total_bytes = _resource_sample()
    case.status = "ok"
    return detector


def _benchmark_pytorch(
    checkpoint: Path,
    image: np.ndarray,
    *,
    confidence: float,
    iterations: int,
    warmup: int,
) -> BenchmarkCase:
    case = BenchmarkCase(
        name="yolo26s-pytorch-cpu",
        backend="pytorch",
        requested_device="cpu",
        image=str(checkpoint),
    )

    def load() -> Any:
        try:
            from ultralytics import YOLO
        except (ImportError, OSError) as exc:
            raise RuntimeError("Ultralytics is required for the PyTorch benchmark") from exc
        return YOLO(str(checkpoint), verbose=False)

    def predict(model: Any, frame: np.ndarray) -> Any:
        return model.predict(
            source=frame,
            conf=confidence,
            imgsz=640,
            device="cpu",
            stream=False,
            verbose=False,
            save=False,
        )

    try:
        detector = _timed_case(
            case,
            image=image,
            load=load,
            predict=predict,
            iterations=iterations,
            warmup=warmup,
            precision="fp32",
        )
        case.effective_precision = "fp32"
        case.device = "cpu"
        case.device_verified = True
        del detector
    except Exception as exc:
        case.error = str(exc)
    return case


def _benchmark_openvino(
    checkpoint: Path,
    image: np.ndarray,
    *,
    device: str,
    precision: str,
    confidence: float,
    iterations: int,
    warmup: int,
) -> BenchmarkCase:
    case = BenchmarkCase(
        name=f"yolo26s-openvino-{device}-{precision}",
        backend="openvino",
        requested_device=device,
        image=str(checkpoint),
    )

    def load() -> OpenVINOPersonDetector:
        return OpenVINOPersonDetector(
            checkpoint,
            confidence_threshold=confidence,
            precision=precision,
            device=device,
            fallback_device="none",
            image_size=640,
            model_root=SCRIPT_ROOT,
        )

    def predict(detector: OpenVINOPersonDetector, frame: np.ndarray) -> Any:
        return detector.detect(frame)

    try:
        detector = _timed_case(
            case,
            image=image,
            load=load,
            predict=predict,
            iterations=iterations,
            warmup=warmup,
            precision=precision,
        )
        case.effective_precision = detector.precision
        case.device = detector.device_used
        case.execution_devices = list(detector.execution_devices)
        case.device_verified = detector.device_verified
        if device == "gpu":
            (
                case.gpu_utilization_percent,
                case.gpu_memory_used_bytes,
                case.gpu_telemetry_reason,
            ) = _intel_gpu_telemetry()
        detector.close()
        if not case.device_verified:
            case.status = "error"
            case.error = "OpenVINO device was not verified by EXECUTION_DEVICES"
    except Exception as exc:
        case.error = str(exc)
    return case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None, help="Local image; defaults to Ultralytics bus.jpg")
    parser.add_argument("--model", type=Path, default=Path("models/yolo26s.pt"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.json:
        try:
            from ultralytics.utils import LOGGER

            LOGGER.setLevel(logging.ERROR)
        except (ImportError, OSError):
            pass
    if args.iterations < 100:
        print("ARGUMENT ERROR: --iterations must be at least 100", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("ARGUMENT ERROR: --warmup cannot be negative", file=sys.stderr)
        return 2
    if not 0 <= args.confidence <= 1:
        print("ARGUMENT ERROR: --confidence must be between 0 and 1", file=sys.stderr)
        return 2

    checkpoint = args.model.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = SCRIPT_ROOT / checkpoint
    try:
        image_path = _resolve_image(args.image)
        image = _load_image(image_path)
    except Exception as exc:
        print(f"BENCHMARK SETUP ERROR: {exc}", file=sys.stderr)
        return 2

    cases: list[BenchmarkCase] = []
    if not args.skip_pytorch:
        cases.append(
            _benchmark_pytorch(
                checkpoint,
                image,
                confidence=args.confidence,
                iterations=args.iterations,
                warmup=args.warmup,
            )
        )
    cases.append(
        _benchmark_openvino(
            checkpoint,
            image,
            device="cpu",
            precision=args.precision,
            confidence=args.confidence,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    )
    if not args.skip_gpu:
        cases.append(
            _benchmark_openvino(
                checkpoint,
                image,
                device="gpu",
                precision=args.precision,
                confidence=args.confidence,
                iterations=args.iterations,
                warmup=args.warmup,
            )
        )

    payload = {
        "image": str(image_path),
        "model": str(checkpoint),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "cases": [asdict(case) for case in cases],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=== YOLO26 PERSON DETECTION BENCHMARK ===")
        print(f"image: {image_path}")
        for case in cases:
            if case.status != "ok":
                print(f"{case.name}: ERROR — {case.error}")
                continue
            print(
                f"{case.name}: mean={case.mean_ms:.2f} ms | p50={case.p50_ms:.2f} ms | "
                f"p95={case.p95_ms:.2f} ms | FPS={case.fps:.2f} | "
                f"load/export={case.load_export_ms:.2f} ms | warmup={case.warmup_ms:.2f} ms | "
                f"device={case.device} verified={case.device_verified}"
            )
            if case.gpu_telemetry_reason:
                print(f"  GPU telemetry: {case.gpu_telemetry_reason}")
    return 0 if all(case.status == "ok" for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

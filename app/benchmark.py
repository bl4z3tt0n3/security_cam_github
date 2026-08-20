"""Deterministic component benchmarks for the local-first pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import time
from typing import Literal

import numpy as np

from app.inference import FakePersonDetector, PersonDetection
from app.face import FaceDetection, FaceMatcher, FakeEmbedder, FakeFaceDetector, PersonStore
from app.metrics import (
    CameraMetrics,
    CameraMetricsSnapshot,
    ResourceSnapshot,
    read_resource_snapshot,
)
from app.tracking import CameraTrackingPipeline


BenchmarkStatus = Literal["measured", "unavailable"]
BenchmarkExecution = Literal["simulated", "real", "real_stream", "unavailable"]
ScalabilityExecution = Literal["simulated", "real"]
ScalabilityStatus = Literal["measured", "unavailable"]


@dataclass(frozen=True)
class BenchmarkMeasurement:
    """Timing summary for one component operation."""

    name: str
    status: BenchmarkStatus
    execution: BenchmarkExecution
    iterations: int
    mean_ms: float | None
    minimum_ms: float | None
    maximum_ms: float | None
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "execution": self.execution,
            "iterations": self.iterations,
            "mean_ms": self.mean_ms,
            "minimum_ms": self.minimum_ms,
            "maximum_ms": self.maximum_ms,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete benchmark output, including resource and camera snapshots."""

    execution: BenchmarkExecution
    iterations_requested: int
    results: tuple[BenchmarkMeasurement, ...]
    resources: ResourceSnapshot
    camera_metrics: CameraMetricsSnapshot | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "execution": self.execution,
            "iterations_requested": self.iterations_requested,
            "results": [result.to_dict() for result in self.results],
            "resources": self.resources.to_dict(),
            "camera_metrics": (
                self.camera_metrics.to_dict() if self.camera_metrics is not None else None
            ),
        }


@dataclass(frozen=True)
class ScalabilityResourceReport:
    """Aggregated host resources sampled during one scalability level."""

    cpu_mean_percent: float | None
    cpu_peak_percent: float | None
    ram_used_bytes: int | None
    ram_percent: float | None
    gpu_status: str
    gpu_mean_percent: float | None
    gpu_peak_percent: float | None
    vram_used_bytes: int | None
    vram_total_bytes: int | None
    gpu_detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "cpu_mean_percent": self.cpu_mean_percent,
            "cpu_peak_percent": self.cpu_peak_percent,
            "ram_used_bytes": self.ram_used_bytes,
            "ram_percent": self.ram_percent,
            "gpu_status": self.gpu_status,
            "gpu_mean_percent": self.gpu_mean_percent,
            "gpu_peak_percent": self.gpu_peak_percent,
            "vram_used_bytes": self.vram_used_bytes,
            "vram_total_bytes": self.vram_total_bytes,
            "gpu_detail": self.gpu_detail,
        }


@dataclass(frozen=True)
class ScalabilityCameraReport:
    """One camera's final metrics and bounded queue observation."""

    camera_id: str
    status: str
    metrics: dict[str, object] | None
    queue_max: int | None
    face_pipeline_concurrency: int | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "status": self.status,
            "metrics": self.metrics,
            "queue_max": self.queue_max,
            "face_pipeline_concurrency": self.face_pipeline_concurrency,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ScalabilityLevelReport:
    """One measured or unavailable camera-count level."""

    camera_count: int
    status: ScalabilityStatus
    scenario: str
    duration_seconds: float
    warmup_seconds: float
    resources: ScalabilityResourceReport | None
    cameras: tuple[ScalabilityCameraReport, ...]
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_count": self.camera_count,
            "status": self.status,
            "scenario": self.scenario,
            "duration_seconds": self.duration_seconds,
            "warmup_seconds": self.warmup_seconds,
            "resources": self.resources.to_dict() if self.resources is not None else None,
            "cameras": [camera.to_dict() for camera in self.cameras],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ScalabilityReport:
    """Complete ordered scalability report for one fake or real run."""

    execution: ScalabilityExecution
    max_cameras_requested: int
    scenario: str
    levels: tuple[ScalabilityLevelReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "execution": self.execution,
            "max_cameras_requested": self.max_cameras_requested,
            "scenario": self.scenario,
            "levels": [level.to_dict() for level in self.levels],
        }


def aggregate_scalability_resources(
    samples: tuple[ResourceSnapshot, ...],
) -> ScalabilityResourceReport | None:
    """Aggregate bounded host samples without retaining them in the report."""

    if not samples:
        return None

    def finite_values(values: list[float | None]) -> list[float]:
        return [value for value in values if value is not None and math.isfinite(value)]

    cpu_values = finite_values([sample.cpu_percent for sample in samples])
    gpu_values = finite_values([sample.gpu_percent for sample in samples])
    latest = samples[-1]
    return ScalabilityResourceReport(
        cpu_mean_percent=(sum(cpu_values) / len(cpu_values) if cpu_values else None),
        cpu_peak_percent=max(cpu_values) if cpu_values else None,
        ram_used_bytes=max(
            (sample.ram_used_bytes for sample in samples if sample.ram_used_bytes is not None),
            default=None,
        ),
        ram_percent=latest.ram_percent,
        gpu_status=("available" if any(sample.gpu_status == "available" for sample in samples) else "unavailable"),
        gpu_mean_percent=(sum(gpu_values) / len(gpu_values) if gpu_values else None),
        gpu_peak_percent=max(gpu_values) if gpu_values else None,
        vram_used_bytes=latest.vram_used_bytes,
        vram_total_bytes=latest.vram_total_bytes,
        gpu_detail=latest.gpu_detail,
    )


def _validate_iterations(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("iterations must be a positive integer")
    return value


def _measure(
    name: str,
    operation: Callable[[], object],
    *,
    iterations: int,
    execution: BenchmarkExecution,
    metrics: CameraMetrics | None,
) -> BenchmarkMeasurement:
    total_ms = 0.0
    minimum_ms: float | None = None
    maximum_ms: float | None = None
    completed = 0
    error: Exception | None = None
    for _ in range(iterations):
        started = time.perf_counter()
        try:
            operation()
        except Exception as exc:  # Report an unavailable component, then continue.
            error = exc
            break
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            error = RuntimeError("benchmark clock returned an invalid duration")
            break
        completed += 1
        total_ms += elapsed_ms
        minimum_ms = elapsed_ms if minimum_ms is None else min(minimum_ms, elapsed_ms)
        maximum_ms = elapsed_ms if maximum_ms is None else max(maximum_ms, elapsed_ms)

    if error is not None or completed == 0:
        detail = f"{type(error).__name__}: {error}" if error is not None else "no samples"
        return BenchmarkMeasurement(
            name=name,
            status="unavailable",
            execution="unavailable",
            iterations=completed,
            mean_ms=None,
            minimum_ms=None,
            maximum_ms=None,
            detail=detail,
        )

    if metrics is not None and name in {
        "person_detection",
        "face_detection",
        "embedding",
        "matching",
        "pipeline",
    }:
        for _ in range(completed):
            # The benchmark has already measured each invocation.  The metric
            # contract needs the same count/rate, while the detailed timing is
            # represented by the benchmark result itself.
            metrics.record_stage(name, total_ms / completed)  # type: ignore[arg-type]

    return BenchmarkMeasurement(
        name=name,
        status="measured",
        execution=execution,
        iterations=completed,
        mean_ms=total_ms / completed,
        minimum_ms=minimum_ms,
        maximum_ms=maximum_ms,
    )


def benchmark_operations(
    operations: Mapping[str, Callable[[], object]],
    *,
    iterations: int = 100,
    execution: BenchmarkExecution = "real",
    camera_id: str = "benchmark-camera",
) -> BenchmarkReport:
    """Benchmark named operations with bounded timing state.

    ``operations`` is intentionally a mapping of already-configured adapters;
    callers can use this for fake components, ONNX components, or a real-stream
    frame without changing the timing/report contract.
    """

    count = _validate_iterations(iterations)
    if execution not in {"simulated", "real", "real_stream"}:
        raise ValueError("execution must be simulated, real, or real_stream")
    metrics = CameraMetrics(camera_id)
    # Prime psutil's interval-free CPU sampler so the returned value reflects
    # the benchmark interval rather than the first-call sentinel value.
    read_resource_snapshot()
    results: list[BenchmarkMeasurement] = []
    for name, operation in operations.items():
        if not callable(operation):
            raise ValueError(f"benchmark operation '{name}' is not callable")
        result = _measure(
            name,
            operation,
            iterations=count,
            execution=execution,
            metrics=metrics,
        )
        results.append(result)
    return BenchmarkReport(
        execution=execution,
        iterations_requested=count,
        results=tuple(results),
        resources=read_resource_snapshot(),
        camera_metrics=metrics.snapshot(),
    )


def run_fake_benchmark(
    *,
    iterations: int = 100,
    camera_id: str = "benchmark-camera",
    persons_root: Path | str | None = None,
) -> BenchmarkReport:
    """Run the repeatable, offline benchmark used by the default CLI mode."""

    count = _validate_iterations(iterations)
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    for y in range(0, frame.shape[0], 10):
        for x in range(0, frame.shape[1], 10):
            if (x // 10 + y // 10) % 2:
                frame[y : y + 10, x : x + 10] = 160
    face_image = frame[20:120, 20:120]
    person_detector = FakePersonDetector(
        [
            PersonDetection(
                bbox=(0, 0, 160, 160),
                confidence=0.95,
                timestamp=datetime.now(timezone.utc),
            )
        ]
    )
    face_detector = FakeFaceDetector([FaceDetection((10, 10, 90, 90), 0.95)])
    embedder = FakeEmbedder(embedding_dimension=4)

    temporary_root: Path | None = None
    if persons_root is None:
        temporary_root = Path.cwd() / ".benchmark-tmp"
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        temporary_root.mkdir(parents=True, exist_ok=True)
        actual_persons_root = temporary_root / "persons"
    else:
        actual_persons_root = Path(persons_root)

    try:
        store = PersonStore(actual_persons_root)
        enrolled_embedding = embedder.embed(face_image)
        store.save(
            name="Benchmark Person",
            person_id="benchmark-person",
            embeddings=np.expand_dims(enrolled_embedding, axis=0),
            model=embedder.metadata,
        )
        matcher = FaceMatcher(embedder, store, threshold=0.8)
        pipeline = CameraTrackingPipeline(camera_id)

        def person_operation() -> object:
            return person_detector.detect(frame)

        def face_operation() -> object:
            return face_detector.detect(face_image)

        def embedding_operation() -> object:
            return embedder.embed(face_image)

        def matching_operation() -> object:
            return matcher.match(face_image)

        def pipeline_operation() -> object:
            detections = person_detector.detect(frame)
            pipeline.update(detections)
            face_detector.detect(face_image)
            embedder.embed(face_image)
            return matcher.match(face_image)

        return benchmark_operations(
            {
                "person_detection": person_operation,
                "face_detection": face_operation,
                "embedding": embedding_operation,
                "matching": matching_operation,
                "pipeline": pipeline_operation,
            },
            iterations=count,
            execution="simulated",
            camera_id=camera_id,
        )
    finally:
        if temporary_root is not None and temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)

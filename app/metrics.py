"""Bounded, thread-safe metrics for independent camera pipelines.

The video worker and sampler already expose acquisition snapshots.  This module
keeps the inference-side counters in a small reusable contract and can ingest
those snapshots without coupling the metrics layer to a concrete worker class.
Host telemetry is optional: CPU/RAM use psutil, NVIDIA uses nvidia-smi when
present, and Windows integrated/discrete GPUs use native GPU Performance
Counter CIM classes through PowerShell.  Shared GPU memory is deliberately
reported separately from system RAM and dedicated VRAM so it is never added to
RAM usage a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Literal


MetricStage = Literal[
    "motion_detection",
    "person_detection",
    "face_detection",
    "embedding",
    "matching",
    "pipeline",
]


def _finite_non_negative(value: float, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return normalized


def _optional_fps(value: float | None, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite_non_negative(value, label=label)


@dataclass(frozen=True)
class CameraMetricsSnapshot:
    """Point-in-time metrics for exactly one camera."""

    camera_id: str
    stream_fps: float | None
    decoded_fps: float
    sampled_fps: float
    person_detection_fps: float
    face_detection_fps: float
    dropped_frames: int
    reconnect_count: int
    queue_size: int
    processing_latency_ms: float | None
    active_tracks: int
    face_quality_reject_count: int
    recognition_attempts: int
    events_generated: int
    decoded_frames: int
    sampled_frames: int
    person_detection_calls: int
    face_detection_calls: int
    embedding_calls: int
    matching_calls: int
    pipeline_calls: int
    elapsed_seconds: float
    motion_detection_fps: float = 0.0
    motion_detection_calls: int = 0
    motion_positive_calls: int = 0
    motion_skipped_samples: int = 0
    last_motion_changed_fraction: float | None = None
    person_detection_inference_ms: float | None = None
    persons_detected: int = 0
    last_person_count: int = 0
    face_landmark_calls: int = 0
    face_landmark_inference_ms: float | None = None
    alignment_calls: int = 0
    alignment_inference_ms: float | None = None
    faces_detected: int = 0
    faces_rejected: int = 0
    embeddings_generated: int = 0
    known_recognitions: int = 0
    unknown_recognitions: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation with stable metric names."""

        return {
            "camera_id": self.camera_id,
            "stream_fps": self.stream_fps,
            "decoded_fps": self.decoded_fps,
            "sampled_fps": self.sampled_fps,
            "person_detection_fps": self.person_detection_fps,
            "person_detection_inference_ms": self.person_detection_inference_ms,
            "persons_detected": self.persons_detected,
            "last_person_count": self.last_person_count,
            "face_detection_fps": self.face_detection_fps,
            "face_landmark_calls": self.face_landmark_calls,
            "face_landmark_inference_ms": self.face_landmark_inference_ms,
            "alignment_calls": self.alignment_calls,
            "alignment_inference_ms": self.alignment_inference_ms,
            "faces_detected": self.faces_detected,
            "faces_rejected": self.faces_rejected,
            "embeddings_generated": self.embeddings_generated,
            "known_recognitions": self.known_recognitions,
            "unknown_recognitions": self.unknown_recognitions,
            "dropped_frames": self.dropped_frames,
            "reconnect_count": self.reconnect_count,
            "queue_size": self.queue_size,
            "processing_latency_ms": self.processing_latency_ms,
            "active_tracks": self.active_tracks,
            "face_quality_reject_count": self.face_quality_reject_count,
            "recognition_attempts": self.recognition_attempts,
            "events_generated": self.events_generated,
            "decoded_frames": self.decoded_frames,
            "sampled_frames": self.sampled_frames,
            "person_detection_calls": self.person_detection_calls,
            "face_detection_calls": self.face_detection_calls,
            "embedding_calls": self.embedding_calls,
            "matching_calls": self.matching_calls,
            "pipeline_calls": self.pipeline_calls,
            "elapsed_seconds": self.elapsed_seconds,
            "motion_detection_fps": self.motion_detection_fps,
            "motion_detection_calls": self.motion_detection_calls,
            "motion_positive_calls": self.motion_positive_calls,
            "motion_skipped_samples": self.motion_skipped_samples,
            "last_motion_changed_fraction": self.last_motion_changed_fraction,
        }


class _Counter:
    """A scalar counter; deliberately no per-sample history is retained."""

    def __init__(self) -> None:
        self.value = 0

    def add(self, amount: int = 1) -> None:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("counter increments must be non-negative integers")
        self.value += amount

    def set(self, amount: int) -> None:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("counter values must be non-negative integers")
        self.value = amount


class CameraMetrics:
    """Collect bounded metrics for one camera without blocking its pipeline.

    Counters are cumulative for the current lifetime/reset interval.  Rates are
    derived from the monotonic elapsed time, so no unbounded rolling sample
    queue is needed.  All mutation and snapshot reads are protected by one
    short-lived lock.
    """

    _STAGES: tuple[MetricStage, ...] = (
        "motion_detection",
        "person_detection",
        "face_detection",
        "embedding",
        "matching",
        "pipeline",
    )

    def __init__(
        self,
        camera_id: str,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        normalized_id = camera_id.strip()
        if not normalized_id:
            raise ValueError("camera_id cannot be empty")
        self._camera_id = normalized_id
        self._clock = clock
        self._lock = threading.RLock()
        self._started = self._read_clock()
        self._stream_fps: float | None = None
        self._decoded_fps_override: float | None = None
        self._sampled_fps_override: float | None = None
        self._dropped_frames = 0
        self._reconnect_count = 0
        self._queue_size = 0
        self._active_tracks = 0
        self._face_quality_reject_count = 0
        self._recognition_attempts = 0
        self._events_generated = 0
        self._motion_positive_calls = 0
        self._motion_skipped_samples = 0
        self._last_motion_changed_fraction: float | None = None
        self._persons_detected = 0
        self._last_person_count = 0
        self._faces_detected = 0
        self._faces_rejected = 0
        self._embeddings_generated = 0
        self._known_recognitions = 0
        self._unknown_recognitions = 0
        self._face_landmark_calls = 0
        self._face_landmark_latency_sum_ms = 0.0
        self._alignment_calls = 0
        self._alignment_latency_sum_ms = 0.0
        self._counters = {
            "decoded": _Counter(),
            "sampled": _Counter(),
            **{stage: _Counter() for stage in self._STAGES},
        }
        self._latency_sum_ms = {stage: 0.0 for stage in self._STAGES}

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def _read_clock(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise ValueError("metrics clock must return a finite value")
        return value

    def reset(self) -> None:
        """Reset counters and rates while retaining the camera identity."""

        with self._lock:
            self._started = self._read_clock()
            self._stream_fps = None
            self._decoded_fps_override = None
            self._sampled_fps_override = None
            self._dropped_frames = 0
            self._reconnect_count = 0
            self._queue_size = 0
            self._active_tracks = 0
            self._face_quality_reject_count = 0
            self._recognition_attempts = 0
            self._events_generated = 0
            self._motion_positive_calls = 0
            self._motion_skipped_samples = 0
            self._last_motion_changed_fraction = None
            self._persons_detected = 0
            self._last_person_count = 0
            self._faces_detected = 0
            self._faces_rejected = 0
            self._embeddings_generated = 0
            self._known_recognitions = 0
            self._unknown_recognitions = 0
            self._face_landmark_calls = 0
            self._face_landmark_latency_sum_ms = 0.0
            self._alignment_calls = 0
            self._alignment_latency_sum_ms = 0.0
            for counter in self._counters.values():
                counter.set(0)
            for stage in self._STAGES:
                self._latency_sum_ms[stage] = 0.0

    def record_decoded(self, count: int = 1) -> None:
        with self._lock:
            self._counters["decoded"].add(count)
            self._decoded_fps_override = None

    def record_sampled(self, count: int = 1) -> None:
        with self._lock:
            self._counters["sampled"].add(count)
            self._sampled_fps_override = None

    def record_stage(self, stage: MetricStage, latency_ms: float) -> None:
        """Record one completed inference/pipeline operation."""

        if stage not in self._STAGES:
            raise ValueError(f"unsupported metric stage: {stage}")
        elapsed = _finite_non_negative(latency_ms, label="latency_ms")
        with self._lock:
            self._counters[stage].add()
            self._latency_sum_ms[stage] += elapsed

    def record_person_detection(self, latency_ms: float, person_count: int = 0) -> None:
        if isinstance(person_count, bool) or not isinstance(person_count, int) or person_count < 0:
            raise ValueError("person_count must be a non-negative integer")
        self.record_stage("person_detection", latency_ms)
        with self._lock:
            self._persons_detected += person_count
            self._last_person_count = person_count

    def record_motion_detection(
        self,
        latency_ms: float,
        *,
        motion_detected: bool,
        skipped_person_detection: bool,
        changed_fraction: float | None = None,
    ) -> None:
        """Record one optional motion gate evaluation."""

        if not isinstance(motion_detected, bool):
            raise ValueError("motion_detected must be a boolean")
        if not isinstance(skipped_person_detection, bool):
            raise ValueError("skipped_person_detection must be a boolean")
        normalized_fraction: float | None = None
        if changed_fraction is not None:
            normalized_fraction = float(changed_fraction)
            if not math.isfinite(normalized_fraction) or not 0 <= normalized_fraction <= 1:
                raise ValueError("changed_fraction must be finite and between 0 and 1")

        self.record_stage("motion_detection", latency_ms)
        with self._lock:
            if motion_detected:
                self._motion_positive_calls += 1
            if skipped_person_detection:
                self._motion_skipped_samples += 1
            self._last_motion_changed_fraction = normalized_fraction

    def record_face_detection(self, latency_ms: float, face_count: int = 0) -> None:
        if isinstance(face_count, bool) or not isinstance(face_count, int) or face_count < 0:
            raise ValueError("face_count must be a non-negative integer")
        self.record_stage("face_detection", latency_ms)
        with self._lock:
            self._faces_detected += face_count

    def record_face_landmark(self, latency_ms: float) -> None:
        elapsed = _finite_non_negative(latency_ms, label="latency_ms")
        with self._lock:
            self._face_landmark_calls += 1
            self._face_landmark_latency_sum_ms += elapsed

    def record_alignment(self, latency_ms: float) -> None:
        elapsed = _finite_non_negative(latency_ms, label="latency_ms")
        with self._lock:
            self._alignment_calls += 1
            self._alignment_latency_sum_ms += elapsed

    def record_embedding(self, latency_ms: float) -> None:
        self.record_stage("embedding", latency_ms)

    def record_matching(self, latency_ms: float) -> None:
        self.record_stage("matching", latency_ms)

    def record_pipeline(self, latency_ms: float) -> None:
        self.record_stage("pipeline", latency_ms)

    def record_face_quality_reject(self, count: int = 1) -> None:
        with self._lock:
            normalized = self._validate_increment(count, "reject count")
            self._face_quality_reject_count += normalized
            self._faces_rejected += normalized

    def record_recognition_attempt(self, count: int = 1) -> None:
        with self._lock:
            self._recognition_attempts += self._validate_increment(count, "recognition count")

    def record_embedding_generated(self, count: int = 1) -> None:
        with self._lock:
            self._embeddings_generated += self._validate_increment(count, "embedding count")

    def record_recognition_result(self, status: str, count: int = 1) -> None:
        normalized = self._validate_increment(count, "recognition result count")
        if status not in {"known", "unknown"}:
            raise ValueError("recognition result status must be known or unknown")
        with self._lock:
            if status == "known":
                self._known_recognitions += normalized
            else:
                self._unknown_recognitions += normalized

    def record_event(self, count: int = 1) -> None:
        with self._lock:
            self._events_generated += self._validate_increment(count, "event count")

    @staticmethod
    def _validate_increment(value: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return value

    def set_active_tracks(self, count: int) -> None:
        with self._lock:
            self._active_tracks = self._validate_increment(count, "active track count")

    def set_queue_size(self, count: int) -> None:
        with self._lock:
            self._queue_size = self._validate_increment(count, "queue size")

    def set_stream_fps(self, value: float | None) -> None:
        with self._lock:
            self._stream_fps = _optional_fps(value, label="stream_fps")

    def set_transport_counters(self, *, dropped_frames: int, reconnect_count: int) -> None:
        with self._lock:
            self._dropped_frames = self._validate_increment(dropped_frames, "dropped_frames")
            self._reconnect_count = self._validate_increment(reconnect_count, "reconnect_count")

    def sync_worker_snapshot(self, worker_snapshot: Any, *, sampler_snapshot: Any = None) -> None:
        """Copy acquisition values from existing worker/sampler snapshots.

        ``Any`` is intentional: this keeps the metrics module independent from
        the video package while still accepting ``CameraWorkerSnapshot`` and
        ``FrameSamplerSnapshot`` instances.
        """

        decoded_frames = self._non_negative_int_attr(worker_snapshot, "frames_received", 0)
        decoded_fps = _optional_fps(
            getattr(worker_snapshot, "decoded_fps", getattr(worker_snapshot, "actual_fps", None)),
            label="decoded_fps",
        )
        dropped_frames = self._non_negative_int_attr(worker_snapshot, "dropped_frames", 0)
        reconnect_count = self._non_negative_int_attr(worker_snapshot, "reconnect_count", 0)
        worker_queue = self._non_negative_int_attr(worker_snapshot, "queue_size", 0)
        stream_fps = _optional_fps(getattr(worker_snapshot, "stream_fps", None), label="stream_fps")

        with self._lock:
            self._counters["decoded"].set(decoded_frames)
            self._decoded_fps_override = decoded_fps
            self._dropped_frames = dropped_frames
            self._reconnect_count = reconnect_count
            self._stream_fps = stream_fps
            if sampler_snapshot is None:
                self._queue_size = worker_queue
                self._sampled_fps_override = None
            else:
                sampled_frames = self._non_negative_int_attr(sampler_snapshot, "frames_sampled", 0)
                sampled_fps = _optional_fps(
                    getattr(sampler_snapshot, "sampled_fps", None), label="sampled_fps"
                )
                self._counters["sampled"].set(sampled_frames)
                self._sampled_fps_override = sampled_fps
                self._queue_size = self._non_negative_int_attr(
                    sampler_snapshot, "queue_size", worker_queue
                )

    @staticmethod
    def _non_negative_int_attr(source: Any, name: str, default: int) -> int:
        value = getattr(source, name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    def snapshot(self, *, now: float | None = None) -> CameraMetricsSnapshot:
        """Return a non-blocking point-in-time snapshot."""

        current = self._read_clock() if now is None else _finite_non_negative(now, label="now")
        with self._lock:
            elapsed = max(0.0, current - self._started)
            counters = {name: counter.value for name, counter in self._counters.items()}
            latency_sum = dict(self._latency_sum_ms)
            stream_fps = self._stream_fps
            decoded_override = self._decoded_fps_override
            sampled_override = self._sampled_fps_override
            dropped_frames = self._dropped_frames
            reconnect_count = self._reconnect_count
            queue_size = self._queue_size
            active_tracks = self._active_tracks
            face_quality_reject_count = self._face_quality_reject_count
            recognition_attempts = self._recognition_attempts
            events_generated = self._events_generated
            motion_positive_calls = self._motion_positive_calls
            motion_skipped_samples = self._motion_skipped_samples
            last_motion_changed_fraction = self._last_motion_changed_fraction
            persons_detected = self._persons_detected
            last_person_count = self._last_person_count
            face_landmark_calls = self._face_landmark_calls
            face_landmark_latency_sum_ms = self._face_landmark_latency_sum_ms
            alignment_calls = self._alignment_calls
            alignment_latency_sum_ms = self._alignment_latency_sum_ms
            faces_detected = self._faces_detected
            faces_rejected = self._faces_rejected
            embeddings_generated = self._embeddings_generated
            known_recognitions = self._known_recognitions
            unknown_recognitions = self._unknown_recognitions

        def rate(counter_name: str) -> float:
            return counters[counter_name] / elapsed if elapsed > 0 else 0.0

        def mean_latency(stage: MetricStage) -> float | None:
            count = counters[stage]
            return latency_sum[stage] / count if count else None

        return CameraMetricsSnapshot(
            camera_id=self._camera_id,
            stream_fps=stream_fps,
            decoded_fps=decoded_override if decoded_override is not None else rate("decoded"),
            sampled_fps=sampled_override if sampled_override is not None else rate("sampled"),
            motion_detection_fps=rate("motion_detection"),
            person_detection_fps=rate("person_detection"),
            person_detection_inference_ms=mean_latency("person_detection"),
            persons_detected=persons_detected,
            last_person_count=last_person_count,
            face_detection_fps=rate("face_detection"),
            face_landmark_calls=face_landmark_calls,
            face_landmark_inference_ms=(
                face_landmark_latency_sum_ms / face_landmark_calls
                if face_landmark_calls else None
            ),
            alignment_calls=alignment_calls,
            alignment_inference_ms=(
                alignment_latency_sum_ms / alignment_calls
                if alignment_calls else None
            ),
            faces_detected=faces_detected,
            faces_rejected=faces_rejected,
            embeddings_generated=embeddings_generated,
            known_recognitions=known_recognitions,
            unknown_recognitions=unknown_recognitions,
            dropped_frames=dropped_frames,
            reconnect_count=reconnect_count,
            queue_size=queue_size,
            processing_latency_ms=mean_latency("pipeline"),
            active_tracks=active_tracks,
            face_quality_reject_count=face_quality_reject_count,
            recognition_attempts=recognition_attempts,
            events_generated=events_generated,
            decoded_frames=counters["decoded"],
            sampled_frames=counters["sampled"],
            person_detection_calls=counters["person_detection"],
            face_detection_calls=counters["face_detection"],
            embedding_calls=counters["embedding"],
            matching_calls=counters["matching"],
            pipeline_calls=counters["pipeline"],
            elapsed_seconds=elapsed,
            motion_detection_calls=counters["motion_detection"],
            motion_positive_calls=motion_positive_calls,
            motion_skipped_samples=motion_skipped_samples,
            last_motion_changed_fraction=last_motion_changed_fraction,
        )


GpuStatus = Literal["available", "unavailable"]


@dataclass(frozen=True)
class ResourceSnapshot:
    """Global host-resource sample with explicit shared-memory semantics.

    ``ram_used_bytes`` is the operating system's system-RAM view.  Integrated
    GPU ``gpu_shared_used_bytes`` is a separate attribution signal and must not
    be added to RAM usage.  ``None`` means the metric is unavailable; it never
    means zero.
    """

    cpu_percent: float | None
    ram_used_bytes: int | None
    ram_total_bytes: int | None
    ram_percent: float | None
    gpu_status: GpuStatus
    gpu_percent: float | None
    vram_used_bytes: int | None
    vram_total_bytes: int | None
    gpu_detail: str
    process_rss_bytes: int | None = None
    ram_available_bytes: int | None = None
    gpu_name: str | None = None
    gpu_shared_used_bytes: int | None = None
    gpu_dedicated_used_bytes: int | None = None
    gpu_metric_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "cpu_percent": self.cpu_percent,
            "ram_used_bytes": self.ram_used_bytes,
            "ram_total_bytes": self.ram_total_bytes,
            "ram_available_bytes": self.ram_available_bytes,
            "ram_percent": self.ram_percent,
            "process_rss_bytes": self.process_rss_bytes,
            "gpu_status": self.gpu_status,
            "gpu_percent": self.gpu_percent,
            "gpu_name": self.gpu_name,
            "vram_used_bytes": self.vram_used_bytes,
            "vram_total_bytes": self.vram_total_bytes,
            "gpu_shared_used_bytes": self.gpu_shared_used_bytes,
            "gpu_dedicated_used_bytes": self.gpu_dedicated_used_bytes,
            "gpu_metric_source": self.gpu_metric_source,
            "gpu_detail": self.gpu_detail,
        }


@dataclass(frozen=True)
class _GpuSample:
    status: GpuStatus
    percent: float | None = None
    vram_used_bytes: int | None = None
    vram_total_bytes: int | None = None
    detail: str = "GPU metrics unavailable"
    name: str | None = None
    shared_used_bytes: int | None = None
    dedicated_used_bytes: int | None = None
    source: str | None = None


_GPU_CACHE_TTL_S = 1.0
_GPU_CACHE_LOCK = threading.Lock()
_gpu_cache_executable: str | None = None
_gpu_cache_at = 0.0
_gpu_cache_value: _GpuSample | None = None


def _read_nvidia_gpu_uncached(executable: str) -> _GpuSample:
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _GpuSample(
            "unavailable",
            detail=f"nvidia-smi unavailable: {type(exc).__name__}",
            source="nvidia-smi",
        )
    if result.returncode != 0:
        return _GpuSample(
            "unavailable",
            detail="nvidia-smi returned an error",
            source="nvidia-smi",
        )
    line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
    fields = [item.strip() for item in line.split(",")]
    if len(fields) < 4:
        return _GpuSample(
            "unavailable",
            detail="GPU metrics were not reported",
            source="nvidia-smi",
        )
    try:
        gpu_percent = _finite_non_negative(float(fields[-3]), label="gpu_percent")
        used_mib = _finite_non_negative(float(fields[-2]), label="vram_used_mib")
        total_mib = _finite_non_negative(float(fields[-1]), label="vram_total_mib")
    except ValueError:
        return _GpuSample(
            "unavailable",
            detail="GPU metrics contained non-numeric values",
            source="nvidia-smi",
        )
    gpu_name = ",".join(fields[:-3]).strip() or None
    used_bytes = int(round(used_mib * 1024 * 1024))
    return _GpuSample(
        "available",
        percent=min(100.0, gpu_percent),
        vram_used_bytes=used_bytes,
        vram_total_bytes=int(round(total_mib * 1024 * 1024)),
        detail="nvidia-smi: device utilization and dedicated VRAM",
        name=gpu_name,
        dedicated_used_bytes=used_bytes,
        source="nvidia-smi",
    )


def _optional_counter_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _read_windows_gpu_uncached(executable: str) -> _GpuSample:
    """Read process GPU counters available on Windows 10/11, including Iris Xe.

    The process' busiest GPU engine is used for utilization instead of summing
    engines, which could exceed 100%.  Shared and dedicated allocations are
    exposed independently.  These are Performance Counter values, not a DXGI
    reservation or a promise that memory is physically dedicated.
    """

    target_pid = os.getpid()
    script = f"""
$ErrorActionPreference = 'Stop'
$targetPid = {target_pid}
$gpuName = (Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name)
$engines = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | Where-Object {{ $_.Name -match \"pid_${{targetPid}}_\" }})
$engineMax = ($engines | Measure-Object -Property UtilizationPercentage -Maximum).Maximum
$mem = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object {{ $_.Name -match \"pid_${{targetPid}}_\" }})
$shared = ($mem | Measure-Object -Property SharedUsage -Sum).Sum
$dedicated = ($mem | Measure-Object -Property DedicatedUsage -Sum).Sum
[pscustomobject]@{{gpu_name=$gpuName; gpu_percent=$engineMax; shared_bytes=$shared; dedicated_bytes=$dedicated}} | ConvertTo-Json -Compress
""".strip()
    try:
        result = subprocess.run(
            [executable, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _GpuSample(
            "unavailable",
            detail=f"Windows GPU Performance Counters unavailable: {type(exc).__name__}",
            source="windows-gpu-performance-counters",
        )
    if result.returncode != 0:
        return _GpuSample(
            "unavailable",
            detail="Windows GPU Performance Counters returned an error",
            source="windows-gpu-performance-counters",
        )
    line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return _GpuSample(
            "unavailable",
            detail="Windows GPU Performance Counters returned invalid JSON",
            source="windows-gpu-performance-counters",
        )
    if not isinstance(payload, dict):
        return _GpuSample(
            "unavailable",
            detail="Windows GPU Performance Counters returned an invalid payload",
            source="windows-gpu-performance-counters",
        )
    name_value = payload.get("gpu_name")
    gpu_name = str(name_value).strip() if name_value is not None else ""
    if not gpu_name:
        return _GpuSample(
            "unavailable",
            detail="Windows did not report a GPU device",
            source="windows-gpu-performance-counters",
        )
    percent = _optional_counter_number(payload.get("gpu_percent"))
    shared = _optional_counter_number(payload.get("shared_bytes"))
    dedicated = _optional_counter_number(payload.get("dedicated_bytes"))
    percent_value = min(100.0, percent) if percent is not None else None
    shared_bytes = int(round(shared)) if shared is not None else None
    dedicated_bytes = int(round(dedicated)) if dedicated is not None else None
    return _GpuSample(
        "available",
        percent=percent_value,
        # Dedicated allocation is exposed through the historical VRAM-used
        # field for compatibility. Total VRAM is intentionally unknown for an
        # iGPU because shared system memory is dynamically managed by Windows.
        vram_used_bytes=dedicated_bytes,
        vram_total_bytes=None,
        detail=(
            "Windows GPU Performance Counters: process max-engine utilization; "
            "shared/dedicated allocations are reported separately"
        ),
        name=gpu_name,
        shared_used_bytes=shared_bytes,
        dedicated_used_bytes=dedicated_bytes,
        source="windows-gpu-performance-counters",
    )


def _cached_gpu_sample(cache_key: str, loader: Callable[[], _GpuSample]) -> _GpuSample:
    global _gpu_cache_executable, _gpu_cache_at, _gpu_cache_value
    now = time.monotonic()
    with _GPU_CACHE_LOCK:
        if (
            _gpu_cache_value is not None
            and _gpu_cache_executable == cache_key
            and now - _gpu_cache_at <= _GPU_CACHE_TTL_S
        ):
            return _gpu_cache_value
    # Do not hold the cache lock while spawning an external telemetry process.
    value = loader()
    with _GPU_CACHE_LOCK:
        _gpu_cache_executable = cache_key
        _gpu_cache_at = time.monotonic()
        _gpu_cache_value = value
    return value


def _read_gpu() -> _GpuSample:
    """Read optional GPU telemetry with a short TTL and Intel/Windows fallback."""

    nvidia = shutil.which("nvidia-smi")
    if nvidia is not None:
        value = _cached_gpu_sample(
            f"nvidia:{nvidia}",
            lambda: _read_nvidia_gpu_uncached(nvidia),
        )
        if value.status == "available" or sys.platform != "win32":
            return value

    if sys.platform == "win32":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is not None:
            return _cached_gpu_sample(
                f"windows:{powershell}",
                lambda: _read_windows_gpu_uncached(powershell),
            )
        detail = "nvidia-smi and PowerShell not found"
        return _GpuSample("unavailable", detail=detail)

    return _GpuSample("unavailable", detail="nvidia-smi not found")


def read_resource_snapshot() -> ResourceSnapshot:
    """Read host/process CPU/RAM and optional GPU metrics.

    Collection is best-effort and bounded.  Missing counters remain ``None``;
    callers must not interpret missing integrated-GPU telemetry as zero load.
    """

    cpu_percent: float | None = None
    ram_used_bytes: int | None = None
    ram_total_bytes: int | None = None
    ram_available_bytes: int | None = None
    ram_percent: float | None = None
    process_rss_bytes: int | None = None
    try:
        import psutil

        cpu_percent = _finite_non_negative(psutil.cpu_percent(interval=None), label="cpu_percent")
        memory = psutil.virtual_memory()
        ram_used_bytes = int(memory.used)
        ram_total_bytes = int(memory.total)
        ram_available_bytes = int(memory.available)
        ram_percent = _finite_non_negative(memory.percent, label="ram_percent")
        try:
            process_rss_bytes = int(psutil.Process(os.getpid()).memory_info().rss)
        except (psutil.Error, OSError, ValueError, AttributeError):
            process_rss_bytes = None
    except (ImportError, OSError, ValueError, AttributeError):
        pass

    gpu = _read_gpu()
    return ResourceSnapshot(
        cpu_percent=cpu_percent,
        ram_used_bytes=ram_used_bytes,
        ram_total_bytes=ram_total_bytes,
        ram_percent=ram_percent,
        process_rss_bytes=process_rss_bytes,
        ram_available_bytes=ram_available_bytes,
        gpu_status=gpu.status,
        gpu_percent=gpu.percent,
        gpu_name=gpu.name,
        vram_used_bytes=gpu.vram_used_bytes,
        vram_total_bytes=gpu.vram_total_bytes,
        gpu_shared_used_bytes=gpu.shared_used_bytes,
        gpu_dedicated_used_bytes=gpu.dedicated_used_bytes,
        gpu_metric_source=gpu.source,
        gpu_detail=gpu.detail,
    )


def format_bytes(value: int | None) -> str:
    """Format a byte count for the human-readable CLI report."""

    if value is None:
        return "n/d"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return "n/d"  # pragma: no cover - loop always returns

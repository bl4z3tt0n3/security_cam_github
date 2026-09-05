"""Immutable Windows-facing state for local person detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from pathlib import Path
from typing import Any, Literal

from app.config import AppConfig
from app.inference.base import PersonDetection
from app.tracking import CameraTrackingUpdate
from app.inference.prompts import normalize_prompts


DetectionBackend = Literal["auto", "openvino", "yoloe", "onnx", "fake"]
DetectionDevice = Literal["auto", "cpu", "gpu", "cuda"]
DetectionPrecision = Literal["fp16", "fp32"]
DetectionFallback = Literal["none", "cpu"]
DEFAULT_YOLOE_MODEL = "models/yoloe-26n-seg.pt"
DEFAULT_OPENVINO_MODEL = "models/yolo26s.pt"
SUPPORTED_YOLOE_MODEL_FILENAMES = (
    "yoloe-26n-seg.pt",
    "yoloe-26s-seg.pt",
    "yoloe-26l-seg.pt",
)
SUPPORTED_YOLOE_MODEL_NAMES = frozenset(
    name.casefold() for name in SUPPORTED_YOLOE_MODEL_FILENAMES
)
SUPPORTED_OPENVINO_MODEL_FILENAMES = ("yolo26s.pt", "yolo26n.pt")
SUPPORTED_OPENVINO_MODEL_NAMES = frozenset(
    name.casefold() for name in SUPPORTED_OPENVINO_MODEL_FILENAMES
)


class PersonDetectionStatus(str, Enum):
    """Lifecycle states exposed by the Windows inference panel."""

    DISABLED = "DISABLED"
    LOADING = "LOADING"
    READY = "READY"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    MODEL_MISSING = "MODEL_MISSING"

    @property
    def label(self) -> str:
        return {
            PersonDetectionStatus.DISABLED: "DISABILITATO",
            PersonDetectionStatus.LOADING: "CARICAMENTO",
            PersonDetectionStatus.READY: "PRONTO",
            PersonDetectionStatus.RUNNING: "IN ESECUZIONE",
            PersonDetectionStatus.ERROR: "ERRORE",
            PersonDetectionStatus.MODEL_MISSING: "MODELLO MANCANTE",
        }[self]


@dataclass(frozen=True)
class PersonDetectionSettings:
    """Validated settings shared by the controller and the contextual panel."""

    enabled: bool = False
    backend: DetectionBackend = "yoloe"
    model: str | None = DEFAULT_YOLOE_MODEL
    confidence_threshold: float = 0.5
    inference_fps: float = 2.0
    device: DetectionDevice = "auto"
    precision: DetectionPrecision = "fp16"
    fallback_device: DetectionFallback = "none"
    image_size: int = 640
    openvino_performance_mode: str = "latency"
    openvino_num_streams: int = 0
    openvino_num_requests: int = 0
    openvino_cpu_threads: int = 0
    max_process_ram_mb: int = 0
    classes: tuple[str, ...] = ("person",)
    show_boxes: bool = True
    prompts: tuple[str, ...] = ("person",)
    show_masks: bool = False

    def __post_init__(self) -> None:
        if self.model is not None:
            normalized_model = str(self.model).strip()
            object.__setattr__(self, "model", normalized_model or None)
        threshold = float(self.confidence_threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        object.__setattr__(self, "confidence_threshold", threshold)

        fps = float(self.inference_fps)
        if not math.isfinite(fps) or not 0 < fps <= 60:
            raise ValueError("inference_fps must be greater than zero and at most 60")
        object.__setattr__(self, "inference_fps", fps)

        normalized_device = str(self.device).strip().lower()
        if normalized_device not in {"auto", "cpu", "gpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, gpu, cuda")
        object.__setattr__(self, "device", normalized_device)
        normalized_backend = str(self.backend).strip().lower()
        if normalized_backend not in {"auto", "openvino", "yoloe", "onnx", "fake"}:
            raise ValueError("backend must be one of: auto, openvino, yoloe, onnx, fake")
        object.__setattr__(self, "backend", normalized_backend)
        normalized_precision = str(self.precision).strip().lower()
        if normalized_precision not in {"fp16", "fp32"}:
            raise ValueError("precision must be one of: fp16, fp32")
        object.__setattr__(self, "precision", normalized_precision)
        normalized_fallback = str(self.fallback_device).strip().lower()
        if normalized_fallback not in {"none", "cpu"}:
            raise ValueError("fallback_device must be one of: none, cpu")
        object.__setattr__(self, "fallback_device", normalized_fallback)
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int):
            raise ValueError("image_size must be a positive integer")
        if not 1 <= self.image_size <= 2048:
            raise ValueError("image_size must be between 1 and 2048")
        performance_mode = str(self.openvino_performance_mode).strip().lower()
        if performance_mode not in {"latency", "throughput"}:
            raise ValueError("openvino_performance_mode must be latency or throughput")
        object.__setattr__(self, "openvino_performance_mode", performance_mode)
        for field_name, maximum in (
            ("openvino_num_streams", 16),
            ("openvino_num_requests", 64),
            ("openvino_cpu_threads", 256),
            ("max_process_ram_mb", 131072),
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f"{field_name} must be between 0 and {maximum}")
        object.__setattr__(self, "classes", normalize_prompts(self.classes))
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "show_boxes", bool(self.show_boxes))
        normalized_prompts = normalize_prompts(self.prompts)
        if normalized_backend == "openvino":
            normalized_prompts = ("person",)
            object.__setattr__(self, "classes", ("person",))
            object.__setattr__(self, "show_masks", False)
        object.__setattr__(self, "prompts", normalized_prompts)
        object.__setattr__(self, "show_masks", bool(self.show_masks))

    @classmethod
    def from_app_config(
        cls,
        config: AppConfig,
        *,
        repo_root: Path | None = None,
    ) -> "PersonDetectionSettings":
        """Project the central YAML configuration into the Windows contract."""

        model = config.person_detection.model
        backend = config.person_detection.backend
        # The first Windows implementation persisted the absent ONNX model.
        # Resolve that legacy value in memory so an existing local config can
        # immediately use the installed YOLOE checkpoint; persistence performs
        # the atomic migration when the UI applies settings.
        if repo_root is not None and (model is None or model.lower().endswith(".onnx")):
            # Keep the Windows contract on the YOLOE path even when the
            # checkpoint is absent: the panel can then show the configured
            # missing .pt path and expose MODEL_MISSING instead of falling
            # back to the retired ONNX branch.
            model = DEFAULT_YOLOE_MODEL
            backend = "yoloe"

        return cls(
            enabled=config.person_detection.enabled,
            backend=backend,
            model=model,
            confidence_threshold=config.person_detection.confidence_threshold,
            inference_fps=config.inference.person_detection_fps,
            device=config.person_detection.device,
            precision=config.person_detection.precision,
            fallback_device=config.person_detection.fallback_device,
            image_size=config.person_detection.image_size,
            openvino_performance_mode=config.person_detection.openvino_performance_mode,
            openvino_num_streams=config.person_detection.openvino_num_streams,
            openvino_num_requests=config.person_detection.openvino_num_requests,
            openvino_cpu_threads=config.person_detection.openvino_cpu_threads,
            max_process_ram_mb=config.person_detection.max_process_ram_mb,
            classes=tuple(config.person_detection.classes),
            show_boxes=config.windows_ui.show_person_boxes,
            prompts=tuple(config.person_detection.prompts),
            show_masks=config.person_detection.show_masks,
        )


@dataclass(frozen=True)
class PersonDetectionSnapshot:
    """Point-in-time result and telemetry for the selected camera.

    ``latency_ms`` and ``batch_duration_ms`` represent wall-clock response
    latency. ``amortized_cost_ms`` is the batch cost divided by its inputs and
    is kept separate because it is a throughput metric, not user-visible
    response latency. ``scheduler_wait_ms`` is the delay after the target due
    time, while ``frame_age_ms`` measures local frame age when the result is
    published.
    """

    camera_id: str | None = None
    status: PersonDetectionStatus = PersonDetectionStatus.DISABLED
    message: str = "Rilevamento persone disabilitato"
    settings: PersonDetectionSettings = field(default_factory=PersonDetectionSettings)
    model_path: str | None = None
    requested_device: str = "auto"
    actual_device: str | None = None
    device_verified: bool = False
    provider: str | None = None
    backend: str | None = None
    precision: str | None = None
    inference_fps: float | None = None
    latency_ms: float | None = None
    batch_duration_ms: float | None = None
    amortized_cost_ms: float | None = None
    scheduler_wait_ms: float | None = None
    frame_age_ms: float | None = None
    person_count: int = 0
    detection_count: int = 0
    detections: tuple[PersonDetection, ...] = ()
    frame_sequence: int | None = None
    frame_timestamp: datetime | None = None
    source_width: int | None = None
    source_height: int | None = None
    result_monotonic: float | None = None
    detector_failures: int = 0
    error: str | None = None
    tracking_update: CameraTrackingUpdate | None = field(default=None, repr=False, compare=False)
    tracking_pipeline: Any | None = field(default=None, repr=False, compare=False)

    @property
    def model_name(self) -> str:
        if not self.model_path:
            return "—"
        return self.model_path.replace("\\", "/").rsplit("/", 1)[-1]

    @property
    def prompt_text(self) -> str:
        return ", ".join(self.settings.prompts)

    @property
    def is_obsolete(self) -> bool:
        if self.result_monotonic is None:
            return False
        return time_monotonic() - self.result_monotonic > max(
            1.0,
            2.0 / self.settings.inference_fps,
        )


def time_monotonic() -> float:
    """Small indirection that keeps snapshot age checks easy to test."""

    import time

    return time.monotonic()

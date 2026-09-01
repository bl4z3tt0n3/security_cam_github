"""Validated YAML configuration and environment-variable expansion."""

from __future__ import annotations

import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
SUPPORTED_STREAM_SCHEMES = frozenset({"http", "https", "rtsp", "rtsps"})
PLACEHOLDER_MARKERS = ("${", "INSERISCI_QUI", "INSERISCI QUI", "<STREAM_URL>")


class ConfigurationError(ValueError):
    """Raised when a configuration file cannot be loaded or used."""


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    name: str | None = None
    enabled: bool = True
    source_type: Literal["opencv"] = "opencv"
    stream_url: str | None = None
    rtsp_transport: Literal["auto", "tcp", "udp"] | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("camera id cannot be empty")
        return normalized

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class VideoConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    open_timeout_seconds: float = Field(default=5.0, gt=0, le=120)
    read_timeout_seconds: float = Field(default=3.0, gt=0, le=120)
    rtsp_transport: Literal["auto", "tcp", "udp"] = "tcp"
    reconnect_delay_seconds: float = Field(default=2.0, ge=0, le=300)
    max_reconnect_attempts: int = Field(default=0, ge=0, le=100)
    max_buffer_frames: int = Field(default=1, ge=1, le=10)
    backend: Literal["auto", "opencv", "ffmpeg"] = "auto"
    hardware_acceleration: Literal["none", "auto", "d3d11", "mfx"] = "none"


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    person_detection_fps: float = Field(default=2.0, gt=0, le=60)

    @field_validator("person_detection_fps")
    @classmethod
    def validate_person_detection_fps(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("person_detection_fps must be finite")
        return value


class PersonDetectionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    backend: Literal["auto", "openvino", "yoloe", "onnx", "fake"] = "auto"
    model: str | None = "models/yoloe-26n-seg.pt"
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    inference_fps: float = Field(default=2.0, gt=0, le=60)
    precision: Literal["fp16", "fp32"] = "fp16"
    device: Literal["auto", "cpu", "gpu", "cuda"] = "auto"
    fallback_device: Literal["none", "cpu"] = "none"
    image_size: int = Field(default=640, gt=0, le=2048)
    openvino_performance_mode: Literal["latency", "throughput"] = "latency"
    openvino_num_streams: int = Field(default=0, ge=0, le=16)
    openvino_num_requests: int = Field(default=0, ge=0, le=64)
    openvino_cpu_threads: int = Field(default=0, ge=0, le=256)
    max_process_ram_mb: int = Field(default=0, ge=0, le=131072)
    classes: list[str] = Field(default_factory=lambda: ["person"])
    prompts: list[str] = Field(default_factory=lambda: ["person"])
    show_masks: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_imgsz_alias(cls, value: Any) -> Any:
        """Accept Ultralytics' ``imgsz`` spelling without persisting two keys."""

        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "image_size" not in normalized and "imgsz" in normalized:
            normalized["image_size"] = normalized["imgsz"]
        return normalized

    @field_validator("classes", mode="before")
    @classmethod
    def normalize_classes(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            raw_values = value.split(",")
        elif isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raise ValueError("person_detection.classes must be a comma-separated string or list")

        normalized: list[str] = []
        seen: set[str] = set()
        for raw in raw_values:
            item = str(raw).strip()
            if not item:
                continue
            if len(item) > 64:
                raise ValueError("person_detection classes must be at most 64 characters")
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        if not normalized:
            raise ValueError("person_detection.classes must contain at least one category")
        if len(normalized) > 20:
            raise ValueError("person_detection.classes cannot contain more than 20 categories")
        return normalized

    @field_validator("prompts", mode="before")
    @classmethod
    def normalize_prompts(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            raw_values = value.split(",")
        elif isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raise ValueError("person_detection.prompts must be a comma-separated string or list")

        normalized: list[str] = []
        seen: set[str] = set()
        for raw in raw_values:
            prompt = str(raw).strip()
            if not prompt:
                continue
            if len(prompt) > 64:
                raise ValueError("person_detection prompts must be at most 64 characters")
            key = prompt.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(prompt)
        if not normalized:
            raise ValueError("person_detection.prompts must contain at least one category")
        if len(normalized) > 20:
            raise ValueError("person_detection.prompts cannot contain more than 20 categories")
        return normalized

    @field_validator("inference_fps")
    @classmethod
    def validate_inference_fps(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("inference_fps must be finite")
        return value

    @model_validator(mode="after")
    def validate_backend_classes(self) -> "PersonDetectionConfig":
        if self.backend == "openvino" and self.classes != ["person"]:
            raise ValueError("person_detection.classes must be ['person'] for the OpenVINO backend")
        return self

    @property
    def imgsz(self) -> int:
        """Compatibility alias used by Ultralytics-facing callers."""

        return self.image_size


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    iou_threshold: float = Field(default=0.30, ge=0, le=1)
    max_center_distance_px: float = Field(default=100.0, gt=0)
    max_missed_samples: int = Field(default=3, ge=0, le=300)

    @field_validator("iou_threshold", "max_center_distance_px")
    @classmethod
    def validate_finite_float(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("tracking thresholds must be finite")
        return value


class MotionDetectionConfig(BaseModel):
    """Cheap frame-difference gate for optional person detection reduction."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    pixel_threshold: int = Field(default=25, ge=1, le=255)
    min_changed_fraction: float = Field(default=0.01, ge=0, le=1)
    resize_width: int = Field(default=320, gt=0, le=2048)
    warmup_frames: int = Field(default=1, ge=1, le=30)

    @field_validator("min_changed_fraction")
    @classmethod
    def validate_changed_fraction(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("motion min_changed_fraction must be finite")
        return value


class FaceDetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    detector_id: str | None = None
    backend: Literal["auto", "onnxruntime", "openvino", "opencv_dnn", "fake"] = "auto"
    model: str | None = None
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    nms_threshold: float = Field(default=0.4, ge=0, le=1)
    device: Literal["auto", "cpu", "gpu", "cuda", "npu"] = "auto"
    inference_fps: float = Field(default=2.0, gt=0, le=60)
    openvino_performance_mode: Literal["latency", "throughput"] = "latency"
    openvino_cpu_threads: int = Field(default=0, ge=0, le=256)
    max_process_ram_mb: int = Field(default=0, ge=0, le=131072)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_face_keys(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        legacy_keys = tuple(
            key for key in ("backend_id", "model_id") if key in normalized
        )
        if legacy_keys:
            warnings.warn(
                "face_detection uses deprecated keys "
                + ", ".join(legacy_keys)
                + "; migrate to backend/model/detector_id",
                DeprecationWarning,
                stacklevel=2,
            )
        legacy_detector = normalized.pop("detector_id", None)
        legacy_backend = normalized.pop("backend_id", None)
        legacy_model_id = normalized.pop("model_id", None)
        if legacy_detector is not None:
            normalized["detector_id"] = legacy_detector
        if legacy_backend is not None:
            normalized["backend"] = legacy_backend
        if legacy_model_id is not None and normalized.get("model") is None:
            model_text = str(legacy_model_id)
            legacy_paths = {
                "scrfd_2.5g_kps": "models/face_detection/scrfd_2.5g_kps/scrfd_2.5g_bnkps.onnx",
                "face_detection_0205": "models/face_detection/face_detection_0205_fp32/face-detection-0205.xml",
                "face-detection-0205": "models/face_detection/face_detection_0205_fp32/face-detection-0205.xml",
                "yunet_2023mar": "models/face_detection/yunet_2023mar/face_detection_yunet_2023mar.onnx",
            }
            if model_text in legacy_paths:
                normalized["detector_id"] = (
                    "face_detection_0205"
                    if model_text == "face-detection-0205"
                    else model_text
                )
                normalized["model"] = legacy_paths[model_text]
            else:
                normalized["model"] = legacy_model_id
        for key in (
            "roi_mode",
            "show_boxes",
            "show_confidence",
            "show_detector",
            "show_inference_time",
            "show_landmarks",
        ):
            if key in normalized:
                warnings.warn(
                    f"face_detection.{key} is deprecated and has no effect; remove it",
                    DeprecationWarning,
                    stacklevel=2,
                )
                normalized.pop(key, None)
        return normalized

    @model_validator(mode="after")
    def validate_enabled_model(self) -> "FaceDetectionConfig":
        if self.enabled and not self.model:
            raise ValueError("face_detection.model is required when detection is enabled")
        if not math.isfinite(self.inference_fps):
            raise ValueError("face_detection.inference_fps must be finite")
        return self


class FaceLandmarksConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: str | None = None
    landmarker_id: str = "landmarks-regression-retail-0009"
    backend: Literal["openvino"] = "openvino"
    device: Literal["auto", "cpu", "gpu", "npu"] = "auto"
    openvino_performance_mode: Literal["latency", "throughput"] = "latency"
    openvino_cpu_threads: int = Field(default=0, ge=0, le=256)
    max_process_ram_mb: int = Field(default=0, ge=0, le=131072)

    @model_validator(mode="after")
    def validate_enabled_model(self) -> "FaceLandmarksConfig":
        if self.enabled and not self.model:
            raise ValueError("face_landmarks.model is required when landmarks are enabled")
        return self


class FaceQualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_width: int = Field(default=80, gt=0)
    min_height: int = Field(default=80, gt=0)
    blur_threshold: float = Field(default=40, ge=0)
    min_brightness: float = Field(default=30, ge=0, le=255)
    max_brightness: float = Field(default=225, ge=0, le=255)
    min_confidence: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_brightness_range(self) -> "FaceQualityConfig":
        if self.min_brightness > self.max_brightness:
            raise ValueError("face quality min_brightness cannot exceed max_brightness")
        return self


class RecognitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    recognizer_id: str | None = None
    backend: Literal["auto", "onnxruntime", "openvino", "fake"] = "auto"
    model: str | None = None
    model_version: str = Field(default="1", min_length=1)
    device: Literal["auto", "cpu", "gpu", "cuda", "npu"] = "auto"
    threshold: float | None = Field(default=None, ge=0)
    min_confirmations: int = Field(default=2, ge=1)
    confirmation_window_seconds: float = Field(default=10.0, gt=0, le=300)
    inference_fps: float = Field(default=1.0, gt=0, le=30)
    openvino_performance_mode: Literal["latency", "throughput"] = "latency"
    openvino_cpu_threads: int = Field(default=0, ge=0, le=256)
    max_process_ram_mb: int = Field(default=0, ge=0, le=131072)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_recognition_keys(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        legacy_keys = tuple(key for key in ("backend_id", "model_id") if key in normalized)
        if legacy_keys:
            warnings.warn(
                "recognition uses deprecated keys "
                + ", ".join(legacy_keys)
                + "; migrate to backend/model/recognizer_id",
                DeprecationWarning,
                stacklevel=2,
            )
        legacy_model_id = normalized.pop("model_id", None)
        legacy_backend = normalized.pop("backend_id", None)
        if legacy_backend is not None:
            normalized["backend"] = legacy_backend
        if legacy_model_id is not None and normalized.get("model") is None:
            model_text = str(legacy_model_id)
            legacy_paths = {
                "face-reidentification-retail-0095": "models/face_embedding/face-reidentification-retail-0095/face-reidentification-retail-0095.xml",
                "facenet-20180402-vggface2": "models/face_embedding/facenet-20180402-vggface2.onnx",
                "arcface-resnet50-webface600k": "models/face_embedding/arcface-resnet50-webface600k.onnx",
            }
            if model_text in legacy_paths:
                normalized["recognizer_id"] = model_text
                normalized["model"] = legacy_paths[model_text]
            else:
                normalized["model"] = legacy_model_id
        return normalized

    @model_validator(mode="after")
    def validate_enabled_model(self) -> "RecognitionConfig":
        if self.enabled and not self.model:
            raise ValueError("recognition.model is required when recognition is enabled")
        if not math.isfinite(self.confirmation_window_seconds) or not math.isfinite(self.inference_fps):
            raise ValueError("recognition timing values must be finite")
        return self


class EventsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    save_snapshot: bool = True
    known_person_cooldown_seconds: float = Field(default=30, ge=0)
    unknown_person_cooldown_seconds: float = Field(default=15, ge=0)


class RecordingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    database_dir: Path = Path("database")
    persons_dir: Path = Path("persons")
    enrollment_dir: Path = Path("enrollment")
    events_dir: Path = Path("events")
    snapshots_dir: Path = Path("snapshots")
    recordings_dir: Path = Path("recordings")
    logs_dir: Path = Path("logs")
    models_dir: Path = Path("models")


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    level: str = "INFO"


class HardwareOptimizationConfig(BaseModel):
    """Conservative profile for a local Intel i7 + Iris Xe + 16 GB system."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    profile: Literal["none", "intel_iris_xe"] = "none"
    adaptive_person_detection: bool = True
    force_face_cpu: bool = True
    decode_acceleration: Literal["auto", "none", "mfx", "d3d11"] = "auto"
    gpu_performance_mode: Literal["latency", "throughput"] = "throughput"
    gpu_streams: int = Field(default=2, ge=0, le=8)
    gpu_num_requests: int = Field(default=2, ge=0, le=16)
    cpu_threads: int = Field(default=0, ge=0, le=256)
    max_process_ram_mb: int = Field(default=6144, ge=1024, le=32768)
    background_preview_fps: float = Field(default=5.0, gt=0, le=30)
    background_preview_max_width: int = Field(default=480, ge=160, le=1920)

    @field_validator("background_preview_fps")
    @classmethod
    def validate_background_preview_fps(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("hardware_optimization.background_preview_fps must be finite")
        return value


class WindowsUiConfig(BaseModel):
    """Configuration that affects only the local Windows monitor UI."""

    model_config = ConfigDict(extra="ignore")

    display_fps: float = Field(default=15.0, gt=0, le=60)
    start_maximized: bool = True
    remember_window_geometry: bool = True
    show_person_boxes: bool = True
    background_preview_fps: float = Field(default=5.0, gt=0, le=30)
    background_preview_max_width: int = Field(default=480, ge=160, le=1920)

    @field_validator("display_fps")
    @classmethod
    def validate_display_fps(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("windows_ui.display_fps must be finite")
        return value


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cameras: list[CameraConfig] = Field(default_factory=list)
    video: VideoConfig = Field(default_factory=VideoConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    person_detection: PersonDetectionConfig = Field(default_factory=PersonDetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    motion_detection: MotionDetectionConfig = Field(default_factory=MotionDetectionConfig)
    face_detection: FaceDetectionConfig = Field(default_factory=FaceDetectionConfig)
    face_landmarks: FaceLandmarksConfig = Field(default_factory=FaceLandmarksConfig)
    face_quality: FaceQualityConfig = Field(default_factory=FaceQualityConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    hardware_optimization: HardwareOptimizationConfig = Field(
        default_factory=HardwareOptimizationConfig
    )
    windows_ui: WindowsUiConfig = Field(default_factory=WindowsUiConfig)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_inference_fps(cls, value: Any) -> Any:
        """Resolve the legacy nested FPS key before Pydantic validation.

        ``person_detection.inference_fps`` predates the dedicated ``inference``
        section. Existing local configurations remain valid, while an explicit
        ``inference.person_detection_fps`` always takes precedence.
        """

        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        if "inference" in normalized:
            raw_inference = normalized["inference"]
            if not isinstance(raw_inference, dict):
                return normalized
            inference = dict(raw_inference)
        else:
            inference = {}

        if "person_detection_fps" not in inference:
            legacy = normalized.get("person_detection")
            if isinstance(legacy, dict) and "inference_fps" in legacy:
                inference["person_detection_fps"] = legacy["inference_fps"]

        normalized["inference"] = inference
        return normalized

    @model_validator(mode="after")
    def mirror_effective_inference_fps(self) -> "AppConfig":
        """Keep the legacy field readable with the effective canonical value."""

        self.person_detection.inference_fps = self.inference.person_detection_fps
        return self

    @model_validator(mode="after")
    def apply_requested_hardware_profile(self) -> "AppConfig":
        """Apply opt-in hardware policy after all canonical settings are validated."""

        from app.hardware import apply_hardware_profile

        return apply_hardware_profile(self)

    @model_validator(mode="after")
    def validate_face_dependencies(self) -> "AppConfig":
        """Reject face stages that cannot be reached by the opt-in pipeline."""

        if self.recognition.enabled and not self.face_detection.enabled:
            raise ValueError(
                "recognition.enabled requires face_detection.enabled=true"
            )
        if self.face_landmarks.enabled and not self.face_detection.enabled:
            raise ValueError(
                "face_landmarks.enabled requires face_detection.enabled=true"
            )
        return self

    @field_validator("cameras")
    @classmethod
    def validate_unique_camera_ids(cls, cameras: list[CameraConfig]) -> list[CameraConfig]:
        ids = [camera.id for camera in cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("camera ids must be unique")
        return cameras


def expand_environment_variables(text: str, environ: dict[str, str] | None = None) -> str:
    """Expand ``${NAME}`` while preserving unresolved variables as placeholders."""

    values = os.environ if environ is None else environ

    def replace(match: re.Match[str]) -> str:
        return values.get(match.group(1), match.group(0))

    return _ENV_PATTERN.sub(replace, text)


def load_config(path: Path | str) -> AppConfig:
    """Load and validate a YAML configuration file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file not found: {config_path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
        raw_data: Any = yaml.safe_load(expand_environment_variables(raw_text))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read configuration {config_path}: {exc}") from exc

    if raw_data is None:
        raw_data = {}
    if not isinstance(raw_data, dict):
        raise ConfigurationError("configuration root must be a YAML mapping")

    try:
        return AppConfig.model_validate(raw_data)
    except Exception as exc:  # Pydantic's validation exception is intentionally wrapped.
        raise ConfigurationError(f"invalid configuration {config_path}: {exc}") from exc


def is_placeholder_url(url: str | None) -> bool:
    """Return whether a URL is absent or still contains the documented placeholder."""

    if url is None:
        return True
    value = url.strip()
    return not value or any(marker in value for marker in PLACEHOLDER_MARKERS)


def validate_stream_url(url: str | None) -> str:
    """Validate a concrete local stream URL and return its trimmed form."""

    if is_placeholder_url(url):
        raise ConfigurationError(
            "stream URL is not configured; set CAMERA_HUAWEI_URL or pass --url"
        )

    assert url is not None
    normalized = url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme.lower() not in SUPPORTED_STREAM_SCHEMES:
        supported = ", ".join(sorted(SUPPORTED_STREAM_SCHEMES))
        raise ConfigurationError(
            f"unsupported stream URL scheme '{parsed.scheme or '<missing>'}'; "
            f"supported schemes: {supported}"
        )
    if not parsed.hostname:
        raise ConfigurationError("stream URL has no host name or IP address")
    return normalized


def get_camera(config: AppConfig, camera_id: str) -> CameraConfig:
    """Find one configured camera by id."""

    for camera in config.cameras:
        if camera.id == camera_id:
            return camera
    raise ConfigurationError(f"camera '{camera_id}' is not configured")


def ensure_runtime_directories(root: Path, storage: StorageConfig) -> list[Path]:
    """Create configured runtime directories and return the paths created/existing."""

    directories = [
        storage.database_dir,
        storage.persons_dir,
        storage.events_dir,
        storage.snapshots_dir,
        storage.recordings_dir,
        storage.logs_dir,
        storage.models_dir,
    ]
    resolved: list[Path] = []
    for directory in directories:
        candidate = directory if directory.is_absolute() else root / directory
        candidate.mkdir(parents=True, exist_ok=True)
        resolved.append(candidate)
    return resolved

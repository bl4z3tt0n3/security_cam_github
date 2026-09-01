"""Construction of the configured person detector with lazy model loading."""

from __future__ import annotations

from pathlib import Path

from app.config import PersonDetectionConfig

from .base import DisabledPersonDetector, PersonDetector
from .fake import FakePersonDetector
from .onnx import OnnxPersonDetector
from .openvino import OpenVINOPersonDetector
from .prompts import normalize_prompts
from .yoloe import YoloEPersonDetector, YoloESegmentationDetector


def _resolve_onnx_model(model: str, model_root: Path | None) -> Path:
    model_path = Path(model).expanduser()
    if not model_path.is_absolute():
        model_path = (model_root or Path.cwd()) / model_path
    return model_path


def _resolve_yoloe_model(model: str, model_root: Path | None) -> str:
    """Prefer a local file, otherwise preserve official model identifiers."""

    model_path = Path(model).expanduser()
    if model_path.is_absolute():
        return str(model_path)

    root = model_root or Path.cwd()
    local_path = root / model_path
    if local_path.is_file() or model_path.parent != Path("."):
        return str(local_path)
    return model.strip()


def _resolve_backend(config: PersonDetectionConfig) -> str:
    if config.backend != "auto":
        return config.backend
    model = (config.model or "").strip().lower()
    if model.endswith(".onnx"):
        return "onnx"
    if model.endswith(("yolo26s.pt", "yolo26n.pt", "_openvino_model", ".xml")):
        return "openvino"
    return "yoloe"


def _needs_segmentation_adapter(config: PersonDetectionConfig) -> bool:
    """Use the richer adapter when masks or multiple prompt classes are requested."""

    return bool(config.show_masks or tuple(config.prompts) != ("person",))


def create_person_detector(
    config: PersonDetectionConfig,
    *,
    model_root: Path | None = None,
) -> PersonDetector:
    """Create the configured detector without touching model dependencies when disabled."""

    if not config.enabled:
        return DisabledPersonDetector()

    backend = _resolve_backend(config)
    if backend == "fake":
        return FakePersonDetector()

    if config.model is None or not config.model.strip():
        raise ValueError("person_detection.model is required when detection is enabled")

    if backend == "onnx":
        return OnnxPersonDetector(
            _resolve_onnx_model(config.model, model_root),
            confidence_threshold=config.confidence_threshold,
            device=config.device,
        )
    if backend == "yoloe":
        prompts = normalize_prompts(config.prompts)
        if _needs_segmentation_adapter(config):
            model_path = Path(config.model).expanduser()
            if not model_path.is_absolute():
                model_path = (model_root or Path.cwd()) / model_path
            root = model_root or Path.cwd()
            return YoloESegmentationDetector(
                model_path,
                prompts=prompts,
                confidence_threshold=config.confidence_threshold,
                device=config.device,
                text_encoder_path=root / "models" / "mobileclip2_b.ts",
            )
        return YoloEPersonDetector(
            _resolve_yoloe_model(config.model, model_root),
            confidence_threshold=config.confidence_threshold,
            device=config.device,
            prompts=prompts,
            image_size=config.image_size,
        )
    if backend == "openvino":
        return OpenVINOPersonDetector(
            config.model,
            confidence_threshold=config.confidence_threshold,
            precision=config.precision,
            device=config.device,
            fallback_device=config.fallback_device,
            classes=config.classes,
            image_size=config.image_size,
            performance_mode=config.openvino_performance_mode,
            num_streams=config.openvino_num_streams,
            num_requests=config.openvino_num_requests,
            cpu_threads=config.openvino_cpu_threads,
            max_process_ram_mb=config.max_process_ram_mb,
            model_root=model_root,
        )
    raise ValueError(f"unsupported person detection backend: {backend}")

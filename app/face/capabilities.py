"""Shared face model/backend/device capability matrix."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np

from .registry import (
    FACE_DETECTOR_SPECS,
    LANDMARKER_SPEC,
    RECOGNIZER_SPECS,
    detector_spec,
    FaceModelSpec,
    RecognizerSpec,
    artifact_sha256,
    model_path,
)
from .openvino_runtime import OpenVINOCoreManager
from .model_resolution import FaceModelResolutionError, resolve_recognizer


@dataclass(frozen=True)
class FaceCapability:
    component: str
    model_id: str
    display_name: str
    backend: str
    device: str
    available: bool
    artifact_present: bool
    reason: str
    landmarks: bool = False
    embedding_dimension: int | None = None
    probed: bool = False
    actual_device: str | None = None
    source: str = ""
    license: str = ""
    model_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _runtime_devices(backend: str) -> tuple[str, ...]:
    normalized = str(backend).strip().lower()
    if normalized == "opencv_dnn":
        return ("cpu",) if hasattr(cv2, "FaceDetectorYN") else ()
    if normalized == "onnxruntime":
        try:
            import onnxruntime as ort

            providers = set(ort.get_available_providers())
        except Exception:
            return ()
        devices: list[str] = []
        if "CPUExecutionProvider" in providers:
            devices.append("cpu")
        if "CUDAExecutionProvider" in providers:
            devices.append("cuda")
        return tuple(devices)
    if normalized == "openvino":
        try:
            from openvino import Core

            devices = {str(value).upper() for value in Core().available_devices}
        except Exception:
            return ()
        result = [device for device in ("cpu", "gpu", "npu") if device.upper() in devices]
        return tuple(result)
    if normalized == "fake":
        return ("cpu",)
    return ()


def _artifact_present(path: Path, backend: str) -> bool:
    if backend == "openvino":
        if path.suffix.lower() != ".xml":
            return False
        return path.is_file() and path.with_suffix(".bin").is_file()
    return path.is_file()


def _static_dimension(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _probe_onnx(
    path: Path,
    device: str,
    *,
    expected_output_dimension: int | None = None,
) -> tuple[bool, str, str | None]:
    import onnxruntime as ort

    providers = tuple(ort.get_available_providers())
    if device == "cuda":
        provider = "CUDAExecutionProvider"
    elif device == "cpu":
        provider = "CPUExecutionProvider"
    elif "CUDAExecutionProvider" in providers:
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"
    if provider not in providers:
        return False, f"provider unavailable: {provider}", None
    session = ort.InferenceSession(str(path), providers=[provider])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        return False, "probe requires exactly one input", None
    info = inputs[0]
    # ONNX face recognizers commonly expose a dynamic batch dimension.  The
    # capability probe is a single-sample probe, so only spatial dimensions
    # receive the model-specific fallback; treating a dynamic batch as 640
    # makes a valid [None, 3, H, W] recognizer look incompatible with its
    # declared per-sample output dimension.
    shape = tuple(
        _static_dimension(value, 1 if index == 0 else 640)
        for index, value in enumerate(info.shape)
    )
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        return False, f"unsupported input shape: {info.shape}", None
    outputs = session.run(None, {info.name: np.zeros(shape, dtype=np.float32)})
    if expected_output_dimension is not None:
        values = np.asarray(outputs[0]) if outputs else np.empty(0, dtype=np.float32)
        actual = (
            int(np.prod(values.shape[1:]))
            if values.ndim > 1
            else int(values.size)
        )
        if not outputs or actual != expected_output_dimension:
            return (
                False,
                f"output dimension {actual} does not match {expected_output_dimension}",
                None,
            )
    actual = "cuda" if provider == "CUDAExecutionProvider" else "cpu"
    return True, "I/O probe passed", actual


def _probe_opencv(path: Path) -> tuple[bool, str, str | None]:
    detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.5, 0.3, 5000)
    detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    return True, "I/O probe passed", "cpu"


def _probe_openvino(
    path: Path,
    device: str,
    *,
    model_id: str,
    expected_output_dimension: int | None = None,
) -> tuple[bool, str, str | None]:
    if path.suffix.lower() != ".xml":
        return False, "OpenVINO probe requires an .xml IR model", None
    xml_path = path
    core = OpenVINOCoreManager.core()
    model = core.read_model(str(xml_path))
    requested = "AUTO" if device == "auto" else device.upper()
    compiled = OpenVINOCoreManager.compile_model(
        core,
        model,
        device=requested,
        model_id=model_id,
        model_sha256=artifact_sha256(xml_path),
        cache_root=None,
    )
    inputs = list(compiled.inputs)
    if len(inputs) != 1:
        return False, "probe requires exactly one input", None
    port = inputs[0]
    raw_shape = tuple(port.shape)
    default_size = 48 if "0009" in model_id else 128 if "0095" in model_id else 416
    shape = tuple(_static_dimension(value, default_size) for value in raw_shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        return False, f"unsupported input shape: {raw_shape}", None
    result = compiled({port.any_name: np.zeros(shape, dtype=np.float32)})
    if result is None:
        return False, "probe returned no result", None
    if expected_output_dimension is not None:
        values = next(iter(result.values())) if hasattr(result, "values") else result[0]
        actual = int(np.asarray(values).size)
        if actual != expected_output_dimension:
            return (
                False,
                f"output dimension {actual} does not match {expected_output_dimension}",
                None,
            )
    actual = None
    try:
        devices = compiled.get_property("EXECUTION_DEVICES")
        if devices:
            actual = OpenVINOCoreManager.normalize_device_name(devices[0])
    except Exception:
        actual = None
    return True, "compile/inference probe passed", actual or device


_PROBE_LOCK = threading.RLock()
_PROBE_CACHE: dict[tuple[str, str, str, str, str], tuple[bool, str, str | None]] = {}


def _probe_spec(
    component: str,
    spec: FaceModelSpec | RecognizerSpec,
    path: Path,
    device: str,
) -> tuple[bool, str, str | None]:
    digest = artifact_sha256(path) or "missing"
    key = (component, spec.model_id if isinstance(spec, FaceModelSpec) else spec.recognizer_id, str(path), digest, device)
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        expected_output_dimension = (
            spec.embedding_dimension
            if isinstance(spec, RecognizerSpec)
            else 10
            if component == "face_landmarks"
            else None
        )
        if spec.backend == "onnxruntime":
            result = _probe_onnx(
                path,
                device,
                expected_output_dimension=expected_output_dimension,
            )
        elif spec.backend == "opencv_dnn":
            result = _probe_opencv(path)
        elif spec.backend == "openvino":
            result = _probe_openvino(
                path,
                device,
                model_id=spec.model_id if isinstance(spec, FaceModelSpec) else spec.recognizer_id,
                expected_output_dimension=expected_output_dimension,
            )
        else:
            result = False, f"unsupported probe backend: {spec.backend}", None
    except Exception as exc:
        result = False, f"probe failed: {type(exc).__name__}: {exc}", None
    with _PROBE_LOCK:
        _PROBE_CACHE[key] = result
    return result


def _capability_for_spec(
    component: str,
    spec: FaceModelSpec | RecognizerSpec,
    *,
    model_root: Path | None,
    artifact_path: Path | None = None,
) -> tuple[FaceCapability, ...]:
    path = artifact_path or model_path(spec, model_root)
    artifact_present = _artifact_present(path, spec.backend)
    runtime_devices = set(_runtime_devices(spec.backend))
    rows: list[FaceCapability] = []
    for device in spec.devices:
        normalized = device.lower()
        device_available = normalized in runtime_devices or (
            normalized == "auto" and bool(runtime_devices)
        )
        if not artifact_present:
            reason = f"artifact missing: {path.name}"
            probed = False
            actual_device = None
        elif not device_available:
            reason = f"runtime device unavailable: {device}"
            probed = False
            actual_device = None
        else:
            probed, reason, actual_device = _probe_spec(component, spec, path, normalized)
            if probed:
                reason = f"{reason}; actual device: {actual_device or normalized}"
        rows.append(
            FaceCapability(
                component=component,
                model_id=spec.model_id if isinstance(spec, FaceModelSpec) else spec.recognizer_id,
                display_name=spec.display_name,
                source=spec.source,
                license=spec.license,
                backend=spec.backend,
                device=normalized,
                available=artifact_present and device_available and probed,
                artifact_present=artifact_present,
                reason=reason,
                landmarks=spec.landmarks if isinstance(spec, FaceModelSpec) else True,
                embedding_dimension=(
                    spec.embedding_dimension if isinstance(spec, RecognizerSpec) else None
                ),
                probed=probed,
                actual_device=actual_device,
                model_path=spec.relative_path,
            )
        )
    return tuple(rows)


def face_capability_matrix(
    model_root: Path | str | None = None,
    *,
    configured_recognition: dict[str, Any] | None = None,
) -> tuple[FaceCapability, ...]:
    """Return the matrix consumed by factories and both Windows UIs."""

    root = Path(model_root).expanduser() if model_root is not None else None
    rows: list[FaceCapability] = []
    for spec in FACE_DETECTOR_SPECS:
        rows.extend(_capability_for_spec("face_detection", spec, model_root=root))
    rows.extend(_capability_for_spec("face_landmarks", LANDMARKER_SPEC, model_root=root))
    for spec in RECOGNIZER_SPECS:
        rows.extend(_capability_for_spec("recognition", spec, model_root=root))
    if configured_recognition:
        configured_path = configured_recognition.get("model")
        if configured_path:
            path = Path(str(configured_path)).expanduser()
            if not path.is_absolute() and root is not None:
                path = root / path
            try:
                resolution, spec = resolve_recognizer(
                    model_id=configured_recognition.get("recognizer_id"),
                    requested_backend=str(configured_recognition.get("backend", "auto")),
                    path=path,
                    device=str(configured_recognition.get("device", "auto")),
                )
                rows = [
                    row
                    for row in rows
                    if not (
                        row.component == "recognition"
                        and row.model_id == resolution.model_id
                    )
                ]
                active_rows = _capability_for_spec(
                    "recognition",
                    spec,
                    model_root=root,
                    artifact_path=path,
                )
                rows.extend(
                    FaceCapability(
                        component=row.component,
                        model_id=row.model_id,
                        display_name=row.display_name,
                        backend=row.backend,
                        device=row.device,
                        available=row.available,
                        artifact_present=row.artifact_present,
                        reason=f"configured model {path.name}: {row.reason}",
                        landmarks=row.landmarks,
                        embedding_dimension=row.embedding_dimension,
                        probed=row.probed,
                        actual_device=row.actual_device,
                        source=row.source,
                        license=row.license,
                    )
                    for row in active_rows
                )
            except (FaceModelResolutionError, ValueError) as exc:
                rows.append(
                    FaceCapability(
                        component="recognition",
                        model_id=str(configured_recognition.get("recognizer_id") or "configured"),
                        display_name="Configured face recognizer",
                        backend=str(configured_recognition.get("backend", "auto")),
                        device=str(configured_recognition.get("device", "auto")),
                        available=False,
                        artifact_present=path.is_file(),
                        reason=f"configured model {path.name}: {exc}",
                        source="configured",
                    )
                )
    return tuple(rows)


def validate_capability(
    *,
    component: str,
    model_id: str,
    backend: str,
    device: str,
    model_root: Path | str | None = None,
    artifact_path: Path | str | None = None,
) -> FaceCapability:
    """Fail closed for an unavailable model/backend/device combination."""

    normalized_device = str(device).strip().lower()
    root = Path(model_root).expanduser() if model_root is not None else None
    specs: tuple[FaceModelSpec | RecognizerSpec, ...]
    if component == "recognition":
        specs = tuple(spec for spec in RECOGNIZER_SPECS if spec.recognizer_id == model_id)
    elif component == "face_landmarks":
        specs = (LANDMARKER_SPEC,) if LANDMARKER_SPEC.model_id == model_id else ()
    else:
        try:
            canonical_id = detector_spec(model_id).model_id
        except ValueError:
            canonical_id = model_id
        specs = tuple(spec for spec in FACE_DETECTOR_SPECS if spec.model_id == canonical_id)
    if not specs:
        raise ValueError(f"unsupported {component} model: {model_id}")
    configured_path = (
        Path(artifact_path).expanduser()
        if artifact_path is not None
        else None
    )
    if configured_path is not None and not configured_path.is_absolute() and root is not None:
        configured_path = root / configured_path
    rows = _capability_for_spec(
        component,
        specs[0],
        model_root=root,
        artifact_path=configured_path,
    )
    for row in rows:
        if row.backend == backend and row.device == normalized_device:
            if not row.available:
                raise ValueError(
                    f"unsupported {component} selection {model_id}/{backend}/{normalized_device}: {row.reason}"
                )
            return row
    raise ValueError(f"unsupported {component} backend/device: {model_id}/{backend}/{normalized_device}")


__all__ = ["FaceCapability", "face_capability_matrix", "validate_capability"]

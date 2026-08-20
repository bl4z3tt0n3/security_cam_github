"""Format- and contract-aware resolution for local face models.

The resolver deliberately does not infer a backend from a filename stem.  The
file format selects the only compatible runtime family and, for ONNX
recognizers, the input/output contract is inspected before a registered
preprocessing/alignment profile is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import (
    FACE_DETECTOR_SPECS,
    RECOGNIZER_SPECS,
    FaceModelSpec,
    RecognizerSpec,
    detector_spec,
    recognizer_spec,
)


class FaceModelResolutionError(ValueError):
    """Raised when a model, backend or registered contract is incompatible."""


def normalize_device_for_backend(device: str, backend: str) -> str:
    requested = str(device).strip().casefold()
    normalized_backend = str(backend).strip().casefold()
    if normalized_backend in {"onnxruntime", "opencv_dnn"}:
        if requested == "gpu":
            requested = "cuda"
        if requested == "npu":
            raise FaceModelResolutionError(
                f"backend {normalized_backend} non espone un device NPU"
            )
    if normalized_backend == "openvino" and requested == "cuda":
        raise FaceModelResolutionError(
            "OpenVINO usa il device gpu/npu, non cuda"
        )
    return requested


@dataclass(frozen=True)
class FaceModelResolution:
    component: str
    requested_model_id: str | None
    model_id: str
    path: Path
    model_format: str
    requested_backend: str
    backend: str


def model_format(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".xml":
        return "openvino_ir"
    return suffix.lstrip(".") or "unknown"


def validate_backend_format(
    path: Path,
    backend: str,
    *,
    component: str,
) -> None:
    """Reject backend/file combinations before any runtime tries to load them."""

    normalized = str(backend).strip().casefold()
    suffix = path.suffix.casefold()
    if normalized == "openvino" and suffix != ".xml":
        raise FaceModelResolutionError(
            f"{component} backend openvino richiede un modello .xml OpenVINO IR "
            f"con .bin adiacente; ricevuto {path.name}"
        )
    if normalized in {"onnxruntime", "opencv_dnn"} and suffix != ".onnx":
        raise FaceModelResolutionError(
            f"{component} backend {normalized} richiede un modello .onnx; "
            f"ricevuto {path.name}"
        )


def resolve_detector(
    *,
    model_id: str | None,
    requested_backend: str,
    path: Path,
) -> FaceModelResolution:
    requested_id = model_id.strip().lower() if model_id else None
    requested_backend = str(requested_backend).strip().casefold()
    canonical_id = None
    if requested_id:
        canonical_id = detector_spec(requested_id).model_id
    suffix = path.suffix.casefold()
    if requested_backend == "auto":
        if canonical_id is None:
            if suffix == ".xml":
                canonical_id = "face_detection_0205"
            elif suffix == ".onnx":
                canonical_id = "yunet_2023mar"
            else:
                raise FaceModelResolutionError(
                    f"formato detector non supportato: {path.name}"
                )
        spec = detector_spec(canonical_id)
        backend = spec.backend
        # YuNet is the only registered detector whose ONNX contract belongs to
        # OpenCV DNN rather than ONNX Runtime.
        if suffix == ".onnx" and backend == "openvino":
            raise FaceModelResolutionError(
                f"il detector {canonical_id} richiede OpenVINO .xml, non {path.name}"
            )
        if suffix == ".xml" and backend != "openvino":
            raise FaceModelResolutionError(
                f"il detector {canonical_id} non supporta il formato OpenVINO .xml"
            )
    else:
        backend = str(requested_backend).strip().casefold()
        if canonical_id is None:
            candidates = [
                spec for spec in FACE_DETECTOR_SPECS if spec.backend == backend
            ]
            if len(candidates) == 1:
                canonical_id = candidates[0].model_id
            else:
                raise FaceModelResolutionError(
                    "face detector id è obbligatorio quando il backend non è auto"
                )
    validate_backend_format(path, backend, component="face detector")
    return FaceModelResolution(
        component="face_detection",
        requested_model_id=requested_id,
        model_id=canonical_id or "",
        path=path,
        model_format=model_format(path),
        requested_backend=requested_backend,
        backend=backend,
    )


def _onnx_contract(path: Path, *, device: str = "auto") -> tuple[int, int, int]:
    """Read one ONNX model's per-sample input shape and flat output dimension."""

    try:
        import onnxruntime as ort

        providers = tuple(ort.get_available_providers())
        provider = (
            "CUDAExecutionProvider"
            if device in {"cuda", "gpu"} and "CUDAExecutionProvider" in providers
            else "CPUExecutionProvider"
        )
        if provider not in providers:
            raise FaceModelResolutionError(
                f"provider ONNX Runtime non disponibile: {provider}"
            )
        session = ort.InferenceSession(str(path), providers=[provider])
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise FaceModelResolutionError(
                f"il recognizer ONNX deve avere un solo input, trovati {len(inputs)}"
            )
        shape = tuple(inputs[0].shape)
        if len(shape) != 4 or shape[1] != 3:
            raise FaceModelResolutionError(
                f"input ONNX non compatibile NCHW a 3 canali: {shape}"
            )
        height = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 0
        width = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else 0
        if not width or not height:
            raise FaceModelResolutionError(
                f"input ONNX con dimensioni spaziali dinamiche non risolvibili: {shape}"
            )
        outputs = session.get_outputs()
        if not outputs:
            raise FaceModelResolutionError("il recognizer ONNX non espone output")
        output_shape = tuple(outputs[0].shape)
        dimension = 1
        for value in output_shape[1:] if len(output_shape) > 1 else output_shape:
            if not isinstance(value, int) or value <= 0:
                raise FaceModelResolutionError(
                    f"dimensione embedding ONNX dinamica o non valida: {output_shape}"
                )
            dimension *= int(value)
        return width, height, dimension
    except FaceModelResolutionError:
        raise
    except Exception as exc:
        raise FaceModelResolutionError(
            f"probe contratto ONNX fallita per {path.name}: {type(exc).__name__}: {exc}"
        ) from exc


def _matching_recognizers(
    path: Path,
    *,
    device: str = "auto",
) -> tuple[RecognizerSpec, ...]:
    width, height, dimension = _onnx_contract(path, device=device)
    return tuple(
        spec
        for spec in RECOGNIZER_SPECS
        if spec.backend == "onnxruntime"
        and spec.input_width == width
        and spec.input_height == height
        and spec.embedding_dimension == dimension
    )


def resolve_recognizer(
    *,
    model_id: str | None,
    requested_backend: str,
    path: Path,
    device: str = "auto",
) -> tuple[FaceModelResolution, RecognizerSpec]:
    requested_id = model_id.strip().lower() if model_id else None
    requested_backend = str(requested_backend).strip().casefold()
    suffix = path.suffix.casefold()

    if requested_backend == "auto":
        backend = "openvino" if suffix == ".xml" else "onnxruntime"
    else:
        backend = requested_backend
    validate_backend_format(path, backend, component="face recognizer")
    normalize_device_for_backend(device, backend)

    if suffix == ".xml":
        if not requested_id:
            raise FaceModelResolutionError(
                "recognition.recognizer_id è obbligatorio per un modello OpenVINO .xml"
            )
        spec = recognizer_spec(requested_id)
        if spec.backend != "openvino":
            raise FaceModelResolutionError(
                f"il recognizer {requested_id} non è compatibile con OpenVINO .xml"
            )
    else:
        matches = _matching_recognizers(path, device=device)
        if requested_id:
            requested_spec = recognizer_spec(requested_id)
            requested_matches = tuple(
                candidate
                for candidate in matches
                if candidate.recognizer_id == requested_spec.recognizer_id
            )
            if requested_backend != "auto":
                if requested_spec.backend != backend or not requested_matches:
                    raise FaceModelResolutionError(
                        f"recognizer {requested_id} non compatibile con {path.name}: "
                        f"contratto richiesto {requested_spec.input_width}x"
                        f"{requested_spec.input_height}/{requested_spec.embedding_dimension}, "
                        f"contratti rilevati "
                        f"{[(item.recognizer_id, item.input_width, item.input_height, item.embedding_dimension) for item in matches]}"
                    )
                spec = requested_spec
            elif requested_matches:
                spec = requested_spec
            elif len(matches) == 1:
                # In auto mode the configured id is a legacy/configuration hint;
                # the unique actual model contract wins without rewriting YAML.
                spec = matches[0]
            else:
                raise FaceModelResolutionError(
                    f"nessun contratto recognizer univoco per {path.name}; "
                    f"richiesto {requested_id}, rilevati "
                    f"{[(item.recognizer_id, item.input_width, item.input_height, item.embedding_dimension) for item in matches]}"
                )
        elif len(matches) == 1:
            spec = matches[0]
        else:
            raise FaceModelResolutionError(
                f"nessun contratto recognizer ONNX univoco per {path.name}: "
                f"{[(item.recognizer_id, item.input_width, item.input_height, item.embedding_dimension) for item in matches]}"
            )

    if spec.backend != backend:
        raise FaceModelResolutionError(
            f"backend {backend} incompatibile con il modello registrato {spec.recognizer_id}"
        )
    return (
        FaceModelResolution(
            component="recognition",
            requested_model_id=requested_id,
            model_id=spec.recognizer_id,
            path=path,
            model_format=model_format(path),
            requested_backend=requested_backend,
            backend=backend,
        ),
        spec,
    )


__all__ = [
    "FaceModelResolution",
    "FaceModelResolutionError",
    "model_format",
    "resolve_detector",
    "resolve_recognizer",
    "normalize_device_for_backend",
    "validate_backend_format",
]

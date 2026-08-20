"""Replaceable local face-embedding adapters.

The enrollment pipeline stores embeddings together with the exact preprocessing
and model identity used to produce them.  This makes an accidental comparison
between incompatible models detectable before a future recognition step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
DEFAULT_INPUT_SIZE = 112
DEFAULT_NORMALIZATION = "arcface_127.5_128"


class FaceEmbeddingError(RuntimeError):
    """Raised when an embedding model cannot load or execute."""


class IncompatibleEmbeddingModelError(ValueError):
    """Raised when stored and requested embedding model metadata differ."""


@dataclass(frozen=True)
class EmbeddingModelMetadata:
    """Identity and preprocessing contract for one embedding model."""

    backend: str
    model_id: str
    model_version: str
    embedding_dimension: int
    input_width: int
    input_height: int
    color_order: str = "RGB"
    normalization: str = DEFAULT_NORMALIZATION
    model_sha256: str | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)
    recognizer_id: str | None = None
    requested_device: str | None = None
    actual_device: str | None = None
    alignment_template: Mapping[str, Any] | None = None
    default_threshold: float | None = None

    def __post_init__(self) -> None:
        for name in ("backend", "model_id", "model_version", "color_order", "normalization"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("embedding_dimension", "input_width", "input_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.color_order.upper() not in {"RGB", "BGR"}:
            raise ValueError("the supported face embedding color order is RGB or BGR")
        if self.model_sha256 is not None:
            normalized_hash = self.model_sha256.strip().lower()
            if len(normalized_hash) != 64 or any(
                character not in "0123456789abcdef" for character in normalized_hash
            ):
                raise ValueError("model_sha256 must be a SHA-256 hexadecimal digest")
            object.__setattr__(self, "model_sha256", normalized_hash)
        for name in ("recognizer_id", "requested_device", "actual_device"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when present")
        if self.default_threshold is not None and (
            not math.isfinite(float(self.default_threshold)) or self.default_threshold < 0
        ):
            raise ValueError("default_threshold must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "embedding_dimension": self.embedding_dimension,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "color_order": self.color_order.upper(),
            "normalization": self.normalization,
            "model_sha256": self.model_sha256,
            "configuration": dict(self.configuration),
            "recognizer_id": self.recognizer_id,
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "alignment_template": (
                dict(self.alignment_template) if self.alignment_template is not None else None
            ),
            "default_threshold": self.default_threshold,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EmbeddingModelMetadata":
        """Parse persisted model metadata and reject incomplete records."""

        try:
            return cls(
                backend=str(value["backend"]),
                model_id=str(value["model_id"]),
                model_version=str(value["model_version"]),
                embedding_dimension=int(value["embedding_dimension"]),
                input_width=int(value["input_width"]),
                input_height=int(value["input_height"]),
                color_order=str(value.get("color_order", "RGB")),
                normalization=str(value.get("normalization", DEFAULT_NORMALIZATION)),
                model_sha256=value.get("model_sha256"),
                configuration=value.get("configuration", {}),
                recognizer_id=value.get("recognizer_id"),
                requested_device=value.get("requested_device"),
                actual_device=value.get("actual_device"),
                alignment_template=value.get("alignment_template"),
                default_threshold=value.get("default_threshold"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid embedding model metadata") from exc

    def compatibility_key(self) -> str:
        """Return a stable key used to compare two model contracts."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def is_compatible_with(self, other: "EmbeddingModelMetadata") -> bool:
        return self.compatibility_key() == other.compatibility_key()

    @property
    def fingerprint(self) -> str:
        """Stable opaque gallery scope for this exact model contract."""

        return hashlib.sha256(self.compatibility_key().encode("utf-8")).hexdigest()[:32]


class FaceEmbedder(ABC):
    """Replaceable face embedding contract used by enrollment."""

    @property
    @abstractmethod
    def metadata(self) -> EmbeddingModelMetadata:
        """Return the model contract used by :meth:`embed`."""

    @abstractmethod
    def embed(self, face_image: np.ndarray) -> np.ndarray:
        """Return one finite, L2-normalized embedding for a face image."""


def _normalize_embedding(value: np.ndarray, expected_dimension: int) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if embedding.size != expected_dimension:
        raise FaceEmbeddingError(
            f"embedding dimension {embedding.size} does not match "
            f"model metadata dimension {expected_dimension}"
        )
    if not np.all(np.isfinite(embedding)):
        raise FaceEmbeddingError("embedding contains non-finite values")
    norm = float(np.linalg.norm(embedding))
    if not math.isfinite(norm) or norm <= 0:
        raise FaceEmbeddingError("embedding has zero or invalid norm")
    return (embedding / norm).astype(np.float32, copy=False)


class FakeEmbedder(FaceEmbedder):
    """Deterministic embedder for offline tests and demos."""

    def __init__(
        self,
        embedding_dimension: int = 4,
        *,
        model_id: str = "fake-face-embedder",
        model_version: str = "1",
        callback: Callable[[np.ndarray], Sequence[float] | np.ndarray] | None = None,
    ) -> None:
        self._metadata = EmbeddingModelMetadata(
            backend="fake",
            model_id=model_id,
            model_version=model_version,
            embedding_dimension=embedding_dimension,
            input_width=DEFAULT_INPUT_SIZE,
            input_height=DEFAULT_INPUT_SIZE,
            configuration={"deterministic": True},
        )
        if callback is not None and not callable(callback):
            raise ValueError("callback must be callable")
        self._callback = callback
        self._calls = 0

    @property
    def metadata(self) -> EmbeddingModelMetadata:
        return self._metadata

    @property
    def calls(self) -> int:
        return self._calls

    def embed(self, face_image: np.ndarray) -> np.ndarray:
        self._calls += 1
        if self._callback is None:
            value = np.arange(1, self.metadata.embedding_dimension + 1, dtype=np.float32)
        else:
            value = self._callback(face_image)
        return _normalize_embedding(np.asarray(value), self.metadata.embedding_dimension)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_device(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return normalized


def _select_provider(device: str, available: Sequence[str]) -> str:
    providers = set(available)
    if device == "cuda":
        if CUDA_PROVIDER not in providers:
            raise FaceEmbeddingError(
                "CUDA device requested but CUDAExecutionProvider is unavailable"
            )
        return CUDA_PROVIDER
    if device == "cpu":
        if available and CPU_PROVIDER not in providers:
            raise FaceEmbeddingError("CPUExecutionProvider is unavailable")
        return CPU_PROVIDER
    if CUDA_PROVIDER in providers:
        return CUDA_PROVIDER
    if CPU_PROVIDER in providers or not providers:
        return CPU_PROVIDER
    raise FaceEmbeddingError(
        "no supported ONNX Runtime provider available; expected CPUExecutionProvider"
    )


def _actual_provider(session: Any, requested: str) -> str:
    providers = tuple(getattr(session, "get_providers", lambda: ())())
    if requested in providers:
        return requested
    if CPU_PROVIDER in providers:
        return CPU_PROVIDER
    return providers[0] if providers else requested


def _static_dimension(value: Any, default: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        if default is None:
            raise FaceEmbeddingError("model input/output shape contains a dynamic dimension")
        return default
    if result <= 0:
        if default is None:
            raise FaceEmbeddingError("model input/output shape contains an invalid dimension")
        return default
    return result


class OnnxFaceEmbedder(FaceEmbedder):
    """ONNX Runtime adapter for the documented 3-channel face model contract."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        model_id: str | None = None,
        model_version: str = "1",
        device: str = "auto",
        embedding_dimension: int | None = None,
        color_order: str = "RGB",
        normalization: str = DEFAULT_NORMALIZATION,
        recognizer_id: str | None = None,
        alignment_template: Mapping[str, Any] | None = None,
        default_threshold: float | None = None,
        session: Any | None = None,
        session_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._model_path = Path(model_path).expanduser()
        self._requested_device = _validate_device(device)
        if session is None and not self._model_path.is_file():
            raise FaceEmbeddingError(f"face embedding model not found: {self._model_path}")

        try:
            if session is None:
                import onnxruntime as ort

                available = tuple(ort.get_available_providers())
                provider = _select_provider(self._requested_device, available)
                factory = session_factory or ort.InferenceSession
                session = factory(str(self._model_path), providers=[provider])
            else:
                available = tuple(getattr(session, "get_providers", lambda: ())())
                provider = _select_provider(self._requested_device, available)
        except FaceEmbeddingError:
            raise
        except Exception as exc:
            raise FaceEmbeddingError(
                f"cannot load face embedding model {self._model_path}: {exc}"
            ) from exc

        self._session = session
        self._input_name, input_height, input_width = self._inspect_input(session)
        inferred_dimension = self._inspect_output_dimension(session)
        dimension = embedding_dimension or inferred_dimension
        if dimension is None:
            raise FaceEmbeddingError(
                "embedding dimension is dynamic; pass embedding_dimension explicitly"
            )
        actual_provider = _actual_provider(session, provider)
        self._provider_used = actual_provider
        self._device_used = "cuda" if actual_provider == CUDA_PROVIDER else "cpu"
        self._metadata = EmbeddingModelMetadata(
            backend="onnxruntime",
            model_id=model_id or self._model_path.stem,
            model_version=model_version,
            embedding_dimension=dimension,
            input_width=input_width,
            input_height=input_height,
            color_order=color_order,
            normalization=normalization,
            configuration={
                "device": self._device_used,
                "provider": actual_provider,
                "input_layout": "NCHW",
            },
            model_sha256=_sha256_file(self._model_path),
            recognizer_id=recognizer_id or model_id or self._model_path.stem,
            requested_device=self._requested_device,
            actual_device=self._device_used,
            alignment_template=alignment_template,
            default_threshold=default_threshold,
        )

    @property
    def metadata(self) -> EmbeddingModelMetadata:
        return self._metadata

    @property
    def device_used(self) -> str:
        return self._device_used

    @property
    def provider_used(self) -> str:
        return self._provider_used

    def close(self) -> None:
        """Release the ONNX Runtime session when the live pipeline is rebuilt."""

        session = getattr(self, "_session", None)
        close = getattr(session, "close", None)
        if callable(close):
            close()
        self._session = None

    @staticmethod
    def _inspect_input(session: Any) -> tuple[str, int, int]:
        try:
            inputs = session.get_inputs()
            if len(inputs) != 1:
                raise FaceEmbeddingError("face embedding model must have exactly one input")
            info = inputs[0]
            shape = tuple(info.shape)
            if len(shape) != 4:
                raise FaceEmbeddingError("face embedding model input must be NCHW")
            channels = _static_dimension(shape[1], 3)
            if channels != 3:
                raise FaceEmbeddingError("face embedding model input must have three channels")
            return (
                str(info.name),
                _static_dimension(shape[2], DEFAULT_INPUT_SIZE),
                _static_dimension(shape[3], DEFAULT_INPUT_SIZE),
            )
        except FaceEmbeddingError:
            raise
        except Exception as exc:
            raise FaceEmbeddingError(f"cannot inspect face embedding model input: {exc}") from exc

    @staticmethod
    def _inspect_output_dimension(session: Any) -> int | None:
        try:
            outputs = session.get_outputs()
            if not outputs:
                raise FaceEmbeddingError("face embedding model has no outputs")
            shape = tuple(outputs[0].shape)
            if not shape:
                raise FaceEmbeddingError("face embedding model output has no dimensions")
            dimensions = shape[1:] if len(shape) > 1 else shape
            if any(not isinstance(value, int) or value <= 0 for value in dimensions):
                return None
            dimension = math.prod(dimensions)
            return int(dimension) if dimension > 0 else None
        except FaceEmbeddingError:
            raise
        except Exception as exc:
            raise FaceEmbeddingError(f"cannot inspect face embedding model output: {exc}") from exc

    def _preprocess(self, face_image: np.ndarray, width: int, height: int) -> np.ndarray:
        image = np.asarray(face_image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise FaceEmbeddingError("face image must be a non-empty BGR or grayscale image")
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        ordered = (
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            if self.metadata.color_order.upper() == "RGB"
            else resized
        )
        values = ordered.astype(np.float32)
        if self.metadata.normalization in {"arcface_127.5_128", "facenet_fixed_standardization"}:
            normalized = (values - 127.5) / 128.0
        elif self.metadata.normalization == "arcface_127.5_127.5":
            normalized = (values - 127.5) / 127.5
        elif self.metadata.normalization in {"none", "openvino_raw_bgr"}:
            normalized = values
        else:
            raise FaceEmbeddingError(
                f"unsupported face embedding normalization: {self.metadata.normalization}"
            )
        return np.transpose(normalized, (2, 0, 1))[None, ...]

    def embed(self, face_image: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(
            face_image, self.metadata.input_width, self.metadata.input_height
        )
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            raise FaceEmbeddingError(f"face embedding inference failed: {exc}") from exc
        if not outputs:
            raise FaceEmbeddingError("face embedding model returned no output")
        return _normalize_embedding(np.asarray(outputs[0]), self.metadata.embedding_dimension)

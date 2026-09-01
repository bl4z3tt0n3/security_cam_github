"""OpenVINO face embedding adapter with explicit preprocessing metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import cv2
import numpy as np

from .embedding import (
    EmbeddingModelMetadata,
    FaceEmbedder,
    FaceEmbeddingError,
    _normalize_embedding,
)
from .registry import RecognizerSpec, artifact_sha256
from .openvino_runtime import OpenVINOCoreManager


class OpenVINOFaceEmbedder(FaceEmbedder):
    """Run a concrete OpenVINO recognizer from an IR or supported model file."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        spec: RecognizerSpec | None = None,
        model_id: str | None = None,
        model_version: str = "1",
        device: str = "auto",
        embedding_dimension: int | None = None,
        input_width: int | None = None,
        input_height: int | None = None,
        color_order: str | None = None,
        normalization: str | None = None,
        compiled_model: Any | None = None,
        core: Any | None = None,
        cache_dir: Path | None = None,
        performance_mode: str = "latency",
        cpu_threads: int = 0,
        max_process_ram_mb: int = 0,
    ) -> None:
        self._model_path = Path(model_path).expanduser()
        self._performance_mode = str(performance_mode).strip().lower()
        self._cpu_threads = int(cpu_threads)
        self._max_process_ram_mb = int(max_process_ram_mb)
        self._spec = spec
        self._requested_device = str(device).strip().lower()
        if self._requested_device not in {"auto", "cpu", "gpu", "npu"}:
            raise ValueError("OpenVINO embedding device must be one of: auto, cpu, gpu, npu")
        self._core = core
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self._compiled = compiled_model
        self._input_name: str | None = None
        self._input_width = input_width or (spec.input_width if spec else 128)
        self._input_height = input_height or (spec.input_height if spec else 128)
        self._embedding_dimension = embedding_dimension or (spec.embedding_dimension if spec else None)
        self._color_order = color_order or (spec.color_order if spec else "BGR")
        self._normalization = normalization or (spec.normalization if spec else "openvino_raw_bgr")
        self._model_id = model_id or (spec.recognizer_id if spec else self._model_path.stem)
        self._model_version = model_version
        self._device_used = ""
        if compiled_model is None:
            self.load()
        else:
            self._inspect_compiled(compiled_model)

    @property
    def metadata(self) -> EmbeddingModelMetadata:
        return self._metadata

    @property
    def device_used(self) -> str:
        return self._device_used

    @property
    def provider_used(self) -> str:
        return "openvino"

    def load(self) -> None:
        if self._compiled is not None:
            return
        model_path = self._model_path
        if model_path.suffix.lower() != ".xml":
            raise FaceEmbeddingError(
                f"OpenVINO embedding richiede un file .xml IR, ricevuto: {model_path}"
            )
        if not model_path.is_file() or not model_path.with_suffix(".bin").is_file():
            raise FaceEmbeddingError(f"OpenVINO embedding IR not found: {model_path}")
        try:
            if self._core is None:
                self._core = OpenVINOCoreManager.core()
            requested = "AUTO" if self._requested_device == "auto" else self._requested_device.upper()
            model = self._core.read_model(str(model_path))
            self._compiled = OpenVINOCoreManager.compile_model(
                self._core,
                model,
                device=requested,
                model_id=self._model_id,
                model_sha256=artifact_sha256(model_path),
                cache_root=self._cache_dir,
                performance_mode=self._performance_mode,
                cpu_threads=self._cpu_threads,
                max_process_ram_mb=self._max_process_ram_mb,
            )
            self._inspect_compiled(self._compiled, fallback_device=requested.lower())
        except FaceEmbeddingError:
            raise
        except Exception as exc:
            raise FaceEmbeddingError(f"cannot load OpenVINO embedding model: {exc}") from exc

    def _inspect_compiled(self, compiled: Any, fallback_device: str | None = None) -> None:
        try:
            inputs = list(compiled.inputs)
            if not inputs:
                raise FaceEmbeddingError("OpenVINO embedding model has no input")
            input_port = inputs[0]
            self._input_name = input_port.any_name
            shape = tuple(int(value) for value in input_port.shape)
            if len(shape) != 4 or shape[1] != 3:
                raise FaceEmbeddingError("OpenVINO embedding input must be NCHW with three channels")
            self._input_height, self._input_width = shape[2], shape[3]
            outputs = list(compiled.outputs)
            if not outputs:
                raise FaceEmbeddingError("OpenVINO embedding model has no output")
            output_shape = tuple(int(value) for value in outputs[0].shape)
            inferred = math.prod(output_shape[1:] if len(output_shape) > 1 else output_shape)
            self._embedding_dimension = self._embedding_dimension or int(inferred)
            if self._embedding_dimension <= 0:
                raise FaceEmbeddingError("OpenVINO embedding dimension must be positive")
            try:
                self._device_used = (
                    OpenVINOCoreManager.execution_device(compiled, fallback_device)
                    or self._requested_device
                )
            except Exception:
                self._device_used = fallback_device or self._requested_device
            template = None
            if self._spec is not None:
                template = {
                    "width": self._spec.alignment_template.width,
                    "height": self._spec.alignment_template.height,
                    "points": [list(point) for point in self._spec.alignment_template.points.points],
                }
            self._metadata = EmbeddingModelMetadata(
                backend="openvino",
                model_id=self._model_id,
                model_version=self._model_version,
                embedding_dimension=int(self._embedding_dimension),
                input_width=self._input_width,
                input_height=self._input_height,
                color_order=self._color_order,
                normalization=self._normalization,
                model_sha256=artifact_sha256(self._model_path),
                configuration={"input_layout": "NCHW"},
                recognizer_id=self._model_id,
                requested_device=self._requested_device,
                actual_device=self._device_used,
                alignment_template=template,
                default_threshold=self._spec.default_threshold if self._spec else None,
            )
        except FaceEmbeddingError:
            raise
        except Exception as exc:
            raise FaceEmbeddingError(f"cannot inspect OpenVINO embedding model: {exc}") from exc

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        value = np.asarray(image)
        if value.ndim == 2:
            value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
        if value.ndim != 3 or value.shape[2] != 3 or value.size == 0:
            raise FaceEmbeddingError("face image must be a non-empty three-channel image")
        resized = cv2.resize(value, (self._input_width, self._input_height), interpolation=cv2.INTER_AREA)
        ordered = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if self._color_order.upper() == "RGB" else resized
        values = ordered.astype(np.float32)
        if self._normalization == "arcface_127.5_127.5":
            values = (values - 127.5) / 127.5
        elif self._normalization in {"none", "openvino_raw_bgr"}:
            pass
        elif self._normalization in {"arcface_127.5_128", "facenet_fixed_standardization"}:
            values = (values - 127.5) / 128.0
        else:
            raise FaceEmbeddingError(f"unsupported OpenVINO normalization: {self._normalization}")
        return np.transpose(values, (2, 0, 1))[None, ...]

    def embed(self, face_image: np.ndarray) -> np.ndarray:
        if self._compiled is None:
            self.load()
        assert self._compiled is not None
        tensor = self._preprocess(face_image)
        try:
            raw = self._compiled({self._input_name: tensor}) if self._input_name else self._compiled(tensor)
        except Exception as exc:
            raise FaceEmbeddingError(f"OpenVINO embedding inference failed: {exc}") from exc
        value = next(iter(raw.values())) if hasattr(raw, "values") else raw[0]
        return _normalize_embedding(np.asarray(value), int(self._embedding_dimension or 0))

    def close(self) -> None:
        self._compiled = None
        self._core = None


__all__ = ["OpenVINOFaceEmbedder"]

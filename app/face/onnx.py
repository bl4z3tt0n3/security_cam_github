"""ONNX Runtime face-detector adapter for local enrollment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import FaceDetection, FaceDetector, FaceDetectorError


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
DEFAULT_INPUT_SIZE = 640


def _validate_device(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return normalized


def _select_provider(device: str, available: Sequence[str]) -> str:
    providers = set(available)
    if device == "cuda":
        if CUDA_PROVIDER not in providers:
            raise FaceDetectorError(
                "CUDA device requested but CUDAExecutionProvider is unavailable"
            )
        return CUDA_PROVIDER
    if device == "cpu":
        if available and CPU_PROVIDER not in providers:
            raise FaceDetectorError("CPUExecutionProvider is unavailable")
        return CPU_PROVIDER
    if CUDA_PROVIDER in providers:
        return CUDA_PROVIDER
    if CPU_PROVIDER in providers or not providers:
        return CPU_PROVIDER
    raise FaceDetectorError(
        "no supported ONNX Runtime provider available; expected CPUExecutionProvider"
    )


def _actual_provider(session: Any, requested: str) -> str:
    providers = tuple(getattr(session, "get_providers", lambda: ())())
    if requested in providers:
        return requested
    if CPU_PROVIDER in providers:
        return CPU_PROVIDER
    return providers[0] if providers else requested


def _dimension(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


class DisabledFaceDetector(FaceDetector):
    """No-op detector used when face detection is explicitly disabled."""

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        del frame
        return []


class OnnxFaceDetector(FaceDetector):
    """Run an ONNX face detector with output rows ``[x1,y1,x2,y2,confidence]``.

    Coordinates are expected in the resized model-input pixel space.  A sixth
    column is accepted for models that append a class id; it is ignored because
    this adapter is dedicated to face detection.
    """

    def __init__(
        self,
        model_path: Path | str,
        *,
        confidence_threshold: float = 0.5,
        device: str = "auto",
        session: Any | None = None,
        session_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not 0 <= float(confidence_threshold) <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self._model_path = Path(model_path).expanduser()
        self._confidence_threshold = float(confidence_threshold)
        requested_device = _validate_device(device)
        if session is None and not self._model_path.is_file():
            raise FaceDetectorError(f"face detection model not found: {self._model_path}")
        try:
            if session is None:
                import onnxruntime as ort

                available = tuple(ort.get_available_providers())
                provider = _select_provider(requested_device, available)
                factory = session_factory or ort.InferenceSession
                session = factory(str(self._model_path), providers=[provider])
            else:
                available = tuple(getattr(session, "get_providers", lambda: ())())
                provider = _select_provider(requested_device, available)
        except FaceDetectorError:
            raise
        except Exception as exc:
            raise FaceDetectorError(
                f"cannot load face detection model {self._model_path}: {exc}"
            ) from exc

        self._session = session
        self._input_name, self._input_height, self._input_width = self._inspect_input(session)
        actual_provider = _actual_provider(session, provider)
        self._provider_used = actual_provider
        self._device_used = "cuda" if actual_provider == CUDA_PROVIDER else "cpu"

    @property
    def device_used(self) -> str:
        return self._device_used

    @property
    def provider_used(self) -> str:
        return self._provider_used

    @staticmethod
    def _inspect_input(session: Any) -> tuple[str, int, int]:
        try:
            inputs = session.get_inputs()
            if len(inputs) != 1:
                raise FaceDetectorError("face detection model must have exactly one input")
            info = inputs[0]
            shape = tuple(info.shape)
            if len(shape) != 4:
                raise FaceDetectorError("face detection model input must be NCHW")
            channels = _dimension(shape[1], 3)
            if channels != 3:
                raise FaceDetectorError("face detection model input must have three channels")
            return (
                str(info.name),
                _dimension(shape[2], DEFAULT_INPUT_SIZE),
                _dimension(shape[3], DEFAULT_INPUT_SIZE),
            )
        except FaceDetectorError:
            raise
        except Exception as exc:
            raise FaceDetectorError(f"cannot inspect face detection model input: {exc}") from exc

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise FaceDetectorError("face detector input must be a non-empty BGR image")
        resized = cv2.resize(
            image,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return np.transpose(normalized, (2, 0, 1))[None, ...]

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        image = np.asarray(frame)
        if image.ndim < 2 or image.shape[0] == 0 or image.shape[1] == 0:
            raise FaceDetectorError("face detector input must have non-zero dimensions")
        tensor = self._preprocess(image)
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            raise FaceDetectorError(f"face detection inference failed: {exc}") from exc
        if not outputs:
            raise FaceDetectorError("face detection model returned no output")
        rows = np.asarray(outputs[0])
        if rows.ndim == 3:
            if rows.shape[0] != 1:
                raise FaceDetectorError("face detection output must contain one batch")
            rows = rows[0]
        if rows.ndim != 2 or rows.shape[1] < 5:
            raise FaceDetectorError(
                "face detection output must be shaped (detections, 5) or (detections, 6)"
            )
        frame_height, frame_width = image.shape[:2]
        scale_x = frame_width / self._input_width
        scale_y = frame_height / self._input_height
        detections: list[FaceDetection] = []
        for row in rows:
            x1, y1, x2, y2, confidence = (float(value) for value in row[:5])
            if not np.isfinite([x1, y1, x2, y2, confidence]).all():
                continue
            if confidence < self._confidence_threshold:
                continue
            detections.append(
                FaceDetection(
                    (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                    confidence,
                )
            )
        return detections

"""ONNX Runtime adapter for end-to-end YOLO26 person detection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import math
import threading
from datetime import datetime
from typing import Any

import numpy as np

from .base import PersonDetection, PersonDetectionError, PersonDetector, utc_now


DEFAULT_INPUT_SIZE = 640
CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"


class OnnxPersonDetector(PersonDetector):
    """Run a YOLO26 end-to-end detection model through ONNX Runtime.

    The supported model contract is the exported YOLO26 detection graph with
    one output shaped ``(batch, detections, 6)``. Each row is
    ``[x1, y1, x2, y2, confidence, class_id]`` in model-input pixels.
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
        self._model_path = Path(model_path).expanduser()
        self._confidence_threshold = self._validate_threshold(confidence_threshold)
        self._requested_device = self._validate_device(device)

        if session is None and not self._model_path.is_file():
            raise PersonDetectionError(f"person detection model not found: {self._model_path}")

        try:
            if session is None:
                import onnxruntime as ort

                available = tuple(ort.get_available_providers())
                provider = self._select_provider(self._requested_device, available)
                factory = session_factory or ort.InferenceSession
                session = factory(str(self._model_path), providers=[provider])
            else:
                available = tuple(getattr(session, "get_providers", lambda: ())())
                provider = self._select_provider(self._requested_device, available)
        except PersonDetectionError:
            raise
        except Exception as exc:
            raise PersonDetectionError(
                f"cannot load person detection model {self._model_path}: {exc}"
            ) from exc

        self._session = session
        self._input_name, self._input_height, self._input_width = self._inspect_input(session)
        actual_provider = self._get_actual_provider(session, provider)
        self._provider_used = actual_provider
        self._device_used = "cuda" if actual_provider == CUDA_PROVIDER else "cpu"
        # Thread-local scratch preserves concurrent ONNX Runtime calls while
        # avoiding a fresh letterbox canvas allocation on every frame.
        self._scratch = threading.local()

    @staticmethod
    def _validate_threshold(value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("confidence_threshold must be a finite number")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence_threshold must be a finite number") from exc
        if not math.isfinite(normalized) or not 0 <= normalized <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return normalized

    @staticmethod
    def _validate_device(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        return normalized

    @staticmethod
    def _select_provider(device: str, available: Sequence[str]) -> str:
        providers = set(available)
        if device == "cuda":
            if CUDA_PROVIDER not in providers:
                raise PersonDetectionError(
                    "CUDA device requested but CUDAExecutionProvider is unavailable"
                )
            return CUDA_PROVIDER
        if device == "cpu":
            if CPU_PROVIDER not in providers:
                raise PersonDetectionError("CPU device requested but CPUExecutionProvider is unavailable")
            return CPU_PROVIDER
        if CUDA_PROVIDER in providers:
            return CUDA_PROVIDER
        if CPU_PROVIDER in providers:
            return CPU_PROVIDER
        raise PersonDetectionError(
            "no supported ONNX Runtime provider available; expected CPUExecutionProvider"
        )

    @staticmethod
    def _get_actual_provider(session: Any, requested: str) -> str:
        providers = tuple(getattr(session, "get_providers", lambda: ())())
        if providers:
            if requested in providers:
                return requested
            if CPU_PROVIDER in providers:
                return CPU_PROVIDER
            return providers[0]
        return requested

    @staticmethod
    def _inspect_input(session: Any) -> tuple[str, int, int]:
        try:
            inputs = session.get_inputs()
            if len(inputs) != 1:
                raise PersonDetectionError("person detection model must have exactly one input")
            input_info = inputs[0]
            shape = tuple(input_info.shape)
            if len(shape) != 4:
                raise PersonDetectionError("person detection model input must be NCHW")
            height = OnnxPersonDetector._static_dimension(shape[2])
            width = OnnxPersonDetector._static_dimension(shape[3])
            return str(input_info.name), height, width
        except PersonDetectionError:
            raise
        except Exception as exc:
            raise PersonDetectionError(f"cannot inspect person detection model input: {exc}") from exc

    @staticmethod
    def _static_dimension(value: Any) -> int:
        if isinstance(value, (int, np.integer)) and int(value) > 0:
            return int(value)
        return DEFAULT_INPUT_SIZE

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def device_used(self) -> str:
        return self._device_used

    @property
    def provider_used(self) -> str:
        return self._provider_used

    @property
    def backend(self) -> str:
        return "onnx"

    @property
    def device_verified(self) -> bool:
        return True

    @property
    def supports_concurrent_inference(self) -> bool:
        # ONNX Runtime sessions permit concurrent Run calls and preprocessing
        # scratch is thread-local.
        return True

    def detect(
        self,
        frame: np.ndarray,
        timestamp: datetime | None = None,
    ) -> list[PersonDetection]:
        image = self._validate_frame(frame)
        detection_timestamp = utc_now() if timestamp is None else timestamp
        if (
            not isinstance(detection_timestamp, datetime)
            or detection_timestamp.tzinfo is None
            or detection_timestamp.utcoffset() is None
        ):
            raise PersonDetectionError("detection timestamp must be timezone-aware")
        tensor, scale, pad_x, pad_y = self._preprocess(image)
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            raise PersonDetectionError(
                f"person detection inference failed for {self._model_path}: {exc}"
            ) from exc

        rows = self._validate_output(outputs)
        height, width = image.shape[:2]
        detections: list[PersonDetection] = []
        for row in rows:
            confidence = float(row[4])
            class_id = float(row[5])
            if not all(math.isfinite(value) for value in row[:6]):
                continue
            if int(class_id) != 0 or confidence < self._confidence_threshold:
                continue

            x1 = (float(row[0]) - pad_x) / scale
            y1 = (float(row[1]) - pad_y) / scale
            x2 = (float(row[2]) - pad_x) / scale
            y2 = (float(row[3]) - pad_y) / scale
            bbox = (
                max(0.0, min(float(width), x1)),
                max(0.0, min(float(height), y1)),
                max(0.0, min(float(width), x2)),
                max(0.0, min(float(height), y2)),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            detections.append(
                PersonDetection(
                    bbox=bbox,
                    confidence=confidence,
                    timestamp=detection_timestamp,
                )
            )
        return detections

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise PersonDetectionError("person detection frame must have shape HxWx3")
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise PersonDetectionError("person detection frame cannot be empty")
        if image.dtype == np.uint8:
            return image
        return np.clip(image, 0, 255).astype(np.uint8)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        scale = min(self._input_width / width, self._input_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = self._resize(image, resized_width, resized_height)

        canvas = getattr(self._scratch, "canvas", None)
        expected_shape = (self._input_height, self._input_width, 3)
        if canvas is None or canvas.shape != expected_shape:
            canvas = np.empty(expected_shape, dtype=np.uint8)
            self._scratch.canvas = canvas
        canvas.fill(114)
        pad_x = (self._input_width - resized_width) // 2
        pad_y = (self._input_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

        try:
            import cv2

            tensor = cv2.dnn.blobFromImage(
                canvas,
                scalefactor=1.0 / 255.0,
                size=(self._input_width, self._input_height),
                mean=(0.0, 0.0, 0.0),
                swapRB=True,
                crop=False,
            )
        except ImportError:
            rgb = canvas[:, :, ::-1]
            tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        return tensor, scale, pad_x, pad_y

    @staticmethod
    def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
        try:
            import cv2

            return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # OpenCV is a project dependency, but this fallback keeps the core
            # adapter testable in a minimal Python environment.
            y_indices = np.linspace(0, image.shape[0] - 1, height).round().astype(int)
            x_indices = np.linspace(0, image.shape[1] - 1, width).round().astype(int)
            return image[y_indices][:, x_indices]

    @staticmethod
    def _validate_output(outputs: Any) -> np.ndarray:
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise PersonDetectionError(
                "unsupported person detection output: expected one YOLO26 output tensor"
            )
        output = np.asarray(outputs[0])
        if output.ndim != 3 or output.shape[0] < 1 or output.shape[2] != 6:
            raise PersonDetectionError(
                "unsupported person detection output: expected shape (batch, detections, 6)"
            )
        return output[0]

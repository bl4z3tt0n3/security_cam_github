"""Local YOLOE adapters for prompted person and segmentation detection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
import gc
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .base import (
    MaskPolygon,
    PersonDetection,
    PersonDetectionError,
    PersonDetector,
    utc_now,
)
from .prompts import normalize_prompts


PERSON_PROMPTS = ("person",)
DEFAULT_IMAGE_SIZE = 640
logger = logging.getLogger(__name__)


def _cuda_is_available() -> bool:
    """Return CUDA availability without importing torch at module import time."""

    try:
        import torch
    except (ImportError, OSError):
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _to_numpy(value: Any, *, label: str) -> np.ndarray:
    """Convert Ultralytics/torch values to a NumPy array for validation."""

    if value is None:
        raise PersonDetectionError(f"YOLOE output is missing {label}")
    try:
        detached = value.detach() if callable(getattr(value, "detach", None)) else value
        on_cpu = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
        return np.asarray(on_cpu)
    except Exception as exc:
        raise PersonDetectionError(f"cannot read YOLOE output field {label}: {exc}") from exc


class YoloEPersonDetector(PersonDetector):
    """Run one prompted YOLOE model and return person detections.

    Ultralytics import and model construction are deliberately lazy from the
    package perspective. ``model_factory`` and ``cuda_available`` are test
    seams and do not affect the production API.
    """

    def __init__(
        self,
        model: str,
        *,
        confidence_threshold: float = 0.5,
        device: str = "auto",
        prompts: Sequence[str] = PERSON_PROMPTS,
        image_size: int = DEFAULT_IMAGE_SIZE,
        model_factory: Callable[[str], Any] | None = None,
        cuda_available: Callable[[], bool] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger("person_detector")
        self._model_spec = self._validate_model_spec(model)
        if self._model_spec.lower().endswith(".onnx"):
            raise PersonDetectionError(
                "YOLOE text prompts require a .pt checkpoint or official model identifier; "
                "use backend=onnx for an exported ONNX model"
            )
        self._confidence_threshold = self._validate_threshold(confidence_threshold)
        self._requested_device = self._validate_device(device)
        self._prompts = self._validate_prompts(prompts)
        self._image_size = self._validate_image_size(image_size)
        self._cuda_available = cuda_available or _cuda_is_available
        self._device_used = self._resolve_device(self._requested_device)
        self._device_verified = False
        self._fallback_attempted = False

        try:
            if model_factory is None:
                try:
                    from ultralytics import YOLOE
                except (ImportError, OSError) as exc:
                    raise PersonDetectionError(
                        "Ultralytics is not installed or cannot be imported; "
                        "install the person-detection extra"
                    ) from exc
                model_factory = YOLOE
            self._model = model_factory(self._model_spec)
            set_classes = getattr(self._model, "set_classes", None)
            if not callable(set_classes):
                raise PersonDetectionError("YOLOE model does not expose set_classes()")
            set_classes(list(self._prompts))
        except PersonDetectionError:
            raise
        except Exception as exc:
            raise PersonDetectionError(
                f"cannot load/download YOLOE model {self._model_spec}: {exc}"
            ) from exc

        self._logger.info(
            "person_detector backend=yoloe model=%s device=%s prompts=%s",
            self._model_spec,
            self._device_used,
            list(self._prompts),
        )

    @staticmethod
    def _validate_model_spec(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("person_detection.model is required when detection is enabled")
        return value.strip()

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
    def _validate_prompts(value: Sequence[str]) -> tuple[str, ...]:
        try:
            return normalize_prompts(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid person detection prompts: {exc}") from exc

    @staticmethod
    def _validate_image_size(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("image_size must be a positive integer")
        if value > 2048:
            raise ValueError("image_size cannot be greater than 2048")
        return value

    def _resolve_device(self, requested: str) -> str:
        if requested == "cpu":
            return "cpu"
        available = bool(self._cuda_available())
        if requested == "cuda" and not available:
            raise PersonDetectionError(
                "CUDA device requested but torch.cuda.is_available() is false"
            )
        return "cuda:0" if available else "cpu"

    @property
    def model_spec(self) -> str:
        return self._model_spec

    @property
    def backend(self) -> str:
        return "yoloe"

    @property
    def prompts(self) -> tuple[str, ...]:
        return self._prompts

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def device_used(self) -> str:
        return self._device_used

    @property
    def device_verified(self) -> bool:
        return self._device_verified

    @property
    def provider_used(self) -> str:
        return "torch.cuda" if self._device_used.startswith("cuda") else "torch.cpu"

    def detect(
        self,
        frame: np.ndarray,
        timestamp: datetime | None = None,
    ) -> list[PersonDetection]:
        image = self._validate_frame(frame)
        detection_timestamp = self._validate_timestamp(timestamp)
        started = perf_counter()
        results = self._predict(image)
        detections = self._extract_detections(results, image.shape[:2], detection_timestamp)
        self._logger.debug(
            "person_detector backend=yoloe persons=%d inference_ms=%.2f device=%s",
            len(detections),
            (perf_counter() - started) * 1000.0,
            self._device_used,
        )
        return detections

    def close(self) -> None:
        self._model = None

    def _predict(self, image: np.ndarray) -> Any:
        predict = getattr(self._model, "predict", None)
        if not callable(predict):
            raise PersonDetectionError("YOLOE model does not expose predict()")

        kwargs = {
            "source": image,
            "conf": self._confidence_threshold,
            "device": self._device_used,
            "imgsz": self._image_size,
            "stream": False,
            "verbose": False,
        }
        try:
            results = predict(**kwargs)
        except Exception as exc:
            if (
                self._requested_device == "auto"
                and self._device_used.startswith("cuda")
                and not self._fallback_attempted
            ):
                self._fallback_attempted = True
                previous_device = self._device_used
                self._device_used = "cpu"
                self._logger.warning(
                    "person_detector CUDA inference failed; retrying once on CPU: %s",
                    exc,
                )
                kwargs["device"] = self._device_used
                try:
                    results = predict(**kwargs)
                except Exception as fallback_exc:
                    raise PersonDetectionError(
                        "YOLOE inference failed on CUDA and CPU fallback"
                    ) from fallback_exc
                self._logger.info(
                    "person_detector device_fallback from=%s to=cpu",
                    previous_device,
                )
            else:
                raise PersonDetectionError(
                    f"YOLOE inference failed on device {self._device_used}: {exc}"
                ) from exc
        self._device_verified = True
        return results

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> np.ndarray:
        try:
            image = np.asarray(frame)
        except Exception as exc:
            raise PersonDetectionError(f"cannot read person detection frame: {exc}") from exc
        if image.ndim != 3 or image.shape[2] != 3:
            raise PersonDetectionError("person detection frame must have shape HxWx3")
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise PersonDetectionError("person detection frame cannot be empty")
        if not np.issubdtype(image.dtype, np.number):
            raise PersonDetectionError("person detection frame must contain numeric pixels")
        try:
            if not bool(np.isfinite(image).all()):
                raise PersonDetectionError("person detection frame must contain finite pixels")
        except TypeError as exc:
            raise PersonDetectionError("person detection frame must contain numeric pixels") from exc
        if image.dtype == np.uint8:
            return image
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _validate_timestamp(timestamp: datetime | None) -> datetime:
        value = utc_now() if timestamp is None else timestamp
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise PersonDetectionError("detection timestamp must be timezone-aware")
        return value

    def _extract_detections(
        self,
        results: Any,
        frame_shape: tuple[int, int],
        timestamp: datetime,
    ) -> list[PersonDetection]:
        if results is None:
            raise PersonDetectionError("YOLOE returned no results")
        try:
            result_list = list(results) if not isinstance(results, (list, tuple)) else list(results)
        except Exception as exc:
            raise PersonDetectionError(f"cannot read YOLOE results: {exc}") from exc
        if len(result_list) != 1:
            raise PersonDetectionError(
                f"YOLOE returned {len(result_list)} results for one frame; expected exactly one"
            )
        result = result_list[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            raise PersonDetectionError("YOLOE result does not contain boxes")

        xyxy = _to_numpy(getattr(boxes, "xyxy", None), label="boxes.xyxy")
        confidence = _to_numpy(getattr(boxes, "conf", None), label="boxes.conf").reshape(-1)
        classes = _to_numpy(getattr(boxes, "cls", None), label="boxes.cls").reshape(-1)
        if xyxy.size == 0 and xyxy.ndim == 1:
            xyxy = np.empty((0, 4), dtype=np.float32)
        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise PersonDetectionError("YOLOE boxes.xyxy must have shape (N, 4)")
        if len(confidence) != len(xyxy) or len(classes) != len(xyxy):
            raise PersonDetectionError("YOLOE box fields have inconsistent lengths")

        height, width = frame_shape
        detections: list[PersonDetection] = []
        for index, row in enumerate(xyxy):
            try:
                class_id = float(classes[index])
                score = float(confidence[index])
                coordinates = tuple(float(value) for value in row)
            except (TypeError, ValueError) as exc:
                raise PersonDetectionError("YOLOE output contains non-numeric box data") from exc
            if not all(math.isfinite(value) for value in (*coordinates, score, class_id)):
                continue
            if abs(class_id - round(class_id)) > 1e-6:
                continue
            integer_class_id = int(round(class_id))
            if self._label_for(result, integer_class_id) not in (None, "person"):
                continue
            if integer_class_id != 0:
                continue
            if score < self._confidence_threshold or score > 1.0:
                continue

            x1, y1, x2, y2 = coordinates
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
                    confidence=score,
                    timestamp=timestamp,
                )
            )
        return detections

    @staticmethod
    def _label_for(result: Any, class_id: int) -> str | None:
        names = getattr(result, "names", None)
        label: Any = None
        if isinstance(names, dict):
            label = names.get(class_id, names.get(str(class_id)))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            label = names[class_id]
        return None if label is None else str(label).strip().lower()


class YoloESegmentationDetector(PersonDetector):
    """Run a local prompted YOLOE segmentation checkpoint for the Windows UI."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        prompts: str | tuple[str, ...] | list[str] = ("person",),
        confidence_threshold: float = 0.0,
        device: str = "auto",
        text_encoder_path: Path | str | None = None,
    ) -> None:
        self._model_path = Path(model_path).expanduser()
        self._requested_device = self._validate_device(device)
        self._confidence_threshold = self._validate_confidence(confidence_threshold)
        self._prompts = normalize_prompts(prompts)
        self._text_encoder_path = (
            Path(text_encoder_path).expanduser()
            if text_encoder_path is not None
            else self._model_path.parent / "mobileclip2_b.ts"
        )
        self._model: Any | None = None
        self._text_encoder: Any | None = None
        self._torch: Any | None = None
        self._torch_device: Any | None = None
        self._fallback_reason: str | None = None

        if self._model_path.suffix.lower() != ".pt":
            raise PersonDetectionError("YOLOE Windows richiede un checkpoint PyTorch .pt")
        if not self._model_path.is_file():
            raise PersonDetectionError(f"YOLOE model not found: {self._model_path}")
        if not self._text_encoder_path.is_file():
            raise PersonDetectionError(
                "encoder prompt YOLOE non trovato: " f"{self._text_encoder_path}"
            )

        try:
            import torch
            from ultralytics import YOLOE
            from ultralytics.nn.text_model import MobileCLIPTS
            from ultralytics.nn.tasks import YOLOEModel
        except ImportError as exc:
            raise PersonDetectionError(
                "YOLOE richiede i pacchetti locali ultralytics e torch"
            ) from exc

        self._torch = torch
        self._yoloe_class = YOLOE
        self._yoloe_model_class = YOLOEModel
        self._text_encoder_class = MobileCLIPTS

        if self._requested_device == "cuda" and not torch.cuda.is_available():
            raise PersonDetectionError("CUDA richiesto ma torch.cuda.is_available() è falso")

        if self._requested_device == "auto" and torch.cuda.is_available():
            try:
                self._load_on_device(torch.device("cuda:0"))
                self._probe_inference()
                return
            except Exception as exc:
                self._fallback_reason = f"CUDA non verificato: {exc}"
                logger.warning("YOLOE auto fallback CPU: %s", exc)
                self._release_resources()

        target = (
            torch.device("cuda:0") if self._requested_device == "cuda" else torch.device("cpu")
        )
        self._load_on_device(target)
        if self._requested_device == "cuda":
            self._probe_inference()

    @staticmethod
    def _validate_device(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        return normalized

    @staticmethod
    def _validate_confidence(value: float) -> float:
        normalized = float(value)
        if not 0 <= normalized <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return normalized

    @property
    def backend(self) -> str:
        return "yoloe"

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def prompts(self) -> tuple[str, ...]:
        return self._prompts

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    @property
    def device_used(self) -> str:
        if self._torch_device is None:
            return "unknown"
        return str(self._torch_device)

    @property
    def provider_used(self) -> str:
        return "PyTorch/Ultralytics"

    def _load_on_device(self, device: Any) -> None:
        assert self._torch is not None
        model = None
        text_encoder = None
        try:
            model = self._yoloe_class(str(self._model_path), verbose=False)
            if str(getattr(model, "task", "")).lower() != "segment":
                raise PersonDetectionError("il checkpoint YOLOE deve avere task=segment")
            core = getattr(model, "model", None)
            if core is None or not hasattr(core, "to"):
                raise PersonDetectionError("checkpoint YOLOE privo di modulo PyTorch")
            if not isinstance(core, self._yoloe_model_class):
                raise PersonDetectionError("il checkpoint .pt non è un modello YOLOE")
            core.to(device)
            core.eval()
            text_encoder = self._text_encoder_class(
                device=device,
                weight=str(self._text_encoder_path),
            )
            self._model = model
            self._text_encoder = text_encoder
            self._torch_device = device
            self._apply_prompts(self._prompts)
        except PersonDetectionError:
            if text_encoder is not None:
                del text_encoder
            if model is not None:
                del model
            self._release_resources()
            raise
        except Exception as exc:
            if text_encoder is not None:
                del text_encoder
            if model is not None:
                del model
            self._release_resources()
            raise PersonDetectionError(
                f"impossibile caricare/configurare YOLOE {self._model_path.name}: {exc}"
            ) from exc

    def _apply_prompts(self, prompts: tuple[str, ...]) -> None:
        if self._model is None or self._text_encoder is None or self._torch is None:
            raise PersonDetectionError("YOLOE non è inizializzato")
        normalized = normalize_prompts(prompts)
        with self._torch.inference_mode():
            tokens = self._text_encoder.tokenize(list(normalized))
            text_features = self._text_encoder.encode_text(tokens).detach()
            if text_features.ndim == 2:
                text_features = text_features.unsqueeze(0)
            core = self._model.model
            parameter = next(core.parameters())
            text_features = text_features.to(device=parameter.device)
            head = core.model[-1]
            embeddings = head.get_tpe(text_features).to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            self._model.set_classes(list(normalized), embeddings=embeddings)
        self._prompts = normalized

    def _probe_inference(self) -> None:
        if self._model is None or self._torch_device is None:
            raise PersonDetectionError("YOLOE non è pronto per la probe")
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        with self._torch.inference_mode():
            results = self._model.predict(
                source=image,
                imgsz=64,
                conf=max(0.01, self._confidence_threshold),
                device=self._torch_device,
                verbose=False,
                save=False,
                batch=1,
            )
        if not results:
            raise PersonDetectionError("probe inferenza YOLOE senza risultato")

    def detect(
        self,
        frame: np.ndarray,
        timestamp: datetime | None = None,
    ) -> list[PersonDetection]:
        if self._model is None or self._torch_device is None or self._torch is None:
            raise PersonDetectionError("YOLOE non è pronto")
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 1 or image.shape[1] < 1:
            raise PersonDetectionError("frame YOLOE non valido: atteso HxWx3")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        try:
            with self._torch.inference_mode():
                results = self._model.predict(
                    source=image,
                    conf=max(0.001, self._confidence_threshold),
                    device=self._torch_device,
                    verbose=False,
                    save=False,
                    batch=1,
                )
        except Exception as exc:
            raise PersonDetectionError(
                f"inferenza YOLOE fallita per {self._model_path.name}: {exc}"
            ) from exc

        if not results:
            return []
        return self._detections_from_result(results[0], image, timestamp)

    def _detections_from_result(
        self,
        result: Any,
        image: np.ndarray,
        timestamp: datetime | None = None,
    ) -> list[PersonDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = self._to_numpy(getattr(boxes, "xyxy", None))
        confidences = self._to_numpy(getattr(boxes, "conf", None)).reshape(-1)
        classes = self._to_numpy(getattr(boxes, "cls", None)).reshape(-1)
        masks = getattr(result, "masks", None)
        polygons = getattr(masks, "xy", None) if masks is not None else None
        names = getattr(result, "names", {})
        height, width = image.shape[:2]
        detection_timestamp = utc_now() if timestamp is None else timestamp
        detections: list[PersonDetection] = []

        for index, raw_box in enumerate(xyxy):
            if len(raw_box) < 4:
                continue
            class_id = int(classes[index]) if index < len(classes) else 0
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
            if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                continue
            if confidence < getattr(self, "_confidence_threshold", 0.0):
                continue
            x1, y1, x2, y2 = [float(value) for value in raw_box[:4]]
            bbox = (
                max(0.0, min(float(width), x1)),
                max(0.0, min(float(height), y1)),
                max(0.0, min(float(width), x2)),
                max(0.0, min(float(height), y2)),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            label = self._class_label(names, class_id)
            mask_polygon = self._mask_polygon(polygons, index, width, height)
            detections.append(
                PersonDetection(
                    bbox=bbox,
                    confidence=confidence,
                    timestamp=detection_timestamp,
                    class_id=class_id,
                    label=label,
                    mask_polygon=mask_polygon,
                )
            )
        return detections

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0,), dtype=np.float32)
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    @staticmethod
    def _class_label(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, names.get(str(class_id), class_id)))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _mask_polygon(
        polygons: Any,
        index: int,
        width: int,
        height: int,
    ) -> MaskPolygon | None:
        if polygons is None or index >= len(polygons):
            return None
        raw = YoloESegmentationDetector._to_numpy(polygons[index])
        if raw.ndim != 2 or raw.shape[1] < 2 or len(raw) < 3:
            return None
        points = tuple(
            (
                max(0.0, min(float(width), float(point[0]))),
                max(0.0, min(float(height), float(point[1]))),
            )
            for point in raw
        )
        return points if len(points) >= 3 else None

    @property
    def device_verified(self) -> bool:
        return self._model is not None and self._torch_device is not None

    def close(self) -> None:
        self._release_resources()

    def _release_resources(self) -> None:
        model, encoder = self._model, self._text_encoder
        self._model = None
        self._text_encoder = None
        self._torch_device = None
        if model is not None:
            del model
        if encoder is not None:
            del encoder
        gc.collect()
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

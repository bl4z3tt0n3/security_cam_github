"""OpenVINO adapter for the YOLO26 person-detection checkpoint.

The adapter deliberately keeps the same small ``PersonDetector`` contract used
by the camera sampler.  Ultralytics and OpenVINO are imported only when an
enabled detector is constructed; the optional dependency therefore remains
optional for YOLOE, ONNX, fake, and disabled installations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml

from .base import PersonDetection, PersonDetectionError, PersonDetector, utc_now


OPENVINO_OFFICIAL_CHECKPOINTS = frozenset({"yolo26s.pt", "yolo26n.pt"})
OPENVINO_PERSON_CLASSES = ("person",)
DEFAULT_OPENVINO_IMAGE_SIZE = 640
_CACHE_MARKER = ".person_detector.json"


def _as_numpy(value: Any, *, label: str) -> np.ndarray:
    """Read tensor-like Ultralytics fields without importing torch here."""

    if value is None:
        raise PersonDetectionError(f"OpenVINO output is missing {label}")
    try:
        detached = value.detach() if callable(getattr(value, "detach", None)) else value
        on_cpu = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
        return np.asarray(on_cpu)
    except Exception as exc:
        raise PersonDetectionError(f"cannot read OpenVINO output field {label}: {exc}") from exc


class OpenVINOPersonDetector(PersonDetector):
    """Run a YOLO26 detection checkpoint through an exported OpenVINO IR.

    ``model`` may be a local ``.pt`` checkpoint, one of the two official
    checkpoint identifiers (``yolo26s.pt``/``yolo26n.pt``), or an existing
    OpenVINO IR directory/XML file.  Missing official checkpoints are fetched
    only through Ultralytics' official asset helper.  A small JSON marker next
    to the IR complements Ultralytics' metadata and prevents an FP16/FP32 or
    input-size mismatch from silently reusing a stale cache.

    The optional factories are intentionally narrow test seams.  Production
    code leaves them unset, preserving lazy imports and the real OpenVINO
    ``EXECUTION_DEVICES`` verification path.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        confidence_threshold: float = 0.45,
        precision: str = "fp16",
        device: str = "auto",
        fallback_device: str = "none",
        classes: Sequence[str] = OPENVINO_PERSON_CLASSES,
        image_size: int = DEFAULT_OPENVINO_IMAGE_SIZE,
        model_root: Path | str | None = None,
        core_factory: Callable[[], Any] | None = None,
        yolo_factory: Callable[..., Any] | None = None,
        download_fn: Callable[[Path], str | Path] | None = None,
        execution_devices_reader: Callable[[Any], Sequence[str] | None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger("person_detector")
        self._model_spec = self._validate_model_spec(model)
        self._confidence_threshold = self._validate_threshold(confidence_threshold)
        self._requested_precision = self._validate_precision(precision)
        self._precision_used = self._requested_precision
        self._requested_device = self._validate_device(device)
        self._fallback_device = self._validate_fallback(fallback_device)
        self._classes = self._validate_classes(classes)
        self._image_size = self._validate_image_size(image_size)
        self._model_root = Path(model_root or Path.cwd()).expanduser().resolve()
        self._core_factory = core_factory
        self._yolo_factory = yolo_factory
        self._download_fn = download_fn
        self._execution_devices_reader = execution_devices_reader
        self._ov: Any | None = None
        self._core: Any | None = None
        self._model: Any | None = None
        self._source_checkpoint: Path | None = None
        self._cache_path: Path | None = None
        self._available_devices: tuple[str, ...] = ()
        self._target_device = ""
        self._device_used = "unknown"
        self._actual_execution_devices: tuple[str, ...] = ()
        self._device_verified = False
        self._fallback_reason: str | None = None

        self._load_core()
        self._resolve_target_device()
        self._prepare_ir_cache(self._requested_precision)

        self._logger.info(
            "person_detector backend=openvino model=%s precision=%s device=%s available=%s",
            self._model_spec,
            self._precision_used,
            self._target_device,
            list(self._available_devices),
        )

    @staticmethod
    def _validate_model_spec(value: str | Path) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("person_detection.model is required when detection is enabled")
        return normalized

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
    def _validate_precision(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"fp16", "fp32"}:
            raise ValueError("precision must be one of: fp16, fp32")
        return normalized

    @staticmethod
    def _validate_device(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"auto", "cpu", "gpu"}:
            raise ValueError("OpenVINO device must be one of: auto, cpu, gpu")
        return normalized

    @staticmethod
    def _validate_fallback(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"none", "cpu"}:
            raise ValueError("fallback_device must be one of: none, cpu")
        return normalized

    @staticmethod
    def _validate_classes(value: Sequence[str]) -> tuple[str, ...]:
        if isinstance(value, str):
            values = (value,)
        else:
            values = tuple(value)
        normalized = tuple(str(item).strip().casefold() for item in values if str(item).strip())
        if normalized != OPENVINO_PERSON_CLASSES:
            raise ValueError("OpenVINO backend supports classes=['person'] only")
        return OPENVINO_PERSON_CLASSES

    @staticmethod
    def _validate_image_size(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("image_size must be a positive integer")
        if value > 2048:
            raise ValueError("image_size cannot be greater than 2048")
        return value

    @property
    def model_spec(self) -> str:
        return self._model_spec

    @property
    def model_path(self) -> Path | None:
        return self._source_checkpoint

    @property
    def cache_path(self) -> Path | None:
        return self._cache_path

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def precision(self) -> str:
        return self._precision_used

    @property
    def requested_precision(self) -> str:
        return self._requested_precision

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    @property
    def available_devices(self) -> tuple[str, ...]:
        return self._available_devices

    @property
    def execution_devices(self) -> tuple[str, ...]:
        return self._actual_execution_devices

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    @property
    def device_used(self) -> str:
        return self._device_used

    @property
    def provider_used(self) -> str:
        if self._actual_execution_devices:
            return f"OpenVINO/{','.join(self._actual_execution_devices)}"
        return f"OpenVINO/{self._target_device or 'unknown'}"

    @property
    def backend(self) -> str:
        return "openvino"

    @property
    def device_verified(self) -> bool:
        return self._device_verified

    def detect(
        self,
        frame: np.ndarray,
        timestamp: datetime | None = None,
    ) -> list[PersonDetection]:
        image = self._validate_frame(frame)
        detection_timestamp = self._validate_timestamp(timestamp)

        try:
            results = self._predict(image)
        except Exception as exc:
            raise PersonDetectionError(
                f"OpenVINO inference failed on {self._target_device}: {exc}"
            ) from exc

        return self._extract_detections(
            results,
            image.shape[:2],
            detection_timestamp,
            self._confidence_threshold,
        )

    def close(self) -> None:
        self._model = None
        self._device_verified = False

    def _load_core(self) -> None:
        try:
            if self._core_factory is not None:
                self._core = self._core_factory()
            else:
                try:
                    import openvino as ov
                except (ImportError, OSError) as exc:
                    raise PersonDetectionError(
                        "OpenVINO non è installato; installare l'extra 'openvino'"
                    ) from exc
                self._ov = ov
                self._core = ov.Core()
            devices = getattr(self._core, "available_devices", ())
            self._available_devices = tuple(str(device) for device in devices)
        except PersonDetectionError:
            raise
        except Exception as exc:
            raise PersonDetectionError(f"cannot initialize OpenVINO Core: {exc}") from exc

        if not self._available_devices:
            raise PersonDetectionError("OpenVINO Core reports no available devices")

    @staticmethod
    def _device_family(device: str) -> str:
        return str(device).strip().split(".", 1)[0].upper()

    def _find_available(self, family: str) -> str | None:
        wanted = family.upper()
        for device in self._available_devices:
            if self._device_family(device) == wanted:
                return device
        return None

    def _resolve_target_device(self) -> None:
        if self._requested_device == "auto":
            selected = self._find_available("GPU") or self._find_available("CPU")
            if selected is None:
                raise PersonDetectionError(
                    "OpenVINO auto device could not resolve CPU or GPU from Core.available_devices"
                )
        elif self._requested_device == "gpu":
            selected = self._find_available("GPU")
            if selected is None:
                if self._fallback_device != "cpu":
                    raise PersonDetectionError(
                        "OpenVINO GPU requested but no GPU is present in Core.available_devices"
                    )
                selected = self._find_available("CPU")
                if selected is None:
                    raise PersonDetectionError(
                        "OpenVINO GPU fallback requested but CPU is not available"
                    )
                self._fallback_reason = "GPU non disponibile in Core.available_devices"
                self._logger.warning(
                    "person_detector backend=openvino device_fallback from=GPU to=%s reason=%s",
                    selected,
                    self._fallback_reason,
                )
        else:
            selected = self._find_available("CPU")
            if selected is None:
                raise PersonDetectionError("OpenVINO CPU is not available in Core.available_devices")

        self._target_device = selected
        self._device_used = selected

    def _resolve_model_path(self) -> tuple[Path | None, Path]:
        candidate = Path(self._model_spec).expanduser()
        if not candidate.is_absolute():
            if (
                candidate.parent == Path(".")
                and candidate.name.casefold() in OPENVINO_OFFICIAL_CHECKPOINTS
            ):
                candidate = self._model_root / "models" / candidate.name
            else:
                candidate = self._model_root / candidate

        if candidate.is_dir():
            return None, candidate
        if candidate.suffix.lower() == ".xml":
            return None, candidate.parent
        if candidate.is_file():
            if candidate.suffix.lower() != ".pt":
                raise PersonDetectionError(
                    "OpenVINO model must be a YOLO26 .pt checkpoint or an OpenVINO IR directory"
                )
            return candidate, self._cache_directory(candidate, self._requested_precision)

        name = candidate.name.casefold()
        if name not in OPENVINO_OFFICIAL_CHECKPOINTS:
            raise PersonDetectionError(
                "OpenVINO official download is restricted to yolo26s.pt and yolo26n.pt"
            )
        candidate.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = self._download_official(candidate)
        except Exception as exc:
            raise PersonDetectionError(
                f"cannot download official OpenVINO checkpoint {candidate.name}: {exc}"
            ) from exc
        if not downloaded.is_file():
            raise PersonDetectionError(
                f"official checkpoint download did not produce a file: {downloaded}"
            )
        return downloaded, self._cache_directory(downloaded, self._requested_precision)

    @staticmethod
    def _cache_directory(checkpoint: Path, precision: str) -> Path:
        suffix = "_fp32_openvino_model" if precision == "fp32" else "_openvino_model"
        return checkpoint.with_name(f"{checkpoint.stem}{suffix}")

    def _download_official(self, target: Path) -> Path:
        if target.name.casefold() not in OPENVINO_OFFICIAL_CHECKPOINTS:
            raise ValueError("only yolo26s.pt and yolo26n.pt may be downloaded")
        if self._download_fn is not None:
            result = Path(self._download_fn(target))
        else:
            try:
                from ultralytics.utils.downloads import attempt_download_asset
            except (ImportError, OSError) as exc:
                raise PersonDetectionError(
                    "Ultralytics is required for the official YOLO26 checkpoint download"
                ) from exc
            result = Path(attempt_download_asset(target))

        if result.resolve() != target.resolve() and result.is_file() and not target.is_file():
            shutil.copy2(result, target)
            result = target
        return result

    def _prepare_ir_cache(self, precision: str) -> None:
        source, cache = self._resolve_model_path() if self._source_checkpoint is None else (
            self._source_checkpoint,
            self._cache_directory(self._source_checkpoint, precision),
        )
        if source is None:
            if not self._cache_is_valid(cache, precision, self._image_size):
                raise PersonDetectionError(
                    f"OpenVINO IR cache is missing or metadata is incompatible: {cache}"
                )
            self._cache_path = cache
            return

        self._source_checkpoint = source
        cache = self._cache_directory(source, precision)
        if self._cache_is_valid(cache, precision, self._image_size):
            self._cache_path = cache
            return

        export_source = source
        temporary_source: Path | None = None
        if precision == "fp32":
            # Ultralytics derives the output directory from the checkpoint stem.
            # Export through a temporary stem so a FP32 fallback never overwrites
            # the primary FP16 ``*_openvino_model`` cache.
            export_source = source.with_name(f"{source.stem}_fp32{source.suffix}")
            if not export_source.exists():
                shutil.copy2(source, export_source)
                temporary_source = export_source

        try:
            model = self._new_yolo(export_source, for_export=True)
            task = str(getattr(model, "task", "detect") or "detect").casefold()
            if task != "detect":
                raise PersonDetectionError(
                    f"OpenVINO requires a detection checkpoint (task=detect), got task={task}"
                )

            kwargs: dict[str, Any] = {
                "format": "openvino",
                "imgsz": self._image_size,
                "dynamic": False,
                "nms": False,
                "device": "cpu",
                "verbose": False,
            }
            if precision == "fp16":
                # Ultralytics maps quantize=16 to OpenVINO's compress_to_fp16=True.
                kwargs["quantize"] = 16
            try:
                exported = model.export(**kwargs)
            except Exception as exc:
                raise PersonDetectionError(
                    f"OpenVINO export failed for {source.name}: {exc}"
                ) from exc
        finally:
            if temporary_source is not None:
                temporary_source.unlink(missing_ok=True)

        exported_path = Path(str(exported)) if exported is not None else cache
        if exported_path.suffix.lower() == ".xml":
            exported_path = exported_path.parent
        if not self._has_ir_files(exported_path) and self._has_ir_files(cache):
            exported_path = cache
        if not self._has_ir_files(exported_path):
            raise PersonDetectionError(
                f"OpenVINO export returned no XML/BIN IR pair: {exported_path}"
            )
        if exported_path.resolve() != cache.resolve():
            self._logger.info(
                "OpenVINO exporter returned %s; publishing cache at %s",
                exported_path,
                cache,
            )
            shutil.copytree(exported_path, cache, dirs_exist_ok=True)
        self._write_cache_marker(cache, source, precision, task)
        if not self._cache_is_valid(cache, precision, self._image_size):
            raise PersonDetectionError(f"OpenVINO cache metadata validation failed: {cache}")
        self._cache_path = cache

    @staticmethod
    def _has_ir_files(path: Path) -> bool:
        return path.is_dir() and any(path.glob("*.xml")) and any(path.glob("*.bin"))

    @classmethod
    def _cache_marker_path(cls, cache: Path) -> Path:
        return cache / _CACHE_MARKER

    @classmethod
    def _cache_is_valid(cls, cache: Path, precision: str, image_size: int) -> bool:
        if not cls._has_ir_files(cache) or not (cache / "metadata.yaml").is_file():
            return False
        try:
            marker = json.loads(cls._cache_marker_path(cache).read_text(encoding="utf-8"))
            if marker.get("task") != "detect" or marker.get("precision") != precision:
                return False
            if int(marker.get("image_size")) != image_size:
                return False
            metadata = yaml.safe_load((cache / "metadata.yaml").read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                task = metadata.get("task")
                if task is None and isinstance(metadata.get("args"), dict):
                    task = metadata["args"].get("task")
                if task is not None and str(task).casefold() not in {"detect", "detection"}:
                    return False
        except (OSError, ValueError, TypeError, yaml.YAMLError, json.JSONDecodeError):
            return False
        return True

    def _write_cache_marker(
        self,
        cache: Path,
        source: Path,
        precision: str,
        task: str,
    ) -> None:
        cache.mkdir(parents=True, exist_ok=True)
        marker = {
            "format": "openvino",
            "source_checkpoint": source.name,
            "task": task,
            "precision": precision,
            "image_size": self._image_size,
        }
        self._cache_marker_path(cache).write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _new_yolo(self, model_path: Path, *, for_export: bool = False) -> Any:
        factory = self._yolo_factory
        if factory is None:
            try:
                from ultralytics import YOLO
            except (ImportError, OSError) as exc:
                raise PersonDetectionError(
                    "Ultralytics non è installato; installare l'extra person-detection"
                ) from exc
            factory = YOLO

        try:
            if for_export:
                return factory(str(model_path), verbose=False)
            return factory(str(model_path), task="detect", verbose=False)
        except TypeError:
            # Small fake factories and older Ultralytics versions may not accept
            # optional constructor keywords; the model contract remains the same.
            try:
                return factory(str(model_path))
            except Exception as exc:
                raise PersonDetectionError(
                    f"cannot load OpenVINO model {model_path}: {exc}"
                ) from exc
        except Exception as exc:
            raise PersonDetectionError(f"cannot load OpenVINO model {model_path}: {exc}") from exc

    def _load_runtime_model(self) -> Any:
        if self._cache_path is None:
            raise PersonDetectionError("OpenVINO IR cache is not prepared")
        model = self._new_yolo(self._cache_path)
        task = str(getattr(model, "task", "detect") or "detect").casefold()
        if task not in {"detect", "detection"}:
            raise PersonDetectionError(
                f"OpenVINO IR must expose task=detect, got task={task}"
            )
        self._model = model
        # A freshly loaded/compiled model must prove its execution device once.
        self._device_verified = False
        self._actual_execution_devices = ()
        return model

    def _predict(self, image: np.ndarray) -> Any:
        device_fallback_attempted = False
        precision_fallback_attempted = False
        while True:
            model = self._model or self._load_runtime_model()
            try:
                predict = getattr(model, "predict", None)
                if not callable(predict):
                    raise PersonDetectionError("OpenVINO model does not expose predict()")
                results = predict(
                    source=image,
                    conf=self._confidence_threshold,
                    imgsz=self._image_size,
                    device=f"intel:{self._target_device}",
                    classes=[0],
                    stream=False,
                    verbose=False,
                    save=False,
                )
                if not self._device_verified:
                    execution_devices = self._read_execution_devices(model)
                    self._verify_execution_devices(execution_devices)
                    self._actual_execution_devices = tuple(execution_devices)
                    matching = next(
                        (
                            device
                            for device in execution_devices
                            if self._device_family(device) == self._device_family(self._target_device)
                        ),
                        execution_devices[0],
                    )
                    self._device_used = matching
                    self._device_verified = True
                return results
            except Exception as exc:
                self._model = None
                if (
                    self._device_family(self._target_device) == "GPU"
                    and self._fallback_device == "cpu"
                    and not device_fallback_attempted
                ):
                    device_fallback_attempted = True
                    self._fallback_to_cpu(str(exc))
                    continue
                if (
                    self._device_family(self._target_device) == "CPU"
                    and self._precision_used == "fp16"
                    and not precision_fallback_attempted
                    and self._source_checkpoint is not None
                ):
                    precision_fallback_attempted = True
                    self._fallback_to_fp32(str(exc))
                    continue
                raise PersonDetectionError(str(exc)) from exc

    def _read_execution_devices(self, model: Any) -> tuple[str, ...]:
        if self._execution_devices_reader is not None:
            values = self._execution_devices_reader(model)
        else:
            values = None
            for owner in (
                getattr(getattr(model, "predictor", None), "model", None),
                getattr(model, "backend", None),
                model,
            ):
                compiled = getattr(owner, "ov_compiled_model", None) if owner is not None else None
                if compiled is not None and callable(getattr(compiled, "get_property", None)):
                    try:
                        values = compiled.get_property("EXECUTION_DEVICES")
                    except Exception:
                        values = None
                if values is None and owner is not None:
                    candidate = getattr(owner, "execution_devices", None)
                    if candidate is not None:
                        values = candidate
                if values is not None:
                    break
        normalized = tuple(str(value) for value in (values or ()) if str(value).strip())
        if not normalized:
            raise PersonDetectionError(
                "OpenVINO inference did not expose EXECUTION_DEVICES; real device verification failed"
            )
        return normalized

    def _verify_execution_devices(self, execution_devices: Sequence[str]) -> None:
        target_family = self._device_family(self._target_device)
        if not any(self._device_family(device) == target_family for device in execution_devices):
            raise PersonDetectionError(
                f"OpenVINO EXECUTION_DEVICES={list(execution_devices)} does not verify requested "
                f"device {self._target_device}"
            )

    def _fallback_to_cpu(self, reason: str) -> None:
        cpu = self._find_available("CPU")
        if cpu is None:
            raise PersonDetectionError("OpenVINO GPU fallback requested but CPU is unavailable")
        previous = self._target_device
        self._target_device = cpu
        self._device_used = cpu
        self._fallback_reason = reason
        self._logger.warning(
            "person_detector backend=openvino device_fallback from=%s to=%s reason=%s",
            previous,
            cpu,
            reason,
        )

    def _fallback_to_fp32(self, reason: str) -> None:
        if self._source_checkpoint is None:
            raise PersonDetectionError(
                f"OpenVINO FP16 cache failed and no checkpoint is available for FP32 fallback: {reason}"
            )
        previous = self._precision_used
        self._precision_used = "fp32"
        self._prepare_ir_cache("fp32")
        self._logger.warning(
            "person_detector backend=openvino precision_fallback from=%s to=fp32 reason=%s",
            previous,
            reason,
        )

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
        if not np.issubdtype(image.dtype, np.number) or np.issubdtype(
            image.dtype, np.complexfloating
        ):
            raise PersonDetectionError("person detection frame must contain real numeric pixels")
        # Integer pixels are finite by construction; scanning an entire uint8
        # frame before each inference is unnecessary work.
        if np.issubdtype(image.dtype, np.floating) and not bool(np.isfinite(image).all()):
            raise PersonDetectionError("person detection frame must contain finite pixels")
        return image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _validate_timestamp(timestamp: datetime | None) -> datetime:
        value = utc_now() if timestamp is None else timestamp
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise PersonDetectionError("detection timestamp must be timezone-aware")
        return value

    @staticmethod
    def _extract_detections(
        results: Any,
        frame_shape: tuple[int, int],
        timestamp: datetime,
        confidence_threshold: float = 0.0,
    ) -> list[PersonDetection]:
        if results is None:
            raise PersonDetectionError("OpenVINO returned no results")
        try:
            result_list = list(results)
        except Exception as exc:
            raise PersonDetectionError(f"cannot read OpenVINO results: {exc}") from exc
        if len(result_list) != 1:
            raise PersonDetectionError(
                f"OpenVINO returned {len(result_list)} results for one frame; expected exactly one"
            )
        result = result_list[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            raise PersonDetectionError("OpenVINO result does not contain detection boxes")
        xyxy = _as_numpy(getattr(boxes, "xyxy", None), label="boxes.xyxy")
        confidence = _as_numpy(getattr(boxes, "conf", None), label="boxes.conf").reshape(-1)
        classes = _as_numpy(getattr(boxes, "cls", None), label="boxes.cls").reshape(-1)
        if xyxy.size == 0 and xyxy.ndim == 1:
            xyxy = np.empty((0, 4), dtype=np.float32)
        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise PersonDetectionError("OpenVINO boxes.xyxy must have shape (N, 4)")
        if len(confidence) != len(xyxy) or len(classes) != len(xyxy):
            raise PersonDetectionError("OpenVINO box fields have inconsistent lengths")

        height, width = frame_shape
        detections: list[PersonDetection] = []
        for index, row in enumerate(xyxy):
            try:
                values = tuple(float(value) for value in row)
                score = float(confidence[index])
                class_id = float(classes[index])
            except (TypeError, ValueError) as exc:
                raise PersonDetectionError("OpenVINO output contains non-numeric box data") from exc
            if not all(math.isfinite(value) for value in (*values, score, class_id)):
                continue
            if abs(class_id - round(class_id)) > 1e-6 or int(round(class_id)) != 0:
                continue
            if score < confidence_threshold or score < 0 or score > 1:
                continue
            x1, y1, x2, y2 = values
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

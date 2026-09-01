"""Model-specific local face detectors and landmark adapters.

The three supported detectors deliberately keep separate preprocessing and
output decoders.  SCRFD, face-detection-0205 and YuNet do not share an output
contract, so a generic tabular ONNX parser would silently produce bad boxes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
import math

import cv2
import numpy as np

from app.hardware import ensure_process_memory_budget, resolve_cpu_thread_budget

from .base import (
    FaceDetection,
    FaceDetector,
    FaceDetectorError,
    FaceLandmark5,
    FaceLandmarker,
    crop_frame,
)
from .onnx import CPU_PROVIDER, CUDA_PROVIDER, _actual_provider, _select_provider
from .openvino_runtime import OpenVINOCoreManager
from .registry import artifact_sha256


def _validate_threshold(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be between 0 and 1")
    return threshold


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise FaceDetectorError("face detector input must be a non-empty BGR image")
    return image


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.4) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = (boxes[:, index] for index in range(4))
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - intersection
        overlap = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[overlap < threshold]
    return keep


class ScrfdFaceDetector(FaceDetector):
    """SCRFD 2.5G KPS ONNX Runtime adapter."""

    detector_id = "scrfd_2.5g_kps"
    backend_id = "onnxruntime"

    def __init__(
        self,
        model_path: Path | str,
        *,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "auto",
        session: Any | None = None,
        session_factory: Callable[..., Any] | None = None,
        input_size: tuple[int, int] = (640, 640),
        cpu_threads: int = 0,
        max_process_ram_mb: int = 0,
    ) -> None:
        self._model_path = Path(model_path).expanduser()
        self._confidence_threshold = _validate_threshold(confidence_threshold)
        self._nms_threshold = _validate_threshold(nms_threshold)
        self._device_requested = str(device).strip().lower()
        if self._device_requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("SCRFD device must be one of: auto, cpu, cuda")
        self._input_width, self._input_height = input_size
        self._cpu_threads = int(cpu_threads)
        self._max_process_ram_mb = int(max_process_ram_mb)
        if self._input_width <= 0 or self._input_height <= 0:
            raise ValueError("SCRFD input size must be positive")
        self._padded = np.empty((self._input_height, self._input_width, 3), dtype=np.uint8)
        self._center_cache: dict[tuple[int, bool], np.ndarray] = {}
        self._session_factory = session_factory
        self._session: Any | None = None
        self._input_name = ""
        self._provider_used = ""
        self._device_used = ""
        if session is None:
            self.load()
        else:
            self._configure_session(session)

    @property
    def device_used(self) -> str:
        return self._device_used

    @property
    def provider_used(self) -> str:
        return self._provider_used

    def load(self) -> None:
        if self._session is not None:
            return
        if not self._model_path.is_file():
            raise FaceDetectorError(f"SCRFD model not found: {self._model_path}")
        try:
            import onnxruntime as ort

            available = tuple(ort.get_available_providers())
            selected = _select_provider(self._device_requested, available)
            factory = self._session_factory or ort.InferenceSession
            kwargs: dict[str, Any] = {"providers": [selected]}
            if selected == CPU_PROVIDER:
                ensure_process_memory_budget(
                    self._max_process_ram_mb,
                    stage="SCRFD ONNX Runtime load",
                )
                options = ort.SessionOptions()
                options.intra_op_num_threads = resolve_cpu_thread_budget(self._cpu_threads)
                options.inter_op_num_threads = 1
                kwargs["sess_options"] = options
            try:
                session = factory(str(self._model_path), **kwargs)
            except TypeError:
                # Small injected test factories may not expose SessionOptions.
                kwargs.pop("sess_options", None)
                session = factory(str(self._model_path), **kwargs)
            self._configure_session(session, requested=selected)
        except FaceDetectorError:
            raise
        except Exception as exc:
            raise FaceDetectorError(f"cannot load SCRFD model {self._model_path}: {exc}") from exc

    def _configure_session(self, session: Any, *, requested: str | None = None) -> None:
        try:
            inputs = session.get_inputs()
            if not inputs:
                raise FaceDetectorError("SCRFD model has no input")
            self._input_name = str(inputs[0].name)
            providers = tuple(getattr(session, "get_providers", lambda: ())())
            selected = requested or _select_provider(self._device_requested, providers)
            self._provider_used = _actual_provider(session, selected)
            self._device_used = "cuda" if self._provider_used == CUDA_PROVIDER else "cpu"
            self._session = session
        except FaceDetectorError:
            raise
        except Exception as exc:
            raise FaceDetectorError(f"cannot inspect SCRFD session: {exc}") from exc

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        scale = min(self._input_width / width, self._input_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        padded = self._padded
        padded.fill(0)
        padded[:resized_height, :resized_width] = resized
        tensor = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 128.0,
            size=(self._input_width, self._input_height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return tensor, scale

    @staticmethod
    def _rows(value: Any, columns: int | None = None) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == 1:
            array = array[0]
        if columns is None:
            return array.reshape(-1)
        if array.size % columns != 0:
            raise FaceDetectorError(f"SCRFD output cannot be reshaped to columns={columns}")
        return array.reshape(-1, columns)

    def _centers_for_stride(self, stride: int, *, doubled: bool) -> np.ndarray:
        key = (stride, doubled)
        cached = self._center_cache.get(key)
        if cached is not None:
            return cached
        grid_width = math.ceil(self._input_width / stride)
        grid_height = math.ceil(self._input_height / stride)
        centers = np.stack(
            np.meshgrid(np.arange(grid_width), np.arange(grid_height)), axis=-1
        ).reshape(-1, 2).astype(np.float32)
        centers *= stride
        if doubled:
            centers = np.repeat(centers, 2, axis=0)
        centers.setflags(write=False)
        self._center_cache[key] = centers
        return centers

    def _decode_feature_outputs(
        self,
        outputs: Sequence[Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if len(outputs) not in {6, 9}:
            raise FaceDetectorError(f"SCRFD expected 6 or 9 outputs, got {len(outputs)}")
        scores: list[np.ndarray] = []
        boxes: list[np.ndarray] = []
        keypoints: list[np.ndarray] = []
        has_keypoints = len(outputs) == 9
        for index, stride in enumerate((8, 16, 32)):
            score = self._rows(outputs[index])
            box = self._rows(outputs[index + 3], 4)
            if len(score) != len(box):
                raise FaceDetectorError(f"SCRFD score/bbox length mismatch at stride {stride}")
            base_centers = self._centers_for_stride(stride, doubled=False)
            centers = (
                self._centers_for_stride(stride, doubled=True)
                if len(box) == len(base_centers) * 2
                else base_centers
            )
            if len(box) != len(centers):
                raise FaceDetectorError(
                    f"SCRFD stride {stride} has {len(box)} boxes; expected {len(centers)}"
                )
            boxes.append(
                np.column_stack(
                    (
                        centers[:, 0] - box[:, 0] * stride,
                        centers[:, 1] - box[:, 1] * stride,
                        centers[:, 0] + box[:, 2] * stride,
                        centers[:, 1] + box[:, 3] * stride,
                    )
                )
            )
            scores.append(score)
            if has_keypoints:
                kp = self._rows(outputs[index + 6], 10)
                if len(kp) != len(centers):
                    raise FaceDetectorError(f"SCRFD keypoint length mismatch at stride {stride}")
                keypoints.append((kp.reshape(-1, 5, 2) * stride + centers[:, None, :]).reshape(-1, 10))
        return (
            np.concatenate(boxes, axis=0),
            np.concatenate(scores, axis=0),
            np.concatenate(keypoints, axis=0) if keypoints else None,
        )

    def detect(self, frame: np.ndarray, threshold: float | None = None) -> list[FaceDetection]:
        image = _validate_frame(frame)
        confidence_threshold = self._confidence_threshold if threshold is None else _validate_threshold(threshold)
        if self._session is None:
            self.load()
        assert self._session is not None
        tensor, scale = self._preprocess(image)
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            raise FaceDetectorError(f"SCRFD inference failed: {exc}") from exc
        try:
            if len(outputs) in {6, 9}:
                boxes, scores, keypoints = self._decode_feature_outputs(outputs)
            else:
                rows = np.asarray(outputs[0])
                if rows.ndim == 3 and rows.shape[0] == 1:
                    rows = rows[0]
                if rows.ndim != 2 or rows.shape[1] < 5:
                    raise FaceDetectorError("SCRFD output is neither feature-map nor row format")
                boxes = rows[:, :4]
                scores = rows[:, 4]
                keypoints = rows[:, 5:15] if rows.shape[1] >= 15 else None
        except FaceDetectorError:
            raise
        except Exception as exc:
            raise FaceDetectorError(f"cannot decode SCRFD outputs: {exc}") from exc
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if np.any((scores < 0) | (scores > 1)):
            scores = 1.0 / (1.0 + np.exp(-scores))
        valid = np.isfinite(scores) & (scores >= confidence_threshold)
        valid &= np.isfinite(boxes).all(axis=1)
        boxes = boxes[valid]
        scores = scores[valid]
        if keypoints is not None:
            keypoints = keypoints[valid]
        frame_boxes = boxes / max(scale, 1e-9)
        keep = _nms(frame_boxes, scores, self._nms_threshold)
        source_height, source_width = image.shape[:2]
        results: list[FaceDetection] = []
        for index in keep:
            box = frame_boxes[index]
            box[[0, 2]] = np.clip(box[[0, 2]], 0, source_width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, source_height)
            landmarks = None
            if keypoints is not None:
                points = keypoints[index].reshape(5, 2) / max(scale, 1e-9)
                landmarks = FaceLandmark5(
                    tuple(
                        (float(np.clip(x, 0, source_width)), float(np.clip(y, 0, source_height)))
                        for x, y in points
                    )  # type: ignore[arg-type]
                )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            results.append(
                FaceDetection(
                    tuple(float(value) for value in box),
                    float(scores[index]),
                    landmarks=landmarks,
                    detector_id=self.detector_id,
                    backend=self.backend_id,
                    device=self._device_used,
                )
            )
        return results

    def close(self) -> None:
        self._session = None


class OpenVINOFaceDetector0205(FaceDetector):
    """Direct OpenVINO IR adapter for Intel face-detection-0205."""

    detector_id = "face_detection_0205"
    backend_id = "openvino"

    def __init__(
        self,
        model_path: Path | str,
        *,
        confidence_threshold: float = 0.5,
        device: str = "auto",
        compiled_model: Any | None = None,
        core: Any | None = None,
        cache_dir: Path | None = None,
        performance_mode: str = "latency",
        cpu_threads: int = 0,
        max_process_ram_mb: int = 0,
    ) -> None:
        self._model_path = Path(model_path).expanduser()
        self._confidence_threshold = _validate_threshold(confidence_threshold)
        self._performance_mode = str(performance_mode).strip().lower()
        self._cpu_threads = int(cpu_threads)
        self._max_process_ram_mb = int(max_process_ram_mb)
        self._device_requested = str(device).strip().lower()
        if self._device_requested not in {"auto", "cpu", "gpu", "npu"}:
            raise ValueError("OpenVINO device must be one of: auto, cpu, gpu, npu")
        self._compiled = compiled_model
        self._core = core
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self._input_name: str | None = None
        self._input_width = 416
        self._input_height = 416
        self._device_used = ""
        if compiled_model is None:
            self.load()
        else:
            self._inspect_compiled(compiled_model)

    @property
    def device_used(self) -> str:
        return self._device_used

    @property
    def provider_used(self) -> str:
        return self.backend_id

    def load(self) -> None:
        if self._compiled is not None:
            return
        xml_path = self._model_path
        bin_path = xml_path.with_suffix(".bin")
        if xml_path.suffix.lower() != ".xml" or not xml_path.is_file() or not bin_path.is_file():
            raise FaceDetectorError(f"OpenVINO IR not found: {xml_path} / {bin_path}")
        try:
            if self._core is None:
                self._core = OpenVINOCoreManager.core()
            requested = "AUTO" if self._device_requested == "auto" else self._device_requested.upper()
            model = self._core.read_model(str(xml_path))
            self._compiled = OpenVINOCoreManager.compile_model(
                self._core,
                model,
                device=requested,
                model_id=self.detector_id,
                model_sha256=artifact_sha256(xml_path),
                cache_root=self._cache_dir,
                performance_mode=self._performance_mode,
                cpu_threads=self._cpu_threads,
                max_process_ram_mb=self._max_process_ram_mb,
            )
            self._inspect_compiled(self._compiled, fallback_device=requested.lower())
        except FaceDetectorError:
            raise
        except Exception as exc:
            raise FaceDetectorError(f"cannot load OpenVINO face model: {exc}") from exc

    def _inspect_compiled(self, compiled: Any, fallback_device: str | None = None) -> None:
        try:
            inputs = list(compiled.inputs)
            if not inputs:
                raise FaceDetectorError("OpenVINO face model has no input")
            input_port = inputs[0]
            self._input_name = input_port.any_name
            shape = tuple(int(value) for value in input_port.shape)
            if len(shape) == 4 and shape[1] == 3:
                self._input_height, self._input_width = shape[2], shape[3]
            self._device_used = (
                OpenVINOCoreManager.execution_device(compiled, fallback_device)
                or self._device_requested
            )
        except FaceDetectorError:
            raise
        except Exception as exc:
            raise FaceDetectorError(f"cannot inspect OpenVINO face model: {exc}") from exc

    @staticmethod
    def _read_execution_device(compiled: Any) -> str | None:
        return OpenVINOCoreManager.execution_device(compiled)

    def detect(self, frame: np.ndarray, threshold: float | None = None) -> list[FaceDetection]:
        image = _validate_frame(frame)
        confidence_threshold = self._confidence_threshold if threshold is None else _validate_threshold(threshold)
        if self._compiled is None:
            self.load()
        assert self._compiled is not None
        resized = cv2.resize(
            image,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0,
            size=(self._input_width, self._input_height),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )
        try:
            raw = self._compiled({self._input_name: tensor}) if self._input_name else self._compiled(tensor)
        except Exception as exc:
            raise FaceDetectorError(f"OpenVINO face inference failed: {exc}") from exc
        named_values = list(raw.items()) if hasattr(raw, "items") else [(None, value) for value in raw]
        boxes_array: np.ndarray | None = None
        labels_array: np.ndarray | None = None
        unnamed: list[np.ndarray] = []
        for name, value in named_values:
            array = np.asarray(value)
            output_name = str(
                getattr(name, "any_name", None)
                or getattr(name, "get_names", lambda: ())()
                or name
                or ""
            ).casefold()
            if "box" in output_name:
                boxes_array = array.reshape(-1, 5)
            elif "label" in output_name:
                labels_array = array.reshape(-1)
            else:
                unnamed.append(array)
        for array in unnamed:
            flat = array.reshape(-1)
            if flat.size >= 5 and flat.size % 5 == 0 and (array.ndim >= 2 or flat.size == 5):
                candidate = flat.reshape(-1, 5)
                if boxes_array is None or candidate.shape[0] >= boxes_array.shape[0]:
                    boxes_array = candidate
            elif labels_array is None:
                labels_array = flat
        if boxes_array is None:
            raise FaceDetectorError("OpenVINO face model returned no boxes output")
        if labels_array is None:
            for array in unnamed:
                flat = array.reshape(-1)
                if flat.size == boxes_array.shape[0]:
                    labels_array = flat
                    break
        source_height, source_width = image.shape[:2]
        results: list[FaceDetection] = []
        for index, row in enumerate(boxes_array):
            if labels_array is not None and index < len(labels_array) and int(round(float(labels_array[index]))) != 0:
                continue
            x1, y1, x2, y2, confidence = (float(value) for value in row[:5])
            if not np.isfinite([x1, y1, x2, y2, confidence]).all() or confidence < confidence_threshold:
                continue
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, x2 = x1 * source_width, x2 * source_width
                y1, y2 = y1 * source_height, y2 * source_height
            else:
                x1, x2 = x1 * source_width / self._input_width, x2 * source_width / self._input_width
                y1, y2 = y1 * source_height / self._input_height, y2 * source_height / self._input_height
            x1, x2 = np.clip((x1, x2), 0, source_width)
            y1, y2 = np.clip((y1, y2), 0, source_height)
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                FaceDetection(
                    (float(x1), float(y1), float(x2), float(y2)),
                    float(confidence),
                    detector_id=self.detector_id,
                    backend=self.backend_id,
                    device=self._device_used,
                )
            )
        return results

    def close(self) -> None:
        self._compiled = None
        self._core = None


OpenVino0205FaceDetector = OpenVINOFaceDetector0205


class YuNetFaceDetector(FaceDetector):
    """OpenCV Zoo YuNet adapter using ``FaceDetectorYN`` on CPU."""

    detector_id = "yunet_2023mar"
    backend_id = "opencv_dnn"

    def __init__(
        self,
        model_path: Path | str,
        *,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        device: str = "cpu",
        detector: Any | None = None,
    ) -> None:
        self._model_path = Path(model_path).expanduser()
        self._confidence_threshold = _validate_threshold(confidence_threshold)
        self._nms_threshold = _validate_threshold(nms_threshold)
        if str(device).strip().lower() not in {"auto", "cpu"}:
            raise ValueError("YuNet supports only the cpu device")
        self._detector = detector
        if detector is None:
            self.load()

    @property
    def device_used(self) -> str:
        return "cpu"

    @property
    def provider_used(self) -> str:
        return self.backend_id

    def load(self) -> None:
        if self._detector is not None:
            return
        if not self._model_path.is_file():
            raise FaceDetectorError(f"YuNet model not found: {self._model_path}")
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(self._model_path), "", (320, 320), self._confidence_threshold, self._nms_threshold, 5000
            )
        except Exception as exc:
            raise FaceDetectorError(f"cannot load YuNet model {self._model_path}: {exc}") from exc

    def detect(self, frame: np.ndarray, threshold: float | None = None) -> list[FaceDetection]:
        image = _validate_frame(frame)
        confidence_threshold = self._confidence_threshold if threshold is None else _validate_threshold(threshold)
        if self._detector is None:
            self.load()
        assert self._detector is not None
        try:
            set_score_threshold = getattr(self._detector, "setScoreThreshold", None)
            if callable(set_score_threshold):
                set_score_threshold(float(confidence_threshold))
            self._detector.setInputSize((int(image.shape[1]), int(image.shape[0])))
            _status, faces = self._detector.detect(image)
        except Exception as exc:
            raise FaceDetectorError(f"YuNet inference failed: {exc}") from exc
        if faces is None:
            return []
        rows = np.asarray(faces)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        results: list[FaceDetection] = []
        for row in rows:
            if row.size < 15:
                continue
            x, y, width, height, confidence = map(float, (row[0], row[1], row[2], row[3], row[14]))
            if not np.isfinite([x, y, width, height, confidence]).all() or confidence < confidence_threshold:
                continue
            raw_points = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            landmarks = None
            if np.isfinite(raw_points).all():
                landmarks = FaceLandmark5(tuple((float(px), float(py)) for px, py in raw_points))  # type: ignore[arg-type]
            results.append(
                FaceDetection(
                    (x, y, x + max(0.0, width), y + max(0.0, height)),
                    confidence,
                    landmarks=landmarks,
                    detector_id=self.detector_id,
                    backend=self.backend_id,
                    device="cpu",
                )
            )
        return results

    def close(self) -> None:
        self._detector = None


class OpenVINOLandmarksRegressor(FaceLandmarker):
    """OpenVINO ``landmarks-regression-retail-0009`` adapter."""

    landmarker_id = "landmarks-regression-retail-0009"

    def __init__(
        self,
        model_path: Path | str,
        *,
        device: str = "auto",
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
        self._device_requested = str(device).strip().lower()
        if self._device_requested not in {"auto", "cpu", "gpu", "npu"}:
            raise ValueError("landmarker device must be one of: auto, cpu, gpu, npu")
        self._compiled = compiled_model
        self._core = core
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self._input_name: str | None = None
        self._device_used = ""
        self._input_size = (48, 48)
        if compiled_model is None:
            self.load()
        else:
            self._inspect_compiled(compiled_model)

    @property
    def device_used(self) -> str:
        return self._device_used

    def load(self) -> None:
        if self._compiled is not None:
            return
        xml_path = self._model_path
        bin_path = xml_path.with_suffix(".bin")
        if xml_path.suffix.lower() != ".xml" or not xml_path.is_file() or not bin_path.is_file():
            raise FaceDetectorError(f"landmarker IR not found: {xml_path} / {bin_path}")
        try:
            if self._core is None:
                self._core = OpenVINOCoreManager.core()
            requested = "AUTO" if self._device_requested == "auto" else self._device_requested.upper()
            model = self._core.read_model(str(xml_path))
            self._compiled = OpenVINOCoreManager.compile_model(
                self._core,
                model,
                device=requested,
                model_id=self.landmarker_id,
                model_sha256=artifact_sha256(xml_path),
                cache_root=self._cache_dir,
                performance_mode=self._performance_mode,
                cpu_threads=self._cpu_threads,
                max_process_ram_mb=self._max_process_ram_mb,
            )
            self._inspect_compiled(self._compiled, fallback_device=requested.lower())
        except FaceDetectorError:
            raise
        except Exception as exc:
            raise FaceDetectorError(f"cannot load landmark model: {exc}") from exc

    def _inspect_compiled(self, compiled: Any, fallback_device: str | None = None) -> None:
        inputs = list(compiled.inputs)
        if not inputs:
            raise FaceDetectorError("landmarker has no input")
        self._input_name = inputs[0].any_name
        shape = tuple(int(value) for value in inputs[0].shape)
        if len(shape) == 4 and shape[1] == 3:
            self._input_size = (shape[3], shape[2])
        try:
            devices = compiled.get_property("EXECUTION_DEVICES")
            self._device_used = (
                OpenVINOCoreManager.execution_device(compiled, fallback_device)
                or self._device_requested
            )
        except Exception:
            self._device_used = fallback_device or self._device_requested

    def landmark(self, image: np.ndarray, detection: FaceDetection) -> FaceLandmark5 | None:
        if detection.landmarks is not None:
            return detection.landmarks
        if self._compiled is None:
            self.load()
        assert self._compiled is not None
        cropped = crop_frame(image, detection.bbox)
        if cropped is None:
            return None
        face = cv2.resize(cropped.image, self._input_size, interpolation=cv2.INTER_LINEAR)
        tensor = np.transpose(face.astype(np.float32), (2, 0, 1))[None, ...]
        try:
            raw = self._compiled({self._input_name: tensor}) if self._input_name else self._compiled(tensor)
        except Exception as exc:
            raise FaceDetectorError(f"landmark inference failed: {exc}") from exc
        values = next(iter(raw.values())) if hasattr(raw, "values") else raw[0]
        points = np.asarray(values, dtype=np.float32).reshape(-1)
        if points.size < 10 or not np.isfinite(points[:10]).all():
            return None
        x1, y1, x2, y2 = detection.bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return FaceLandmark5(
            tuple((float(x1 + points[index] * width), float(y1 + points[index + 1] * height)) for index in range(0, 10, 2))  # type: ignore[arg-type]
        )

    def close(self) -> None:
        self._compiled = None
        self._core = None


__all__ = [
    "OpenVINOFaceDetector0205",
    "OpenVINOLandmarksRegressor",
    "OpenVino0205FaceDetector",
    "ScrfdFaceDetector",
    "YuNetFaceDetector",
]

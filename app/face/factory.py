"""Factories for configured local face-analysis adapters."""

from __future__ import annotations

from pathlib import Path

from app.config import FaceDetectionConfig, FaceLandmarksConfig, RecognitionConfig

from .capabilities import face_capability_matrix, validate_capability
from .alignment import SimilarityFaceAligner
from .detectors import (
    OpenVINOFaceDetector0205,
    OpenVINOLandmarksRegressor,
    ScrfdFaceDetector,
    YuNetFaceDetector,
)
from .embedding import FaceEmbedder, FaceEmbeddingError, OnnxFaceEmbedder
from .matcher import FaceMatcher
from .onnx import DisabledFaceDetector
from .openvino import OpenVINOFaceEmbedder
from .registry import detector_spec, recognizer_spec
from .model_resolution import (
    FaceModelResolutionError,
    resolve_detector,
    resolve_recognizer,
    normalize_device_for_backend,
    validate_backend_format,
)
from .storage import PersonStore
from .service import FaceAnalysisService
from .orchestrator import FaceRecognitionOrchestrator
from .base import FaceDetector, FaceQualityEvaluator
from app.metrics import CameraMetrics
from app.inference.synchronization import InferenceGate


def _resolve_model_path(model: str, model_root: Path | None) -> Path:
    path = Path(model).expanduser()
    if path.is_absolute():
        return path
    return (model_root or Path.cwd()) / path


def _require_artifact(path: Path, backend: str, *, label: str) -> None:
    validate_backend_format(path, backend, component=label)
    candidates = (path, path.with_suffix(".bin")) if backend == "openvino" else (path,)
    missing = tuple(candidate for candidate in candidates if not candidate.is_file())
    if missing:
        raise FileNotFoundError(
            f"{label} artifact missing: " + ", ".join(str(candidate) for candidate in missing)
        )


def _detector_id(config: FaceDetectionConfig) -> str | None:
    if config.detector_id:
        return detector_spec(config.detector_id).model_id
    model = (config.model or "").casefold()
    if "yunet" in model:
        return "yunet_2023mar"
    if "0205" in model or model.endswith(".xml"):
        return "face_detection_0205"
    if "scrfd" in model:
        return "scrfd_2.5g_kps"
    return None


def _detector_backend(config: FaceDetectionConfig, detector_id: str | None) -> str:
    if config.backend != "auto":
        return config.backend
    if detector_id is not None:
        return detector_spec(detector_id).backend
    return "onnxruntime"


def create_face_detector(
    config: FaceDetectionConfig,
    *,
    model_root: Path | None = None,
) -> FaceDetector:
    """Create one model-specific detector and fail closed on invalid selection."""

    if not config.enabled:
        return DisabledFaceDetector()
    if config.model is None or not config.model.strip():
        raise ValueError("face_detection.model is required when detection is enabled")
    path = _resolve_model_path(config.model, model_root)
    resolution = resolve_detector(
        model_id=_detector_id(config),
        requested_backend=config.backend,
        path=path,
    )
    detector_id = resolution.model_id
    backend = resolution.backend
    effective_device = normalize_device_for_backend(config.device, backend)
    _require_artifact(path, backend, label="face detector")
    if detector_id is not None:
        validate_capability(
            component="face_detection",
            model_id=detector_id,
            backend=backend,
            device=effective_device,
            model_root=model_root,
            artifact_path=path,
        )
    if detector_id == "scrfd_2.5g_kps" and backend == "onnxruntime":
        return ScrfdFaceDetector(
            path,
            confidence_threshold=config.confidence_threshold,
            nms_threshold=config.nms_threshold,
            device=effective_device,
        )
    if detector_id == "face_detection_0205" and backend == "openvino":
        return OpenVINOFaceDetector0205(
            path,
            confidence_threshold=config.confidence_threshold,
            device=effective_device,
            cache_dir=(model_root or Path.cwd()) / ".cache" / "openvino",
            performance_mode=config.openvino_performance_mode,
            cpu_threads=config.openvino_cpu_threads,
            max_process_ram_mb=config.max_process_ram_mb,
        )
    if detector_id == "yunet_2023mar" and backend == "opencv_dnn":
        return YuNetFaceDetector(
            path,
            confidence_threshold=config.confidence_threshold,
            nms_threshold=config.nms_threshold,
            device=effective_device,
        )
    raise ValueError(f"unsupported face detector selection: {detector_id}/{backend}")


def create_face_landmarker(
    config: FaceLandmarksConfig | None,
    *,
    model_root: Path | None = None,
) -> object | None:
    """Create the optional landmarker; absence is explicit detector-only mode."""

    if config is None or not config.enabled:
        return None
    if not config.model:
        raise ValueError("face_landmarks.model is required when landmarks are enabled")
    path = _resolve_model_path(config.model, model_root)
    _require_artifact(path, config.backend, label="face landmarker")
    validate_capability(
        component="face_landmarks",
        model_id=config.landmarker_id,
        backend=config.backend,
        device=config.device,
        model_root=model_root,
        artifact_path=path,
    )
    return OpenVINOLandmarksRegressor(
        path,
        device=config.device,
        cache_dir=(model_root or Path.cwd()) / ".cache" / "openvino",
        performance_mode=config.openvino_performance_mode,
        cpu_threads=config.openvino_cpu_threads,
        max_process_ram_mb=config.max_process_ram_mb,
    )


def _recognizer_id(config: RecognitionConfig) -> str | None:
    if config.recognizer_id:
        return config.recognizer_id.strip().lower()
    model = (config.model or "").casefold()
    if "retail-0095" in model:
        return "face-reidentification-retail-0095"
    if "facenet" in model:
        return "facenet-20180402-vggface2"
    if "arcface" in model or "w600k" in model:
        return "arcface-resnet50-webface600k"
    return None


def create_face_embedder(
    config: RecognitionConfig,
    *,
    model_root: Path | None = None,
) -> FaceEmbedder:
    """Create the configured local embedder used by enrollment and live matching."""

    if config.model is None or not config.model.strip():
        raise FaceEmbeddingError("recognition.model is required for enrollment/recognition")
    path = _resolve_model_path(config.model, model_root)
    try:
        resolution, spec = resolve_recognizer(
            model_id=config.recognizer_id,
            requested_backend=config.backend,
            path=path,
            device=config.device,
        )
    except (FaceModelResolutionError, ValueError) as exc:
        raise FaceEmbeddingError(str(exc)) from exc
    recognizer_id = resolution.model_id
    backend = resolution.backend
    _require_artifact(path, backend, label="face recognizer")
    validate_capability(
        component="recognition",
        model_id=recognizer_id,
        backend=backend,
        device=normalize_device_for_backend(config.device, backend),
        model_root=model_root,
        artifact_path=path,
    )
    template = {
        "width": spec.alignment_template.width,
        "height": spec.alignment_template.height,
        "points": [list(point) for point in spec.alignment_template.points.points],
    }
    if backend == "openvino":
        return OpenVINOFaceEmbedder(
            path,
            spec=spec,
            model_id=recognizer_id,
            model_version=config.model_version,
            device=normalize_device_for_backend(config.device, backend),
            cache_dir=(model_root or Path.cwd()) / ".cache" / "openvino",
            performance_mode=config.openvino_performance_mode,
            cpu_threads=config.openvino_cpu_threads,
            max_process_ram_mb=config.max_process_ram_mb,
        )
    if backend == "onnxruntime":
        return OnnxFaceEmbedder(
            path,
            model_id=recognizer_id,
            model_version=config.model_version,
            device=normalize_device_for_backend(config.device, backend),
            embedding_dimension=spec.embedding_dimension,
            color_order=spec.color_order,
            normalization=spec.normalization,
            recognizer_id=recognizer_id,
            alignment_template=template,
            default_threshold=config.threshold,
        )
    raise FaceEmbeddingError(f"unsupported face recognizer backend: {backend}")


def create_face_matcher(
    config: RecognitionConfig,
    *,
    persons_root: Path | str,
    model_root: Path | None = None,
    embedder: FaceEmbedder | None = None,
    metrics: CameraMetrics | None = None,
    inference_gate: InferenceGate | None = None,
) -> FaceMatcher:
    """Create the local matcher and validate enrolled model compatibility."""

    actual_embedder = embedder or create_face_embedder(config, model_root=model_root)
    recognizer_id = actual_embedder.metadata.recognizer_id or actual_embedder.metadata.model_id
    store = PersonStore(
        persons_root,
        scope=Path(recognizer_id) / actual_embedder.metadata.fingerprint,
    )
    threshold = config.threshold if config.threshold is not None else actual_embedder.metadata.default_threshold
    return FaceMatcher(
        actual_embedder,
        store,
        threshold=threshold,
        metrics=metrics,
        inference_gate=inference_gate,
    )


def create_face_orchestrator(
    config: object,
    camera_id: str,
    *,
    model_root: Path | None = None,
    metrics: CameraMetrics | None = None,
    inference_gate: InferenceGate | None = None,
) -> FaceRecognitionOrchestrator | None:
    """Build the single live face pipeline for one camera."""

    face_config = getattr(config, "face_detection")
    if not face_config.enabled:
        return None
    detector = create_face_detector(face_config, model_root=model_root)
    landmarks_config = getattr(config, "face_landmarks", None)
    landmarker = None
    recognition_config = getattr(config, "recognition")
    matcher: FaceMatcher | None = None
    aligner = None
    recognition_error: str | None = None
    try:
        landmarker = create_face_landmarker(landmarks_config, model_root=model_root)
    except Exception as exc:
        # A native-landmark detector can still support recognition.  For a
        # detector without native landmarks this becomes a recognition-only
        # diagnostic below; detection itself remains usable.
        recognition_error = f"landmarker unavailable: {type(exc).__name__}: {exc}"
    if recognition_config.enabled:
        detector_id = _detector_id(face_config)
        native_landmarks = (
            detector_id is not None and detector_spec(detector_id).landmarks
        )
        if not native_landmarks and landmarker is None:
            recognition_error = recognition_error or (
                "recognition requires face_landmarks for a detector without native landmarks"
            )
        else:
            try:
                matcher = create_face_matcher(
                    recognition_config,
                    persons_root=(model_root or Path.cwd()) / getattr(config.storage, "persons_dir"),
                    model_root=model_root,
                    metrics=metrics,
                    inference_gate=inference_gate,
                )
                recognizer_id = matcher.embedder.metadata.recognizer_id or matcher.embedder.metadata.model_id
                aligner = SimilarityFaceAligner(recognizer_spec(recognizer_id).alignment_template)
                recognition_error = None
            except Exception as exc:
                recognition_error = (
                    f"recognizer unavailable: {type(exc).__name__}: {exc}"
                )
    service = FaceAnalysisService(
        camera_id,
        detector,
        aligner=aligner,
        landmarker=landmarker,
        matcher=matcher,
        evaluator=FaceQualityEvaluator(
            min_width=config.face_quality.min_width,
            min_height=config.face_quality.min_height,
            blur_threshold=config.face_quality.blur_threshold,
            min_brightness=config.face_quality.min_brightness,
            max_brightness=config.face_quality.max_brightness,
            min_confidence=config.face_quality.min_confidence,
        ),
        metrics=metrics,
        inference_gate=inference_gate,
    )
    return FaceRecognitionOrchestrator(
        service,
        face_fps=face_config.inference_fps,
        recognition_fps=recognition_config.inference_fps if recognition_config.enabled else None,
        recognition_error=recognition_error,
        enabled=True,
    )


__all__ = [
    "create_face_detector",
    "create_face_embedder",
    "create_face_landmarker",
    "create_face_matcher",
    "create_face_orchestrator",
    "face_capability_matrix",
]

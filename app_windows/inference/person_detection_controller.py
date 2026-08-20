"""Asynchronous person detection bound to an already-running video provider."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
import threading
import time
from typing import Protocol

from PySide6.QtCore import QObject, Signal

from app.config import PersonDetectionConfig
from app.inference import InferenceGate, PersonDetector, create_person_detector
from app.metrics import CameraMetrics
from app.tracking import CameraTrackingPipeline, IoUGreedyTracker
from app_windows.models.person_detection_state import (
    PersonDetectionSettings,
    PersonDetectionSnapshot,
    PersonDetectionStatus,
)


class InferenceFrameSource(Protocol):
    """Minimum provider contract needed by the background inference loop."""

    camera_id: str

    def latest_frame(self):
        """Return the newest bounded frame packet, or ``None``."""


class PersonDetectionController(QObject):
    """Run one configured detector against the selected existing camera provider.

    Camera acquisition remains owned by ``CameraMonitorController``. This
    object only consumes the provider's latest frame and never starts, stops,
    or reconnects a camera.
    """

    snapshot_changed = Signal(object)

    def __init__(
        self,
        *,
        repo_root: Path,
        settings: PersonDetectionSettings | None = None,
        inference_gate: InferenceGate | None = None,
        logger: logging.Logger | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._repo_root = Path(repo_root)
        self._logger = logger or logging.getLogger(__name__)
        self._settings = settings or PersonDetectionSettings()
        self._active_camera_id: str | None = None
        self._provider: InferenceFrameSource | None = None
        self._snapshot = PersonDetectionSnapshot(
            settings=self._settings,
            model_path=self._settings.model,
            requested_device=self._settings.device,
        )
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._gate = inference_gate or InferenceGate()
        self._metrics: dict[str, CameraMetrics] = {}
        self._tracking_pipeline: CameraTrackingPipeline | None = None

    @property
    def settings(self) -> PersonDetectionSettings:
        with self._lock:
            return self._settings

    @property
    def snapshot(self) -> PersonDetectionSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def is_started(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def inference_gate(self) -> InferenceGate:
        """Gate shared with optional face stages for this Windows session."""

        return self._gate

    @property
    def tracking_pipeline(self) -> CameraTrackingPipeline | None:
        """Return the single tracker used for the active camera session."""

        with self._lock:
            return self._tracking_pipeline

    def start(self) -> None:
        """Start model loading/inference without blocking the GUI."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._wake.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="windows-person-detection",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def stop(self, timeout_s: float | None = 1.5) -> None:
        """Stop inference while leaving camera providers untouched."""

        self._stop_event.set()
        self._wake.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)

    def update_settings(self, settings: PersonDetectionSettings) -> None:
        """Apply settings asynchronously and invalidate older generations."""

        if not isinstance(settings, PersonDetectionSettings):
            raise TypeError("settings must be PersonDetectionSettings")
        with self._lock:
            self._settings = settings
            self._generation += 1
        self._wake.set()

    def set_active_camera(
        self,
        camera_id: str | None,
        provider: InferenceFrameSource | None,
    ) -> None:
        """Analyze one existing provider, or pause when ``camera_id`` is null."""

        normalized_id = camera_id.strip() if camera_id else None
        if normalized_id is None:
            provider = None
        with self._lock:
            self._active_camera_id = normalized_id
            self._provider = provider
            if normalized_id is None or (
                self._tracking_pipeline is not None
                and self._tracking_pipeline.camera_id != normalized_id
            ):
                self._tracking_pipeline = None
            self._generation += 1
        self._wake.set()

    def _state(
        self,
    ) -> tuple[PersonDetectionSettings, str | None, InferenceFrameSource | None, int]:
        with self._lock:
            return self._settings, self._active_camera_id, self._provider, self._generation

    def _publish_if_current(
        self,
        snapshot: PersonDetectionSnapshot,
        *,
        generation: int,
        camera_id: str | None,
    ) -> bool:
        """Publish only if the result still belongs to the active generation."""

        with self._lock:
            if (
                self._generation != generation
                or self._active_camera_id != camera_id
            ):
                return False
            self._snapshot = snapshot
        self.snapshot_changed.emit(snapshot)
        return True

    def _base_snapshot(
        self,
        settings: PersonDetectionSettings,
        camera_id: str | None,
        *,
        status: PersonDetectionStatus,
        message: str,
        error: str | None = None,
        detector: PersonDetector | None = None,
        detections=(),
        frame_sequence: int | None = None,
        frame_timestamp=None,
        source_width: int | None = None,
        source_height: int | None = None,
        result_monotonic: float | None = None,
        latency_ms: float | None = None,
        person_count: int = 0,
        detection_count: int = 0,
        detector_failures: int = 0,
        metrics: CameraMetrics | None = None,
        tracking_update=None,
        tracking_pipeline: CameraTrackingPipeline | None = None,
    ) -> PersonDetectionSnapshot:
        metric_snapshot = metrics.snapshot() if metrics is not None else None
        actual_device = detector.device_used if detector is not None else None
        device_verified = detector.device_verified if detector is not None else False
        provider = detector.provider_used if detector is not None else None
        backend = detector.backend if detector is not None else settings.backend
        precision = getattr(detector, "precision", None) if detector is not None else settings.precision
        return PersonDetectionSnapshot(
            camera_id=camera_id,
            status=status,
            message=message,
            error=(
                error
                if error is not None
                else (message if status in {
                    PersonDetectionStatus.ERROR,
                    PersonDetectionStatus.MODEL_MISSING,
                } else None)
            ),
            settings=settings,
            model_path=settings.model,
            requested_device=settings.device,
            actual_device=actual_device,
            device_verified=device_verified,
            provider=provider,
            backend=backend,
            precision=precision,
            inference_fps=(
                metric_snapshot.person_detection_fps
                if metric_snapshot is not None
                else None
            ),
            latency_ms=(
                latency_ms
                if latency_ms is not None
                else (
                    metric_snapshot.processing_latency_ms
                    if metric_snapshot is not None
                    else None
                )
            ),
            person_count=person_count,
            detection_count=detection_count,
            detections=tuple(detections),
            frame_sequence=frame_sequence,
            frame_timestamp=frame_timestamp,
            source_width=source_width,
            source_height=source_height,
            result_monotonic=result_monotonic,
            detector_failures=detector_failures,
            tracking_update=tracking_update,
            tracking_pipeline=(
                tracking_pipeline
                if tracking_pipeline is not None
                else self._tracking_pipeline
            ),
        )

    def _resolve_model_path(self, model: str | None) -> Path | None:
        if not model:
            return None
        path = Path(model).expanduser()
        return path if path.is_absolute() else self._repo_root / path

    @staticmethod
    def _model_signature(
        settings: PersonDetectionSettings,
    ) -> tuple[object, ...]:
        return (
            settings.enabled,
            settings.backend,
            settings.model,
            settings.device,
            settings.precision,
            settings.fallback_device,
            settings.image_size,
            settings.classes,
            settings.prompts,
            settings.confidence_threshold,
            settings.inference_fps,
            settings.show_masks,
        )

    def _load_detector(
        self,
        settings: PersonDetectionSettings,
        camera_id: str | None,
        generation: int,
    ) -> tuple[PersonDetector | None, PersonDetectionSnapshot | None]:
        if not settings.enabled:
            return None, self._base_snapshot(
                settings,
                camera_id,
                status=PersonDetectionStatus.DISABLED,
                message="Rilevamento persone disabilitato",
            )

        backend = settings.backend
        model_path = self._resolve_model_path(settings.model)
        if backend == "auto":
            model_name = (settings.model or "").casefold()
            if model_name.endswith(".onnx"):
                backend = "onnx"
            elif model_name.endswith(("yolo26s.pt", "yolo26n.pt", "_openvino_model", ".xml")):
                backend = "openvino"
            else:
                backend = "yoloe"

        # Local YOLOE/ONNX files retain the existing MODEL_MISSING status. An
        # official OpenVINO checkpoint is intentionally allowed to be absent:
        # the adapter owns its official first-use download and cache lifecycle.
        if backend == "yoloe":
            if model_path is None or model_path.suffix.lower() != ".pt":
                return None, self._base_snapshot(
                    settings,
                    camera_id,
                    status=PersonDetectionStatus.ERROR,
                    message="Selezionare un checkpoint YOLOE segmentation .pt",
                )
            if not model_path.is_file():
                return None, self._base_snapshot(
                    settings,
                    camera_id,
                    status=PersonDetectionStatus.MODEL_MISSING,
                    message=f"Modello YOLOE non trovato: {settings.model}",
                )
        elif backend == "onnx":
            if model_path is None or model_path.suffix.lower() != ".onnx":
                return None, self._base_snapshot(
                    settings,
                    camera_id,
                    status=PersonDetectionStatus.ERROR,
                    message="Selezionare un modello ONNX per il backend ONNX",
                )
            if not model_path.is_file():
                return None, self._base_snapshot(
                    settings,
                    camera_id,
                    status=PersonDetectionStatus.MODEL_MISSING,
                    message=f"Modello ONNX non trovato: {settings.model}",
                )
        elif backend == "fake":
            model_path = None

        loading_snapshot = self._base_snapshot(
            settings,
            camera_id,
            status=PersonDetectionStatus.LOADING,
            message=f"Caricamento backend {backend}"
            + (f" · {model_path.name}…" if model_path is not None else "…"),
        )
        self._publish_if_current(
            loading_snapshot,
            generation=generation,
            camera_id=camera_id,
        )

        try:
            detector_config = PersonDetectionConfig(
                enabled=True,
                backend=settings.backend,
                model=settings.model,
                confidence_threshold=settings.confidence_threshold,
                inference_fps=settings.inference_fps,
                precision=settings.precision,
                device=settings.device,
                fallback_device=settings.fallback_device,
                image_size=settings.image_size,
                classes=list(settings.classes),
                prompts=list(settings.prompts),
                show_masks=settings.show_masks,
            )
            detector = create_person_detector(
                detector_config,
                model_root=self._repo_root,
            )
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            self._logger.error("person detection model load failed backend=%s: %s", backend, message)
            return None, self._base_snapshot(
                settings,
                camera_id,
                status=PersonDetectionStatus.ERROR,
                message=message,
            )

        return detector, self._base_snapshot(
            settings,
            camera_id,
            status=PersonDetectionStatus.READY,
            message=f"Backend {detector.backend} pronto; in attesa del frame",
            detector=detector,
        )

    @staticmethod
    def _close_detector(detector: PersonDetector | None) -> None:
        if detector is None:
            return
        try:
            detector.close()
        except Exception:
            logging.getLogger(__name__).debug("detector close failed", exc_info=True)

    def _run(self) -> None:
        detector: PersonDetector | None = None
        loaded_signature: tuple[object, ...] | None = None
        loaded_camera_id: str | None = None
        last_packet_sequence: int | None = None
        next_inference_at = 0.0
        detector_failures = 0

        try:
            while not self._stop_event.is_set():
                settings, camera_id, provider, generation = self._state()
                signature = self._model_signature(settings)

                if signature != loaded_signature:
                    self._close_detector(detector)
                    detector = None
                    loaded_signature = signature
                    loaded_camera_id = None
                    last_packet_sequence = None
                    next_inference_at = 0.0
                    detector_failures = 0
                    if camera_id is not None:
                        self._tracking_for(camera_id).reset(reason="person model changed")
                    detector, load_snapshot = self._load_detector(
                        settings,
                        camera_id,
                        generation,
                    )
                    if load_snapshot is not None:
                        self._publish_if_current(
                            load_snapshot,
                            generation=generation,
                            camera_id=camera_id,
                        )
                    (
                        current_settings,
                        current_camera_id,
                        _current_provider,
                        current_generation,
                    ) = self._state()
                    if (
                        self._model_signature(current_settings) != signature
                        or current_camera_id != camera_id
                        or current_generation != generation
                    ):
                        self._close_detector(detector)
                        detector = None
                        loaded_signature = None
                        continue
                    if self._stop_event.is_set():
                        break
                    if detector is None:
                        self._wake.wait(0.25)
                        self._wake.clear()
                        continue

                if camera_id != loaded_camera_id:
                    loaded_camera_id = camera_id
                    last_packet_sequence = None
                    next_inference_at = 0.0
                    tracking_pipeline = (
                        self._tracking_for(camera_id)
                        if camera_id is not None
                        else None
                    )
                    if detector is not None:
                        if camera_id is None or provider is None:
                            self._publish_if_current(
                                self._base_snapshot(
                                    settings,
                                    camera_id,
                                    status=PersonDetectionStatus.READY,
                                    message=f"Backend {detector.backend} pronto; aprire una camera in focus",
                                    detector=detector,
                                ),
                                generation=generation,
                                camera_id=camera_id,
                            )
                        else:
                            self._publish_if_current(
                                self._base_snapshot(
                                    settings,
                                    camera_id,
                                    status=PersonDetectionStatus.READY,
                                    message="Pronto; attesa del primo frame",
                                    detector=detector,
                                    metrics=self._metrics_for(camera_id),
                                ),
                                generation=generation,
                                camera_id=camera_id,
                            )
                    elif self.snapshot.status in {
                        PersonDetectionStatus.DISABLED,
                        PersonDetectionStatus.ERROR,
                        PersonDetectionStatus.MODEL_MISSING,
                    }:
                        previous = self.snapshot
                        self._publish_if_current(
                            self._base_snapshot(
                                settings,
                                camera_id,
                                status=previous.status,
                                message=previous.message,
                                detector_failures=previous.detector_failures,
                            ),
                            generation=generation,
                            camera_id=camera_id,
                        )

                if detector is None:
                    self._wake.wait(0.25)
                    self._wake.clear()
                    continue
                if camera_id is None or provider is None:
                    self._wake.wait(0.15)
                    self._wake.clear()
                    continue

                now = time.monotonic()
                if now < next_inference_at:
                    self._wake.wait(min(0.05, next_inference_at - now))
                    self._wake.clear()
                    continue

                try:
                    packet = provider.latest_frame()
                except Exception as exc:
                    detector_failures += 1
                    if not self._publish_if_current(
                        self._base_snapshot(
                            settings,
                            camera_id,
                            status=PersonDetectionStatus.ERROR,
                            message=f"Lettura frame per inferenza fallita: {exc}",
                            detector=detector,
                            detector_failures=detector_failures,
                        ),
                        generation=generation,
                        camera_id=camera_id,
                    ):
                        continue
                    self._wake.wait(0.25)
                    self._wake.clear()
                    continue

                if packet is None or packet.sequence == last_packet_sequence:
                    self._wake.wait(0.05)
                    self._wake.clear()
                    continue

                started = time.perf_counter()
                try:
                    raw_detections = self._gate.run(detector.detect, packet.frame)
                    detections = tuple(
                        replace(detection, timestamp=packet.received_at_utc)
                        for detection in raw_detections
                        if detection.confidence >= settings.confidence_threshold
                    )
                    latency_ms = (time.perf_counter() - started) * 1000.0
                except Exception as exc:
                    detector_failures += 1
                    last_packet_sequence = packet.sequence
                    next_inference_at = time.monotonic() + 0.5
                    if not self._publish_if_current(
                        self._base_snapshot(
                            settings,
                            camera_id,
                            status=PersonDetectionStatus.ERROR,
                            message=f"Inferenza persone ({detector.backend}) fallita: {exc}",
                            detector=detector,
                            frame_sequence=packet.sequence,
                            frame_timestamp=packet.received_at_utc,
                            detector_failures=detector_failures,
                            metrics=self._metrics_for(camera_id),
                        ),
                        generation=generation,
                        camera_id=camera_id,
                    ):
                        continue
                    continue

                current_settings, current_camera_id, _current_provider, current_generation = self._state()
                if current_generation != generation or current_camera_id != camera_id:
                    continue

                metrics = self._metrics_for(camera_id)
                metrics.record_person_detection(latency_ms)
                last_packet_sequence = packet.sequence
                next_inference_at = time.monotonic() + 1.0 / settings.inference_fps

                tracking_pipeline = self._tracking_for(camera_id)
                tracking_update = tracking_pipeline.update(
                    tuple(
                        detection
                        for detection in detections
                        if detection.label.casefold() == "person"
                    )
                )

                source_width: int | None = None
                source_height: int | None = None
                try:
                    shape = packet.frame.shape
                    source_height = int(shape[0])
                    source_width = int(shape[1])
                except (AttributeError, IndexError, TypeError, ValueError):
                    pass

                person_count = sum(
                    detection.label.casefold() == "person"
                    for detection in detections
                )
                self._publish_if_current(
                    self._base_snapshot(
                        settings,
                        camera_id,
                        status=PersonDetectionStatus.RUNNING,
                        message=f"Inferenza persone ({detector.backend}) attiva",
                        detector=detector,
                        detections=detections,
                        frame_sequence=packet.sequence,
                        frame_timestamp=packet.received_at_utc,
                        source_width=source_width,
                        source_height=source_height,
                        result_monotonic=time.monotonic(),
                        latency_ms=latency_ms,
                        person_count=person_count,
                        detection_count=len(detections),
                        detector_failures=detector_failures,
                        metrics=metrics,
                        tracking_update=tracking_update,
                        tracking_pipeline=tracking_pipeline,
                    ),
                    generation=generation,
                    camera_id=camera_id,
                )
        finally:
            self._close_detector(detector)

    def _tracking_for(self, camera_id: str) -> CameraTrackingPipeline:
        with self._lock:
            pipeline = self._tracking_pipeline
            if pipeline is None or pipeline.camera_id != camera_id:
                pipeline = CameraTrackingPipeline(
                    camera_id,
                    tracker=IoUGreedyTracker(),
                    metrics=self._metrics_for(camera_id),
                )
                self._tracking_pipeline = pipeline
            return pipeline

    def _metrics_for(self, camera_id: str) -> CameraMetrics:
        with self._lock:
            metrics = self._metrics.get(camera_id)
            if metrics is None:
                metrics = CameraMetrics(camera_id)
                self._metrics[camera_id] = metrics
            return metrics

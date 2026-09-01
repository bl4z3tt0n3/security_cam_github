"""Windows front-end for the shared core face orchestrator.

The controller consumes person detections already produced by
``PersonDetectionController`` and the existing provider's latest frame.  It
does not create a person detector or a second acquisition/tracking source.
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
from typing import Any, Protocol

from PySide6.QtCore import QObject, Signal

from app.config import AppConfig
from app.face import (
    FaceRecognitionOrchestrator,
    PersonStore,
    TrackRecognitionConfirmer,
    create_face_orchestrator,
    face_capability_matrix,
)
from app.inference import InferenceGate
from app.metrics import CameraMetrics
from app.tracking import CameraTrackingPipeline
from app_windows.models.face_recognition_state import (
    FaceGalleryState,
    FaceOverlayState,
    FaceRecognitionSettings,
    FaceRecognitionSnapshot,
    FaceRecognitionStatus,
)
from app_windows.inference.face_gallery import scan_enrollment_people
from app_windows.models.person_detection_state import PersonDetectionSnapshot


class InferenceFrameSource(Protocol):
    camera_id: str

    def latest_frame(self):
        ...


class FaceRecognitionController(QObject):
    """Run the opt-in face stages on the latest person-tracked frame."""

    snapshot_changed = Signal(object)
    capabilities_changed = Signal(object)
    gallery_changed = Signal(object)

    def __init__(
        self,
        *,
        repo_root: Path,
        config: AppConfig,
        settings: FaceRecognitionSettings | None = None,
        inference_gate: InferenceGate | None = None,
        logger: logging.Logger | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._repo_root = Path(repo_root).resolve()
        self._base_config = config
        self._settings = settings or FaceRecognitionSettings.from_app_config(config)
        configured_enrollment_root = Path(config.storage.enrollment_dir).expanduser()
        self._enrollment_root = (
            configured_enrollment_root
            if configured_enrollment_root.is_absolute()
            else self._repo_root / configured_enrollment_root
        ).resolve()
        self._logger = logger or logging.getLogger(__name__)
        self._gate = inference_gate or InferenceGate()
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._camera_id: str | None = None
        self._provider: InferenceFrameSource | None = None
        self._person_snapshot: PersonDetectionSnapshot | None = None
        self._snapshot = FaceRecognitionSnapshot()
        self._gallery = FaceGalleryState()
        self._metrics: dict[str, CameraMetrics] = {}
        self._orchestrator: FaceRecognitionOrchestrator | None = None
        self._pipeline: CameraTrackingPipeline | None = None
        self._tracking_pipeline: CameraTrackingPipeline | None = None

    @property
    def settings(self) -> FaceRecognitionSettings:
        with self._lock:
            return self._settings

    @property
    def snapshot(self) -> FaceRecognitionSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def gallery(self) -> FaceGalleryState:
        with self._lock:
            return self._gallery

    @property
    def enrollment_root(self) -> Path:
        with self._lock:
            return self._enrollment_root

    def set_enrollment_root(self, root: Path | str) -> None:
        """Update the source used by gallery scans and explicit enrollment."""

        candidate = Path(root).expanduser()
        with self._lock:
            self._enrollment_root = candidate.resolve()

    @property
    def inference_gate(self) -> InferenceGate:
        return self._gate

    @property
    def is_started(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._wake.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="windows-face-recognition",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def stop(self, timeout_s: float | None = 1.5) -> None:
        self._stop_event.set()
        self._wake.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)
        self._close_runtime()

    def update_settings(self, settings: FaceRecognitionSettings) -> None:
        if not isinstance(settings, FaceRecognitionSettings):
            raise TypeError("settings must be FaceRecognitionSettings")
        with self._lock:
            self._settings = settings
            self._generation += 1
        self._wake.set()

    def update_config(self, config: AppConfig) -> FaceRecognitionSettings:
        """Replace the validated base config and apply its effective face settings."""

        if not isinstance(config, AppConfig):
            raise TypeError("config must be AppConfig")
        settings = FaceRecognitionSettings.from_app_config(config)
        enrollment = Path(config.storage.enrollment_dir).expanduser()
        enrollment = (
            enrollment
            if enrollment.is_absolute()
            else self._repo_root / enrollment
        ).resolve()
        with self._lock:
            self._base_config = config
            self._settings = settings
            self._enrollment_root = enrollment
            self._generation += 1
        self._wake.set()
        return settings

    def set_active_camera(
        self,
        camera_id: str | None,
        provider: InferenceFrameSource | None,
    ) -> None:
        normalized = camera_id.strip() if camera_id else None
        if normalized is None:
            provider = None
        with self._lock:
            self._camera_id = normalized
            self._provider = provider
            self._person_snapshot = None
            self._tracking_pipeline = None
            self._generation += 1
        self._wake.set()

    def set_person_snapshot(self, value: PersonDetectionSnapshot | None) -> None:
        if value is not None and not isinstance(value, PersonDetectionSnapshot):
            return
        with self._lock:
            if value is None or value.camera_id == self._camera_id:
                self._person_snapshot = value
                self._tracking_pipeline = (
                    value.tracking_pipeline if value is not None else None
                )
        self._wake.set()

    def refresh_capabilities(self) -> tuple[dict[str, Any], ...]:
        settings = self.settings
        rows = tuple(
            row.to_dict()
            for row in face_capability_matrix(
                self._repo_root,
                configured_recognition={
                    "recognizer_id": settings.recognizer_id,
                    "backend": settings.recognizer_backend,
                    "model": settings.recognizer_model,
                    "device": settings.recognizer_device,
                },
            )
        )
        self.capabilities_changed.emit(rows)
        return rows

    def refresh_gallery(self) -> FaceGalleryState:
        with self._lock:
            matcher = (
                self._orchestrator.service.matcher
                if self._orchestrator is not None
                else None
            )
            enrollment_root = self._enrollment_root
        active_persons: tuple[dict[str, Any], ...] = ()
        recognizer_id: str | None = None
        fingerprint: str | None = None
        gallery_error: str | None = None
        if matcher is None:
            with self._lock:
                orchestrator = self._orchestrator
            gallery_error = (
                orchestrator.recognition_error
                if orchestrator is not None and orchestrator.recognition_error
                else "recognition is not loaded"
            )
        else:
            try:
                records = matcher.refresh()
                metadata = matcher.embedder.metadata
                recognizer_id = metadata.recognizer_id or metadata.model_id
                fingerprint = metadata.fingerprint
                active_persons = tuple(
                    {
                        "person_id": record.person_id,
                        "name": record.name,
                        "embedding_count": int(record.embeddings.shape[0]),
                        "fingerprint": record.model.fingerprint,
                    }
                    for record in records
                )
            except Exception as exc:
                gallery_error = f"gallery refresh failed: {type(exc).__name__}: {exc}"

        scan = scan_enrollment_people(
            enrollment_root,
            active_people=active_persons,
        )
        message_parts = ["gallery refreshed"]
        if gallery_error:
            message_parts.append(gallery_error)
        if scan.error:
            message_parts.append(scan.error)
        gallery = FaceGalleryState(
            recognizer_id=recognizer_id,
            fingerprint=fingerprint,
            persons=active_persons,
            enrollment_people=scan.people,
            enrollment_root=str(enrollment_root),
            enrollment_root_present=scan.root_present,
            status="ready",
            message="; ".join(message_parts),
            error=scan.error,
        )
        with self._lock:
            self._gallery = gallery
        self.gallery_changed.emit(gallery)
        return gallery

    def remove_person(self, person_id: str) -> FaceGalleryState:
        with self._lock:
            matcher = (
                self._orchestrator.service.matcher
                if self._orchestrator is not None
                else None
            )
        if matcher is None:
            raise RuntimeError("recognition is not loaded")
        matcher.person_store.delete(person_id)
        return self.refresh_gallery()

    def enroll_person(
        self,
        name: str,
        images_directory: Path | str,
        *,
        person_id: str | None = None,
        overwrite: bool = False,
    ):
        """Run explicit local enrollment; never consumes a live camera frame."""

        settings = self.settings
        if not settings.face_detection_enabled:
            raise RuntimeError("face detection must be enabled for enrollment")
        service, resources = self._build_enrollment_service(settings)
        try:
            report = service.enroll(
                name,
                images_directory,
                person_id=person_id,
                overwrite=overwrite,
            )
        finally:
            for resource in resources:
                close = getattr(resource, "close", None)
                if callable(close):
                    close()
        self.refresh_gallery()
        return report

    def import_enrollment(self, root: Path | str | None = None):
        """Import every configured enrollment ``<person_id>`` folder explicitly."""

        from app.face import EnrollmentBatchService

        settings = self.settings
        if not settings.face_detection_enabled:
            raise RuntimeError("face detection must be enabled for enrollment import")
        service, resources = self._build_enrollment_service(settings)
        try:
            report = EnrollmentBatchService(service).import_tree(
                root or self.enrollment_root
            )
        finally:
            for resource in resources:
                close = getattr(resource, "close", None)
                if callable(close):
                    close()
        self.refresh_gallery()
        return report

    def _build_enrollment_service(self, settings: FaceRecognitionSettings):
        from app.face import (
            EnrollmentService,
            FaceQualityEvaluator,
            SimilarityFaceAligner,
            create_face_detector,
            create_face_embedder,
            create_face_landmarker,
        )
        from app.face.registry import recognizer_spec

        config = self._effective_config(settings)
        detector = create_face_detector(config.face_detection, model_root=self._repo_root)
        embedder = None
        landmarker = None
        try:
            embedder = create_face_embedder(config.recognition, model_root=self._repo_root)
            try:
                landmarker = create_face_landmarker(
                    config.face_landmarks,
                    model_root=self._repo_root,
                )
            except Exception as exc:
                # SCRFD/YuNet already carry five points.  A broken optional
                # OpenVINO landmarker must not make manual enrollment or its
                # per-image diagnostics unavailable.
                self._logger.warning("face landmarker unavailable for enrollment: %s", exc)
            recognizer_id = embedder.metadata.recognizer_id or embedder.metadata.model_id
            aligner = SimilarityFaceAligner(recognizer_spec(recognizer_id).alignment_template)
            persons_root = self._repo_root / config.storage.persons_dir
            store = PersonStore(
                persons_root,
                scope=Path(recognizer_id) / embedder.metadata.fingerprint,
            )
            service = EnrollmentService(
                detector,
                embedder,
                store,
                aligner=aligner,
                landmarker=landmarker,
                evaluator=FaceQualityEvaluator(
                    min_width=config.face_quality.min_width,
                    min_height=config.face_quality.min_height,
                    blur_threshold=config.face_quality.blur_threshold,
                    min_brightness=config.face_quality.min_brightness,
                    max_brightness=config.face_quality.max_brightness,
                    min_confidence=config.face_quality.min_confidence,
                ),
            )
        except Exception:
            for resource in (detector, landmarker, embedder):
                close = getattr(resource, "close", None)
                if callable(close):
                    close()
            raise
        return service, (detector, landmarker, embedder)

    def _state(self):
        with self._lock:
            return (
                self._settings,
                self._camera_id,
                self._provider,
                self._person_snapshot,
                self._tracking_pipeline,
                self._generation,
            )

    def _effective_config(self, settings: FaceRecognitionSettings) -> AppConfig:
        face = self._base_config.face_detection.model_copy(
            update={
                "enabled": settings.face_detection_enabled,
                "detector_id": settings.detector_id,
                "backend": settings.detector_backend,
                "model": settings.detector_model,
                "device": settings.detector_device,
                "confidence_threshold": settings.detector_confidence_threshold,
                "inference_fps": settings.detector_inference_fps,
            }
        )
        landmarks = self._base_config.face_landmarks.model_copy(
            update={
                "enabled": settings.landmarks_enabled,
                "landmarker_id": settings.landmarker_id,
                "backend": settings.landmarker_backend,
                "model": settings.landmarker_model,
                "device": settings.landmarker_device,
            }
        )
        recognition = self._base_config.recognition.model_copy(
            update={
                "enabled": settings.recognition_enabled,
                "recognizer_id": settings.recognizer_id,
                "backend": settings.recognizer_backend,
                "model": settings.recognizer_model,
                "device": settings.recognizer_device,
                "threshold": settings.recognition_threshold,
                "inference_fps": settings.recognition_inference_fps,
                "min_confirmations": settings.min_confirmations,
                "confirmation_window_seconds": settings.confirmation_window_seconds,
            }
        )
        return self._base_config.model_copy(
            update={
                "face_detection": face,
                "face_landmarks": landmarks,
                "recognition": recognition,
            }
        )

    def _build_runtime(
        self,
        settings: FaceRecognitionSettings,
        camera_id: str,
        tracking_pipeline: CameraTrackingPipeline | None,
    ):
        config = self._effective_config(settings)
        if tracking_pipeline is None or tracking_pipeline.camera_id != camera_id:
            raise RuntimeError("face pipeline is waiting for the person tracking pipeline")
        metrics = tracking_pipeline.metrics or self._metrics.setdefault(camera_id, CameraMetrics(camera_id))
        self._metrics[camera_id] = metrics
        orchestrator = create_face_orchestrator(
            config,
            camera_id,
            model_root=self._repo_root,
            metrics=metrics,
            inference_gate=self._gate,
        )
        if orchestrator is None:
            raise RuntimeError("face detection is disabled in the effective configuration")
        tracking_pipeline.set_recognition_confirmer(
            TrackRecognitionConfirmer(
                camera_id=camera_id,
                min_confirmations=settings.min_confirmations,
                confirmation_window_seconds=settings.confirmation_window_seconds,
            )
            if settings.recognition_enabled and orchestrator.service.matcher is not None
            else None
        )
        return tracking_pipeline, orchestrator

    def _close_runtime(self) -> None:
        with self._lock:
            orchestrator = self._orchestrator
            pipeline = self._pipeline
            self._orchestrator = None
            self._pipeline = None
        if pipeline is not None:
            pipeline.set_recognition_confirmer(None)
        if orchestrator is None:
            return
        try:
            orchestrator.service.detector.close()
        except Exception:
            self._logger.debug("face detector close failed", exc_info=True)
        landmarker = orchestrator.service.landmarker
        close_landmarker = getattr(landmarker, "close", None)
        if callable(close_landmarker):
            try:
                close_landmarker()
            except Exception:
                self._logger.debug("face landmarker close failed", exc_info=True)
        matcher = orchestrator.service.matcher
        if matcher is not None:
            close = getattr(matcher.embedder, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    self._logger.debug("face embedder close failed", exc_info=True)

    def _publish(self, snapshot: FaceRecognitionSnapshot, *, generation: int) -> bool:
        with self._lock:
            if generation != self._generation:
                return False
            self._snapshot = snapshot
        self.snapshot_changed.emit(snapshot)
        return True

    def _run(self) -> None:
        loaded_signature: tuple[Any, ...] | None = None
        last_sequence: int | None = None
        try:
            while not self._stop_event.is_set():
                (
                    settings,
                    camera_id,
                    provider,
                    person_snapshot,
                    tracking_pipeline,
                    generation,
                ) = self._state()
                signature = (
                    settings,
                    camera_id,
                    id(tracking_pipeline),
                )
                if signature != loaded_signature:
                    self._close_runtime()
                    loaded_signature = signature
                    last_sequence = None
                    if not settings.face_detection_enabled:
                        if tracking_pipeline is not None:
                            tracking_pipeline.set_recognition_confirmer(None)
                        self._publish(
                            FaceRecognitionSnapshot(
                                camera_id=camera_id,
                                status=FaceRecognitionStatus.DISABLED,
                                message="Rilevamento facciale disabilitato",
                            ),
                            generation=generation,
                        )
                    elif camera_id is None:
                        self._publish(
                            FaceRecognitionSnapshot(
                                status=FaceRecognitionStatus.READY,
                                message="Face pipeline pronto; aprire una camera in focus",
                                detector_id=settings.detector_id,
                                recognizer_id=settings.recognizer_id,
                            ),
                            generation=generation,
                        )
                    elif tracking_pipeline is None:
                        self._publish(
                            FaceRecognitionSnapshot(
                                camera_id=camera_id,
                                status=FaceRecognitionStatus.READY,
                                message="Attesa del tracking persone condiviso",
                                detector_id=settings.detector_id,
                                recognizer_id=settings.recognizer_id,
                            ),
                            generation=generation,
                        )
                    else:
                        self._publish(
                            FaceRecognitionSnapshot(
                                camera_id=camera_id,
                                status=FaceRecognitionStatus.LOADING,
                                message="Caricamento face pipeline…",
                                detector_id=settings.detector_id,
                                recognizer_id=settings.recognizer_id,
                            ),
                            generation=generation,
                        )
                        try:
                            pipeline, orchestrator = self._build_runtime(
                                settings,
                                camera_id,
                                tracking_pipeline,
                            )
                            with self._lock:
                                self._pipeline = pipeline
                                self._orchestrator = orchestrator
                            if orchestrator is not None:
                                try:
                                    self.refresh_gallery()
                                except Exception as exc:
                                    self._logger.warning(
                                        "face gallery refresh failed: %s",
                                        exc,
                                    )
                                detector = orchestrator.service.detector
                                matcher = orchestrator.service.matcher
                                recognizer_status = (
                                    FaceRecognitionStatus.READY
                                    if matcher is not None and orchestrator.recognition_error is None
                                    else (
                                        FaceRecognitionStatus.UNSUPPORTED
                                        if settings.recognition_enabled
                                        else FaceRecognitionStatus.DISABLED
                                    )
                                )
                                recognizer_message = (
                                    "Recognizer pronto"
                                    if recognizer_status is FaceRecognitionStatus.READY
                                    else orchestrator.recognition_error
                                    or "Riconoscimento facciale disabilitato"
                                )
                                metadata = matcher.embedder.metadata if matcher is not None else None
                                self._publish(
                                    FaceRecognitionSnapshot(
                                        camera_id=camera_id,
                                        status=FaceRecognitionStatus.READY,
                                        message="Face pipeline pronto; attesa di persona",
                                        detection_status=FaceRecognitionStatus.READY,
                                        detection_message="Face detector pronto; attesa di persona",
                                        recognition_status=recognizer_status,
                                        recognition_message=recognizer_message,
                                        recognition_error=(
                                            orchestrator.recognition_error
                                            if recognizer_status is not FaceRecognitionStatus.READY
                                            else None
                                        ),
                                        detector_id=settings.detector_id,
                                        recognizer_id=settings.recognizer_id,
                                        effective_recognizer_id=(
                                            metadata.recognizer_id if metadata is not None else None
                                        ),
                                        detector_backend=getattr(detector, "backend_id", None),
                                        detector_model=getattr(detector, "detector_id", None),
                                        recognizer_backend=(metadata.backend if metadata is not None else None),
                                        recognizer_model=(metadata.model_id if metadata is not None else None),
                                        requested_detector_device=settings.detector_device,
                                        requested_recognizer_device=settings.recognizer_device,
                                    ),
                                    generation=generation,
                                )
                        except FileNotFoundError as exc:
                            self._publish(
                                FaceRecognitionSnapshot(
                                    camera_id=camera_id,
                                    status=FaceRecognitionStatus.MODEL_MISSING,
                                    message=str(exc),
                                    error=str(exc),
                                ),
                                generation=generation,
                            )
                        except (ValueError, RuntimeError) as exc:
                            self._publish(
                                FaceRecognitionSnapshot(
                                    camera_id=camera_id,
                                    status=FaceRecognitionStatus.UNSUPPORTED,
                                    message=str(exc),
                                    error=str(exc),
                                ),
                                generation=generation,
                            )
                if camera_id is None or provider is None or not settings.face_detection_enabled:
                    self._wake.wait(0.15)
                    self._wake.clear()
                    continue
                with self._lock:
                    pipeline = self._pipeline
                    orchestrator = self._orchestrator
                if pipeline is None or orchestrator is None:
                    self._wake.wait(0.25)
                    self._wake.clear()
                    continue
                try:
                    packet = provider.latest_frame()
                except Exception as exc:
                    self._publish(
                        FaceRecognitionSnapshot(
                            camera_id=camera_id,
                            status=FaceRecognitionStatus.ERROR,
                            message=f"Lettura frame face fallita: {exc}",
                            error=str(exc),
                        ),
                        generation=generation,
                    )
                    self._wake.wait(0.25)
                    self._wake.clear()
                    continue
                if packet is None or packet.sequence == last_sequence:
                    self._wake.wait(0.05)
                    self._wake.clear()
                    continue
                # Face sampling is independent from person FPS. Always use
                # the latest immutable track snapshot from the shared
                # pipeline, while the frame comes from the bounded latest
                # frame source. A reset clears ``latest_update`` and blocks
                # stale face work until person tracking publishes again.
                if person_snapshot is None:
                    self._wake.wait(0.02)
                    self._wake.clear()
                    continue
                last_sequence = packet.sequence
                if tracking_pipeline is None or pipeline is not tracking_pipeline:
                    self._wake.wait(0.02)
                    self._wake.clear()
                    continue
                update = tracking_pipeline.latest_update
                if update is None or update.camera_id != camera_id:
                    self._wake.wait(0.02)
                    self._wake.clear()
                    continue
                try:
                    result = orchestrator.process(
                        packet.frame,
                        update,
                        pipeline,
                        timestamp=packet.received_at_utc,
                    )
                except Exception as exc:
                    self._publish(
                        FaceRecognitionSnapshot(
                            camera_id=camera_id,
                            status=FaceRecognitionStatus.ERROR,
                            message=f"Inferenza face fallita: {exc}",
                            error=str(exc),
                            detector_id=settings.detector_id,
                            recognizer_id=settings.recognizer_id,
                            requested_detector_device=settings.detector_device,
                            requested_recognizer_device=settings.recognizer_device,
                        ),
                        generation=generation,
                    )
                    self._wake.wait(0.25)
                    self._wake.clear()
                    continue
                metrics_snapshot = self._metrics[camera_id].snapshot()
                overlays: list[FaceOverlayState] = []
                final_recognitions = {
                    track_id: recognition
                    for track_id, recognition in result.final_recognitions
                }
                detector_device = getattr(orchestrator.service.detector, "device_used", None)
                actual_detector_devices: set[str] = (
                    {str(detector_device)} if detector_device else set()
                )
                matcher = orchestrator.service.matcher
                recognition_error = result.analysis.recognition_error or orchestrator.recognition_error
                recognizer_device = (
                    matcher.embedder.metadata.actual_device
                    if matcher is not None
                    else None
                )
                actual_recognizer_devices: set[str] = (
                    {str(recognizer_device)} if recognizer_device else set()
                )
                for tracked in result.analysis.results:
                    for decision in tracked.decisions:
                        detection = decision.detection
                        recognition = final_recognitions.get(tracked.track_id)
                        if recognition is None and decision.recognition is not None:
                            recognition = result.final_for_track(tracked.track_id)
                        if detection.device:
                            actual_detector_devices.add(detection.device)
                        if recognition is not None and recognition.actual_device:
                            actual_recognizer_devices.add(recognition.actual_device)
                        overlays.append(
                            FaceOverlayState(
                                camera_id=camera_id,
                                track_id=tracked.track_id,
                                bbox=decision.frame_bbox,
                                landmarks=(
                                    detection.landmarks.points
                                    if detection.landmarks is not None
                                    else ()
                                ),
                                recognition_status=(
                                    recognition.status if recognition is not None else "unknown"
                                ),
                                person_id=recognition.person_id if recognition is not None else None,
                                person_name=recognition.person_name if recognition is not None else None,
                                score=recognition.score if recognition is not None else None,
                                threshold=recognition.threshold if recognition is not None else None,
                            )
                        )
                known = sum(overlay.recognition_status == "known" for overlay in overlays)
                unknown = sum(overlay.recognition_status == "unknown" for overlay in overlays)
                self._publish(
                    FaceRecognitionSnapshot(
                        camera_id=camera_id,
                        status=FaceRecognitionStatus.RUNNING,
                        message="Face analysis attiva",
                        detection_status=FaceRecognitionStatus.RUNNING,
                        detection_message="Face detection attiva",
                        recognition_status=(
                            FaceRecognitionStatus.RUNNING
                            if matcher is not None and recognition_error is None
                            else (
                                FaceRecognitionStatus.UNSUPPORTED
                                if settings.recognition_enabled
                                else FaceRecognitionStatus.DISABLED
                            )
                        ),
                        recognition_message=(
                            "Face recognition attiva"
                            if matcher is not None and recognition_error is None
                            else recognition_error or "Riconoscimento facciale disabilitato"
                        ),
                        recognition_error=recognition_error,
                        detector_id=settings.detector_id,
                        recognizer_id=settings.recognizer_id,
                        effective_recognizer_id=(
                            matcher.embedder.metadata.recognizer_id
                            if matcher is not None
                            else None
                        ),
                        detector_backend=getattr(orchestrator.service.detector, "backend_id", None),
                        detector_model=getattr(orchestrator.service.detector, "detector_id", None),
                        recognizer_backend=(
                            matcher.embedder.metadata.backend if matcher is not None else None
                        ),
                        recognizer_model=(
                            matcher.embedder.metadata.model_id if matcher is not None else None
                        ),
                        requested_detector_device=settings.detector_device,
                        actual_detector_device=(
                            next(iter(actual_detector_devices), None)
                            if len(actual_detector_devices) == 1
                            else ("mixed" if actual_detector_devices else None)
                        ),
                        requested_recognizer_device=settings.recognizer_device,
                        actual_recognizer_device=(
                            next(iter(actual_recognizer_devices), None)
                            if len(actual_recognizer_devices) == 1
                            else ("mixed" if actual_recognizer_devices else None)
                        ),
                        frame_sequence=packet.sequence,
                        frame_timestamp=packet.received_at_utc,
                        face_count=len(overlays),
                        recognized_count=known,
                        unknown_count=unknown,
                        gallery_count=len(self._gallery.persons),
                        overlays=tuple(overlays),
                        telemetry=metrics_snapshot.to_dict(),
                    ),
                    generation=generation,
                )
        finally:
            self._close_runtime()


__all__ = ["FaceRecognitionController", "InferenceFrameSource"]

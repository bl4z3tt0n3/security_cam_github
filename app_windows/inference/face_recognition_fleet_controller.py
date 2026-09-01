"""Fleet face-recognition controller with one shared set of model resources."""

from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
from typing import Mapping

from app.face import (
    FaceRecognitionOrchestrator,
    TrackRecognitionConfirmer,
    create_face_orchestrator,
)
from app.face.service import FaceAnalysisService
from app.inference import InferenceGate
from app.metrics import CameraMetrics
from app.tracking import CameraTrackingPipeline
from app_windows.models.face_recognition_state import (
    FaceOverlayState,
    FaceRecognitionSettings,
    FaceRecognitionSnapshot,
    FaceRecognitionStatus,
)
from app_windows.models.person_detection_state import PersonDetectionSnapshot

from .face_recognition_controller import (
    FaceRecognitionController,
    InferenceFrameSource,
)


class FleetFaceRecognitionController(FaceRecognitionController):
    """Analyze every camera with persons while sharing one face model set.

    The class preserves FaceRecognitionController's gallery/enrollment/config API.
    Only the live scheduler changes: one detector/landmarker/embedder is loaded,
    and lightweight per-camera FaceRecognitionOrchestrators share those objects.
    Live work remains single-threaded inside this controller, so sharing model
    instances does not introduce concurrent calls into backend objects.
    """

    def __init__(
        self,
        *,
        repo_root: Path,
        config,
        settings: FaceRecognitionSettings | None = None,
        inference_gate: InferenceGate | None = None,
        logger: logging.Logger | None = None,
        parent=None,
    ) -> None:
        super().__init__(
            repo_root=repo_root,
            config=config,
            settings=settings,
            inference_gate=inference_gate,
            logger=logger,
            parent=parent,
        )
        self._providers_by_camera: dict[str, InferenceFrameSource] = {}
        self._person_by_camera: dict[str, PersonDetectionSnapshot] = {}
        self._snapshots_by_camera: dict[str, FaceRecognitionSnapshot] = {}
        self._orchestrators_by_camera: dict[str, FaceRecognitionOrchestrator] = {}
        self._pipelines_by_camera: dict[str, CameraTrackingPipeline] = {}
        self._template_orchestrator: FaceRecognitionOrchestrator | None = None
        self._source_generation = 0

    @property
    def snapshot(self) -> FaceRecognitionSnapshot:
        with self._lock:
            if self._camera_id is not None:
                value = self._snapshots_by_camera.get(self._camera_id)
                if value is not None:
                    return value
            return self._snapshot

    @property
    def snapshots(self) -> Mapping[str, FaceRecognitionSnapshot]:
        with self._lock:
            return dict(self._snapshots_by_camera)

    def set_sources(self, providers: Mapping[str, InferenceFrameSource]) -> None:
        normalized = {
            str(camera_id): provider
            for camera_id, provider in providers.items()
            if provider is not None
        }
        with self._lock:
            changed = (
                set(self._providers_by_camera) != set(normalized)
                or any(
                    self._providers_by_camera.get(key) is not value
                    for key, value in normalized.items()
                )
            )
            self._providers_by_camera = normalized
            if changed:
                self._source_generation += 1
            removed = set(self._person_by_camera).difference(normalized)
            for camera_id in removed:
                self._person_by_camera.pop(camera_id, None)
                self._snapshots_by_camera.pop(camera_id, None)
                self._detach_camera_runtime(camera_id)
        self._wake.set()

    def set_active_camera(
        self,
        camera_id: str | None,
        provider: InferenceFrameSource | None,
    ) -> None:
        """Select which already-running fleet snapshot the UI exposes."""

        normalized = camera_id.strip() if camera_id else None
        with self._lock:
            self._camera_id = normalized
            self._provider = provider
            if normalized is not None and provider is not None:
                self._providers_by_camera[normalized] = provider
            cached = (
                self._snapshots_by_camera.get(normalized)
                if normalized is not None
                else None
            )
            if cached is not None:
                self._snapshot = cached
        if cached is not None:
            self.snapshot_changed.emit(cached)
        self._wake.set()

    def set_person_snapshot(self, value: PersonDetectionSnapshot | None) -> None:
        if value is None or not isinstance(value, PersonDetectionSnapshot):
            return
        camera_id = value.camera_id
        if camera_id is None:
            return
        with self._lock:
            self._person_by_camera[camera_id] = value
            if value.tracking_pipeline is not None:
                self._pipelines_by_camera[camera_id] = value.tracking_pipeline
            if camera_id == self._camera_id:
                self._person_snapshot = value
                self._tracking_pipeline = value.tracking_pipeline
        self._wake.set()

    def _fleet_state(self):
        with self._lock:
            return (
                self._settings,
                dict(self._providers_by_camera),
                dict(self._person_by_camera),
                self._generation,
                self._source_generation,
                self._camera_id,
            )

    def _publish_fleet(self, snapshot: FaceRecognitionSnapshot, *, generation: int) -> bool:
        camera_id = snapshot.camera_id
        with self._lock:
            if generation != self._generation:
                return False
            if camera_id is not None:
                self._snapshots_by_camera[camera_id] = snapshot
            if camera_id == self._camera_id or camera_id is None:
                self._snapshot = snapshot
        self.snapshot_changed.emit(snapshot)
        return True

    def _build_shared_runtime(
        self,
        settings: FaceRecognitionSettings,
    ) -> FaceRecognitionOrchestrator:
        config = self._effective_config(settings)
        template = create_face_orchestrator(
            config,
            "__fleet_shared__",
            model_root=self._repo_root,
            metrics=None,
            inference_gate=self._gate,
        )
        if template is None:
            raise RuntimeError("face detection is disabled in the effective configuration")
        with self._lock:
            self._template_orchestrator = template
            # Keep inherited gallery/enrollment helpers pointing at the shared
            # matcher. The template is never used for live camera processing.
            self._orchestrator = template
        return template

    def _camera_orchestrator(
        self,
        camera_id: str,
        pipeline: CameraTrackingPipeline,
        settings: FaceRecognitionSettings,
        template: FaceRecognitionOrchestrator,
    ) -> FaceRecognitionOrchestrator:
        with self._lock:
            existing = self._orchestrators_by_camera.get(camera_id)
            previous_pipeline = self._pipelines_by_camera.get(camera_id)
            if existing is not None and previous_pipeline is pipeline:
                return existing

        if previous_pipeline is not None and previous_pipeline is not pipeline:
            previous_pipeline.set_recognition_confirmer(None)
            if existing is not None:
                existing.reset()

        metrics = pipeline.metrics or self._metrics.setdefault(
            camera_id,
            CameraMetrics(camera_id),
        )
        self._metrics[camera_id] = metrics
        shared = template.service
        service = FaceAnalysisService(
            camera_id,
            shared.detector,
            aligner=shared.aligner,
            landmarker=shared.landmarker,
            matcher=shared.matcher,
            evaluator=shared.evaluator,
            metrics=metrics,
            inference_gate=self._gate,
        )
        orchestrator = FaceRecognitionOrchestrator(
            service,
            face_fps=template.face_fps,
            recognition_fps=template.recognition_fps,
            recognition_error=template.recognition_error,
            enabled=True,
        )
        pipeline.set_recognition_confirmer(
            TrackRecognitionConfirmer(
                camera_id=camera_id,
                min_confirmations=settings.min_confirmations,
                confirmation_window_seconds=settings.confirmation_window_seconds,
            )
            if settings.recognition_enabled and shared.matcher is not None
            else None
        )
        with self._lock:
            self._orchestrators_by_camera[camera_id] = orchestrator
            self._pipelines_by_camera[camera_id] = pipeline
            if camera_id == self._camera_id:
                self._pipeline = pipeline
        return orchestrator

    def _detach_camera_runtime(self, camera_id: str) -> None:
        pipeline = self._pipelines_by_camera.pop(camera_id, None)
        orchestrator = self._orchestrators_by_camera.pop(camera_id, None)
        if pipeline is not None:
            pipeline.set_recognition_confirmer(None)
        if orchestrator is not None:
            orchestrator.reset()

    def _close_runtime(self) -> None:
        with self._lock:
            template = self._template_orchestrator
            pipelines = tuple(self._pipelines_by_camera.values())
            self._template_orchestrator = None
            self._orchestrators_by_camera.clear()
            self._pipelines_by_camera.clear()
            self._orchestrator = None
            self._pipeline = None
        for pipeline in pipelines:
            pipeline.set_recognition_confirmer(None)
        if template is None:
            return

        try:
            template.service.detector.close()
        except Exception:
            self._logger.debug("shared face detector close failed", exc_info=True)
        landmarker = template.service.landmarker
        close_landmarker = getattr(landmarker, "close", None)
        if callable(close_landmarker):
            try:
                close_landmarker()
            except Exception:
                self._logger.debug("shared face landmarker close failed", exc_info=True)
        matcher = template.service.matcher
        if matcher is not None:
            close = getattr(matcher.embedder, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    self._logger.debug("shared face embedder close failed", exc_info=True)

    def refresh_gallery(self):
        gallery = super().refresh_gallery()
        # Every camera service shares the exact same matcher object, therefore
        # one atomic matcher.refresh() updates the entire fleet.
        return gallery

    def _snapshot_from_result(
        self,
        *,
        camera_id: str,
        settings: FaceRecognitionSettings,
        orchestrator: FaceRecognitionOrchestrator,
        packet,
        result,
    ) -> FaceRecognitionSnapshot:
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
        recognition_error = (
            result.analysis.recognition_error or orchestrator.recognition_error
        )
        recognizer_device = (
            matcher.embedder.metadata.actual_device if matcher is not None else None
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
                        person_id=(
                            recognition.person_id if recognition is not None else None
                        ),
                        person_name=(
                            recognition.person_name if recognition is not None else None
                        ),
                        score=recognition.score if recognition is not None else None,
                        threshold=(
                            recognition.threshold if recognition is not None else None
                        ),
                    )
                )
        known = sum(item.recognition_status == "known" for item in overlays)
        unknown = sum(item.recognition_status == "unknown" for item in overlays)
        return FaceRecognitionSnapshot(
            camera_id=camera_id,
            status=FaceRecognitionStatus.RUNNING,
            message="Face analysis fleet attiva",
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
        )

    def _run(self) -> None:
        loaded_settings: FaceRecognitionSettings | None = None
        loaded_generation = -1
        observed_source_generation = -1
        template: FaceRecognitionOrchestrator | None = None
        last_sequences: dict[str, int] = {}

        try:
            while not self._stop_event.is_set():
                (
                    settings,
                    providers,
                    person_snapshots,
                    generation,
                    source_generation,
                    active_camera_id,
                ) = self._fleet_state()

                if loaded_settings != settings or loaded_generation != generation:
                    self._close_runtime()
                    template = None
                    loaded_settings = settings
                    loaded_generation = generation
                    last_sequences.clear()
                    if not settings.face_detection_enabled:
                        for camera_id in providers:
                            self._publish_fleet(
                                FaceRecognitionSnapshot(
                                    camera_id=camera_id,
                                    status=FaceRecognitionStatus.DISABLED,
                                    message="Rilevamento facciale disabilitato",
                                ),
                                generation=generation,
                            )
                        self._wake.wait(0.1)
                        self._wake.clear()
                        continue

                if source_generation != observed_source_generation:
                    observed_source_generation = source_generation
                    last_sequences.clear()

                eligible = [
                    camera_id
                    for camera_id, snapshot in person_snapshots.items()
                    if camera_id in providers
                    and snapshot.tracking_pipeline is not None
                    and snapshot.person_count > 0
                ]
                should_load = bool(eligible or active_camera_id)
                if template is None and should_load and settings.face_detection_enabled:
                    target = active_camera_id or (eligible[0] if eligible else None)
                    if target is not None:
                        self._publish_fleet(
                            FaceRecognitionSnapshot(
                                camera_id=target,
                                status=FaceRecognitionStatus.LOADING,
                                message="Caricamento face pipeline condivisa…",
                                detector_id=settings.detector_id,
                                recognizer_id=settings.recognizer_id,
                            ),
                            generation=generation,
                        )
                    try:
                        template = self._build_shared_runtime(settings)
                        try:
                            self.refresh_gallery()
                        except Exception as exc:
                            self._logger.warning("face fleet gallery refresh failed: %s", exc)
                    except FileNotFoundError as exc:
                        if target is not None:
                            self._publish_fleet(
                                FaceRecognitionSnapshot(
                                    camera_id=target,
                                    status=FaceRecognitionStatus.MODEL_MISSING,
                                    message=str(exc),
                                    error=str(exc),
                                ),
                                generation=generation,
                            )
                        self._wake.wait(0.25)
                        self._wake.clear()
                        continue
                    except (ValueError, RuntimeError) as exc:
                        if target is not None:
                            self._publish_fleet(
                                FaceRecognitionSnapshot(
                                    camera_id=target,
                                    status=FaceRecognitionStatus.UNSUPPORTED,
                                    message=str(exc),
                                    error=str(exc),
                                ),
                                generation=generation,
                            )
                        self._wake.wait(0.25)
                        self._wake.clear()
                        continue

                if template is None:
                    self._wake.wait(0.05)
                    self._wake.clear()
                    continue

                did_work = False
                for camera_id in tuple(eligible):
                    current_settings, current_providers, current_person, current_generation, current_source_generation, _active = self._fleet_state()
                    if (
                        current_generation != generation
                        or current_settings != settings
                        or current_source_generation != source_generation
                    ):
                        break
                    provider = current_providers.get(camera_id)
                    person = current_person.get(camera_id)
                    if provider is None or person is None:
                        continue
                    pipeline = person.tracking_pipeline
                    if pipeline is None:
                        continue
                    update = pipeline.latest_update
                    if update is None or not update.active_tracks:
                        orchestrator = self._orchestrators_by_camera.get(camera_id)
                        if orchestrator is not None:
                            orchestrator.reset()
                        continue
                    try:
                        packet = provider.latest_frame()
                    except Exception as exc:
                        self._publish_fleet(
                            FaceRecognitionSnapshot(
                                camera_id=camera_id,
                                status=FaceRecognitionStatus.ERROR,
                                message=f"Lettura frame face fallita: {exc}",
                                error=str(exc),
                            ),
                            generation=generation,
                        )
                        continue
                    if packet is None or packet.sequence == last_sequences.get(camera_id):
                        continue

                    orchestrator = self._camera_orchestrator(
                        camera_id,
                        pipeline,
                        settings,
                        template,
                    )
                    last_sequences[camera_id] = packet.sequence
                    did_work = True
                    try:
                        result = orchestrator.process(
                            packet.frame,
                            update,
                            pipeline,
                            timestamp=packet.received_at_utc,
                        )
                    except Exception as exc:
                        self._publish_fleet(
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
                        continue
                    self._publish_fleet(
                        self._snapshot_from_result(
                            camera_id=camera_id,
                            settings=settings,
                            orchestrator=orchestrator,
                            packet=packet,
                            result=result,
                        ),
                        generation=generation,
                    )

                if not did_work:
                    self._wake.wait(0.02)
                    self._wake.clear()
        finally:
            self._close_runtime()


__all__ = ["FleetFaceRecognitionController"]

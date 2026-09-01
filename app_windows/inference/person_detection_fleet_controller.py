"""Fleet person-detection controller for the native Windows monitor.

One detector instance consumes the latest frames already owned by
CameraMonitorController.  It never opens a camera itself.  When the detector
supports true batch inference (OpenVINO GPU), frames due at the same sampling
instant are grouped into one backend batch.
"""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
import threading
import time
from typing import Mapping

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

from .person_detection_controller import InferenceFrameSource


class FleetPersonDetectionController(QObject):
    """Run one shared person detector across every already-open camera provider."""

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
        self._gate = inference_gate or InferenceGate()
        self._active_camera_id: str | None = None
        self._providers: dict[str, InferenceFrameSource] = {}
        self._snapshots: dict[str, PersonDetectionSnapshot] = {}
        self._metrics: dict[str, CameraMetrics] = {}
        self._pipelines: dict[str, CameraTrackingPipeline] = {}
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._settings_generation = 0
        self._source_generation = 0

    @property
    def settings(self) -> PersonDetectionSettings:
        with self._lock:
            return self._settings

    @property
    def snapshot(self) -> PersonDetectionSnapshot:
        with self._lock:
            if self._active_camera_id is not None:
                value = self._snapshots.get(self._active_camera_id)
                if value is not None:
                    return value
            return PersonDetectionSnapshot(
                settings=self._settings,
                model_path=self._settings.model,
                requested_device=self._settings.device,
            )

    @property
    def snapshots(self) -> Mapping[str, PersonDetectionSnapshot]:
        with self._lock:
            return dict(self._snapshots)

    @property
    def inference_gate(self) -> InferenceGate:
        return self._gate

    @property
    def tracking_pipeline(self) -> CameraTrackingPipeline | None:
        with self._lock:
            if self._active_camera_id is None:
                return None
            return self._pipelines.get(self._active_camera_id)

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
                name="windows-person-detection-fleet",
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

    def update_settings(self, settings: PersonDetectionSettings) -> None:
        if not isinstance(settings, PersonDetectionSettings):
            raise TypeError("settings must be PersonDetectionSettings")
        with self._lock:
            if settings == self._settings:
                return
            self._settings = settings
            self._settings_generation += 1
        self._wake.set()

    def set_sources(self, providers: Mapping[str, InferenceFrameSource]) -> None:
        """Replace the provider view without opening/stopping any stream."""

        normalized = {
            str(camera_id): provider
            for camera_id, provider in providers.items()
            if provider is not None
        }
        with self._lock:
            changed = (
                set(self._providers) != set(normalized)
                or any(self._providers.get(key) is not value for key, value in normalized.items())
            )
            self._providers = normalized
            if changed:
                self._source_generation += 1
            removed = set(self._pipelines).difference(normalized)
            for camera_id in removed:
                self._pipelines.pop(camera_id, None)
        self._wake.set()

    def set_active_camera(
        self,
        camera_id: str | None,
        provider: InferenceFrameSource | None,
    ) -> None:
        """Select the snapshot/tracker consumed by face recognition and focus UI."""

        normalized = camera_id.strip() if camera_id else None
        with self._lock:
            self._active_camera_id = normalized
            if normalized is not None and provider is not None:
                self._providers[normalized] = provider
        self._wake.set()

    def _state(self):
        with self._lock:
            return (
                self._settings,
                dict(self._providers),
                self._settings_generation,
                self._source_generation,
            )

    def _resolve_model_path(self, model: str | None) -> Path | None:
        if not model:
            return None
        path = Path(model).expanduser()
        return path if path.is_absolute() else self._repo_root / path

    @staticmethod
    def _model_signature(settings: PersonDetectionSettings) -> tuple[object, ...]:
        return (
            settings.enabled,
            settings.backend,
            settings.model,
            settings.device,
            settings.precision,
            settings.fallback_device,
            settings.image_size,
            settings.openvino_performance_mode,
            settings.openvino_num_streams,
            settings.openvino_num_requests,
            settings.openvino_cpu_threads,
            settings.max_process_ram_mb,
            settings.classes,
            settings.prompts,
            settings.confidence_threshold,
            settings.inference_fps,
            settings.show_masks,
        )

    def _build_detector(self, settings: PersonDetectionSettings) -> PersonDetector | None:
        if not settings.enabled:
            return None
        backend = settings.backend
        model_path = self._resolve_model_path(settings.model)
        if backend == "auto":
            name = (settings.model or "").casefold()
            if name.endswith(".onnx"):
                backend = "onnx"
            elif name.endswith(("yolo26s.pt", "yolo26n.pt", "_openvino_model", ".xml")):
                backend = "openvino"
            else:
                backend = "yoloe"
        if backend == "yoloe" and (model_path is None or not model_path.is_file()):
            raise FileNotFoundError(f"Modello YOLOE non trovato: {settings.model}")
        if backend == "onnx" and (model_path is None or not model_path.is_file()):
            raise FileNotFoundError(f"Modello ONNX non trovato: {settings.model}")

        return create_person_detector(
            PersonDetectionConfig(
                enabled=True,
                backend=settings.backend,
                model=settings.model,
                confidence_threshold=settings.confidence_threshold,
                inference_fps=settings.inference_fps,
                precision=settings.precision,
                device=settings.device,
                fallback_device=settings.fallback_device,
                image_size=settings.image_size,
                openvino_performance_mode=settings.openvino_performance_mode,
                openvino_num_streams=settings.openvino_num_streams,
                openvino_num_requests=settings.openvino_num_requests,
                openvino_cpu_threads=settings.openvino_cpu_threads,
                max_process_ram_mb=settings.max_process_ram_mb,
                classes=list(settings.classes),
                prompts=list(settings.prompts),
                show_masks=settings.show_masks,
            ),
            model_root=self._repo_root,
        )

    @staticmethod
    def _close_detector(detector: PersonDetector | None) -> None:
        if detector is None:
            return
        try:
            detector.close()
        except Exception:
            logging.getLogger(__name__).debug("fleet detector close failed", exc_info=True)

    def _metrics_for(self, camera_id: str) -> CameraMetrics:
        with self._lock:
            value = self._metrics.get(camera_id)
            if value is None:
                value = CameraMetrics(camera_id)
                self._metrics[camera_id] = value
            return value

    def _tracking_for(self, camera_id: str) -> CameraTrackingPipeline:
        with self._lock:
            value = self._pipelines.get(camera_id)
            if value is None:
                value = CameraTrackingPipeline(
                    camera_id,
                    tracker=IoUGreedyTracker(),
                    metrics=self._metrics_for(camera_id),
                )
                self._pipelines[camera_id] = value
            return value

    def _snapshot_for(
        self,
        settings: PersonDetectionSettings,
        camera_id: str,
        *,
        status: PersonDetectionStatus,
        message: str,
        detector: PersonDetector | None = None,
        detections=(),
        packet=None,
        latency_ms: float | None = None,
        detector_failures: int = 0,
        tracking_update=None,
        tracking_pipeline=None,
    ) -> PersonDetectionSnapshot:
        metrics = self._metrics_for(camera_id)
        metric_snapshot = metrics.snapshot()
        source_width = source_height = None
        if packet is not None:
            try:
                source_height = int(packet.frame.shape[0])
                source_width = int(packet.frame.shape[1])
            except (AttributeError, IndexError, TypeError, ValueError):
                pass
        return PersonDetectionSnapshot(
            camera_id=camera_id,
            status=status,
            message=message,
            error=message if status in {PersonDetectionStatus.ERROR, PersonDetectionStatus.MODEL_MISSING} else None,
            settings=settings,
            model_path=settings.model,
            requested_device=settings.device,
            actual_device=detector.device_used if detector is not None else None,
            device_verified=detector.device_verified if detector is not None else False,
            provider=detector.provider_used if detector is not None else None,
            backend=detector.backend if detector is not None else settings.backend,
            precision=getattr(detector, "precision", settings.precision) if detector is not None else settings.precision,
            inference_fps=metric_snapshot.person_detection_fps,
            latency_ms=latency_ms,
            person_count=sum(item.label.casefold() == "person" for item in detections),
            detection_count=len(detections),
            detections=tuple(detections),
            frame_sequence=packet.sequence if packet is not None else None,
            frame_timestamp=packet.received_at_utc if packet is not None else None,
            source_width=source_width,
            source_height=source_height,
            result_monotonic=time.monotonic() if packet is not None else None,
            detector_failures=detector_failures,
            tracking_update=tracking_update,
            tracking_pipeline=tracking_pipeline,
        )

    def _publish(self, snapshot: PersonDetectionSnapshot) -> None:
        camera_id = snapshot.camera_id
        if camera_id is None:
            return
        with self._lock:
            self._snapshots[camera_id] = snapshot
        self.snapshot_changed.emit(snapshot)

    def _publish_state_for_all(
        self,
        settings: PersonDetectionSettings,
        providers: Mapping[str, InferenceFrameSource],
        *,
        status: PersonDetectionStatus,
        message: str,
        detector: PersonDetector | None = None,
    ) -> None:
        for camera_id in providers:
            self._publish(
                self._snapshot_for(
                    settings,
                    camera_id,
                    status=status,
                    message=message,
                    detector=detector,
                )
            )

    def _run(self) -> None:
        detector: PersonDetector | None = None
        loaded_signature: tuple[object, ...] | None = None
        loaded_generation = -1
        observed_source_generation = -1
        last_sequences: dict[str, int] = {}
        next_at: dict[str, float] = {}
        failures: dict[str, int] = {}

        try:
            while not self._stop_event.is_set():
                settings, providers, generation, source_generation = self._state()
                signature = self._model_signature(settings)
                if source_generation != observed_source_generation:
                    observed_source_generation = source_generation
                    last_sequences.clear()
                    next_at.clear()

                if signature != loaded_signature or generation != loaded_generation:
                    self._close_detector(detector)
                    detector = None
                    loaded_signature = signature
                    loaded_generation = generation
                    last_sequences.clear()
                    next_at.clear()
                    failures.clear()
                    for pipeline in tuple(self._pipelines.values()):
                        pipeline.reset(reason="person model changed")
                    if not settings.enabled:
                        self._publish_state_for_all(
                            settings,
                            providers,
                            status=PersonDetectionStatus.DISABLED,
                            message="Rilevamento persone disabilitato",
                        )
                        self._wake.wait(0.15)
                        self._wake.clear()
                        continue
                    self._publish_state_for_all(
                        settings,
                        providers,
                        status=PersonDetectionStatus.LOADING,
                        message="Caricamento modello persone condiviso…",
                    )
                    try:
                        detector = self._build_detector(settings)
                    except FileNotFoundError as exc:
                        self._publish_state_for_all(
                            settings,
                            providers,
                            status=PersonDetectionStatus.MODEL_MISSING,
                            message=str(exc),
                        )
                    except Exception as exc:
                        self._logger.error("fleet person model load failed: %s", exc)
                        self._publish_state_for_all(
                            settings,
                            providers,
                            status=PersonDetectionStatus.ERROR,
                            message=str(exc) or type(exc).__name__,
                        )
                    if detector is None:
                        self._wake.wait(0.25)
                        self._wake.clear()
                        continue
                    self._publish_state_for_all(
                        settings,
                        providers,
                        status=PersonDetectionStatus.READY,
                        message=f"Backend {detector.backend} condiviso pronto",
                        detector=detector,
                    )

                if detector is None or not providers:
                    self._wake.wait(0.1)
                    self._wake.clear()
                    continue

                now = time.monotonic()
                due: list[tuple[str, InferenceFrameSource, object]] = []
                for camera_id, provider in providers.items():
                    if now < next_at.get(camera_id, 0.0):
                        continue
                    try:
                        packet = provider.latest_frame()
                    except Exception as exc:
                        failures[camera_id] = failures.get(camera_id, 0) + 1
                        self._publish(
                            self._snapshot_for(
                                settings,
                                camera_id,
                                status=PersonDetectionStatus.ERROR,
                                message=f"Lettura frame inferenza fallita: {exc}",
                                detector=detector,
                                detector_failures=failures[camera_id],
                            )
                        )
                        next_at[camera_id] = now + 0.25
                        continue
                    if packet is None or packet.sequence == last_sequences.get(camera_id):
                        continue
                    due.append((camera_id, provider, packet))

                if not due:
                    self._wake.wait(0.01)
                    self._wake.clear()
                    continue

                batch_size = (
                    min(len(due), detector.preferred_batch_size)
                    if detector.supports_batch_inference
                    else 1
                )
                batch_size = max(1, batch_size)
                for offset in range(0, len(due), batch_size):
                    chunk = due[offset : offset + batch_size]
                    frames = [entry[2].frame for entry in chunk]
                    timestamps = [entry[2].received_at_utc for entry in chunk]
                    started = time.perf_counter()
                    try:
                        if len(chunk) > 1 and detector.supports_batch_inference:
                            results = self._gate.run(
                                detector.detect_batch,
                                frames,
                                timestamps,
                            )
                        else:
                            results = [
                                self._gate.run(detector.detect, frames[0], timestamps[0])
                            ]
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        per_frame_latency = elapsed_ms / max(1, len(chunk))
                    except Exception as exc:
                        for camera_id, _provider, packet in chunk:
                            failures[camera_id] = failures.get(camera_id, 0) + 1
                            last_sequences[camera_id] = packet.sequence
                            next_at[camera_id] = time.monotonic() + 0.5
                            self._publish(
                                self._snapshot_for(
                                    settings,
                                    camera_id,
                                    status=PersonDetectionStatus.ERROR,
                                    message=f"Inferenza persone ({detector.backend}) fallita: {exc}",
                                    detector=detector,
                                    packet=packet,
                                    detector_failures=failures[camera_id],
                                )
                            )
                        continue

                    if len(results) != len(chunk):
                        self._logger.error(
                            "fleet detector result count mismatch inputs=%s outputs=%s",
                            len(chunk),
                            len(results),
                        )
                        continue

                    current_settings, current_providers, current_generation, current_source_generation = self._state()
                    if (
                        current_generation != generation
                        or self._model_signature(current_settings) != signature
                        or current_source_generation != source_generation
                    ):
                        # Configuration/provider changes invalidate in-flight
                        # results; the next loop uses the new fleet snapshot.
                        continue

                    completed_at = time.monotonic()
                    for (camera_id, _provider, packet), raw in zip(chunk, results, strict=True):
                        if camera_id not in current_providers:
                            continue
                        detections = tuple(
                            replace(item, timestamp=packet.received_at_utc)
                            for item in raw
                            if item.confidence >= settings.confidence_threshold
                        )
                        metrics = self._metrics_for(camera_id)
                        metrics.record_person_detection(per_frame_latency)
                        pipeline = self._tracking_for(camera_id)
                        tracking_update = pipeline.update(
                            tuple(
                                item
                                for item in detections
                                if item.label.casefold() == "person"
                            )
                        )
                        last_sequences[camera_id] = packet.sequence
                        next_at[camera_id] = completed_at + 1.0 / settings.inference_fps
                        self._publish(
                            self._snapshot_for(
                                settings,
                                camera_id,
                                status=PersonDetectionStatus.RUNNING,
                                message=f"Inferenza persone ({detector.backend}) attiva",
                                detector=detector,
                                detections=detections,
                                packet=packet,
                                latency_ms=per_frame_latency,
                                detector_failures=failures.get(camera_id, 0),
                                tracking_update=tracking_update,
                                tracking_pipeline=pipeline,
                            )
                        )
        finally:
            self._close_detector(detector)


__all__ = ["FleetPersonDetectionController"]

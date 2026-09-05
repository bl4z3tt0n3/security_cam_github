"""Fleet person-detection controller for the native Windows monitor.

One detector instance consumes the latest frames already owned by
CameraMonitorController.  It never opens a camera itself.  When the detector
supports true batch inference (OpenVINO GPU), frames due at the same sampling
instant are grouped into one backend batch.
"""

from __future__ import annotations

from dataclasses import replace
import logging
import math
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
    _LOAD_RETRY_MIN_S = 0.5
    _LOAD_RETRY_MAX_S = 8.0

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
        self._source_sessions: dict[str, int] = {}
        self._provider_markers: dict[str, tuple[object, ...]] = {}
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

    @staticmethod
    def _provider_marker(provider: InferenceFrameSource) -> tuple[object, ...]:
        """Return a cheap source-session marker, including reconnects when exposed."""

        marker: list[object] = [id(provider)]
        snapshot_method = getattr(provider, "snapshot", None)
        if callable(snapshot_method):
            try:
                snapshot = snapshot_method()
                worker = getattr(snapshot, "worker", None)
                reconnect_count = getattr(worker, "reconnect_count", None)
                if reconnect_count is not None:
                    marker.append(("reconnect_count", int(reconnect_count)))
            except Exception:
                # Session invalidation must never make frame acquisition fail.
                pass
        for attribute in ("session_generation", "source_generation"):
            value = getattr(provider, attribute, None)
            if value is not None:
                try:
                    marker.append((attribute, int(value)))
                except (TypeError, ValueError):
                    marker.append((attribute, str(value)))
        return tuple(marker)

    def _invalidate_source_locked(self, camera_id: str, *, reason: str) -> int:
        self._source_sessions[camera_id] = self._source_sessions.get(camera_id, 0) + 1
        self._source_generation += 1
        pipeline = self._pipelines.get(camera_id)
        if pipeline is not None:
            pipeline.reset(reason=reason)
        return self._source_sessions[camera_id]

    def invalidate_source(self, camera_id: str, *, reason: str = "camera source reconnected") -> None:
        """Explicitly invalidate per-camera tracking/recognition state on reconnect."""

        normalized = camera_id.strip()
        if not normalized:
            return
        with self._lock:
            if normalized not in self._providers:
                return
            self._provider_markers[normalized] = self._provider_marker(
                self._providers[normalized]
            )
            self._invalidate_source_locked(normalized, reason=reason)
        self._wake.set()

    def _observe_provider_marker(
        self,
        camera_id: str,
        provider: InferenceFrameSource,
    ) -> int:
        marker = self._provider_marker(provider)
        with self._lock:
            if self._providers.get(camera_id) is not provider:
                return -1
            previous = self._provider_markers.get(camera_id)
            if previous is None:
                self._provider_markers[camera_id] = marker
                self._source_sessions.setdefault(camera_id, 1)
            elif previous != marker:
                self._provider_markers[camera_id] = marker
                self._invalidate_source_locked(
                    camera_id,
                    reason="camera source reconnect/session changed",
                )
            return self._source_sessions.get(camera_id, 1)

    def set_sources(self, providers: Mapping[str, InferenceFrameSource]) -> None:
        """Replace the provider view without opening/stopping any stream."""

        normalized = {
            str(camera_id): provider
            for camera_id, provider in providers.items()
            if provider is not None
        }
        markers = {key: self._provider_marker(value) for key, value in normalized.items()}
        with self._lock:
            previous = dict(self._providers)
            changed = set(previous) != set(normalized)
            for camera_id, provider in normalized.items():
                old_provider = previous.get(camera_id)
                old_marker = self._provider_markers.get(camera_id)
                new_marker = markers[camera_id]
                if old_provider is None:
                    self._source_sessions[camera_id] = max(
                        1, self._source_sessions.get(camera_id, 0)
                    )
                elif old_provider is not provider or old_marker != new_marker:
                    self._invalidate_source_locked(
                        camera_id,
                        reason="camera provider/session replaced",
                    )
                    changed = True
                self._provider_markers[camera_id] = new_marker
            self._providers = normalized
            removed = set(previous).difference(normalized)
            for camera_id in removed:
                self._pipelines.pop(camera_id, None)
                self._provider_markers.pop(camera_id, None)
                self._source_sessions.pop(camera_id, None)
                self._snapshots.pop(camera_id, None)
            if changed and not removed:
                # Per-camera invalidations already increment the global source
                # generation.  Key-set additions still need one global bump.
                added = set(normalized).difference(previous)
                if added:
                    self._source_generation += 1
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
                old = self._providers.get(normalized)
                marker = self._provider_marker(provider)
                if old is None:
                    self._source_sessions[normalized] = max(
                        1, self._source_sessions.get(normalized, 0)
                    )
                    self._source_generation += 1
                elif old is not provider or self._provider_markers.get(normalized) != marker:
                    self._invalidate_source_locked(
                        normalized,
                        reason="active camera provider/session replaced",
                    )
                self._providers[normalized] = provider
                self._provider_markers[normalized] = marker
        self._wake.set()

    def _state(self):
        with self._lock:
            return (
                self._settings,
                dict(self._providers),
                self._settings_generation,
                self._source_generation,
                dict(self._source_sessions),
            )

    def _resolve_model_path(self, model: str | None) -> Path | None:
        if not model:
            return None
        path = Path(model).expanduser()
        return path if path.is_absolute() else self._repo_root / path

    @staticmethod
    def _model_signature(settings: PersonDetectionSettings) -> tuple[object, ...]:
        """Fields that actually require rebuilding the detector instance."""

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
            # Threshold is part of backend config, so lowering it cannot be
            # implemented safely by post-filtering an already-stricter model.
            settings.confidence_threshold,
            settings.show_masks,
        )

    @classmethod
    def _recoverable_load_error(cls, exc: BaseException) -> bool:
        if isinstance(exc, (MemoryError, TimeoutError)):
            return True
        text = f"{type(exc).__name__}: {exc}".casefold()
        return any(
            token in text
            for token in (
                "out of memory",
                "cannot allocate",
                "insufficient memory",
                "resource temporarily",
                "device busy",
            )
        )

    @staticmethod
    def _next_deadline(previous_due: float, completed_at: float, fps: float) -> float:
        """Advance a target-frequency deadline without replaying missed slots."""

        interval = 1.0 / fps
        if previous_due <= 0:
            return completed_at + interval
        deadline = previous_due + interval
        if deadline <= completed_at:
            missed = math.floor((completed_at - deadline) / interval) + 1
            deadline += missed * interval
        return deadline

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
        batch_duration_ms: float | None = None,
        amortized_cost_ms: float | None = None,
        scheduler_wait_ms: float | None = None,
        frame_age_ms: float | None = None,
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
            batch_duration_ms=batch_duration_ms,
            amortized_cost_ms=amortized_cost_ms,
            scheduler_wait_ms=scheduler_wait_ms,
            frame_age_ms=frame_age_ms,
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
        requested_signature: tuple[object, ...] | None = None
        loaded_signature: tuple[object, ...] | None = None
        permanent_failure_signature: tuple[object, ...] | None = None
        load_failures = 0
        next_load_attempt = 0.0
        observed_inference_fps: float | None = None
        last_sequences: dict[str, int] = {}
        last_received_monotonic: dict[str, float] = {}
        next_at: dict[str, float] = {}
        failures: dict[str, int] = {}

        try:
            while not self._stop_event.is_set():
                settings, providers, _generation, _source_generation, _sessions = self._state()
                signature = self._model_signature(settings)

                # Scheduling/view changes do not rebuild the detector or reset
                # tracking.  A changed target FPS only re-anchors deadlines.
                if observed_inference_fps != settings.inference_fps:
                    observed_inference_fps = settings.inference_fps
                    next_at.clear()

                if signature != requested_signature:
                    old_requested = requested_signature
                    requested_signature = signature
                    permanent_failure_signature = None
                    load_failures = 0
                    next_load_attempt = 0.0
                    if detector is not None and loaded_signature != signature:
                        self._close_detector(detector)
                        detector = None
                        loaded_signature = None
                    last_sequences.clear()
                    last_received_monotonic.clear()
                    next_at.clear()
                    failures.clear()
                    if old_requested is not None:
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

                if detector is None:
                    if permanent_failure_signature == signature:
                        self._wake.wait(0.25)
                        self._wake.clear()
                        continue
                    now = time.monotonic()
                    if now < next_load_attempt:
                        self._wake.wait(min(0.25, next_load_attempt - now))
                        self._wake.clear()
                        continue
                    self._publish_state_for_all(
                        settings,
                        providers,
                        status=PersonDetectionStatus.LOADING,
                        message=(
                            "Caricamento modello persone condiviso…"
                            if load_failures == 0
                            else "Attesa risorse per nuovo tentativo di caricamento…"
                        ),
                    )
                    try:
                        candidate = self._build_detector(settings)
                    except FileNotFoundError as exc:
                        permanent_failure_signature = signature
                        self._publish_state_for_all(
                            settings,
                            providers,
                            status=PersonDetectionStatus.MODEL_MISSING,
                            message=str(exc),
                        )
                        continue
                    except Exception as exc:
                        if self._recoverable_load_error(exc):
                            load_failures += 1
                            delay = min(
                                self._LOAD_RETRY_MAX_S,
                                self._LOAD_RETRY_MIN_S * (2 ** (load_failures - 1)),
                            )
                            next_load_attempt = time.monotonic() + delay
                            self._logger.warning(
                                "fleet person model load temporarily unavailable; retry in %.2fs: %s",
                                delay,
                                exc,
                            )
                            self._publish_state_for_all(
                                settings,
                                providers,
                                status=PersonDetectionStatus.LOADING,
                                message=f"Risorse temporaneamente insufficienti; nuovo tentativo tra {delay:.1f}s",
                            )
                            continue
                        permanent_failure_signature = signature
                        self._logger.error("fleet person model load failed: %s", exc)
                        self._publish_state_for_all(
                            settings,
                            providers,
                            status=PersonDetectionStatus.ERROR,
                            message=str(exc) or type(exc).__name__,
                        )
                        continue
                    if candidate is None:
                        permanent_failure_signature = signature
                        continue
                    detector = candidate
                    loaded_signature = signature
                    load_failures = 0
                    next_load_attempt = 0.0
                    self._publish_state_for_all(
                        settings,
                        providers,
                        status=PersonDetectionStatus.READY,
                        message=f"Backend {detector.backend} condiviso pronto",
                        detector=detector,
                    )

                if not providers:
                    self._wake.wait(0.1)
                    self._wake.clear()
                    continue

                now = time.monotonic()
                due: list[tuple[str, InferenceFrameSource, object, int, float]] = []
                for camera_id, provider in providers.items():
                    due_at = next_at.get(camera_id, 0.0)
                    if now < due_at:
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
                    session = self._observe_provider_marker(camera_id, provider)
                    if session < 0:
                        continue
                    current_last = last_sequences.get(camera_id)
                    current_received = last_received_monotonic.get(camera_id)
                    if packet is None:
                        continue
                    if current_last is not None and packet.sequence < current_last:
                        self.invalidate_source(
                            camera_id,
                            reason="camera frame sequence restarted",
                        )
                        session = self._state()[4].get(camera_id, session + 1)
                        last_sequences.pop(camera_id, None)
                        last_received_monotonic.pop(camera_id, None)
                        next_at.pop(camera_id, None)
                        due_at = 0.0
                    elif (
                        current_received is not None
                        and packet.sequence != current_last
                        and packet.received_monotonic <= current_received
                    ):
                        self.invalidate_source(
                            camera_id,
                            reason="camera frame clock discontinuity",
                        )
                        session = self._state()[4].get(camera_id, session + 1)
                        last_sequences.pop(camera_id, None)
                        last_received_monotonic.pop(camera_id, None)
                        next_at.pop(camera_id, None)
                        due_at = 0.0
                    if (
                        packet.sequence == last_sequences.get(camera_id)
                        and packet.received_monotonic
                        == last_received_monotonic.get(camera_id)
                    ):
                        continue
                    due.append((camera_id, provider, packet, session, due_at))

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
                    started_monotonic = time.monotonic()
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
                        amortized_cost_ms = elapsed_ms / max(1, len(chunk))
                    except Exception as exc:
                        for camera_id, provider, packet, session, _due_at in chunk:
                            current = self._state()
                            if (
                                current[1].get(camera_id) is not provider
                                or current[4].get(camera_id) != session
                            ):
                                continue
                            failures[camera_id] = failures.get(camera_id, 0) + 1
                            last_sequences[camera_id] = packet.sequence
                            last_received_monotonic[camera_id] = packet.received_monotonic
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

                    current_settings, current_providers, _current_generation, _current_source_generation, current_sessions = self._state()
                    if self._model_signature(current_settings) != signature:
                        # A runtime-model change invalidates the whole batch.
                        continue

                    completed_at = time.monotonic()
                    for (camera_id, provider, packet, session, due_at), raw in zip(
                        chunk, results, strict=True
                    ):
                        if (
                            current_providers.get(camera_id) is not provider
                            or current_sessions.get(camera_id) != session
                        ):
                            # A late result from an old camera session must not
                            # update tracking, recognition confirmation or UI.
                            continue
                        detections = tuple(
                            replace(item, timestamp=packet.received_at_utc)
                            for item in raw
                            if item.confidence >= current_settings.confidence_threshold
                        )
                        metrics = self._metrics_for(camera_id)
                        # Response latency is the batch wall-clock duration.  Do
                        # not divide it by camera count; amortized cost is a
                        # separate throughput metric below.
                        metrics.record_person_detection(elapsed_ms)
                        pipeline = self._tracking_for(camera_id)
                        tracking_update = pipeline.update(
                            tuple(
                                item
                                for item in detections
                                if item.label.casefold() == "person"
                            )
                        )
                        last_sequences[camera_id] = packet.sequence
                        last_received_monotonic[camera_id] = packet.received_monotonic
                        next_at[camera_id] = self._next_deadline(
                            due_at,
                            completed_at,
                            current_settings.inference_fps,
                        )
                        scheduler_wait_ms = (
                            max(0.0, started_monotonic - due_at) * 1000.0
                            if due_at > 0
                            else 0.0
                        )
                        frame_age_ms = max(
                            0.0,
                            (completed_at - packet.received_monotonic) * 1000.0,
                        )
                        self._publish(
                            self._snapshot_for(
                                current_settings,
                                camera_id,
                                status=PersonDetectionStatus.RUNNING,
                                message=f"Inferenza persone ({detector.backend}) attiva",
                                detector=detector,
                                detections=detections,
                                packet=packet,
                                latency_ms=elapsed_ms,
                                batch_duration_ms=elapsed_ms,
                                amortized_cost_ms=amortized_cost_ms,
                                scheduler_wait_ms=scheduler_wait_ms,
                                frame_age_ms=frame_age_ms,
                                detector_failures=failures.get(camera_id, 0),
                                tracking_update=tracking_update,
                                tracking_pipeline=pipeline,
                            )
                        )
        finally:
            self._close_detector(detector)


__all__ = ["FleetPersonDetectionController"]

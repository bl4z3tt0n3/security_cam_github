"""Independent camera sessions coordinated by one shared model instance."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Protocol

from app.config import TrackingConfig
from app.inference import (
    PersonDetector,
    InferenceGate,
    SynchronizedPersonDetector,
)
from app.metrics import CameraMetrics, CameraMetricsSnapshot
from app.tracking import (
    CameraState,
    CameraTrackingPipeline,
    CameraTrackingUpdate,
    EventPublisherLike,
    IoUGreedyTracker,
    TrackRecognitionConfirmerLike,
)
from app.inference.base import PersonDetection
from app.logging_setup import redact_log_text
from app.face.orchestrator import FaceRecognitionOrchestrator
from app.video.base import VideoSource
from app.video.sampler import FrameSampler, FrameSamplerSnapshot
from app.video.motion import MotionDetector
from app.video.worker import CameraWorker, CameraWorkerSnapshot, WorkerState


class FaceAnalysisRequestHook(Protocol):
    """Select an active track for an explicit face-analysis request."""

    def select_track(
        self,
        camera_id: str,
        update: CameraTrackingUpdate,
    ) -> int | None:
        """Return a track id, or ``None`` when no request should be made."""


@dataclass(frozen=True)
class CameraRuntimeSnapshot:
    """Point-in-time state for one independent camera session."""

    camera_id: str
    worker: CameraWorkerSnapshot
    sampler: FrameSamplerSnapshot
    metrics: CameraMetricsSnapshot
    state: CameraState
    processed_samples: int
    detector_failures: int
    last_error: str | None
    thread_alive: bool


class CameraRuntime:
    """Own the acquisition, sampling and tracking lifecycle of one camera."""

    def __init__(
        self,
        camera_id: str,
        source: VideoSource,
        *,
        target_fps: float,
        detector: PersonDetector | None = None,
        motion_detector: MotionDetector | None = None,
        read_timeout_s: float = 3.0,
        reconnect_delay_s: float = 2.0,
        max_reconnect_attempts: int = 0,
        max_buffer_frames: int = 1,
        stop_timeout_s: float | None = None,
        tracking_config: TrackingConfig | None = None,
        metrics: CameraMetrics | None = None,
        event_publisher: EventPublisherLike | None = None,
        recognition_confirmer: TrackRecognitionConfirmerLike | None = None,
        face_analysis_hook: FaceAnalysisRequestHook | None = None,
        face_orchestrator: FaceRecognitionOrchestrator | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        normalized_id = camera_id.strip()
        if not normalized_id:
            raise ValueError("camera_id cannot be empty")
        if source is None:
            raise ValueError("source is required")

        self._camera_id = normalized_id
        self._logger = logger or logging.getLogger(__name__)
        self._read_timeout_s = read_timeout_s
        self._stop_timeout_s = stop_timeout_s or max(1.0, read_timeout_s + 1.0)
        self._metrics = metrics or CameraMetrics(normalized_id)
        tracking = tracking_config or TrackingConfig()
        self._pipeline = CameraTrackingPipeline(
            normalized_id,
            tracker=IoUGreedyTracker(
                iou_threshold=tracking.iou_threshold,
                max_center_distance_px=tracking.max_center_distance_px,
                max_missed_samples=tracking.max_missed_samples,
            ),
            recognition_confirmer=recognition_confirmer,
            event_publisher=event_publisher,
            metrics=self._metrics,
        )
        self._worker = CameraWorker(
            normalized_id,
            source,
            read_timeout_s=read_timeout_s,
            reconnect_delay_s=reconnect_delay_s,
            max_reconnect_attempts=max_reconnect_attempts,
            max_buffer_frames=max_buffer_frames,
            stop_timeout_s=self._stop_timeout_s,
            logger=self._logger,
        )
        self._sampler = FrameSampler(
            self._worker,
            target_fps=target_fps,
            input_wait_timeout_s=min(0.1, read_timeout_s),
            stop_timeout_s=self._stop_timeout_s,
            thread_name=f"frame-sampler-{normalized_id}",
            logger=self._logger,
        )

        self._detector: PersonDetector | None = detector
        self._motion_detector = motion_detector
        self._face_analysis_hook: FaceAnalysisRequestHook | None = face_analysis_hook
        if face_orchestrator is not None and face_orchestrator.camera_id != normalized_id:
            raise ValueError("face orchestrator camera_id must match camera runtime")
        self._face_orchestrator = face_orchestrator
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._processed_samples = 0
        self._detector_failures = 0
        self._last_error: str | None = None
        self._latest_frame: Any = None
        self._latest_detections: tuple[PersonDetection, ...] = ()
        self._last_reconnect_count = 0

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def worker(self) -> CameraWorker:
        return self._worker

    @property
    def sampler(self) -> FrameSampler:
        return self._sampler

    @property
    def pipeline(self) -> CameraTrackingPipeline:
        return self._pipeline

    @property
    def metrics(self) -> CameraMetrics:
        return self._metrics

    @property
    def detector(self) -> PersonDetector | None:
        return self._detector

    @property
    def motion_detector(self) -> MotionDetector | None:
        return self._motion_detector

    @property
    def camera_state(self) -> CameraState:
        return self._pipeline.state

    @property
    def face_orchestrator(self) -> FaceRecognitionOrchestrator | None:
        return self._face_orchestrator

    def latest_result(self) -> tuple[Any, tuple[PersonDetection, ...]]:
        """Return the latest frame/detections for an optional UI preview."""

        with self._condition:
            frame = self._latest_frame
            detections = self._latest_detections
        if frame is not None and callable(getattr(frame, "copy", None)):
            frame = frame.copy()
        return frame, detections

    @property
    def is_alive(self) -> bool:
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    def _set_detector(self, detector: PersonDetector) -> None:
        if detector is None:
            raise ValueError("detector is required")
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("cannot change detector while camera runtime is active")
            self._detector = detector

    def _set_face_analysis_hook(self, hook: FaceAnalysisRequestHook | None) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("cannot change face-analysis hook while runtime is active")
            self._face_analysis_hook = hook

    def _set_face_orchestrator(self, orchestrator: FaceRecognitionOrchestrator | None) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("cannot change face orchestrator while runtime is active")
            if orchestrator is not None and orchestrator.camera_id != self._camera_id:
                raise ValueError("face orchestrator camera_id must match camera runtime")
            self._face_orchestrator = orchestrator

    def _record_error(self, error: BaseException | str, *, detector: bool = False) -> None:
        message = redact_log_text(str(error) or type(error).__name__)
        with self._condition:
            self._last_error = message
            if detector:
                self._detector_failures += 1
            self._condition.notify_all()
        self._logger.error("camera=%s runtime error: %s", self._camera_id, message)

    def start(self) -> None:
        """Start this camera's worker, sampler and processing thread."""

        with self._condition:
            if self._detector is None:
                raise RuntimeError("a person detector must be configured before start")
            if self._thread is not None and self._thread.is_alive():
                return
            self._pipeline.reset(reason="camera runtime restarted")
            if self._face_orchestrator is not None:
                self._face_orchestrator.reset()
            self._stop_event.clear()
            self._last_error = None
            self._processed_samples = 0
            self._detector_failures = 0
            self._last_reconnect_count = 0
            if self._motion_detector is not None:
                self._motion_detector.reset()

        self._worker.start()
        try:
            self._sampler.start()
        except Exception:
            self._worker.stop(timeout_s=self._stop_timeout_s)
            raise

        with self._condition:
            self._thread = threading.Thread(
                target=self._run,
                name=f"camera-runtime-{self._camera_id}",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def stop(self, timeout_s: float | None = None) -> None:
        """Stop this camera without affecting any other runtime."""

        self._stop_event.set()
        timeout = timeout_s if timeout_s is not None else self._stop_timeout_s
        self._sampler.stop(timeout_s=timeout)
        self._worker.stop(timeout_s=timeout)
        with self._condition:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def snapshot(self) -> CameraRuntimeSnapshot:
        """Return one consistent-enough, non-blocking camera snapshot."""

        worker_snapshot = self._worker.snapshot()
        sampler_snapshot = self._sampler.snapshot()
        self._metrics.sync_worker_snapshot(
            worker_snapshot,
            sampler_snapshot=sampler_snapshot,
        )
        with self._condition:
            processed_samples = self._processed_samples
            detector_failures = self._detector_failures
            runtime_error = self._last_error
            thread = self._thread
        return CameraRuntimeSnapshot(
            camera_id=self._camera_id,
            worker=worker_snapshot,
            sampler=sampler_snapshot,
            metrics=self._metrics.snapshot(),
            state=self._pipeline.state,
            processed_samples=processed_samples,
            detector_failures=detector_failures,
            last_error=runtime_error or worker_snapshot.last_error,
            thread_alive=thread is not None and thread.is_alive(),
        )

    def _select_face_track(self, update: CameraTrackingUpdate) -> int | None:
        hook = self._face_analysis_hook
        if hook is None:
            return None
        selector = getattr(hook, "select_track", None)
        if callable(selector):
            return selector(self._camera_id, update)
        if callable(hook):
            return hook(self._camera_id, update)  # type: ignore[misc]
        raise TypeError("face analysis hook must provide callable select_track()")

    def _run(self) -> None:
        detector = self._detector
        if detector is None:  # pragma: no cover - start() guards this invariant
            self._record_error("person detector is not configured")
            return

        try:
            while not self._stop_event.is_set():
                packet = self._sampler.get_latest(
                    timeout_s=min(0.1, self._read_timeout_s)
                )
                if packet is None:
                    if self._worker.state is WorkerState.FAILED:
                        break
                    continue

                worker_snapshot = self._worker.snapshot()
                if worker_snapshot.reconnect_count != self._last_reconnect_count:
                    if self._motion_detector is not None:
                        self._motion_detector.reset()
                    self._pipeline.reset(reason="camera source reconnected")
                    if self._face_orchestrator is not None:
                        self._face_orchestrator.reset()
                    self._last_reconnect_count = worker_snapshot.reconnect_count

                with self._condition:
                    self._latest_frame = packet.frame

                if self._motion_detector is not None:
                    motion_started = time.perf_counter()
                    try:
                        motion = self._motion_detector.detect(packet.frame)
                    except Exception as exc:
                        # A cheap optimization must never suppress the safety-critical
                        # baseline detector when its own implementation fails.
                        self._record_error(exc)
                        self._metrics.record_motion_detection(
                            (time.perf_counter() - motion_started) * 1000.0,
                            motion_detected=True,
                            skipped_person_detection=False,
                        )
                    else:
                        self._metrics.record_motion_detection(
                            (time.perf_counter() - motion_started) * 1000.0,
                            motion_detected=motion.motion_detected,
                            skipped_person_detection=not motion.motion_detected,
                            changed_fraction=motion.changed_fraction,
                        )
                        if not motion.motion_detected:
                            with self._condition:
                                self._processed_samples += 1
                                self._condition.notify_all()
                            continue

                detection_started = time.perf_counter()
                try:
                    detections = detector.detect(packet.frame, packet.received_at_utc)
                except Exception as exc:
                    self._metrics.record_person_detection(
                        (time.perf_counter() - detection_started) * 1000.0,
                        person_count=0,
                    )
                    self._record_error(exc, detector=True)
                    continue
                self._metrics.record_person_detection(
                    (time.perf_counter() - detection_started) * 1000.0,
                    person_count=len(detections),
                )
                with self._condition:
                    self._latest_frame = packet.frame
                    self._latest_detections = tuple(detections)

                pipeline_started = time.perf_counter()
                try:
                    update = self._pipeline.update(detections)
                    if self._face_orchestrator is not None:
                        self._face_orchestrator.process(
                            packet.frame,
                            update,
                            self._pipeline,
                            timestamp=packet.received_at_utc,
                        )
                    elif self._pipeline.state in {
                        CameraState.TRACKING,
                        CameraState.KNOWN,
                        CameraState.UNKNOWN,
                    }:
                        track_id = self._select_face_track(update)
                        if track_id is not None:
                            self._pipeline.begin_face_analysis(track_id)
                except Exception as exc:
                    self._record_error(exc)
                    continue
                finally:
                    self._metrics.record_pipeline(
                        (time.perf_counter() - pipeline_started) * 1000.0
                    )

                with self._condition:
                    self._processed_samples += 1
                    self._condition.notify_all()
        except Exception as exc:  # pragma: no cover - final isolation guard
            if not self._stop_event.is_set():
                self._record_error(exc)


class MultiCameraRuntime:
    """Run independent camera sessions with one serialized detector."""

    def __init__(
        self,
        cameras: Iterable[CameraRuntime],
        *,
        detector: PersonDetector,
        inference_gate: InferenceGate | None = None,
        face_analysis_hook: FaceAnalysisRequestHook | None = None,
        face_orchestrators: Mapping[str, FaceRecognitionOrchestrator] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        runtimes = tuple(cameras)
        if not runtimes:
            raise ValueError("at least one camera runtime is required")
        ids = [runtime.camera_id for runtime in runtimes]
        if len(ids) != len(set(ids)):
            raise ValueError("camera runtime ids must be unique")
        if detector is None:
            raise ValueError("detector is required")

        self._logger = logger or logging.getLogger(__name__)
        self._cameras = {runtime.camera_id: runtime for runtime in runtimes}
        if isinstance(detector, SynchronizedPersonDetector):
            self._detector = detector
            self._inference_gate = detector.gate
        else:
            self._inference_gate = inference_gate or InferenceGate()
            self._detector = SynchronizedPersonDetector(detector, self._inference_gate)
        self._face_analysis_hook = face_analysis_hook
        self._face_orchestrators = dict(face_orchestrators or {})
        for runtime in runtimes:
            runtime._set_detector(self._detector)
            runtime._set_face_analysis_hook(face_analysis_hook)
            orchestrator = self._face_orchestrators.get(runtime.camera_id)
            if orchestrator is not None and runtime.face_orchestrator is not orchestrator:
                runtime._set_face_orchestrator(orchestrator)

    @property
    def cameras(self) -> Mapping[str, CameraRuntime]:
        return self._cameras

    @property
    def detector(self) -> PersonDetector:
        return self._detector

    @property
    def inference_gate(self) -> InferenceGate:
        return self._inference_gate

    def start(self) -> None:
        """Start every camera, preserving progress when one start fails."""

        for runtime in self._cameras.values():
            try:
                runtime.start()
            except Exception as exc:
                runtime._record_error(exc)
                self._logger.error(
                    "camera=%s could not start; other cameras remain active: %s",
                    runtime.camera_id,
                    exc,
                )

    def stop(self, timeout_s: float | None = None) -> None:
        """Stop all cameras independently, attempting every shutdown."""

        for runtime in self._cameras.values():
            try:
                runtime.stop(timeout_s=timeout_s)
            except Exception as exc:  # pragma: no cover - defensive shutdown isolation
                runtime._record_error(exc)
                self._logger.error("camera=%s stop failed: %s", runtime.camera_id, exc)

    def snapshot(self) -> dict[str, CameraRuntimeSnapshot]:
        """Return snapshots keyed by camera id."""

        return {
            camera_id: runtime.snapshot()
            for camera_id, runtime in self._cameras.items()
        }

    def has_failed_camera(self) -> bool:
        """Return whether a source exhausted reconnect attempts."""

        return any(
            snapshot.worker.state is WorkerState.FAILED
            for snapshot in self.snapshot().values()
        )

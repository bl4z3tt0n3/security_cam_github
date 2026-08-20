from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from app.config import load_config
from app.face import (
    FaceAnalysisService,
    FaceDetection,
    FaceLandmark5,
    FaceRecognitionOrchestrator,
    FakeFaceDetector,
)
from app.inference import FakePersonDetector, InferenceGate, PersonDetection
from app.metrics import CameraMetrics
from app.tracking import CameraTrackingPipeline
from app_windows.inference import FaceRecognitionController, PersonDetectionController
import app_windows.inference.face_recognition_controller as face_controller_module
import app_windows.inference.person_detection_controller as person_controller_module
from app_windows.models.camera_view_state import CameraSlot
from app_windows.models.face_recognition_state import (
    FaceRecognitionSettings,
    FaceRecognitionStatus,
)
from app_windows.models.person_detection_state import (
    PersonDetectionSettings,
    PersonDetectionStatus,
)
from app_windows.video.fake_provider import FakeFrameProvider


ROOT = Path(__file__).resolve().parents[2]
CAMERA_ID = "cam_1"
PERSON_BBOX = (100.0, 40.0, 300.0, 300.0)
FACE_DETECTION = FaceDetection(
    bbox=(40.0, 40.0, 140.0, 180.0),
    confidence=0.95,
    landmarks=FaceLandmark5(
        (
            (60.0, 70.0),
            (120.0, 70.0),
            (90.0, 105.0),
            (70.0, 150.0),
            (110.0, 150.0),
        )
    ),
    detector_id="fake-face",
    backend="fake",
    device="cpu",
)


def _face_settings(**overrides: object) -> FaceRecognitionSettings:
    values: dict[str, object] = {
        "face_detection_enabled": True,
        "detector_id": "fake-face",
        "detector_backend": "fake",
        "detector_model": "fake-face",
        "detector_device": "cpu",
        "detector_confidence_threshold": 0.2143884892086336,
        "detector_inference_fps": 20.0,
        "landmarks_enabled": False,
        "recognition_enabled": False,
    }
    values.update(overrides)
    return FaceRecognitionSettings(**values)


def _person_settings() -> PersonDetectionSettings:
    return PersonDetectionSettings(
        enabled=True,
        backend="fake",
        model=None,
        confidence_threshold=0.5,
        inference_fps=20.0,
        device="cpu",
        precision="fp32",
    )


def _camera_slot() -> CameraSlot:
    return CameraSlot(
        slot_index=1,
        camera_id=CAMERA_ID,
        name="Camera 1",
        enabled=True,
        configured=True,
        stream_url=f"fake://{CAMERA_ID}/live",
    )


def _person_detection() -> PersonDetection:
    return PersonDetection(
        bbox=PERSON_BBOX,
        confidence=0.95,
        timestamp=datetime.now(timezone.utc),
    )


def _patch_fake_face_factory(monkeypatch):
    captured_metrics: list[CameraMetrics | None] = []
    detectors: list[FakeFaceDetector] = []

    def create_fake_orchestrator(
        _config,
        camera_id: str,
        *,
        metrics: CameraMetrics | None = None,
        inference_gate: InferenceGate | None = None,
        **_kwargs,
    ) -> FaceRecognitionOrchestrator:
        captured_metrics.append(metrics)
        detector = FakeFaceDetector([FACE_DETECTION])
        detectors.append(detector)
        service = FaceAnalysisService(
            camera_id,
            detector,
            metrics=metrics,
            inference_gate=inference_gate,
        )
        return FaceRecognitionOrchestrator(
            service,
            face_fps=20.0,
            enabled=True,
        )

    monkeypatch.setattr(
        face_controller_module,
        "create_face_orchestrator",
        create_fake_orchestrator,
    )
    return captured_metrics, detectors


def test_face_controller_reuses_person_metrics_and_survives_runtime_lifecycle(
    monkeypatch,
    qapp,
    qtbot,
    tmp_path: Path,
) -> None:
    del qapp
    person_detector = FakePersonDetector([_person_detection()])
    monkeypatch.setattr(
        person_controller_module,
        "create_person_detector",
        lambda *_args, **_kwargs: person_detector,
    )
    captured_metrics, face_detectors = _patch_fake_face_factory(monkeypatch)

    provider = FakeFrameProvider(_camera_slot(), camera_index=0, fps=30.0)
    gate = InferenceGate()
    person_controller = PersonDetectionController(
        repo_root=tmp_path,
        settings=_person_settings(),
        inference_gate=gate,
    )
    face_controller = FaceRecognitionController(
        repo_root=tmp_path,
        config=load_config(ROOT / "config" / "config.example.yaml"),
        settings=_face_settings(),
        inference_gate=gate,
    )

    provider.start()
    person_controller.set_active_camera(CAMERA_ID, provider)
    person_controller.start()
    face_controller.set_active_camera(CAMERA_ID, provider)
    face_controller.start()
    try:
        qtbot.waitUntil(
            lambda: (
                person_controller.snapshot.status is PersonDetectionStatus.RUNNING
                and person_controller.snapshot.tracking_pipeline is not None
            ),
            timeout=5000,
        )
        person_snapshot = person_controller.snapshot
        tracking_pipeline = person_snapshot.tracking_pipeline
        assert tracking_pipeline is not None
        assert tracking_pipeline.metrics is not None

        face_controller.set_person_snapshot(person_snapshot)
        qtbot.waitUntil(
            lambda: (
                face_controller.is_started
                and face_controller.snapshot.status is FaceRecognitionStatus.RUNNING
                and face_controller.snapshot.detection_status
                is FaceRecognitionStatus.RUNNING
                and face_controller.snapshot.face_count == 1
            ),
            timeout=5000,
        )

        snapshot = face_controller.snapshot
        assert face_controller._metrics[CAMERA_ID] is tracking_pipeline.metrics
        assert captured_metrics[-1] is tracking_pipeline.metrics
        assert len(snapshot.overlays) == 1
        assert len(snapshot.overlays[0].landmarks) == 5
        assert face_detectors[-1].calls > 0

        # A detector/backend/device edit must rebuild the runtime while
        # retaining the one metrics object owned by person tracking.
        face_controller.update_settings(
            replace(
                _face_settings(),
                detector_backend="onnxruntime",
                detector_device="gpu",
            )
        )
        qtbot.waitUntil(
            lambda: (
                len(face_detectors) >= 2
                and face_controller.is_started
                and face_controller.snapshot.status is FaceRecognitionStatus.RUNNING
            ),
            timeout=5000,
        )
        assert captured_metrics[-1] is tracking_pipeline.metrics
        assert face_controller._metrics[CAMERA_ID] is tracking_pipeline.metrics

        # Stop/remove/restart must drop runtime resources and allow the same
        # active camera to build a fresh runtime without a stale reference.
        face_controller.stop()
        assert not face_controller.is_started
        face_controller.set_active_camera(None, None)
        face_controller.start()
        qtbot.waitUntil(
            lambda: (
                face_controller.snapshot.camera_id is None
                and face_controller.snapshot.status is FaceRecognitionStatus.READY
            ),
            timeout=3000,
        )
        face_controller.set_active_camera(CAMERA_ID, provider)
        face_controller.set_person_snapshot(person_controller.snapshot)
        qtbot.waitUntil(
            lambda: (
                face_controller.is_started
                and face_controller.snapshot.status is FaceRecognitionStatus.RUNNING
                and face_controller.snapshot.face_count == 1
            ),
            timeout=5000,
        )
        assert len(face_detectors) >= 3
        assert captured_metrics[-1] is tracking_pipeline.metrics
        assert face_controller._metrics[CAMERA_ID] is tracking_pipeline.metrics
    finally:
        face_controller.stop()
        person_controller.stop()
        provider.stop(timeout_s=1.0)


def test_face_runtime_registers_created_and_recreated_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_metrics, _detectors = _patch_fake_face_factory(monkeypatch)
    controller = FaceRecognitionController(
        repo_root=tmp_path,
        config=load_config(ROOT / "config" / "config.example.yaml"),
        settings=_face_settings(),
    )
    settings = _face_settings()

    first_metrics = CameraMetrics(CAMERA_ID)
    first_pipeline = CameraTrackingPipeline(CAMERA_ID, metrics=first_metrics)
    _, first_orchestrator = controller._build_runtime(
        settings,
        CAMERA_ID,
        first_pipeline,
    )
    assert controller._metrics[CAMERA_ID] is first_metrics
    assert captured_metrics[-1] is first_metrics
    assert first_orchestrator.service.metrics is first_metrics

    second_metrics = CameraMetrics(CAMERA_ID)
    second_pipeline = CameraTrackingPipeline(CAMERA_ID, metrics=second_metrics)
    _, second_orchestrator = controller._build_runtime(
        settings,
        CAMERA_ID,
        second_pipeline,
    )
    assert controller._metrics[CAMERA_ID] is second_metrics
    assert captured_metrics[-1] is second_metrics
    assert second_orchestrator.service.metrics is second_metrics
    assert controller._metrics[CAMERA_ID] is not first_metrics

    created_pipeline = CameraTrackingPipeline(CAMERA_ID)
    _, created_orchestrator = controller._build_runtime(
        settings,
        CAMERA_ID,
        created_pipeline,
    )
    created_metrics = controller._metrics[CAMERA_ID]
    assert isinstance(created_metrics, CameraMetrics)
    assert captured_metrics[-1] is created_metrics
    assert created_orchestrator.service.metrics is created_metrics

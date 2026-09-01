from __future__ import annotations

from datetime import datetime, timezone
import time

import numpy as np

from app.config import AppConfig
from app.face.base import FaceDetection, FaceDetector, FaceQualityEvaluator
from app.face.orchestrator import FaceRecognitionOrchestrator
from app.face.service import FaceAnalysisService
from app.inference import PersonDetection
from app.tracking import CameraTrackingPipeline
from app.video.base import FramePacket
from app_windows.inference.face_recognition_fleet_controller import (
    FleetFaceRecognitionController,
)
import app_windows.inference.face_recognition_fleet_controller as fleet_module
from app_windows.models.face_recognition_state import (
    FaceRecognitionSettings,
    FaceRecognitionStatus,
)
from app_windows.models.person_detection_state import PersonDetectionSnapshot


class SharedDetector(FaceDetector):
    detector_id = "fake-face"
    backend_id = "fake"

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    @property
    def device_used(self) -> str:
        return "cpu"

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        self.calls += 1
        height, width = frame.shape[:2]
        return [
            FaceDetection(
                (1.0, 1.0, max(2.0, width - 1.0), max(2.0, height - 1.0)),
                0.95,
                detector_id=self.detector_id,
                backend=self.backend_id,
                device="cpu",
            )
        ]

    def close(self) -> None:
        self.closed = True


class StaticProvider:
    def __init__(self, camera_id: str, sequence: int) -> None:
        self.camera_id = camera_id
        self.packet = FramePacket(
            frame=np.full((120, 160, 3), 128, dtype=np.uint8),
            sequence=sequence,
            received_at_utc=datetime.now(timezone.utc),
            received_monotonic=time.monotonic(),
            read_duration_ms=1.0,
        )

    def latest_frame(self):
        return self.packet


def _person_snapshot(camera_id: str) -> PersonDetectionSnapshot:
    pipeline = CameraTrackingPipeline(camera_id)
    detection = PersonDetection(
        bbox=(10.0, 10.0, 120.0, 110.0),
        confidence=0.95,
        timestamp=datetime.now(timezone.utc),
    )
    update = pipeline.update((detection,))
    return PersonDetectionSnapshot(
        camera_id=camera_id,
        person_count=1,
        detections=(detection,),
        tracking_update=update,
        tracking_pipeline=pipeline,
    )


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_face_fleet_shares_one_detector_across_person_cameras(monkeypatch, tmp_path) -> None:
    detector = SharedDetector()
    template = FaceRecognitionOrchestrator(
        FaceAnalysisService(
            "__template__",
            detector,
            evaluator=FaceQualityEvaluator(
                min_width=1,
                min_height=1,
                blur_threshold=0,
                min_brightness=0,
                max_brightness=255,
                min_confidence=0,
            ),
        ),
        face_fps=100,
        enabled=True,
    )
    builds = 0

    def build(*_args, **_kwargs):
        nonlocal builds
        builds += 1
        return template

    monkeypatch.setattr(fleet_module, "create_face_orchestrator", build)
    controller = FleetFaceRecognitionController(
        repo_root=tmp_path,
        config=AppConfig(),
        settings=FaceRecognitionSettings(
            face_detection_enabled=True,
            detector_id="fake-face",
            detector_backend="fake",
            detector_model="fake",
            detector_device="cpu",
            recognition_enabled=False,
        ),
    )
    controller.set_sources(
        {
            "cam_1": StaticProvider("cam_1", 1),
            "cam_2": StaticProvider("cam_2", 1),
        }
    )
    controller.set_person_snapshot(_person_snapshot("cam_1"))
    controller.set_person_snapshot(_person_snapshot("cam_2"))
    controller.set_active_camera("cam_1", None)
    controller.start()
    try:
        assert _wait_until(
            lambda: (
                len(controller.snapshots) == 2
                and all(
                    value.status is FaceRecognitionStatus.RUNNING
                    for value in controller.snapshots.values()
                )
            )
        )
        assert builds == 1
        assert detector.calls == 2
        assert controller.snapshots["cam_1"].face_count == 1
        assert controller.snapshots["cam_2"].face_count == 1
    finally:
        controller.stop(timeout_s=1.0)

    assert detector.closed is True

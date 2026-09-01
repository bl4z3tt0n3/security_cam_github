from __future__ import annotations

from datetime import datetime, timezone
import time

import numpy as np

from app.inference import PersonDetection, PersonDetector
from app.video.base import FramePacket
from app_windows.inference.person_detection_fleet_controller import (
    FleetPersonDetectionController,
)
from app_windows.models.person_detection_state import (
    PersonDetectionSettings,
    PersonDetectionStatus,
)
import app_windows.inference.person_detection_fleet_controller as fleet_module


class StaticProvider:
    def __init__(self, camera_id: str, value: int) -> None:
        self.camera_id = camera_id
        self.packet = FramePacket(
            frame=np.full((8, 8, 3), value, dtype=np.uint8),
            sequence=1,
            received_at_utc=datetime.now(timezone.utc),
            received_monotonic=time.monotonic(),
            read_duration_ms=1.0,
        )

    def latest_frame(self):
        return self.packet


class BatchDetector(PersonDetector):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.closed = False

    @property
    def backend(self) -> str:
        return "openvino"

    @property
    def device_used(self) -> str:
        return "GPU.0"

    @property
    def provider_used(self) -> str:
        return "OpenVINO/GPU.0"

    @property
    def device_verified(self) -> bool:
        return True

    @property
    def supports_batch_inference(self) -> bool:
        return True

    @property
    def preferred_batch_size(self) -> int:
        return 2

    def detect(self, frame: np.ndarray, timestamp=None):
        raise AssertionError("fleet should batch two due cameras")

    def detect_batch(self, frames, timestamps=None):
        self.batch_sizes.append(len(frames))
        stamps = timestamps or [datetime.now(timezone.utc)] * len(frames)
        return [
            [
                PersonDetection(
                    bbox=(0.0, 0.0, 4.0, 4.0),
                    confidence=0.9,
                    timestamp=stamp,
                )
            ]
            for stamp in stamps
        ]

    def close(self) -> None:
        self.closed = True


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_fleet_controller_batches_existing_camera_providers(monkeypatch, tmp_path) -> None:
    detector = BatchDetector()
    monkeypatch.setattr(fleet_module, "create_person_detector", lambda *_a, **_k: detector)
    controller = FleetPersonDetectionController(
        repo_root=tmp_path,
        settings=PersonDetectionSettings(
            enabled=True,
            backend="fake",
            model=None,
            inference_fps=10,
        ),
    )
    controller.set_sources(
        {
            "cam_1": StaticProvider("cam_1", 1),
            "cam_2": StaticProvider("cam_2", 2),
        }
    )
    controller.set_active_camera("cam_1", None)
    controller.start()
    try:
        assert _wait_until(
            lambda: (
                len(controller.snapshots) == 2
                and all(
                    item.status is PersonDetectionStatus.RUNNING
                    for item in controller.snapshots.values()
                )
            )
        )
        assert detector.batch_sizes == [2]
        assert controller.tracking_pipeline is not None
        assert controller.tracking_pipeline.camera_id == "cam_1"
        assert controller.snapshots["cam_2"].person_count == 1
    finally:
        controller.stop(timeout_s=1.0)

    assert detector.closed is True

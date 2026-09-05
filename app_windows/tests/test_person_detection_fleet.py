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
    def __init__(self, camera_id: str, value: int, *, sequence: int = 1) -> None:
        self.camera_id = camera_id
        self.packet = FramePacket(
            frame=np.full((8, 8, 3), value, dtype=np.uint8),
            sequence=sequence,
            received_at_utc=datetime.now(timezone.utc),
            received_monotonic=time.monotonic(),
            read_duration_ms=1.0,
        )

    def latest_frame(self):
        return self.packet


class BatchDetector(PersonDetector):
    def __init__(self, *, delay_s: float = 0.0) -> None:
        self.batch_sizes: list[int] = []
        self.closed = False
        self.delay_s = delay_s

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
        if self.delay_s:
            time.sleep(self.delay_s)
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


def _settings(**overrides) -> PersonDetectionSettings:
    values = dict(
        enabled=True,
        backend="fake",
        model=None,
        inference_fps=10,
    )
    values.update(overrides)
    return PersonDetectionSettings(**values)


def test_fleet_controller_batches_existing_camera_providers(monkeypatch, tmp_path) -> None:
    detector = BatchDetector()
    monkeypatch.setattr(fleet_module, "create_person_detector", lambda *_a, **_k: detector)
    controller = FleetPersonDetectionController(repo_root=tmp_path, settings=_settings())
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


def test_fleet_provider_replacement_invalidates_tracking_session(monkeypatch, tmp_path) -> None:
    detector = BatchDetector()
    monkeypatch.setattr(fleet_module, "create_person_detector", lambda *_a, **_k: detector)
    first = StaticProvider("cam_1", 1)
    controller = FleetPersonDetectionController(repo_root=tmp_path, settings=_settings())
    controller.set_sources({"cam_1": first})
    controller.set_active_camera("cam_1", None)
    controller.start()
    try:
        assert _wait_until(
            lambda: controller.snapshot.status is PersonDetectionStatus.RUNNING
        )
        pipeline = controller.tracking_pipeline
        assert pipeline is not None
        assert pipeline.latest_update is not None
        old_session = pipeline.session_generation

        controller.set_sources({"cam_1": StaticProvider("cam_1", 2, sequence=1)})
        assert pipeline.session_generation == old_session + 1
        assert pipeline.latest_update is None
        assert _wait_until(
            lambda: (
                controller.snapshot.status is PersonDetectionStatus.RUNNING
                and controller.snapshot.frame_sequence == 1
                and pipeline.latest_update is not None
            )
        )
    finally:
        controller.stop(timeout_s=1.0)


def test_view_and_fps_changes_do_not_reload_detector_or_reset_tracks(monkeypatch, tmp_path) -> None:
    detector = BatchDetector()
    builds: list[int] = []

    def build(*_args, **_kwargs):
        builds.append(1)
        return detector

    monkeypatch.setattr(fleet_module, "create_person_detector", build)
    controller = FleetPersonDetectionController(repo_root=tmp_path, settings=_settings())
    controller.set_sources({"cam_1": StaticProvider("cam_1", 1)})
    controller.set_active_camera("cam_1", None)
    controller.start()
    try:
        assert _wait_until(
            lambda: controller.snapshot.status is PersonDetectionStatus.RUNNING
        )
        pipeline = controller.tracking_pipeline
        assert pipeline is not None
        session = pipeline.session_generation
        controller.update_settings(_settings(show_boxes=False, inference_fps=7.0))
        time.sleep(0.15)
        assert builds == [1]
        assert controller.tracking_pipeline is pipeline
        assert pipeline.session_generation == session
        assert detector.closed is False
    finally:
        controller.stop(timeout_s=1.0)


def test_transient_model_load_failure_retries_without_settings_change(monkeypatch, tmp_path) -> None:
    detector = BatchDetector()
    attempts: list[int] = []

    def build(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise MemoryError("simulated temporary memory pressure")
        return detector

    monkeypatch.setattr(fleet_module, "create_person_detector", build)
    monkeypatch.setattr(FleetPersonDetectionController, "_LOAD_RETRY_MIN_S", 0.01)
    monkeypatch.setattr(FleetPersonDetectionController, "_LOAD_RETRY_MAX_S", 0.02)
    controller = FleetPersonDetectionController(repo_root=tmp_path, settings=_settings())
    controller.set_sources({"cam_1": StaticProvider("cam_1", 1)})
    controller.start()
    try:
        assert _wait_until(lambda: len(attempts) >= 2)
        assert _wait_until(
            lambda: controller.snapshots.get("cam_1") is not None
            and controller.snapshots["cam_1"].status is PersonDetectionStatus.RUNNING
        )
        assert len(attempts) == 2
    finally:
        controller.stop(timeout_s=1.0)


def test_batch_latency_is_not_divided_by_camera_count(monkeypatch, tmp_path) -> None:
    detector = BatchDetector(delay_s=0.08)
    monkeypatch.setattr(fleet_module, "create_person_detector", lambda *_a, **_k: detector)
    controller = FleetPersonDetectionController(repo_root=tmp_path, settings=_settings())
    controller.set_sources(
        {
            "cam_1": StaticProvider("cam_1", 1),
            "cam_2": StaticProvider("cam_2", 2),
        }
    )
    controller.start()
    try:
        assert _wait_until(
            lambda: len(controller.snapshots) == 2
            and all(
                item.status is PersonDetectionStatus.RUNNING
                for item in controller.snapshots.values()
            )
        )
        first = controller.snapshots["cam_1"]
        assert first.latency_ms is not None and first.latency_ms >= 70.0
        assert first.batch_duration_ms == first.latency_ms
        assert first.amortized_cost_ms is not None
        assert 30.0 <= first.amortized_cost_ms < first.latency_ms
        assert first.frame_age_ms is not None and first.frame_age_ms >= first.latency_ms
    finally:
        controller.stop(timeout_s=1.0)


def test_target_fps_deadline_skips_missed_slots_without_adding_inference_time() -> None:
    next_due = FleetPersonDetectionController._next_deadline(10.0, 10.2, 2.0)
    assert next_due == 10.5
    late_due = FleetPersonDetectionController._next_deadline(10.0, 11.2, 2.0)
    assert late_due == 11.5

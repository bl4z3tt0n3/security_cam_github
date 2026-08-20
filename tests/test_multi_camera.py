from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from app.camera import CameraRuntime, MultiCameraRuntime
from app.config import ConfigurationError
from app.events import EventManager
from app.face import RecognitionResult, TrackRecognitionConfirmer
from app.inference import FakePersonDetector, PersonDetection, PersonDetectionError
from app.tracking import CameraState, CameraTrackingPipeline, IoUGreedyTracker
from app.video.base import ReadResult, ReadStatus, StreamInfo, VideoSource, VideoSourceError
from app.video.fake_source import FakeVideoSource
from app.video.motion import MotionDetector
from app.video.worker import WorkerState
from scripts._common import resolve_targets


UTC_START = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def make_detection(value: int = 0) -> PersonDetection:
    return PersonDetection(
        bbox=(float(value), 0.0, float(value + 20), 20.0),
        confidence=0.9,
        timestamp=datetime.now(timezone.utc),
    )


def test_resolve_targets_returns_all_enabled_cameras_and_supports_selection(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "multi.yaml"
    config_path.write_text(
        """
cameras:
  - id: cam_1
    enabled: true
    stream_url: rtsp://cam-1.local/live
  - id: cam_2
    enabled: true
    stream_url: rtsp://cam-2.local/live
  - id: disabled
    enabled: false
    stream_url: rtsp://disabled.local/live
""".lstrip(),
        encoding="utf-8",
    )

    all_targets = resolve_targets(
        SimpleNamespace(config=config_path, camera_id=None, url=None)
    )
    selected = resolve_targets(
        SimpleNamespace(config=config_path, camera_id="cam_2", url=None)
    )

    assert [target.camera.id for target in all_targets if target.camera is not None] == [
        "cam_1",
        "cam_2",
    ]
    assert selected[0].camera is not None
    assert selected[0].camera.id == "cam_2"


def test_resolve_targets_reports_invalid_url_for_the_specific_camera(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
cameras:
  - id: cam_1
    enabled: true
    stream_url: rtsp://cam-1.local/live
  - id: cam_2
    enabled: true
    stream_url: ${CAMERA_CAM_2_URL}
""".lstrip(),
        encoding="utf-8",
    )

    try:
        resolve_targets(SimpleNamespace(config=config_path, camera_id=None, url=None))
    except ConfigurationError as exc:
        assert "camera 'cam_2'" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("placeholder camera URL should be rejected")


class SelectSecondCamera:
    def select_track(self, camera_id: str, update: object) -> int | None:
        if camera_id != "cam_2":
            return None
        new_tracks = getattr(update, "new_tracks")
        return new_tracks[0].track_id if new_tracks else None


def test_two_cameras_keep_scan_and_face_analysis_states_independent() -> None:
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    person = np.ones((32, 32, 3), dtype=np.uint8)

    def detect(frame: np.ndarray) -> list[PersonDetection]:
        return [make_detection()] if int(frame[0, 0, 0]) == 1 else []

    detector = FakePersonDetector(callback=detect)
    first = CameraRuntime(
        "cam_1",
        FakeVideoSource([blank], read_delay_s=0.002),
        target_fps=20.0,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
    )
    second = CameraRuntime(
        "cam_2",
        FakeVideoSource([person], read_delay_s=0.002),
        target_fps=20.0,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
    )
    fleet = MultiCameraRuntime(
        [first, second],
        detector=detector,
        face_analysis_hook=SelectSecondCamera(),
    )

    fleet.start()
    try:
        assert wait_until(
            lambda: (
                fleet.snapshot()["cam_1"].processed_samples > 0
                and fleet.snapshot()["cam_2"].state is CameraState.FACE_ANALYSIS
            )
        )
        snapshots = fleet.snapshot()
        assert snapshots["cam_1"].state is CameraState.PERSON_SCAN
        assert snapshots["cam_2"].state is CameraState.FACE_ANALYSIS
        assert snapshots["cam_1"].metrics.active_tracks == 0
        assert snapshots["cam_2"].metrics.active_tracks == 1
        assert snapshots["cam_1"].metrics.face_detection_calls == 0
        assert snapshots["cam_2"].metrics.face_detection_calls == 0
        assert first.sampler.thread_name == "frame-sampler-cam_1"
        assert second.sampler.thread_name == "frame-sampler-cam_2"
        assert first.detector is second.detector is fleet.detector
    finally:
        fleet.stop(timeout_s=1.0)


def test_static_motion_gate_skips_person_detection_and_preserves_track() -> None:
    frame = np.ones((32, 32, 3), dtype=np.uint8)
    detector = FakePersonDetector(callback=lambda image: [make_detection()] if image is not None else [])
    runtime = CameraRuntime(
        "motion-camera",
        FakeVideoSource([frame], read_delay_s=0.002),
        target_fps=20.0,
        detector=detector,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
        motion_detector=MotionDetector(resize_width=32, min_changed_fraction=0.1),
    )
    runtime.start()
    try:
        assert wait_until(
            lambda: runtime.snapshot().metrics.motion_skipped_samples > 0
        )
        snapshot = runtime.snapshot()
        assert detector.calls == 1
        assert snapshot.metrics.active_tracks == 1
        assert snapshot.metrics.motion_detection_calls > 1
        assert snapshot.metrics.motion_skipped_samples > 0
        assert snapshot.thread_alive is True
    finally:
        runtime.stop(timeout_s=1.0)


def test_motion_gate_failure_falls_back_to_person_detection() -> None:
    class BrokenMotion:
        def reset(self) -> None:
            return None

        def detect(self, frame: object) -> object:
            del frame
            raise RuntimeError("synthetic motion failure")

    frame = np.ones((16, 16, 3), dtype=np.uint8)
    detector = FakePersonDetector(callback=lambda image: [make_detection()] if image is not None else [])
    runtime = CameraRuntime(
        "broken-motion-camera",
        FakeVideoSource([frame], read_delay_s=0.003),
        target_fps=10.0,
        detector=detector,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
        motion_detector=BrokenMotion(),  # type: ignore[arg-type]
    )
    runtime.start()
    try:
        assert wait_until(lambda: detector.calls > 0)
        assert runtime.snapshot().last_error == "synthetic motion failure"
    finally:
        runtime.stop(timeout_s=1.0)


def test_camera_reconnect_and_buffer_state_are_independent() -> None:
    first_source = FakeVideoSource(
        [np.zeros((8, 8, 3), dtype=np.uint8)],
        read_delay_s=0.002,
    )
    second_source = FakeVideoSource(
        [np.full((8, 8, 3), 2, dtype=np.uint8)],
        read_delay_s=0.002,
        fail_after_frames=1,
    )
    first = CameraRuntime(
        "cam_1",
        first_source,
        target_fps=20.0,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
    )
    second = CameraRuntime(
        "cam_2",
        second_source,
        target_fps=20.0,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
    )
    fleet = MultiCameraRuntime(
        [first, second],
        detector=FakePersonDetector(),
    )

    fleet.start()
    try:
        assert wait_until(
            lambda: (
                fleet.snapshot()["cam_1"].worker.frames_received > 0
                and fleet.snapshot()["cam_2"].worker.successful_reconnects > 0
            )
        )
        snapshots = fleet.snapshot()
        assert snapshots["cam_1"].worker.reconnect_count == 0
        assert snapshots["cam_2"].worker.reconnect_count > 0
        first_frame, _ = first.latest_result()
        second_frame, _ = second.latest_result()
        assert first_frame is not None and int(first_frame[0, 0, 0]) == 0
        assert second_frame is not None and int(second_frame[0, 0, 0]) == 2
    finally:
        fleet.stop(timeout_s=1.0)


class AlwaysFailSource(VideoSource):
    def __init__(self) -> None:
        self.reconnect_calls = 0

    def open(self) -> StreamInfo:
        raise VideoSourceError("source is offline", code="offline")

    def read(self, timeout_s: float) -> ReadResult:
        del timeout_s
        return ReadResult.status_result(ReadStatus.DISCONNECTED, "source is offline")

    def reconnect(self) -> StreamInfo:
        self.reconnect_calls += 1
        raise VideoSourceError("source is still offline", code="offline")

    def close(self) -> None:
        return None


def test_failed_camera_does_not_stop_healthy_camera() -> None:
    healthy = CameraRuntime(
        "healthy",
        FakeVideoSource(
            [np.zeros((8, 8, 3), dtype=np.uint8)],
            read_delay_s=0.002,
        ),
        target_fps=20.0,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
        max_reconnect_attempts=1,
    )
    failed = CameraRuntime(
        "failed",
        AlwaysFailSource(),
        target_fps=20.0,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
        max_reconnect_attempts=1,
    )
    fleet = MultiCameraRuntime([healthy, failed], detector=FakePersonDetector())

    fleet.start()
    try:
        assert wait_until(
            lambda: (
                fleet.snapshot()["failed"].worker.state.value == "failed"
                and fleet.snapshot()["healthy"].worker.frames_received > 0
            )
        )
        snapshots = fleet.snapshot()
        assert snapshots["failed"].worker.state.value == "failed"
        assert snapshots["healthy"].worker.state.value == "running"
        assert snapshots["healthy"].worker.reconnect_count == 0
    finally:
        fleet.stop(timeout_s=1.0)


def test_detector_error_isolated_to_one_camera() -> None:
    healthy_frame = np.ones((8, 8, 3), dtype=np.uint8)
    failing_frame = np.full((8, 8, 3), 2, dtype=np.uint8)

    def detect(frame: np.ndarray) -> list[PersonDetection]:
        if int(frame[0, 0, 0]) == 2:
            raise PersonDetectionError("synthetic detector failure")
        return []

    healthy = CameraRuntime(
        "healthy",
        FakeVideoSource([healthy_frame], read_delay_s=0.002),
        target_fps=20.0,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
    )
    failing = CameraRuntime(
        "failing",
        FakeVideoSource([failing_frame], read_delay_s=0.002),
        target_fps=20.0,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
    )
    fleet = MultiCameraRuntime(
        [healthy, failing],
        detector=FakePersonDetector(callback=detect),
    )

    fleet.start()
    try:
        assert wait_until(
            lambda: (
                fleet.snapshot()["failing"].detector_failures > 0
                and fleet.snapshot()["healthy"].processed_samples > 0
            )
        )
        snapshots = fleet.snapshot()
        assert snapshots["failing"].detector_failures > 0
        assert snapshots["healthy"].detector_failures == 0
        assert snapshots["healthy"].state is CameraState.PERSON_SCAN
        assert snapshots["healthy"].worker.state is not WorkerState.FAILED
    finally:
        fleet.stop(timeout_s=1.0)


def test_shared_detector_access_is_serialized() -> None:
    active = 0
    maximum_active = 0
    calls = 0
    state_lock = threading.Lock()

    def detect(frame: np.ndarray) -> list[PersonDetection]:
        del frame
        nonlocal active, maximum_active, calls
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            calls += 1
        time.sleep(0.01)
        with state_lock:
            active -= 1
        return []

    runtimes = [
        CameraRuntime(
            f"cam_{index}",
            FakeVideoSource(
                [np.full((8, 8, 3), index, dtype=np.uint8)],
                read_delay_s=0.002,
            ),
            target_fps=20.0,
            read_timeout_s=0.05,
            reconnect_delay_s=0.0,
        )
        for index in (1, 2)
    ]
    fleet = MultiCameraRuntime(
        runtimes,
        detector=FakePersonDetector(callback=detect),
    )

    fleet.start()
    try:
        assert wait_until(lambda: calls >= 4)
        assert maximum_active == 1
        assert runtimes[0].detector is runtimes[1].detector
    finally:
        fleet.stop(timeout_s=1.0)


def make_known_result() -> RecognitionResult:
    return RecognitionResult(
        status="known",
        person_id="mario_rossi",
        person_name="Mario Rossi",
        score=0.97,
        threshold=0.8,
    )


def test_shared_event_manager_keeps_camera_cooldowns_independent(tmp_path: Path) -> None:
    with EventManager(
        tmp_path / "events",
        save_snapshot=False,
        known_person_cooldown_seconds=30,
    ) as manager:
        pipelines = [
            CameraTrackingPipeline(
                camera_id,
                tracker=IoUGreedyTracker(max_missed_samples=0),
                recognition_confirmer=TrackRecognitionConfirmer(min_confirmations=1),
                event_publisher=manager,
            )
            for camera_id in ("cam_1", "cam_2")
        ]
        for pipeline in pipelines:
            update = pipeline.update([make_detection()])
            pipeline.begin_face_analysis(update.active_tracks[0].track_id)
            confirmation = pipeline.observe_recognition(
                update.active_tracks[0].track_id,
                make_known_result(),
                timestamp=UTC_START,
            )
            assert confirmation.confirmed is True

        duplicate = pipelines[0].observe_recognition(
            1,
            make_known_result(),
            timestamp=UTC_START + timedelta(seconds=1),
        )

    assert duplicate.confirmed is True
    assert len(list((tmp_path / "events").rglob("metadata.json"))) == 2

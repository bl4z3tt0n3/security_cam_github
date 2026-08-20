from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.config import AppConfig, TrackingConfig, load_config
from app.inference import PersonDetection
from app.tracking import (
    CameraState,
    CameraStateMachine,
    CameraTrackingPipeline,
    FaceAnalysisOutcome,
    InvalidStateTransitionError,
    IoUGreedyTracker,
)


def make_detection(
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.9,
    offset_seconds: int = 0,
) -> PersonDetection:
    return PersonDetection(
        bbox=bbox,
        confidence=confidence,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds),
    )


def make_pipeline(
    camera_id: str = "camera-1",
    *,
    max_missed_samples: int = 3,
) -> CameraTrackingPipeline:
    return CameraTrackingPipeline(
        camera_id,
        tracker=IoUGreedyTracker(max_missed_samples=max_missed_samples),
    )


def test_consecutive_detections_keep_the_same_track_id() -> None:
    pipeline = make_pipeline()

    first = pipeline.update([make_detection((0, 0, 20, 20))])
    second = pipeline.update([make_detection((1, 1, 21, 21), offset_seconds=1)])

    assert first.state is CameraState.TRACKING
    assert [transition.current for transition in first.transitions] == [
        CameraState.PERSON_DETECTED,
        CameraState.TRACKING,
    ]
    assert [track.track_id for track in first.new_tracks] == [1]
    assert [track.track_id for track in second.active_tracks] == [1]
    assert second.new_tracks == ()
    assert second.lost_tracks == ()


def test_a_track_survives_missing_samples_and_then_closes() -> None:
    pipeline = make_pipeline(max_missed_samples=2)
    pipeline.update([make_detection((0, 0, 20, 20))])

    first_missing = pipeline.update([])
    second_missing = pipeline.update([])
    closed = pipeline.update([])

    assert [track.missed_samples for track in first_missing.active_tracks] == [1]
    assert [track.missed_samples for track in second_missing.active_tracks] == [2]
    assert [track.track_id for track in closed.lost_tracks] == [1]
    assert closed.active_tracks == ()
    assert closed.state is CameraState.PERSON_SCAN


def test_new_person_after_track_closure_gets_a_new_id() -> None:
    pipeline = make_pipeline(max_missed_samples=0)
    pipeline.update([make_detection((0, 0, 20, 20))])
    closed = pipeline.update([])
    replacement = pipeline.update([make_detection((0, 0, 20, 20), offset_seconds=1)])

    assert [track.track_id for track in closed.lost_tracks] == [1]
    assert [track.track_id for track in replacement.new_tracks] == [2]


def test_multiple_people_are_associated_one_to_one_even_when_detection_order_changes() -> None:
    pipeline = make_pipeline()
    first = pipeline.update(
        [
            make_detection((0, 0, 20, 20)),
            make_detection((100, 0, 120, 20)),
        ]
    )
    second = pipeline.update(
        [
            make_detection((101, 0, 121, 20), offset_seconds=1),
            make_detection((1, 0, 21, 20), offset_seconds=1),
        ]
    )

    first_by_position = {track.bbox[0]: track.track_id for track in first.active_tracks}
    second_by_position = {track.bbox[0]: track.track_id for track in second.active_tracks}
    assert second_by_position[1] == first_by_position[0]
    assert second_by_position[101] == first_by_position[100]
    assert second.new_tracks == ()


def test_center_distance_fallback_keeps_a_track_when_boxes_do_not_overlap() -> None:
    pipeline = make_pipeline()
    first = pipeline.update([make_detection((0, 0, 20, 20))])
    second = pipeline.update([make_detection((25, 0, 45, 20), offset_seconds=1)])

    assert [track.track_id for track in first.new_tracks] == [1]
    assert [track.track_id for track in second.active_tracks] == [1]
    assert second.new_tracks == ()


def test_two_camera_pipelines_are_isolated() -> None:
    camera_one = make_pipeline("camera-1")
    camera_three = make_pipeline("camera-3")

    camera_one.update([make_detection((0, 0, 20, 20))])
    camera_three_update = camera_three.update([])

    assert camera_one.state is CameraState.TRACKING
    assert [track.track_id for track in camera_one.state_machine.active_tracks] == [1]
    assert camera_three_update.state is CameraState.PERSON_SCAN
    assert camera_three_update.active_tracks == ()


def test_new_and_lost_track_notifications_are_emitted_once() -> None:
    pipeline = make_pipeline(max_missed_samples=0)

    created = pipeline.update([make_detection((0, 0, 20, 20))])
    updated = pipeline.update([make_detection((0, 0, 20, 20), offset_seconds=1)])
    lost = pipeline.update([])
    already_lost = pipeline.update([])

    assert [track.track_id for track in created.new_tracks] == [1]
    assert updated.new_tracks == ()
    assert [track.track_id for track in lost.lost_tracks] == [1]
    assert already_lost.lost_tracks == ()


class KnownFaceHook:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, track: object) -> FaceAnalysisOutcome:
        del track
        self.calls += 1
        return FaceAnalysisOutcome.KNOWN


def test_face_analysis_hook_supports_cooldown_and_a_future_retry() -> None:
    pipeline = make_pipeline()
    update = pipeline.update([make_detection((0, 0, 20, 20))])
    track_id = update.active_tracks[0].track_id
    hook = KnownFaceHook()

    pipeline.begin_face_analysis(track_id)
    assert pipeline.state is CameraState.FACE_ANALYSIS
    assert pipeline.analyze_current_track(hook) is FaceAnalysisOutcome.KNOWN
    assert pipeline.state is CameraState.KNOWN
    pipeline.start_cooldown(track_id)
    assert pipeline.state is CameraState.COOLDOWN
    pipeline.finish_cooldown()
    assert pipeline.state is CameraState.TRACKING

    pipeline.begin_face_analysis(track_id)
    pipeline.analyze_current_track(hook)

    assert hook.calls == 2
    assert pipeline.state_machine.analysis_attempts(track_id) == 2


def test_invalid_state_transition_does_not_change_state() -> None:
    state_machine = CameraStateMachine()

    with pytest.raises(InvalidStateTransitionError):
        state_machine.begin_face_analysis(1)

    assert state_machine.state is CameraState.PERSON_SCAN
    assert state_machine.transition_history == ()


def test_tracking_configuration_defaults_and_validation() -> None:
    config = AppConfig.model_validate({})
    assert config.tracking == TrackingConfig()
    assert config.tracking.iou_threshold == pytest.approx(0.30)
    assert config.tracking.max_center_distance_px == pytest.approx(100.0)
    assert config.tracking.max_missed_samples == 3

    with pytest.raises(ValueError):
        TrackingConfig(max_missed_samples=-1)
    with pytest.raises(ValueError):
        TrackingConfig(iou_threshold=1.1)


def test_example_configuration_contains_tracking_settings() -> None:
    config = load_config("config/config.example.yaml")
    assert config.tracking.max_missed_samples == 3
    assert config.tracking.iou_threshold == pytest.approx(0.30)

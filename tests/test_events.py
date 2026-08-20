from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from app.events import EventManager, EventType, SnapshotWriter
from app.face import RecognitionResult, TrackRecognitionConfirmer
from app.inference import PersonDetection
from app.tracking import CameraTrackingPipeline, IoUGreedyTracker


UTC_START = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def make_known_result() -> RecognitionResult:
    return RecognitionResult(
        status="known",
        person_id="mario_rossi",
        person_name="Mario Rossi",
        score=0.97,
        threshold=0.8,
    )


def make_unknown_result() -> RecognitionResult:
    return RecognitionResult(
        status="unknown",
        person_id=None,
        person_name=None,
        score=0.31,
        threshold=0.8,
    )


def event_directory(root: Path, event_id: str, timestamp: datetime) -> Path:
    utc_timestamp = timestamp.astimezone(timezone.utc)
    return (
        root
        / f"{utc_timestamp.year:04d}"
        / f"{utc_timestamp.month:02d}"
        / f"{utc_timestamp.day:02d}"
        / event_id
    )


def test_known_event_writes_metadata_and_snapshot(tmp_path) -> None:
    frame = np.full((48, 64, 3), 120, dtype=np.uint8)
    with EventManager(tmp_path / "events") as manager:
        event = manager.publish_recognition(
            camera_id="front-door",
            track_id=4,
            result=make_known_result(),
            frame=frame,
            timestamp=UTC_START,
        )
        assert event is not None
        assert event.type is EventType.KNOWN_PERSON
        assert event.person_name == "Mario Rossi"
        assert event.snapshot_path == f"2026/08/11/{event.id}/snapshot.jpg"
        assert manager.flush(2.0)

    directory = event_directory(tmp_path / "events", event.id, UTC_START)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["type"] == "known_person"
    assert metadata["camera_id"] == "front-door"
    assert metadata["track_id"] == 4
    assert metadata["person_id"] == "mario_rossi"
    assert metadata["snapshot_path"] == f"2026/08/11/{event.id}/snapshot.jpg"
    assert (directory / "snapshot.jpg").is_file()
    assert (directory / "snapshot.jpg").stat().st_size > 0


def test_unknown_event_has_no_person_identity(tmp_path) -> None:
    with EventManager(tmp_path / "events", save_snapshot=False) as manager:
        event = manager.publish_recognition(
            camera_id="back-yard",
            track_id=2,
            result=make_unknown_result(),
            timestamp=UTC_START,
        )

    assert event is not None
    assert event.type is EventType.UNKNOWN_PERSON
    assert event.person_id is None
    assert event.person_name is None
    metadata_path = event_directory(tmp_path / "events", event.id, UTC_START) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["type"] == "unknown_person"
    assert metadata["recognition_score"] == 0.31
    assert metadata["snapshot_path"] is None


def test_same_track_is_deduplicated_until_cooldown_expires(tmp_path) -> None:
    with EventManager(
        tmp_path / "events",
        save_snapshot=False,
        known_person_cooldown_seconds=30,
    ) as manager:
        first = manager.publish_recognition(
            camera_id="camera-1",
            track_id=1,
            result=make_known_result(),
            timestamp=UTC_START,
        )
        duplicate = manager.publish_recognition(
            camera_id="camera-1",
            track_id=1,
            result=make_known_result(),
            timestamp=UTC_START + timedelta(seconds=29),
        )
        after_cooldown = manager.publish_recognition(
            camera_id="camera-1",
            track_id=1,
            result=make_known_result(),
            timestamp=UTC_START + timedelta(seconds=30),
        )

    assert first is not None
    assert duplicate is None
    assert after_cooldown is not None
    assert after_cooldown.id != first.id


def test_new_track_same_person_is_not_deduplicated(tmp_path) -> None:
    with EventManager(tmp_path / "events", save_snapshot=False) as manager:
        first = manager.publish_recognition(
            camera_id="camera-1",
            track_id=1,
            result=make_known_result(),
            timestamp=UTC_START,
        )
        second = manager.publish_recognition(
            camera_id="camera-1",
            track_id=2,
            result=make_known_result(),
            timestamp=UTC_START + timedelta(seconds=1),
        )

    assert first is not None
    assert second is not None


def test_storage_error_does_not_escape_event_manager(tmp_path) -> None:
    events_path = tmp_path / "events"
    events_path.write_text("not a directory", encoding="utf-8")
    manager = EventManager(events_path, save_snapshot=False)
    try:
        event = manager.publish_recognition(
            camera_id="camera-1",
            track_id=1,
            result=make_unknown_result(),
            timestamp=UTC_START,
        )
    finally:
        manager.close()

    assert event is None


def test_snapshot_writer_error_does_not_escape_event_manager(tmp_path) -> None:
    def broken_encoder(frame: np.ndarray) -> bytes:
        del frame
        raise OSError("simulated disk encoder failure")

    writer = SnapshotWriter(encoder=broken_encoder)
    with EventManager(
        tmp_path / "events",
        snapshot_writer=writer,
    ) as manager:
        event = manager.publish_recognition(
            camera_id="camera-1",
            track_id=1,
            result=make_unknown_result(),
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
            timestamp=UTC_START,
        )
        assert event is not None
        assert manager.flush(2.0)

    assert len(list((tmp_path / "events").rglob("metadata.json"))) == 1


def test_pipeline_publishes_only_confirmed_recognition(tmp_path) -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    with EventManager(tmp_path / "events", save_snapshot=False) as manager:
        pipeline = CameraTrackingPipeline(
            "camera-1",
            tracker=IoUGreedyTracker(max_missed_samples=0),
            recognition_confirmer=TrackRecognitionConfirmer(min_confirmations=2),
            event_publisher=manager,
        )
        update = pipeline.update(
            [
                PersonDetection(
                    bbox=(0, 0, 20, 20),
                    confidence=0.9,
                    timestamp=UTC_START,
                )
            ]
        )
        track_id = update.active_tracks[0].track_id
        pipeline.begin_face_analysis(track_id)

        first = pipeline.observe_recognition(
            track_id,
            make_unknown_result(),
            frame=frame,
            timestamp=UTC_START,
        )
        second = pipeline.observe_recognition(
            track_id,
            make_unknown_result(),
            frame=frame,
            timestamp=UTC_START + timedelta(seconds=1),
        )

    assert first.confirmed is False
    assert second.confirmed is True
    assert len(list((tmp_path / "events").rglob("metadata.json"))) == 1


def test_event_manager_isolates_cameras_with_same_track_id(tmp_path) -> None:
    with EventManager(tmp_path / "events", save_snapshot=False) as manager:
        for camera_id in ("camera-1", "camera-2"):
            manager.publish_recognition(
                camera_id=camera_id,
                track_id=1,
                result=make_unknown_result(),
                timestamp=UTC_START,
            )

    assert len(list((tmp_path / "events").rglob("metadata.json"))) == 2

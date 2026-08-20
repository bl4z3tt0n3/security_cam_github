from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import json

import numpy as np
import pytest

from app.benchmark import run_fake_benchmark
from app.face import (
    FaceAnalysisService,
    FaceDetection,
    FaceMatcher,
    FakeEmbedder,
    FakeFaceDetector,
    PersonStore,
)
from app.inference import PersonDetection
from app.metrics import CameraMetrics, read_resource_snapshot
from app.tracking import CameraState, CameraTrackingPipeline, Track
import app.metrics as metrics_module


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_camera_metrics_calculates_rates_and_latency_without_sample_history() -> None:
    clock = FakeClock()
    metrics = CameraMetrics("cam-a", clock=clock)

    metrics.record_decoded(4)
    metrics.record_sampled(2)
    metrics.record_person_detection(10.0, person_count=1)
    metrics.record_person_detection(20.0, person_count=2)
    metrics.record_face_detection(5.0)
    metrics.record_embedding(3.0)
    metrics.record_matching(2.0)
    metrics.record_pipeline(30.0)
    metrics.record_face_quality_reject(2)
    metrics.record_recognition_attempt()
    metrics.record_event()
    metrics.record_motion_detection(
        4.0,
        motion_detected=False,
        skipped_person_detection=True,
        changed_fraction=0.002,
    )
    metrics.set_active_tracks(3)
    metrics.set_queue_size(1)
    metrics.set_stream_fps(24.8)
    metrics.set_transport_counters(dropped_frames=7, reconnect_count=2)

    clock.value = 2.0
    snapshot = metrics.snapshot()

    assert snapshot.decoded_fps == pytest.approx(2.0)
    assert snapshot.sampled_fps == pytest.approx(1.0)
    assert snapshot.person_detection_fps == pytest.approx(1.0)
    assert snapshot.person_detection_inference_ms == pytest.approx(15.0)
    assert snapshot.persons_detected == 3
    assert snapshot.last_person_count == 2
    assert snapshot.face_detection_fps == pytest.approx(0.5)
    assert snapshot.processing_latency_ms == pytest.approx(30.0)
    assert snapshot.dropped_frames == 7
    assert snapshot.reconnect_count == 2
    assert snapshot.queue_size == 1
    assert snapshot.active_tracks == 3
    assert snapshot.face_quality_reject_count == 2
    assert snapshot.recognition_attempts == 1
    assert snapshot.events_generated == 1
    assert snapshot.motion_detection_fps == pytest.approx(0.5)
    assert snapshot.motion_detection_calls == 1
    assert snapshot.motion_positive_calls == 0
    assert snapshot.motion_skipped_samples == 1
    assert snapshot.last_motion_changed_fraction == pytest.approx(0.002)
    assert len(metrics._latency_sum_ms) == 6  # bounded stage state, not a sample queue


def test_camera_metrics_syncs_existing_worker_and_sampler_snapshots() -> None:
    metrics = CameraMetrics("cam-a")
    worker = SimpleNamespace(
        frames_received=100,
        decoded_fps=24.0,
        dropped_frames=12,
        reconnect_count=3,
        queue_size=1,
        stream_fps=25.0,
    )
    sampler = SimpleNamespace(frames_sampled=8, sampled_fps=2.0, queue_size=1)

    metrics.sync_worker_snapshot(worker, sampler_snapshot=sampler)
    snapshot = metrics.snapshot()

    assert snapshot.decoded_frames == 100
    assert snapshot.decoded_fps == 24.0
    assert snapshot.sampled_frames == 8
    assert snapshot.sampled_fps == 2.0
    assert snapshot.stream_fps == 25.0
    assert snapshot.dropped_frames == 12
    assert snapshot.reconnect_count == 3
    assert snapshot.queue_size == 1


def test_face_analysis_and_tracking_update_optional_camera_metrics() -> None:
    metrics = CameraMetrics("cam-a")
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    detector = FakeFaceDetector([FaceDetection((0, 0, 10, 10), 0.2)])
    service = FaceAnalysisService("cam-a", detector, metrics=metrics)
    timestamp = datetime.now(timezone.utc)
    track = Track(
        track_id=1,
        bbox=(0, 0, 120, 120),
        confidence=0.9,
        first_seen_at=timestamp,
        last_seen_at=timestamp,
    )

    result = service.process(frame, state=CameraState.TRACKING, tracks=(track,))
    pipeline = CameraTrackingPipeline("cam-a", metrics=metrics)
    pipeline.update(
        [
            PersonDetection(
                bbox=(0, 0, 120, 120),
                confidence=0.9,
                timestamp=timestamp,
            )
        ]
    )
    snapshot = metrics.snapshot()

    assert result.results[0].decisions[0].quality.accepted is False
    assert snapshot.face_detection_calls == 1
    assert snapshot.face_quality_reject_count == 1
    assert snapshot.active_tracks == 1


def test_matcher_updates_embedding_matching_and_attempt_metrics(tmp_path) -> None:
    embedder = FakeEmbedder(embedding_dimension=3)
    store = PersonStore(tmp_path / "persons")
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    store.save(
        name="Benchmark Person",
        person_id="benchmark-person",
        embeddings=np.expand_dims(embedder.embed(image), axis=0),
        model=embedder.metadata,
    )
    metrics = CameraMetrics("cam-a")
    matcher = FaceMatcher(embedder, store, threshold=0.8, metrics=metrics)

    result = matcher.match(image)
    snapshot = metrics.snapshot()

    assert result.status == "known"
    assert snapshot.embedding_calls == 1
    assert snapshot.matching_calls == 1
    assert snapshot.recognition_attempts == 1


def test_camera_metrics_reset_isolated_between_cameras() -> None:
    clock_a = FakeClock()
    clock_b = FakeClock()
    first = CameraMetrics("cam-a", clock=clock_a)
    second = CameraMetrics("cam-b", clock=clock_b)

    first.record_person_detection(1.0)
    second.record_face_detection(1.0)
    clock_a.value = 1.0
    clock_b.value = 1.0

    assert first.snapshot().person_detection_calls == 1
    assert first.snapshot().face_detection_calls == 0
    assert second.snapshot().face_detection_calls == 1
    assert second.snapshot().person_detection_calls == 0

    first.reset()
    assert first.snapshot().person_detection_calls == 0
    assert second.snapshot().face_detection_calls == 1


def test_resource_snapshot_reports_absent_gpu_without_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics_module.shutil, "which", lambda command: None)

    snapshot = read_resource_snapshot()

    assert snapshot.gpu_status == "unavailable"
    assert snapshot.gpu_percent is None
    assert snapshot.vram_used_bytes is None
    assert "not found" in snapshot.gpu_detail


def test_fake_benchmark_measures_all_required_components_and_is_json_safe(tmp_path) -> None:
    report = run_fake_benchmark(iterations=4, persons_root=tmp_path / "persons")

    assert report.execution == "simulated"
    assert [result.name for result in report.results] == [
        "person_detection",
        "face_detection",
        "embedding",
        "matching",
        "pipeline",
    ]
    assert all(result.status == "measured" for result in report.results)
    assert all(result.execution == "simulated" for result in report.results)
    assert all(result.iterations == 4 for result in report.results)
    assert report.camera_metrics is not None
    assert report.camera_metrics.pipeline_calls == 4
    json.dumps(report.to_dict())

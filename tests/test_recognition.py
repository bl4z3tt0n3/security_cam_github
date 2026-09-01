from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.face import (
    FaceMatcher,
    FakeEmbedder,
    IncompatibleEmbeddingModelError,
    PersonStore,
    RecognitionResult,
    TrackRecognitionConfirmer,
)
from app.inference import PersonDetection
from app.tracking import CameraState, CameraTrackingPipeline, IoUGreedyTracker


def make_result(
    status: str,
    *,
    person_id: str | None = None,
    person_name: str | None = None,
    score: float | None = 0.9,
    threshold: float | None = 0.8,
) -> RecognitionResult:
    return RecognitionResult(
        status=status,  # type: ignore[arg-type]
        person_id=person_id,
        person_name=person_name,
        score=score,
        threshold=threshold,
    )


def save_person(store: PersonStore, embedder: FakeEmbedder, name: str, vector: list[float]) -> None:
    store.save(
        name=name,
        embeddings=np.asarray([vector], dtype=np.float32),
        model=embedder.metadata,
    )


def test_match_above_threshold_returns_known_person(tmp_path) -> None:
    stored_embedder = FakeEmbedder(embedding_dimension=3, model_id="face-model")
    store = PersonStore(tmp_path / "persons")
    save_person(store, stored_embedder, "Mario Rossi", [1, 0, 0])
    save_person(store, stored_embedder, "Lucia Bianchi", [0, 1, 0])
    query_embedder = FakeEmbedder(
        embedding_dimension=3,
        model_id="face-model",
        callback=lambda image: [1, 0, 0],
    )

    result = FaceMatcher(query_embedder, store, threshold=0.8).match(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )

    assert result.status == "known"
    assert result.person_id == "mario_rossi"
    assert result.person_name == "Mario Rossi"
    assert result.score == pytest.approx(1.0)
    assert result.threshold == pytest.approx(0.8)


def test_match_below_threshold_is_unknown_without_assigning_nearest_name(tmp_path) -> None:
    embedder = FakeEmbedder(
        embedding_dimension=3,
        model_id="face-model",
        callback=lambda image: [1, 1, 0],
    )
    store = PersonStore(tmp_path / "persons")
    save_person(store, embedder, "Mario Rossi", [1, 0, 0])

    result = FaceMatcher(embedder, store, threshold=0.9).match(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )

    assert result.status == "unknown"
    assert result.person_id is None
    assert result.person_name is None
    assert result.score == pytest.approx(2**-0.5)


def test_empty_store_returns_unknown(tmp_path) -> None:
    embedder = FakeEmbedder(embedding_dimension=3, callback=lambda image: [1, 0, 0])

    result = FaceMatcher(embedder, PersonStore(tmp_path / "persons"), threshold=0.8).match(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )

    assert result == RecognitionResult("unknown", None, None, None, 0.8)


def test_unconfigured_threshold_cannot_produce_known_result(tmp_path) -> None:
    embedder = FakeEmbedder(embedding_dimension=3, callback=lambda image: [1, 0, 0])
    store = PersonStore(tmp_path / "persons")
    save_person(store, embedder, "Mario Rossi", [1, 0, 0])

    result = FaceMatcher(embedder, store, threshold=None).match(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )

    assert result.status == "unknown"
    assert result.person_id is None
    assert result.person_name is None
    assert result.score == pytest.approx(1.0)
    assert result.threshold is None


def test_matcher_rejects_incompatible_embedding_model(tmp_path) -> None:
    store = PersonStore(tmp_path / "persons")
    first = FakeEmbedder(embedding_dimension=3, model_id="model-a")
    save_person(store, first, "Mario Rossi", [1, 0, 0])
    second = FakeEmbedder(embedding_dimension=3, model_id="model-b")

    with pytest.raises(IncompatibleEmbeddingModelError, match="incompatible"):
        FaceMatcher(second, store, threshold=0.8)


def test_confirmation_requires_minimum_coherent_observations() -> None:
    confirmer = TrackRecognitionConfirmer(min_confirmations=2)
    result = make_result("known", person_id="mario", person_name="Mario")

    first = confirmer.observe(1, result)
    second = confirmer.observe(1, result)

    assert first.confirmed is False
    assert first.consecutive_count == 1
    assert first.stable_result is None
    assert second.confirmed is True
    assert second.consecutive_count == 2
    assert second.stable_result == result


def test_incoherent_results_reset_pending_streak_without_flipping_stable_identity() -> None:
    confirmer = TrackRecognitionConfirmer(min_confirmations=2)
    mario = make_result("known", person_id="mario", person_name="Mario")
    lucia = make_result("known", person_id="lucia", person_name="Lucia")

    assert confirmer.observe(1, mario).confirmed is False
    assert confirmer.observe(1, mario).confirmed is True
    conflicting = confirmer.observe(1, lucia)
    assert conflicting.confirmed is False
    assert conflicting.consecutive_count == 1
    assert conflicting.stable_result == mario
    assert confirmer.observe(1, lucia).confirmed is True
    assert confirmer.stable_result(1) == lucia


def test_two_tracks_have_independent_confirmation_state() -> None:
    confirmer = TrackRecognitionConfirmer(min_confirmations=2)
    mario = make_result("known", person_id="mario", person_name="Mario")
    unknown = make_result("unknown", score=0.2)

    assert confirmer.observe(1, mario).confirmed is False
    assert confirmer.observe(2, unknown).confirmed is False
    assert confirmer.observe(1, mario).confirmed is True
    assert confirmer.observe(2, unknown).confirmed is True


def make_detection(x: float) -> PersonDetection:
    return PersonDetection(
        bbox=(x, 0, x + 20, 20),
        confidence=0.9,
        timestamp=datetime.now(timezone.utc),
    )


def test_pipeline_moves_face_analysis_to_known_only_after_confirmation() -> None:
    pipeline = CameraTrackingPipeline(
        "camera-1",
        tracker=IoUGreedyTracker(max_missed_samples=0),
        recognition_confirmer=TrackRecognitionConfirmer(min_confirmations=2),
    )
    update = pipeline.update([make_detection(0)])
    track_id = update.active_tracks[0].track_id
    known = make_result("known", person_id="mario", person_name="Mario")

    pipeline.begin_face_analysis(track_id)
    assert pipeline.observe_recognition(track_id, known).confirmed is False
    assert pipeline.state is CameraState.FACE_ANALYSIS
    assert pipeline.observe_recognition(track_id, known).confirmed is True
    assert pipeline.state is CameraState.KNOWN
    assert pipeline.state_machine.face_analysis_outcome(track_id).value == "known"


def test_pipeline_can_confirm_two_tracks_without_sharing_identity_state() -> None:
    pipeline = CameraTrackingPipeline(
        "camera-1",
        tracker=IoUGreedyTracker(max_missed_samples=0),
        recognition_confirmer=TrackRecognitionConfirmer(min_confirmations=2),
    )
    update = pipeline.update([make_detection(0), make_detection(100)])
    first_id, second_id = [track.track_id for track in update.active_tracks]
    known = make_result("known", person_id="mario", person_name="Mario")
    unknown = make_result("unknown", score=0.1)

    pipeline.begin_face_analysis(first_id)
    pipeline.observe_recognition(first_id, known)
    pipeline.observe_recognition(first_id, known)
    assert pipeline.state is CameraState.KNOWN

    pipeline.begin_face_analysis(second_id)
    pipeline.observe_recognition(second_id, unknown)
    pipeline.observe_recognition(second_id, unknown)

    assert pipeline.state is CameraState.UNKNOWN
    assert pipeline.state_machine.face_analysis_outcome(first_id).value == "known"
    assert pipeline.state_machine.face_analysis_outcome(second_id).value == "unknown"


def test_pipeline_forgets_confirmation_when_track_closes() -> None:
    confirmer = TrackRecognitionConfirmer(min_confirmations=2)
    pipeline = CameraTrackingPipeline(
        "camera-1",
        tracker=IoUGreedyTracker(max_missed_samples=0),
        recognition_confirmer=confirmer,
    )
    first = pipeline.update([make_detection(0)])
    first_id = first.active_tracks[0].track_id
    pending = make_result("unknown", score=0.1)
    pipeline.begin_face_analysis(first_id)
    pipeline.observe_recognition(first_id, pending)
    pipeline.update([])

    assert confirmer.count(first_id) == 0
    replacement = pipeline.update([make_detection(0)])
    second_id = replacement.active_tracks[0].track_id
    pipeline.begin_face_analysis(second_id)
    confirmation = pipeline.observe_recognition(second_id, pending)

    assert second_id != first_id
    assert confirmation.confirmed is False
    assert confirmation.consecutive_count == 1


def test_two_camera_pipelines_keep_confirmation_state_independent() -> None:
    camera_one = CameraTrackingPipeline(
        "camera-1",
        recognition_confirmer=TrackRecognitionConfirmer(min_confirmations=2),
    )
    camera_two = CameraTrackingPipeline(
        "camera-2",
        recognition_confirmer=TrackRecognitionConfirmer(min_confirmations=2),
    )
    camera_one.update([make_detection(0)])
    camera_two.update([make_detection(0)])
    known = make_result("known", person_id="mario", person_name="Mario")

    camera_one.begin_face_analysis(1)
    camera_two.begin_face_analysis(1)
    camera_one.observe_recognition(1, known)
    camera_two.observe_recognition(1, known)
    assert camera_one.state is CameraState.FACE_ANALYSIS
    assert camera_two.state is CameraState.FACE_ANALYSIS

    camera_one.observe_recognition(1, known)

    assert camera_one.state is CameraState.KNOWN
    assert camera_two.state is CameraState.FACE_ANALYSIS

def test_matcher_normalizes_gallery_only_during_refresh(tmp_path, monkeypatch) -> None:
    import app.face.matcher as matcher_module

    embedder = FakeEmbedder(
        embedding_dimension=3,
        model_id="face-model",
        callback=lambda image: [1, 0, 0],
    )
    store = PersonStore(tmp_path / "persons")
    save_person(store, embedder, "Mario Rossi", [1, 0, 0])
    save_person(store, embedder, "Lucia Bianchi", [0, 1, 0])

    original = matcher_module._normalize_matrix
    calls = 0

    def counted(value, dimension, *, label):
        nonlocal calls
        calls += 1
        return original(value, dimension, label=label)

    monkeypatch.setattr(matcher_module, "_normalize_matrix", counted)
    matcher = FaceMatcher(embedder, store, threshold=0.8)
    calls_after_refresh = calls

    for _ in range(4):
        result = matcher.match(np.zeros((32, 32, 3), dtype=np.uint8))
        assert result.status == "known"

    assert calls_after_refresh == 2
    assert calls == calls_after_refresh

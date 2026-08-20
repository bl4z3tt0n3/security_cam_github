from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.face import (
    FaceDetection,
    FaceQualityEvaluator,
    FakeEmbedder,
    FakeFaceDetector,
    IncompatibleEmbeddingModelError,
    EnrollmentService,
    EnrollmentBatchService,
    PersonStore,
)


def quality_image(value: int = 128) -> np.ndarray:
    image = np.full((160, 160, 3), value, dtype=np.uint8)
    for y in range(0, 150, 10):
        for x in range(0, 150, 10):
            if (x // 10 + y // 10) % 2:
                image[y : y + 10, x : x + 10] = min(255, value + 35)
    return image


def write_image(directory: Path, name: str, image: np.ndarray) -> None:
    assert cv2.imwrite(str(directory / name), image)


def make_service(
    directory: Path,
    store_directory: Path,
    detections: list[FaceDetection],
    *,
    evaluator: FaceQualityEvaluator | None = None,
    embedder: FakeEmbedder | None = None,
) -> tuple[EnrollmentService, FakeEmbedder]:
    actual_embedder = embedder or FakeEmbedder(embedding_dimension=4)
    service = EnrollmentService(
        FakeFaceDetector(detections),
        actual_embedder,
        PersonStore(store_directory),
        evaluator=evaluator
        or FaceQualityEvaluator(
            min_width=80,
            min_height=80,
            blur_threshold=40,
            min_brightness=30,
            max_brightness=225,
            min_confidence=0.5,
        ),
    )
    return service, actual_embedder


def test_empty_directory_produces_report_without_record(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    service, embedder = make_service(
        tmp_path / "images",
        tmp_path / "persons",
        [FaceDetection((10, 10, 140, 140), 0.9)],
    )
    report = service.enroll("Mario Rossi", tmp_path / "images")
    assert report.processed_count == 0
    assert report.accepted_count == 0
    assert report.record is None
    assert embedder.calls == 0


def test_image_without_face_is_reported(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    write_image(images, "empty.png", quality_image())
    service, embedder = make_service(images, tmp_path / "persons", [])
    report = service.enroll("Mario Rossi", images)
    assert report.images[0].reasons == ("no_face",)
    assert report.record is None
    assert embedder.calls == 0


def test_image_with_multiple_faces_is_reported(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    write_image(images, "group.jpg", quality_image())
    service, embedder = make_service(
        images,
        tmp_path / "persons",
        [FaceDetection((10, 10, 100, 100), 0.9), FaceDetection((60, 60, 150, 150), 0.9)],
    )
    report = service.enroll("Mario Rossi", images)
    assert report.images[0].reasons == ("multiple_faces",)
    assert embedder.calls == 0


def test_low_quality_face_is_rejected(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    write_image(images, "dark.png", np.full((160, 160, 3), 10, dtype=np.uint8))
    service, embedder = make_service(
        images,
        tmp_path / "persons",
        [FaceDetection((10, 10, 140, 140), 0.9)],
    )
    report = service.enroll("Mario Rossi", images)
    assert report.images[0].accepted is False
    assert "too_dark" in report.images[0].reasons
    assert embedder.calls == 0


def test_multiple_valid_images_are_stored_with_metadata(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    write_image(images, "b.png", quality_image(128))
    write_image(images, "a.jpg", quality_image(140))
    service, embedder = make_service(
        images,
        tmp_path / "persons",
        [FaceDetection((10, 10, 140, 140), 0.9)],
    )
    report = service.enroll("Mario Rossi", images)
    assert report.accepted_count == 2
    assert report.rejected_count == 0
    assert embedder.calls == 2
    assert report.record is not None
    assert report.record.person_id == "mario_rossi"
    assert report.record.embeddings.shape == (2, 4)
    metadata_path = tmp_path / "persons" / "mario_rossi" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["embedding_count"] == 2
    assert metadata["embedding_file"] == "embeddings.npz"
    assert metadata["model"]["model_id"] == "fake-face-embedder"
    with np.load(tmp_path / "persons" / "mario_rossi" / "embeddings.npz") as archive:
        assert archive["embeddings"].shape == (2, 4)


def test_storage_rejects_incompatible_model(tmp_path: Path) -> None:
    store = PersonStore(tmp_path / "persons")
    first = FakeEmbedder(embedding_dimension=4, model_id="model-a")
    store.save(
        name="Mario Rossi",
        embeddings=np.ones((2, 4), dtype=np.float32),
        model=first.metadata,
    )
    second = FakeEmbedder(embedding_dimension=4, model_id="model-b")
    with pytest.raises(IncompatibleEmbeddingModelError, match="incompatible"):
        store.assert_compatible("mario_rossi", second.metadata)


def test_person_store_delete_keeps_enrollment_source(tmp_path: Path) -> None:
    source = tmp_path / "enrollment" / "roberto"
    source.mkdir(parents=True)
    source_image = source / "portrait.jpg"
    source_image.write_bytes(b"source remains")

    store = PersonStore(tmp_path / "persons")
    embedder = FakeEmbedder(embedding_dimension=4)
    store.save(
        name="Roberto",
        embeddings=np.ones((1, 4), dtype=np.float32),
        model=embedder.metadata,
        person_id="roberto",
    )

    store.delete("roberto")

    assert not (tmp_path / "persons" / "roberto").exists()
    assert source_image.is_file()


def test_batch_enrollment_scans_person_folders_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "enrollment"
    images = root / "roberto" / "nested"
    images.mkdir(parents=True)
    write_image(images, "face.jpg", quality_image())
    write_image(images, "face-2.png", quality_image(140))
    write_image(images, "broken.jpg", np.zeros((2, 2, 3), dtype=np.uint8))
    (images / "not-an-image.jpg").write_bytes(b"not an image")
    service, embedder = make_service(
        root,
        tmp_path / "persons",
        [FaceDetection((10, 10, 140, 140), 0.9)],
    )

    first = EnrollmentBatchService(service).import_tree(root)
    assert first.accepted_count == 2
    assert first.persons[0].person_id == "roberto"
    record = PersonStore(tmp_path / "persons").load("roberto")
    assert record.embeddings.shape == (2, 4)

    second = EnrollmentBatchService(service).import_tree(root)
    assert second.accepted_count == 2
    record_after = PersonStore(tmp_path / "persons").load("roberto")
    assert record_after.embeddings.shape == (2, 4)
    assert embedder.calls == 4

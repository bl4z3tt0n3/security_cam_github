"""Enrollment orchestration for known people."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .base import (
    FaceAligner,
    FaceCropper,
    FaceDetection,
    FaceDetector,
    FaceDetectorError,
    FaceQualityEvaluator,
    FaceLandmarker,
    NoOpFaceAligner,
)
from .alignment import FaceAlignmentError, localize_detection
from .embedding import FaceEmbedder, FaceEmbeddingError
from .storage import PersonRecord, PersonStorageError, PersonStore


SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class EnrollmentError(RuntimeError):
    """Raised for invalid enrollment input or an unusable enrollment setup."""


@dataclass(frozen=True)
class EnrollmentImageResult:
    image_name: str
    accepted: bool
    reasons: tuple[str, ...] = ()
    embedding_dimension: int | None = None


@dataclass(frozen=True)
class EnrollmentReport:
    name: str
    images: tuple[EnrollmentImageResult, ...]
    record: PersonRecord | None = None

    @property
    def processed_count(self) -> int:
        return len(self.images)

    @property
    def accepted_count(self) -> int:
        return sum(result.accepted for result in self.images)

    @property
    def rejected_count(self) -> int:
        return self.processed_count - self.accepted_count


@dataclass(frozen=True)
class EnrollmentBatchPersonReport:
    person_id: str
    name: str
    images: tuple[EnrollmentImageResult, ...] = ()
    record: PersonRecord | None = None
    error: str | None = None

    @property
    def accepted_count(self) -> int:
        return sum(result.accepted for result in self.images)

    @property
    def rejected_count(self) -> int:
        return len(self.images) - self.accepted_count


@dataclass(frozen=True)
class EnrollmentBatchReport:
    root: str
    persons: tuple[EnrollmentBatchPersonReport, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def accepted_count(self) -> int:
        return sum(person.accepted_count for person in self.persons)

    @property
    def rejected_count(self) -> int:
        return sum(person.rejected_count for person in self.persons)

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "errors": list(self.errors),
            "persons": [
                {
                    "person_id": person.person_id,
                    "name": person.name,
                    "accepted_count": person.accepted_count,
                    "rejected_count": person.rejected_count,
                    "error": person.error,
                    "images": [
                        {
                            "image_name": image.image_name,
                            "accepted": image.accepted,
                            "reasons": list(image.reasons),
                            "embedding_dimension": image.embedding_dimension,
                        }
                        for image in person.images
                    ],
                }
                for person in self.persons
            ],
        }


class EnrollmentService:
    """Validate reference images and persist every accepted embedding."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        store: PersonStore,
        *,
        cropper: FaceCropper | None = None,
        aligner: FaceAligner | None = None,
        landmarker: FaceLandmarker | None = None,
        evaluator: FaceQualityEvaluator | None = None,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.store = store
        self.cropper = cropper or FaceCropper()
        self.aligner = aligner or NoOpFaceAligner()
        self._requires_landmarks = aligner is not None and not isinstance(aligner, NoOpFaceAligner)
        self.landmarker = landmarker
        self.evaluator = evaluator or FaceQualityEvaluator()

    def enroll(
        self,
        name: str,
        images_directory: Path | str,
        *,
        person_id: str | None = None,
        overwrite: bool = False,
    ) -> EnrollmentReport:
        normalized_name = name.strip()
        if not normalized_name:
            raise EnrollmentError("person name cannot be empty")
        directory = Path(images_directory).expanduser()
        if not directory.is_dir():
            raise EnrollmentError(f"enrollment image directory not found: {directory}")
        image_paths = tuple(
            sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
                ),
                key=lambda path: path.name.lower(),
            )
        )
        results: list[EnrollmentImageResult] = []
        embeddings: list[np.ndarray] = []
        for image_path in image_paths:
            result, embedding = self._process_image(image_path)
            results.append(result)
            if embedding is not None:
                embeddings.append(embedding)

        record: PersonRecord | None = None
        if embeddings:
            record = self.store.save(
                name=normalized_name,
                embeddings=np.stack(embeddings, axis=0),
                model=self.embedder.metadata,
                person_id=person_id,
                overwrite=overwrite,
            )
        return EnrollmentReport(normalized_name, tuple(results), record)

    def _process_image(self, path: Path) -> tuple[EnrollmentImageResult, np.ndarray | None]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return EnrollmentImageResult(path.name, False, ("unreadable_image",)), None
        detections = self.detector.detect(image)
        if not detections:
            return EnrollmentImageResult(path.name, False, ("no_face",)), None
        if len(detections) > 1:
            return EnrollmentImageResult(path.name, False, ("multiple_faces",)), None
        detection = detections[0]
        cropped = self.cropper.crop(image, detection.bbox)
        if cropped is None:
            return EnrollmentImageResult(path.name, False, ("invalid_face_crop",)), None
        quality = self.evaluator.evaluate(
            image,
            detection,
            partial_bbox=cropped.was_partial,
        )
        if not quality.accepted:
            return EnrollmentImageResult(
                path.name,
                False,
                tuple(reason.value for reason in quality.reasons),
            ), None
        try:
            source_x1, source_y1, _, _ = cropped.bbox
            enriched = detection
            if enriched.landmarks is None and self.landmarker is not None:
                landmarks = self.landmarker.landmark(image, enriched)
                if landmarks is not None:
                    enriched = FaceDetection(
                        enriched.bbox,
                        enriched.confidence,
                        landmarks=landmarks,
                        detector_id=enriched.detector_id,
                        backend=enriched.backend,
                        device=enriched.device,
                    )
            if enriched.landmarks is None and self._requires_landmarks:
                return EnrollmentImageResult(path.name, False, ("landmarks_missing",)), None
            local_detection = localize_detection(enriched, source_x1, cropped.bbox[1])
            aligned = self.aligner.align(cropped.image, local_detection)
            embedding = self.embedder.embed(aligned)
        except FaceDetectorError:
            return EnrollmentImageResult(path.name, False, ("landmarks_missing",)), None
        except FaceAlignmentError:
            return EnrollmentImageResult(path.name, False, ("alignment_error",)), None
        except FaceEmbeddingError:
            return EnrollmentImageResult(path.name, False, ("embedding_error",)), None
        except (TypeError, ValueError):
            return EnrollmentImageResult(path.name, False, ("embedding_error",)), None
        return (
            EnrollmentImageResult(
                path.name,
                True,
                embedding_dimension=int(embedding.size),
            ),
            embedding,
        )


class EnrollmentBatchService:
    """Import ``enrollment/<person_id>`` folders without duplicate records."""

    def __init__(self, service: EnrollmentService) -> None:
        self.service = service

    def import_tree(self, root: Path | str) -> EnrollmentBatchReport:
        directory = Path(root).expanduser()
        if not directory.is_dir():
            raise EnrollmentError(f"enrollment root not found: {directory}")
        reports: list[EnrollmentBatchPersonReport] = []
        errors: list[str] = []
        for person_directory in sorted(
            (path for path in directory.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        ):
            person_id = person_directory.name
            try:
                self.service.store.validate_person_id(person_id)
            except PersonStorageError as exc:
                message = f"{person_id}: {exc}"
                errors.append(message)
                reports.append(
                    EnrollmentBatchPersonReport(
                        person_id=person_id,
                        name=person_id,
                        error=str(exc),
                    )
                )
                continue
            image_paths = tuple(
                sorted(
                    (
                        path
                        for path in person_directory.rglob("*")
                        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
                    ),
                    key=lambda path: str(path).casefold(),
                )
            )
            results: list[EnrollmentImageResult] = []
            embeddings: list[np.ndarray] = []
            for image_path in image_paths:
                try:
                    result, embedding = self.service._process_image(image_path)
                except Exception as exc:
                    result, embedding = (
                        EnrollmentImageResult(
                            image_path.name,
                            False,
                            (f"processing_error:{type(exc).__name__}",),
                        ),
                        None,
                    )
                results.append(result)
                if embedding is not None:
                    embeddings.append(embedding)
            if not embeddings:
                reports.append(
                    EnrollmentBatchPersonReport(
                        person_id=person_id,
                        name=person_id,
                        images=tuple(results),
                    )
                )
                continue
            try:
                record = self.service.store.merge(
                    name=person_id,
                    person_id=person_id,
                    embeddings=np.stack(embeddings, axis=0),
                    model=self.service.embedder.metadata,
                )
            except (PersonStorageError, ValueError) as exc:
                message = f"{person_id}: {exc}"
                errors.append(message)
                reports.append(
                    EnrollmentBatchPersonReport(
                        person_id=person_id,
                        name=person_id,
                        images=tuple(results),
                        error=str(exc),
                    )
                )
                continue
            reports.append(
                EnrollmentBatchPersonReport(
                    person_id=person_id,
                    name=person_id,
                    images=tuple(results),
                    record=record,
                )
            )
        return EnrollmentBatchReport(str(directory), tuple(reports), tuple(errors))


def iter_supported_images(directory: Path | str) -> Iterable[Path]:
    """Yield supported image files in deterministic order."""

    path = Path(directory)
    if not path.is_dir():
        raise EnrollmentError(f"enrollment image directory not found: {path}")
    yield from sorted(
        (
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda candidate: candidate.name.lower(),
    )

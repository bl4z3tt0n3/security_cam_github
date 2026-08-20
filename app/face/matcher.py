"""Local face matching against enrolled person embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Literal

import numpy as np

from app.metrics import CameraMetrics
from app.inference.synchronization import InferenceGate

from .embedding import FaceEmbedder
from .storage import PersonRecord, PersonStore


RecognitionStatus = Literal["known", "unknown"]


class FaceMatcherError(RuntimeError):
    """Raised when a face cannot be matched safely."""


@dataclass(frozen=True)
class RecognitionResult:
    """One conservative recognition decision.

    ``score`` is cosine similarity: higher values are better.  An unknown
    result may still expose the best candidate score, but never exposes that
    candidate's identity.
    """

    status: RecognitionStatus
    person_id: str | None
    person_name: str | None
    score: float | None
    threshold: float | None
    recognizer_id: str | None = None
    requested_device: str | None = None
    actual_device: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"known", "unknown"}:
            raise ValueError("recognition status must be 'known' or 'unknown'")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("recognition score must be finite when present")
        if self.threshold is not None and (
            not math.isfinite(self.threshold) or self.threshold < 0
        ):
            raise ValueError("recognition threshold must be finite and non-negative")
        if self.status == "known":
            if not self.person_id or not self.person_name:
                raise ValueError("known recognition requires person_id and person_name")
            if self.score is None:
                raise ValueError("known recognition requires a score")
            if self.threshold is None:
                raise ValueError("known recognition requires a configured threshold")
        elif self.person_id is not None or self.person_name is not None:
            raise ValueError("unknown recognition cannot contain a person identity")


def _normalize_vector(value: np.ndarray, dimension: int, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != dimension:
        raise FaceMatcherError(
            f"{label} dimension {vector.size} does not match model dimension {dimension}"
        )
    if not np.all(np.isfinite(vector)):
        raise FaceMatcherError(f"{label} contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise FaceMatcherError(f"{label} has zero or invalid norm")
    return vector / norm


def _normalize_matrix(value: np.ndarray, dimension: int, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimension or matrix.shape[0] == 0:
        raise FaceMatcherError(
            f"{label} must be a non-empty matrix with dimension {dimension}"
        )
    if not np.all(np.isfinite(matrix)):
        raise FaceMatcherError(f"{label} contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise FaceMatcherError(f"{label} contains a zero or invalid norm")
    return matrix / norms[:, None]


class FaceMatcher:
    """Compare one face embedding with all compatible enrolled persons."""

    def __init__(
        self,
        embedder: FaceEmbedder,
        person_store: PersonStore,
        *,
        threshold: float | None,
        metrics: CameraMetrics | None = None,
        inference_gate: InferenceGate | None = None,
    ) -> None:
        if threshold is not None and (
            not math.isfinite(threshold) or threshold < 0
        ):
            raise ValueError("recognition threshold must be finite and non-negative")
        self.embedder = embedder
        self.person_store = person_store
        self.threshold = threshold
        self.metrics = metrics
        self.inference_gate = inference_gate
        self._records: tuple[PersonRecord, ...] = ()
        self.refresh()

    @property
    def records(self) -> tuple[PersonRecord, ...]:
        """Return the immutable snapshot used for matching."""

        return self._records

    def refresh(self) -> tuple[PersonRecord, ...]:
        """Reload enrolled records and validate their model contract."""

        self._records = self.person_store.load_all(expected_model=self.embedder.metadata)
        return self._records

    def match(self, face_image: np.ndarray) -> RecognitionResult:
        """Return a thresholded result without assigning names to unknown faces."""

        if self.metrics is not None:
            self.metrics.record_recognition_attempt()
        # The fake embedder intentionally has no model identity.  Keeping the
        # optional metadata empty preserves the small public contract used by
        # existing offline callers while concrete runtime embedders always
        # expose their recognizer and device fingerprint.
        recognizer_id = (
            self.embedder.metadata.recognizer_id or self.embedder.metadata.model_id
            if self.embedder.metadata.backend != "fake"
            else None
        )
        requested_device = (
            self.embedder.metadata.requested_device
            if self.embedder.metadata.backend != "fake"
            else None
        )
        actual_device = (
            self.embedder.metadata.actual_device
            if self.embedder.metadata.backend != "fake"
            else None
        )
        embedding_started = time.perf_counter()
        try:
            embedding_value = (
                self.inference_gate.run(self.embedder.embed, face_image)
                if self.inference_gate is not None
                else self.embedder.embed(face_image)
            )
            embedding = _normalize_vector(
                embedding_value,
                self.embedder.metadata.embedding_dimension,
                label="query embedding",
            )
            if self.metrics is not None:
                self.metrics.record_embedding_generated()
        finally:
            if self.metrics is not None:
                self.metrics.record_embedding(
                    (time.perf_counter() - embedding_started) * 1000.0
                )

        matching_started = time.perf_counter()
        try:
            best_record: PersonRecord | None = None
            best_score: float | None = None
            for record in self._records:
                enrolled = _normalize_matrix(
                    record.embeddings,
                    self.embedder.metadata.embedding_dimension,
                    label=f"embeddings for person '{record.person_id}'",
                )
                score = float(np.max(enrolled @ embedding))
                score = max(-1.0, min(1.0, score))
                if best_score is None or score > best_score:
                    best_record = record
                    best_score = score

            if (
                best_record is not None
                and best_score is not None
                and self.threshold is not None
                and best_score >= self.threshold
            ):
                result = RecognitionResult(
                    status="known",
                    person_id=best_record.person_id,
                    person_name=best_record.name,
                    score=best_score,
                    threshold=self.threshold,
                    recognizer_id=recognizer_id,
                    requested_device=requested_device,
                    actual_device=actual_device,
                )
                if self.metrics is not None:
                    self.metrics.record_recognition_result("known")
                return result
            result = RecognitionResult(
                status="unknown",
                person_id=None,
                person_name=None,
                score=best_score,
                threshold=self.threshold,
                recognizer_id=recognizer_id,
                requested_device=requested_device,
                actual_device=actual_device,
            )
            if self.metrics is not None:
                self.metrics.record_recognition_result("unknown")
            return result
        finally:
            if self.metrics is not None:
                self.metrics.record_matching(
                    (time.perf_counter() - matching_started) * 1000.0
                )

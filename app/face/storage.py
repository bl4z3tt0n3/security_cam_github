"""Local, atomic persistence for enrolled-person embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid

import numpy as np

from .embedding import (
    EmbeddingModelMetadata,
    IncompatibleEmbeddingModelError,
)


PERSON_SCHEMA_VERSION = 1
_PERSON_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class PersonStorageError(RuntimeError):
    """Raised when a person record cannot be safely persisted or loaded."""


class PersonAlreadyExistsError(PersonStorageError):
    """Raised when an enrollment would overwrite an existing person."""


@dataclass(frozen=True)
class PersonRecord:
    """One persisted person and all original enrollment embeddings."""

    person_id: str
    name: str
    model: EmbeddingModelMetadata
    embeddings: np.ndarray
    directory: Path
    created_at: str


def _validate_embeddings(
    embeddings: np.ndarray,
    expected_dimension: int,
) -> np.ndarray:
    value = np.asarray(embeddings, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] == 0:
        raise PersonStorageError("embeddings must be a non-empty two-dimensional array")
    if value.shape[1] != expected_dimension:
        raise PersonStorageError(
            f"stored embedding dimension {value.shape[1]} does not match "
            f"model metadata dimension {expected_dimension}"
        )
    if not np.all(np.isfinite(value)):
        raise PersonStorageError("embeddings contain non-finite values")
    return value


def _safe_person_id(value: str) -> str:
    candidate = value.strip()
    if not _PERSON_ID_PATTERN.fullmatch(candidate):
        raise PersonStorageError(
            "person_id must contain only letters, numbers, '_' or '-' and be at most 64 characters"
        )
    return candidate


def person_id_from_name(name: str) -> str:
    """Create a stable filesystem-safe identifier from a display name."""

    import unicodedata

    normalized = unicodedata.normalize("NFKD", name.strip())
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    candidate = re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")
    candidate = candidate[:64]
    return candidate or "person"


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_write_npz(target: Path, embeddings: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, embeddings=embeddings)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        except OSError:
            pass


class PersonStore:
    """Persist and load local biometric records under one configured directory."""

    def __init__(self, root: Path | str, *, scope: str | Path | None = None) -> None:
        self.root = Path(root).expanduser()
        self.scope = self._safe_scope(scope) if scope else None
        self.records_root = self.root / self.scope if self.scope else self.root
        try:
            self.records_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PersonStorageError(f"cannot create persons directory {self.root}: {exc}") from exc

    @staticmethod
    def _safe_scope(value: str | Path) -> Path:
        if isinstance(value, Path):
            if value.is_absolute():
                raise PersonStorageError("gallery scope must be relative")
            parts = value.parts
        else:
            normalized = str(value).strip().replace("\\", "/")
            parts = tuple(part for part in normalized.split("/") if part)
        if not parts or any(
            part in {".", ".."}
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", part)
            for part in parts
        ):
            raise PersonStorageError("gallery scope contains unsupported path characters")
        return Path(*parts)

    def _record_directory(self, identifier: str) -> Path:
        """Resolve one record path and prove it remains below the active scope."""

        root = self.records_root.resolve()
        directory = (self.records_root / identifier).resolve()
        if root not in directory.parents:
            raise PersonStorageError("person path escapes the gallery root")
        return directory

    @staticmethod
    def validate_person_id(value: str) -> str:
        """Validate an enrollment-folder identifier using the store policy."""

        return _safe_person_id(value)

    def save(
        self,
        *,
        name: str,
        embeddings: np.ndarray,
        model: EmbeddingModelMetadata,
        person_id: str | None = None,
        overwrite: bool = False,
    ) -> PersonRecord:
        normalized_name = name.strip()
        if not normalized_name:
            raise PersonStorageError("person name cannot be empty")
        identifier = _safe_person_id(person_id or person_id_from_name(normalized_name))
        value = _validate_embeddings(embeddings, model.embedding_dimension)
        directory = self._record_directory(identifier)
        metadata_path = directory / "metadata.json"
        embeddings_path = directory / "embeddings.npz"
        if directory.exists() and not overwrite:
            raise PersonAlreadyExistsError(
                f"person '{identifier}' already exists; use --overwrite only intentionally"
            )
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PersonStorageError(
                f"cannot create person directory '{identifier}': {exc}"
            ) from exc
        created_at = datetime.now(timezone.utc).isoformat()
        metadata: dict[str, Any] = {
            "schema_version": PERSON_SCHEMA_VERSION,
            "person_id": identifier,
            "name": normalized_name,
            "created_at": created_at,
            "embedding_count": int(value.shape[0]),
            "embedding_file": "embeddings.npz",
            "model": model.to_dict(),
        }
        metadata_bytes = json.dumps(
            metadata, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        try:
            _atomic_write_npz(embeddings_path, value)
            _atomic_write_bytes(metadata_path, metadata_bytes)
        except OSError as exc:
            raise PersonStorageError(f"cannot write person '{identifier}': {exc}") from exc
        return PersonRecord(identifier, normalized_name, model, value.copy(), directory, created_at)

    def merge(
        self,
        *,
        name: str,
        embeddings: np.ndarray,
        model: EmbeddingModelMetadata,
        person_id: str | None = None,
        cosine_tolerance: float = 1e-6,
    ) -> PersonRecord:
        """Idempotently append only genuinely new normalized embeddings."""

        if cosine_tolerance < 0 or not np.isfinite(cosine_tolerance):
            raise PersonStorageError("cosine_tolerance must be finite and non-negative")
        identifier = _safe_person_id(person_id or person_id_from_name(name))
        incoming = _validate_embeddings(embeddings, model.embedding_dimension)
        directory = self._record_directory(identifier)
        if not directory.exists():
            return self.save(
                name=name,
                embeddings=incoming,
                model=model,
                person_id=identifier,
                overwrite=False,
            )
        existing = self.load(identifier, expected_model=model)
        accepted: list[np.ndarray] = []
        for candidate in incoming:
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm <= 0:
                continue
            normalized = candidate / candidate_norm
            duplicate = False
            for prior in (*existing.embeddings, *accepted):
                prior_norm = float(np.linalg.norm(prior))
                if prior_norm <= 0:
                    continue
                similarity = float(np.dot(normalized, prior / prior_norm))
                if similarity >= 1.0 - cosine_tolerance:
                    duplicate = True
                    break
            if not duplicate:
                accepted.append(candidate)
        if not accepted:
            return existing
        return self.save(
            name=name or existing.name,
            embeddings=np.vstack((existing.embeddings, np.stack(accepted, axis=0))),
            model=model,
            person_id=identifier,
            overwrite=True,
        )

    def load(
        self,
        person_id: str,
        *,
        expected_model: EmbeddingModelMetadata | None = None,
    ) -> PersonRecord:
        identifier = _safe_person_id(person_id)
        directory = self._record_directory(identifier)
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise PersonStorageError(f"person metadata not found: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") != PERSON_SCHEMA_VERSION:
                raise PersonStorageError("unsupported person metadata schema version")
            if metadata.get("person_id") != identifier:
                raise PersonStorageError("person metadata id does not match its directory")
            model = EmbeddingModelMetadata.from_dict(metadata["model"])
            embedding_file = metadata.get("embedding_file", "embeddings.npz")
            if embedding_file != "embeddings.npz":
                raise PersonStorageError("unsupported embedding file name")
            with np.load(directory / embedding_file, allow_pickle=False) as archive:
                if "embeddings" not in archive:
                    raise PersonStorageError("embedding archive does not contain 'embeddings'")
                embeddings = _validate_embeddings(
                    archive["embeddings"], model.embedding_dimension
                )
            if metadata.get("embedding_count") != int(embeddings.shape[0]):
                raise PersonStorageError("embedding count does not match persisted data")
            name = str(metadata["name"]).strip()
            created_at = str(metadata["created_at"])
        except PersonStorageError:
            raise
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise PersonStorageError(f"cannot read person '{identifier}': {exc}") from exc
        record = PersonRecord(identifier, name, model, embeddings.copy(), directory, created_at)
        if expected_model is not None and not model.is_compatible_with(expected_model):
            raise IncompatibleEmbeddingModelError(
                f"person '{identifier}' was enrolled with an incompatible embedding model"
            )
        return record

    def load_all(
        self,
        *,
        expected_model: EmbeddingModelMetadata | None = None,
    ) -> tuple[PersonRecord, ...]:
        """Load every enrolled person in deterministic id order.

        A directory below ``persons`` is treated as an enrolled record.  A
        malformed or model-incompatible record therefore fails closed instead
        of being silently skipped during recognition.
        """

        try:
            directories = sorted(
                (path for path in self.records_root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            )
        except OSError as exc:
            raise PersonStorageError(
                f"cannot enumerate persons directory {self.root}: {exc}"
            ) from exc
        return tuple(
            self.load(directory.name, expected_model=expected_model)
            for directory in directories
        )

    def delete(self, person_id: str) -> None:
        """Delete one enrolled record below the configured scope."""

        identifier = _safe_person_id(person_id)
        directory = self._record_directory(identifier)
        root = self.records_root.resolve()
        if not directory.is_dir():
            raise PersonStorageError(f"person not found: {identifier}")
        try:
            for path in sorted(directory.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            directory.rmdir()
        except OSError as exc:
            raise PersonStorageError(f"cannot delete person '{identifier}': {exc}") from exc

    def assert_compatible(
        self,
        person_id: str,
        expected_model: EmbeddingModelMetadata,
    ) -> PersonRecord:
        """Load a record only when its model contract exactly matches."""

        return self.load(person_id, expected_model=expected_model)

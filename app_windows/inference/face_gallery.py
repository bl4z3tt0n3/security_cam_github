"""Read-only enrollment source scanning for the Windows face gallery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.face.enrollment import SUPPORTED_IMAGE_SUFFIXES
from app.face.storage import PersonStorageError, PersonStore


@dataclass(frozen=True)
class EnrollmentScan:
    """A safe, UI-ready snapshot of the enrollment source tree."""

    root_present: bool
    people: tuple[dict[str, Any], ...]
    error: str | None = None


def scan_enrollment_people(
    root: Path | str,
    *,
    active_people: Iterable[Mapping[str, Any]] = (),
) -> EnrollmentScan:
    """Combine source folders with active biometric records.

    The scan never mutates either ``enrollment/`` or ``persons/``.  Invalid
    IDs, empty folders and active records whose source folder disappeared are
    represented as rows so the WPF user can understand why an operation is
    unavailable.
    """

    source_root = Path(root).expanduser()
    active: dict[str, dict[str, Any]] = {}
    for value in active_people:
        person_id = str(value.get("person_id") or "").strip()
        if not person_id:
            continue
        active[person_id] = {
            "name": str(value.get("name") or person_id),
            "embedding_count": int(value.get("embedding_count") or 0),
        }

    if not source_root.exists() or not source_root.is_dir():
        return EnrollmentScan(
            root_present=False,
            people=tuple(_missing_source_row(person_id, value) for person_id, value in sorted(active.items())),
            error="enrollment source folder is missing",
        )

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        directories = sorted(
            (path for path in source_root.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )
    except OSError as exc:
        return EnrollmentScan(
            root_present=True,
            people=tuple(_missing_source_row(person_id, value) for person_id, value in sorted(active.items())),
            error=f"cannot read enrollment source folder: {type(exc).__name__}",
        )

    for directory in directories:
        person_id = directory.name
        seen.add(person_id)
        active_value = active.get(person_id)
        try:
            PersonStore.validate_person_id(person_id)
        except PersonStorageError:
            rows.append(
                _row(
                    person_id=person_id,
                    name=active_value["name"] if active_value else person_id,
                    image_count=0,
                    embedding_count=active_value["embedding_count"] if active_value else 0,
                    active=active_value is not None,
                    valid=False,
                    source_available=True,
                    status="invalid",
                )
            )
            continue

        try:
            image_count = sum(
                1
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
        except OSError:
            image_count = 0
            status = "unreadable"
            valid = False
        else:
            status = "empty" if image_count == 0 else ("active" if active_value else "not_active")
            valid = image_count > 0

        rows.append(
            _row(
                person_id=person_id,
                name=active_value["name"] if active_value else person_id,
                image_count=image_count,
                embedding_count=active_value["embedding_count"] if active_value else 0,
                active=active_value is not None,
                valid=valid,
                source_available=True,
                status=status,
            )
        )

    for person_id, value in sorted(active.items()):
        if person_id not in seen:
            rows.append(_missing_source_row(person_id, value))

    rows.sort(key=lambda value: str(value["person_id"]).casefold())
    return EnrollmentScan(root_present=True, people=tuple(rows))


def _missing_source_row(person_id: str, value: Mapping[str, Any]) -> dict[str, Any]:
    return _row(
        person_id=person_id,
        name=str(value.get("name") or person_id),
        image_count=0,
        embedding_count=int(value.get("embedding_count") or 0),
        active=True,
        valid=False,
        source_available=False,
        status="missing",
    )


def _row(
    *,
    person_id: str,
    name: str,
    image_count: int,
    embedding_count: int,
    active: bool,
    valid: bool,
    source_available: bool,
    status: str,
) -> dict[str, Any]:
    return {
        "person_id": person_id,
        "name": name,
        "image_count": image_count,
        "embedding_count": embedding_count,
        "active": active,
        "valid": valid,
        "source_available": source_available,
        "status": status,
    }


__all__ = ["EnrollmentScan", "scan_enrollment_people"]

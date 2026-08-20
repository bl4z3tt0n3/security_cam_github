from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import app.face as face_package
from app.config import load_config
from app.face.registry import LANDMARKER_SPEC
from app_windows.config.persistence import CameraConfigRepository
from app_windows.inference.face_gallery import scan_enrollment_people
from app_windows.inference.face_recognition_controller import FaceRecognitionController
from app_windows.models.face_recognition_state import (
    FaceRecognitionSettings,
    fraction_to_percentage,
    percentage_to_fraction,
)
from app_windows.wpf_bridge import (
    recognition_threshold_from_payload,
    resolve_landmarker_model,
)


def test_face_percentage_contract_uses_normalized_runtime_values() -> None:
    assert percentage_to_fraction(1) == pytest.approx(0.01)
    assert percentage_to_fraction(21) == pytest.approx(0.21)
    assert percentage_to_fraction(100) == pytest.approx(1.0)
    assert fraction_to_percentage(0.214, minimum=1) == 21


def test_face_settings_persistence_preserves_decimal_and_null_threshold(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    source = config_dir / "config.example.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "face_detection": {},
                "face_landmarks": {},
                "recognition": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    settings = FaceRecognitionSettings(
        detector_confidence_threshold=0.214,
        detector_inference_fps=2.5,
        landmarker_id=LANDMARKER_SPEC.model_id,
        landmarker_model=LANDMARKER_SPEC.relative_path,
        recognition_threshold=None,
    )

    target = CameraConfigRepository(tmp_path).save_face_analysis(
        settings,
        current_path=source,
    )
    document = yaml.safe_load(target.read_text(encoding="utf-8"))

    assert document["face_detection"]["confidence_threshold"] == pytest.approx(0.214)
    assert document["face_detection"]["inference_fps"] == pytest.approx(2.5)
    assert document["face_landmarks"]["landmarker_id"] == LANDMARKER_SPEC.model_id
    assert document["recognition"]["threshold"] is None


def test_face_gallery_root_persistence_preserves_existing_yaml_sections(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    source = config_dir / "config.example.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "cameras": [{"id": "cam_1", "stream_url": "fake://cam_1/live"}],
                "storage": {"persons_dir": "persons", "custom": "preserve"},
                "custom_section": {"enabled": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    root = tmp_path / "people-source"
    root.mkdir()

    target = CameraConfigRepository(tmp_path).save_face_gallery_root(
        root,
        current_path=source,
    )
    document = yaml.safe_load(target.read_text(encoding="utf-8"))

    assert target == config_dir / "config.local.yaml"
    assert document["storage"]["enrollment_dir"] == str(root)
    assert document["storage"]["custom"] == "preserve"
    assert document["cameras"][0]["id"] == "cam_1"
    assert document["custom_section"] == {"enabled": True}


def test_face_gallery_root_defaults_and_controller_snapshot_use_configured_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(repo_root / "config" / "config.example.yaml")
    assert config.storage.enrollment_dir == Path("enrollment")

    root = tmp_path / "selected-root"
    (root / "mario").mkdir(parents=True)
    (root / "mario" / "portrait.jpg").write_bytes(b"image")
    configured = config.model_copy(
        update={
            "storage": config.storage.model_copy(update={"enrollment_dir": root}),
        }
    )
    controller = FaceRecognitionController(repo_root=tmp_path, config=configured)

    gallery = controller.refresh_gallery()

    assert gallery.enrollment_root == str(root.resolve())
    assert [row["person_id"] for row in gallery.enrollment_people] == ["mario"]
    assert gallery.enrollment_people[0]["status"] == "not_active"


def test_face_gallery_import_uses_configured_root(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(repo_root / "config" / "config.example.yaml")
    root = tmp_path / "selected-root"
    (root / "mario").mkdir(parents=True)
    configured = config.model_copy(
        update={
            "storage": config.storage.model_copy(update={"enrollment_dir": root}),
        }
    )
    controller = FaceRecognitionController(repo_root=tmp_path, config=configured)
    controller._settings = replace(controller.settings, face_detection_enabled=True)
    imported_roots: list[Path] = []

    class StubBatchService:
        def __init__(self, _service) -> None:
            pass

        def import_tree(self, selected_root: Path) -> object:
            imported_roots.append(Path(selected_root))
            return object()

    monkeypatch.setattr(face_package, "EnrollmentBatchService", StubBatchService)
    monkeypatch.setattr(
        controller,
        "_build_enrollment_service",
        lambda _settings: (object(), ()),
    )

    controller.import_enrollment()

    assert imported_roots == [root.resolve()]


def test_bridge_threshold_update_distinguishes_missing_from_explicit_null() -> None:
    assert recognition_threshold_from_payload(0.73, {}) == pytest.approx(0.73)
    assert recognition_threshold_from_payload(0.73, {"threshold": 0.41}) == pytest.approx(0.41)
    assert recognition_threshold_from_payload(0.73, {"threshold": None}) is None


def test_landmarker_selection_resolves_registry_path_and_rejects_unknown(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert resolve_landmarker_model(LANDMARKER_SPEC.model_id, repo_root) == LANDMARKER_SPEC.relative_path
    with pytest.raises(ValueError, match="unsupported face landmarker"):
        resolve_landmarker_model("unknown-landmarker", repo_root)


def test_enrollment_gallery_scan_handles_empty_multiple_invalid_and_active_rows(tmp_path: Path) -> None:
    root = tmp_path / "enrollment"
    missing = scan_enrollment_people(
        root,
        active_people=({"person_id": "legacy", "name": "Legacy", "embedding_count": 2},),
    )
    assert not missing.root_present
    assert missing.people[0]["status"] == "missing"

    root.mkdir()
    empty = scan_enrollment_people(root)
    assert empty.root_present
    assert empty.people == ()

    (root / "roberto").mkdir()
    (root / "roberto" / "portrait.jpg").write_bytes(b"not decoded by the read-only scanner")
    (root / "roberto" / "notes.txt").write_text("ignored", encoding="utf-8")
    (root / "alice").mkdir()
    (root / "bad id").mkdir()
    (root / "charlie" / "nested").mkdir(parents=True)
    (root / "charlie" / "nested" / "portrait.png").write_bytes(b"image")

    scan = scan_enrollment_people(
        root,
        active_people=({"person_id": "roberto", "name": "Roberto", "embedding_count": 4},),
    )
    by_id = {row["person_id"]: row for row in scan.people}

    assert by_id["roberto"] == {
        "person_id": "roberto",
        "name": "Roberto",
        "image_count": 1,
        "embedding_count": 4,
        "active": True,
        "valid": True,
        "source_available": True,
        "status": "active",
    }
    assert by_id["alice"]["status"] == "empty"
    assert not by_id["alice"]["valid"]
    assert by_id["bad id"]["status"] == "invalid"
    assert by_id["charlie"]["status"] == "not_active"
    assert by_id["charlie"]["image_count"] == 1

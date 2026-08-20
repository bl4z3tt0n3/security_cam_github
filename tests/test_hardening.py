from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import RecognitionConfig, load_config
from app.face import IncompatibleEmbeddingModelError
from app.hardening import HardeningCheck, HardeningReport, normalize_status
import app.face
import scripts.check_environment as check_environment
from scripts.check_environment import (
    _check_hardening,
    _probe_camera_isolation,
    _probe_incompatible_embeddings,
    _probe_shutdown,
    _probe_storage_failure_isolation,
)


ROOT = Path(__file__).resolve().parents[1]


def test_hardening_status_contract_serializes_and_aggregates() -> None:
    report = HardeningReport(
        (
            HardeningCheck("safe", "PASS", "ok"),
            HardeningCheck("deferred", "DEFERRED", "needs LAN"),
        )
    )

    assert report.status == "PASS"
    assert report.failed is False
    assert report.to_dict()["counts"] == {
        "PASS": 1,
        "INFO": 0,
        "DEFERRED": 1,
        "FAIL": 0,
    }
    json.dumps(report.to_dict())
    assert normalize_status("OK") == "PASS"
    assert normalize_status("NOT CONFIGURED") == "INFO"


def test_hardening_offline_probes_cover_residual_failure_modes() -> None:
    assert _probe_incompatible_embeddings()[0] is True
    assert _probe_storage_failure_isolation()[0] is True
    assert _probe_camera_isolation()[0] is True
    assert _probe_shutdown()[0] is True


def test_hardening_report_has_explicit_checks_without_secrets() -> None:
    config = load_config(ROOT / "config" / "config.example.yaml")
    report = _check_hardening(config, emit=False)
    names = {check.name for check in report.checks}

    assert report.failed is False
    assert {
        "Secrets ignore rules",
        "Secret redaction",
        "Storage failure isolation",
        "Embedding compatibility",
        "Offline camera isolation",
        "Graceful shutdown",
        "Disabled components",
    } <= names
    serialized = json.dumps(report.to_dict(), ensure_ascii=False)
    assert "plain-secret" not in serialized
    assert "token-secret" not in serialized
    assert "admin:secret@" not in serialized


def test_hardening_report_does_not_leave_probe_directories() -> None:
    config = load_config(ROOT / "config" / "config.example.yaml")
    _check_hardening(config, emit=False)
    leftovers = list((ROOT / ".test-tmp").glob("hardening-*-*"))
    assert leftovers == []


def test_enabled_recognition_marks_incompatible_configured_records_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "config" / "config.example.yaml")
    model_path = tmp_path / "face.onnx"
    model_path.write_bytes(b"synthetic model placeholder")
    enabled_recognition = RecognitionConfig(
        enabled=True,
        model=str(model_path),
        threshold=0.8,
    )
    config = config.model_copy(update={"recognition": enabled_recognition})
    monkeypatch.setattr(
        check_environment.importlib.util,
        "find_spec",
        lambda name: object() if name == "onnxruntime" else None,
    )

    def reject_incompatible(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise IncompatibleEmbeddingModelError("synthetic incompatible configured record")

    monkeypatch.setattr(app.face, "create_face_matcher", reject_incompatible)
    report = check_environment._check_hardening(config, emit=False)

    configured = next(
        check for check in report.checks if check.name == "Configured embedding records"
    )
    assert configured.status == "FAIL"

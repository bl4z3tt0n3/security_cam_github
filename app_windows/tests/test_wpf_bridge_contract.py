from __future__ import annotations

import pytest

from app.face import capabilities as face_capabilities
from app.face.capabilities import FaceCapability
from app.face.registry import FACE_DETECTOR_SPECS, RECOGNIZER_SPECS
from app_windows.wpf_bridge import face_capability_rows


def test_face_capability_rows_serializes_dataclasses_for_wpf() -> None:
    capability = FaceCapability(
        component="face_detection",
        model_id="scrfd_2.5g_kps",
        display_name="SCRFD",
        backend="onnxruntime",
        device="cpu",
        available=True,
        artifact_present=True,
        reason="I/O probe passed",
        probed=True,
        actual_device="cpu",
        model_path="models/face_detection/scrfd_2.5g_kps/scrfd_2.5g_bnkps.onnx",
    )

    rows = face_capability_rows((capability,))

    assert rows == [capability.to_dict()]
    assert rows[0]["model_id"] == "scrfd_2.5g_kps"
    assert rows[0]["actual_device"] == "cpu"
    assert rows[0]["model_path"].endswith("scrfd_2.5g_bnkps.onnx")


@pytest.mark.parametrize("spec", FACE_DETECTOR_SPECS, ids=lambda value: value.model_id)
def test_face_detector_capabilities_publish_registry_model_path(
    spec,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        face_capabilities,
        "_runtime_devices",
        lambda _backend: ("cpu", "cuda", "gpu", "npu"),
    )

    rows = face_capabilities._capability_for_spec(
        "face_detection",
        spec,
        model_root=tmp_path,
    )

    assert rows
    assert {row.model_path for row in rows} == {spec.relative_path}
    assert all(row.to_dict()["model_path"] == spec.relative_path for row in rows)


@pytest.mark.parametrize("spec", RECOGNIZER_SPECS, ids=lambda value: value.recognizer_id)
def test_face_recognizer_capabilities_publish_registry_model_path(
    spec,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        face_capabilities,
        "_runtime_devices",
        lambda _backend: ("cpu", "cuda", "gpu", "npu"),
    )

    rows = face_capabilities._capability_for_spec(
        "recognition",
        spec,
        model_root=tmp_path,
    )

    assert rows
    assert {row.model_path for row in rows} == {spec.relative_path}
    assert all(row.to_dict()["model_path"] == spec.relative_path for row in rows)


def test_face_capability_rows_preserves_mapping_payloads() -> None:
    payload = {"model_id": "yunet_2023mar", "available": False}

    assert face_capability_rows([payload]) == [payload]


def test_face_capability_rows_rejects_unknown_rows() -> None:
    with pytest.raises(TypeError, match="unsupported face capability row"):
        face_capability_rows([object()])

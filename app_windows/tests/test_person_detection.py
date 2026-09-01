from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QSlider, QSpinBox

from app.inference import (
    FakePersonDetector,
    PersonDetection,
    PersonDetectionError,
    YoloESegmentationDetector,
    normalize_prompts,
)
from app.video.base import FramePacket, utc_now
from app_windows.config.credentials import InMemoryCredentialStore
from app_windows.config.persistence import CameraConfigRepository
from app_windows.inference import PersonDetectionController
import app_windows.inference.person_detection_controller as controller_module
from app_windows.models.camera_display_transform import CameraDisplayTransform
from app_windows.models.camera_view_state import CameraSlot, CameraViewSnapshot, CameraViewStatus
from app_windows.models.person_detection_state import (
    PersonDetectionSnapshot,
    PersonDetectionSettings,
    PersonDetectionStatus,
)
from app_windows.ui.camera_configuration_panel import CameraConfigurationPanel
from app_windows.ui.camera_focus_view import CameraFocusView
from app_windows.ui.main_window import MainWindow
from app_windows.ui.person_detection_panel import (
    PersonDetectionPanel,
    discover_openvino_models,
    discover_yoloe_models,
)
from app_windows.video.detection_geometry import (
    effective_frame_size,
    map_detection_bbox_to_widget,
    map_detection_polygon_to_widget,
    transform_detection_bbox,
    transform_detection_polygon,
)
from app_windows.video.fake_provider import FakeFrameProvider
from app_windows.monitor_controller import CameraMonitorController
from app_windows.config.ui_config import UiSettings


def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _detection(confidence: float = 0.87) -> PersonDetection:
    return PersonDetection(
        bbox=(100.0, 40.0, 220.0, 300.0),
        confidence=confidence,
        timestamp=datetime.now(timezone.utc),
    )


def _slot(camera_id: str, index: int = 1) -> CameraSlot:
    return CameraSlot(
        slot_index=index,
        camera_id=camera_id,
        name=f"Camera {index}",
        enabled=True,
        configured=True,
        stream_url=f"fake://{camera_id}/live",
    )


def test_model_catalog_lists_only_yoloe_segmentation_and_keeps_missing_configured_model(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "valid.onnx").write_bytes(b"onnx")
    (models / "yoloe-26n-seg.pt").write_bytes(b"pt")
    (models / "yoloe-26s-seg.pt").write_bytes(b"pt")
    (models / "yoloe-26l-seg-pf.pt").write_bytes(b"pt")
    (models / "other-detection.pt").write_bytes(b"pt")

    catalog = discover_yoloe_models(tmp_path, "models/yoloe-26l-seg.pt")
    values = {value for _display, value in catalog}

    assert values == {
        "models/yoloe-26l-seg.pt",
        "models/yoloe-26n-seg.pt",
        "models/yoloe-26s-seg.pt",
    }
    assert all(value.lower().endswith(".pt") for value in values)


def test_openvino_catalog_lists_only_yolo26_official_checkpoints(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo26s.pt").write_bytes(b"pt")
    (models / "yolo26n.pt").write_bytes(b"pt")
    (models / "yolo26l.pt").write_bytes(b"pt")

    catalog = discover_openvino_models(tmp_path, "models/yolo26s.pt")
    assert {value for _display, value in catalog} == {
        "models/yolo26n.pt",
        "models/yolo26s.pt",
    }


def test_prompt_normalization_is_bounded_and_deduplicated() -> None:
    assert normalize_prompts(" person, bottle, PERSON, smartphone ") == (
        "person",
        "bottle",
        "smartphone",
    )
    with pytest.raises(ValueError, match="at most 20"):
        normalize_prompts(",".join(f"item-{index}" for index in range(21)))


def test_yoloe_result_adapter_keeps_labels_boxes_and_mask_polygon() -> None:
    detector = object.__new__(YoloESegmentationDetector)
    result = SimpleNamespace(
        names={0: "person", 1: "bottle"},
        boxes=SimpleNamespace(
            xyxy=np.array([[10.0, 20.0, 80.0, 100.0], [30.0, 40.0, 90.0, 120.0]]),
            conf=np.array([0.91, 0.72]),
            cls=np.array([0.0, 1.0]),
        ),
        masks=SimpleNamespace(
            xy=[
                np.array([[10.0, 20.0], [80.0, 20.0], [80.0, 100.0], [10.0, 100.0]]),
                np.array([[30.0, 40.0], [90.0, 40.0], [90.0, 120.0]]),
            ]
        ),
    )

    detections = detector._detections_from_result(
        result,
        np.zeros((160, 240, 3), dtype=np.uint8),
    )

    assert [item.label for item in detections] == ["person", "bottle"]
    assert detections[0].class_id == 0
    assert detections[0].mask_polygon is not None
    assert detections[1].mask_polygon == ((30.0, 40.0), (90.0, 40.0), (90.0, 120.0))


def test_yoloe_adapter_rejects_onnx_contract(tmp_path: Path) -> None:
    with pytest.raises(PersonDetectionError, match=r"\.pt"):
        YoloESegmentationDetector(
            tmp_path / "model.onnx",
            text_encoder_path=tmp_path / "mobileclip2_b.ts",
        )


def test_person_detection_controller_does_not_load_when_disabled(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []

    def fail_if_loaded(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("disabled detection must not load a model")

    monkeypatch.setattr(controller_module, "create_person_detector", fail_if_loaded)
    controller = PersonDetectionController(
        repo_root=tmp_path,
        settings=PersonDetectionSettings(enabled=False),
    )
    controller.start()
    try:
        assert _wait_until(lambda: controller.snapshot.status is PersonDetectionStatus.DISABLED)
        assert calls == []
    finally:
        controller.stop()


def test_person_detection_uses_existing_provider_without_restarting_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models" / "yoloe-26n-seg.pt"
    model_path.parent.mkdir()
    model_path.write_bytes(b"test model")
    fake_detector = FakePersonDetector([_detection()])
    monkeypatch.setattr(
        controller_module,
        "create_person_detector",
        lambda *_args, **_kwargs: fake_detector,
    )

    first = FakeFrameProvider(_slot("cam_1"), camera_index=0, fps=30)
    second = FakeFrameProvider(_slot("cam_2", 2), camera_index=1, fps=30)
    settings = PersonDetectionSettings(
        enabled=True,
        model="models/yoloe-26n-seg.pt",
        confidence_threshold=0.5,
        inference_fps=20,
        device="auto",
    )
    controller = PersonDetectionController(repo_root=tmp_path, settings=settings)
    first.start()
    second.start()
    controller.start()
    try:
        controller.set_active_camera("cam_1", first)
        assert _wait_until(
            lambda: controller.snapshot.status is PersonDetectionStatus.RUNNING
            and controller.snapshot.person_count == 1
        )
        first_starts = first.start_calls
        second_starts = second.start_calls
        controller.set_active_camera("cam_2", second)
        assert _wait_until(lambda: controller.snapshot.camera_id == "cam_2")
        assert first.start_calls == first_starts
        assert second.start_calls == second_starts
        assert first.snapshot().worker is not None
        assert second.snapshot().worker is not None
        assert fake_detector.calls > 0
    finally:
        controller.stop()
        first.stop(timeout_s=1.0)
        second.stop(timeout_s=1.0)


def test_controller_forwards_openvino_profile_without_local_checkpoint_precheck(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: list[object] = []
    fake_detector = FakePersonDetector()

    def create(config, **_kwargs):
        captured.append(config)
        return fake_detector

    monkeypatch.setattr(controller_module, "create_person_detector", create)
    settings = PersonDetectionSettings(
        enabled=True,
        backend="openvino",
        model="models/yolo26s.pt",
        device="gpu",
        precision="fp16",
        fallback_device="cpu",
        image_size=512,
    )
    controller = PersonDetectionController(repo_root=tmp_path, settings=settings)

    detector, snapshot = controller._load_detector(settings, None, 0)

    assert detector is fake_detector
    assert snapshot is not None and snapshot.status is PersonDetectionStatus.READY
    config = captured[0]
    assert config.backend == "openvino"
    assert config.precision == "fp16"
    assert config.device == "gpu"
    assert config.fallback_device == "cpu"
    assert config.image_size == 512


def test_person_detection_controller_reports_missing_model_without_stopping_stream(
    tmp_path: Path,
) -> None:
    provider = FakeFrameProvider(_slot("cam_1"), camera_index=0, fps=20)
    controller = PersonDetectionController(
        repo_root=tmp_path,
        settings=PersonDetectionSettings(
            enabled=True,
            model="models/not-present.pt",
        ),
    )
    provider.start()
    controller.start()
    try:
        controller.set_active_camera("cam_1", provider)
        assert _wait_until(
            lambda: controller.snapshot.status is PersonDetectionStatus.MODEL_MISSING
        )
        assert provider.snapshot().worker is not None
        assert provider.snapshot().worker.thread_alive is True
    finally:
        controller.stop()
        provider.stop(timeout_s=1.0)


def test_detection_geometry_follows_rotation_and_mirror() -> None:
    source_width, source_height = 640, 360
    bbox = _detection().bbox

    assert transform_detection_bbox(
        bbox,
        source_width,
        source_height,
        CameraDisplayTransform(),
    ) == pytest.approx((100.0, 40.0, 220.0, 300.0))
    assert transform_detection_bbox(
        bbox,
        source_width,
        source_height,
        CameraDisplayTransform(90),
    ) == pytest.approx((40.0, 420.0, 300.0, 540.0))
    assert transform_detection_bbox(
        bbox,
        source_width,
        source_height,
        CameraDisplayTransform(90, True),
    ) == pytest.approx((60.0, 420.0, 320.0, 540.0))
    assert effective_frame_size(source_width, source_height, CameraDisplayTransform(90)) == (
        360,
        640,
    )

    polygon = ((100.0, 40.0), (220.0, 40.0), (220.0, 300.0), (100.0, 300.0))
    assert transform_detection_polygon(
        polygon,
        source_width,
        source_height,
        CameraDisplayTransform(90, True),
    ) == ((320.0, 540.0), (320.0, 420.0), (60.0, 420.0), (60.0, 540.0))

    rendered = map_detection_bbox_to_widget(
        bbox,
        source_width,
        source_height,
        700,
        500,
        CameraDisplayTransform(),
    )
    assert rendered.x == pytest.approx(109.375)
    assert rendered.y == pytest.approx(96.7777778)
    assert rendered.width == pytest.approx(131.25)
    assert rendered.height == pytest.approx(284.5555556, abs=0.01)


def test_focus_renders_person_box_on_the_transformed_video(qapp, qtbot, tmp_path: Path) -> None:
    slot = _slot("cam_1")
    monitor = CameraMonitorController(
        (slot,),
        lambda current: FakeFrameProvider(current, camera_index=0),
        display_fps=15,
        read_timeout_s=0.25,
    )
    focus = CameraFocusView(
        controller=monitor,
        config_path=tmp_path / "config.local.yaml",
        repo_root=tmp_path,
        credentials=InMemoryCredentialStore(),
    )
    qtbot.addWidget(focus)
    focus.resize(900, 600)
    focus.show()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    focus.set_snapshot(
        CameraViewSnapshot(
            slot,
            CameraViewStatus.LIVE,
            "Flusso attivo",
            frame=FramePacket(
                frame=frame,
                sequence=1,
                received_at_utc=utc_now(),
                received_monotonic=time.monotonic(),
                read_duration_ms=0.0,
            ),
        )
    )
    focus.set_person_detection_snapshot(
        PersonDetectionSnapshot(
            camera_id="cam_1",
            status=PersonDetectionStatus.RUNNING,
            message="Inferenza persone attiva",
            settings=PersonDetectionSettings(
                enabled=True,
                model="models/yoloe-26n-seg.pt",
                show_boxes=True,
            ),
            detections=(_detection(),),
            frame_sequence=1,
            source_width=640,
            source_height=360,
            result_monotonic=time.monotonic(),
            person_count=1,
        )
    )
    qapp.processEvents()

    pixmap = focus._video_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    image = pixmap.toImage()
    green_pixels = 0
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if color.green() > 180 and color.green() > color.red() + 30:
                green_pixels += 1
    assert green_pixels > 20


def test_person_detection_panel_exposes_live_controls(qapp, qtbot, tmp_path: Path) -> None:
    config = tmp_path / "config.local.yaml"
    config.write_text(
        "person_detection:\n  enabled: false\n  model: models/yolo26n.onnx\n"
        "  confidence_threshold: 0.5\n  device: auto\ninference:\n  person_detection_fps: 2\n",
        encoding="utf-8",
    )
    controller = PersonDetectionController(repo_root=tmp_path)
    panel = CameraConfigurationPanel(
        CameraMonitorController(
            (_slot("cam_1"),),
            lambda slot: FakeFrameProvider(slot, camera_index=0),
            display_fps=15,
            read_timeout_s=0.25,
        ),
        config_path=config,
        repo_root=tmp_path,
        credentials=InMemoryCredentialStore(),
        person_detection_controller=controller,
    )
    qtbot.addWidget(panel)
    assert panel.person_detection_panel is not None
    detection_panel = panel.person_detection_panel
    assert detection_panel is not None
    assert detection_panel.findChild(QCheckBox, "personDetectionEnabled") is not None
    assert detection_panel.findChild(QComboBox, "personDetectionModel") is not None
    assert detection_panel.findChild(QSlider, "personDetectionConfidence") is not None
    assert detection_panel.findChild(QSpinBox, "personDetectionFps") is not None
    assert detection_panel.findChild(QLineEdit, "personDetectionPrompts") is not None
    assert detection_panel.findChild(QCheckBox, "personDetectionShowMasks") is not None


def test_person_detection_panel_switches_to_openvino_controls(qapp, qtbot, tmp_path: Path) -> None:
    controller = PersonDetectionController(
        repo_root=tmp_path,
        settings=PersonDetectionSettings(
            enabled=False,
            backend="openvino",
            model="models/yolo26s.pt",
            device="gpu",
            precision="fp16",
            fallback_device="cpu",
            image_size=512,
        ),
    )
    panel = PersonDetectionPanel(
        controller,
        repository=CameraConfigRepository(tmp_path),
        config_path=tmp_path / "config.local.yaml",
        repo_root=tmp_path,
    )
    qtbot.addWidget(panel)

    backend = panel.findChild(QComboBox, "personDetectionBackend")
    device = panel.findChild(QComboBox, "personDetectionDevice")
    prompts = panel.findChild(QLineEdit, "personDetectionPrompts")
    masks = panel.findChild(QCheckBox, "personDetectionShowMasks")
    precision = panel.findChild(QComboBox, "personDetectionPrecision")
    fallback = panel.findChild(QComboBox, "personDetectionFallbackDevice")
    image_size = panel.findChild(QSpinBox, "personDetectionImageSize")

    assert backend is not None and backend.currentData() == "openvino"
    assert device is not None and [device.itemData(i) for i in range(device.count())] == [
        "auto",
        "cpu",
        "gpu",
    ]
    assert prompts is not None and prompts.text() == "person" and not prompts.isEnabled()
    assert masks is not None and not masks.isEnabled() and not masks.isChecked()
    assert precision is not None and precision.isEnabled()
    assert fallback is not None and fallback.isEnabled()
    assert image_size is not None and image_size.value() == 512


def test_main_window_wires_focus_analysis_and_keeps_missing_model_separate_from_stream(
    qapp,
    qtbot,
    tmp_path: Path,
) -> None:
    slots = (
        _slot("cam_1"),
        *(
            CameraSlot(index, f"slot_{index}", f"Camera {index}", False, False, None)
            for index in range(2, 7)
        ),
    )
    monitor = CameraMonitorController(
        slots,
        lambda slot: FakeFrameProvider(slot, camera_index=slot.slot_index - 1),
        display_fps=15,
        read_timeout_s=0.25,
    )
    detection = PersonDetectionController(
        repo_root=tmp_path,
        settings=PersonDetectionSettings(
            enabled=True,
            model="models/missing.pt",
        ),
    )
    window = MainWindow(
        slots,
        monitor,
        ui_settings=UiSettings(start_maximized=False),
        config_path=tmp_path / "config.local.yaml",
        repo_root=tmp_path,
        credentials=InMemoryCredentialStore(),
        person_detection_controller=detection,
    )
    qtbot.addWidget(window)
    window.show()
    window.show_focus("cam_1")
    window.start_monitoring()
    try:
        qtbot.waitUntil(
            lambda: detection.snapshot.status is PersonDetectionStatus.MODEL_MISSING,
            timeout=3000,
        )
        qtbot.waitUntil(
            lambda: (
                window.focus_view.person_detection_snapshot is not None
                and window.focus_view.person_detection_snapshot.status
                is PersonDetectionStatus.MODEL_MISSING
            ),
            timeout=3000,
        )
        assert window.focus_view.configuration_panel.person_detection_panel is not None
        assert monitor.provider_for("cam_1") is not None
        assert monitor.provider_for("cam_1").start_calls == 1
        assert window.focus_view.person_detection_snapshot is not None
        assert window.focus_view.person_detection_snapshot.status is PersonDetectionStatus.MODEL_MISSING
    finally:
        window.close()


def test_person_detection_settings_persist_without_touching_camera_or_custom_sections(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.local.yaml"
    config.write_text(
        "cameras:\n  - id: cam_1\n    stream_url: fake://cam_1/live\n"
        "person_detection:\n  enabled: false\n"
        "custom_section:\n  keep: true\n",
        encoding="utf-8",
    )
    settings = PersonDetectionSettings(
        enabled=True,
        model="models/yoloe-26n-seg.pt",
        confidence_threshold=0.65,
        inference_fps=6,
        device="cpu",
        show_boxes=False,
        prompts=("person", "bottle"),
        show_masks=True,
    )
    result = CameraConfigRepository(tmp_path).save_person_detection(
        settings,
        current_path=config,
    )
    rendered = result.read_text(encoding="utf-8")
    assert "id: cam_1" in rendered
    assert "keep: true" in rendered
    assert "enabled: true" in rendered
    assert "backend: yoloe" in rendered
    assert "confidence_threshold: 0.65" in rendered
    assert "person_detection_fps: 6.0" in rendered
    assert "show_person_boxes: false" in rendered
    assert "prompts:" in rendered
    assert "- bottle" in rendered
    assert "show_masks: true" in rendered


def test_legacy_onnx_config_migrates_to_local_yoloe_without_touching_example(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "yoloe-26n-seg.pt").write_bytes(b"pt")
    example = tmp_path / "config" / "config.example.yaml"
    example.parent.mkdir()
    example.write_text(
        "cameras:\n  - id: cam_1\n    stream_url: fake://cam_1/live\n"
        "person_detection:\n  enabled: true\n  model: models/yolo26n.onnx\n"
        "custom_section:\n  keep: true\n",
        encoding="utf-8",
    )

    repository = CameraConfigRepository(tmp_path)
    target = repository.migrate_person_detection(current_path=example)

    assert target == tmp_path / "config" / "config.local.yaml"
    rendered = target.read_text(encoding="utf-8")
    assert "model: models/yoloe-26n-seg.pt" in rendered
    assert "- person" in rendered
    assert "show_masks: false" in rendered
    assert "keep: true" in rendered
    assert "model: models/yolo26n.onnx" in example.read_text(encoding="utf-8")

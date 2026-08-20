from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import yaml

from PySide6.QtWidgets import QSplitter, QToolBar

from app.config import AppConfig, CameraConfig, ConfigurationError
from app.video.base import (
    FramePacket,
    ReadResult,
    ReadStatus,
    StreamInfo,
    VideoSource,
    utc_now,
    redact_url,
)

from app_windows.config.camera_config import (
    CameraDraft,
    draft_from_slot,
    parse_camera_url,
    runtime_stream_url,
    validate_camera_draft,
)
from app_windows.config.connection_test import (
    AsyncConnectionTester,
    ConnectionTestResult,
)
from app_windows.config.credentials import DpapiCredentialStore, InMemoryCredentialStore
from app_windows.config.persistence import CameraConfigRepository
from app_windows.config.ui_config import UiSettings
from app_windows.models.camera_view_state import camera_slots_from_config
from app_windows.monitor_controller import CameraMonitorController
from app_windows.ui.camera_configuration_dialog import CameraConfigurationDialog
from app_windows.ui.main_window import MainWindow
from app_windows.video.fake_provider import FakeFrameProvider, fake_connection_source_factory


def _write_example(root: Path) -> Path:
    path = root / "config" / "config.example.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """cameras:
  - id: huawei
    name: Huawei P30 Pro
    enabled: true
    source_type: opencv
    stream_url: rtsp://utente:old-secret@192.168.1.20:8554/stream
  - id: cam_2
    name: Camera 2
    enabled: false
    source_type: opencv
    stream_url: ${CAMERA_CAM_2_URL}
video:
  rtsp_transport: tcp
windows_ui:
  display_fps: 17
  start_maximized: false
custom_section:
  keep: true
""",
        encoding="utf-8",
    )
    return path


def _slots() -> tuple:
    config = AppConfig(
        cameras=[
            CameraConfig(
                id="huawei",
                name="Huawei P30 Pro",
                enabled=True,
                stream_url="rtsp://utente:secret@192.168.1.20:8554/stream",
            )
        ]
    )
    return camera_slots_from_config(config)


def test_existing_values_load_without_exposing_password() -> None:
    slot = _slots()[0]
    credentials = InMemoryCredentialStore({"huawei": "secret"})
    draft = draft_from_slot(slot, credentials)

    assert draft.camera_id == "huawei"
    assert draft.name == "Huawei P30 Pro"
    assert draft.host == "192.168.1.20"
    assert draft.port == 8554
    assert draft.path == "/stream"
    assert draft.username == "utente"
    assert draft.password == ""
    assert draft.password_is_stored is True
    assert "secret" not in repr(draft)

    validated = validate_camera_draft(draft)
    assert validated.stream_url == "rtsp://utente@192.168.1.20:8554/stream"
    assert "secret" not in validated.stream_url
    assert "secret" not in redact_url(
        runtime_stream_url(validated.stream_url, validated.credential_value)
    )


def test_fallback_names_and_transport_are_central() -> None:
    config = AppConfig(
        cameras=[
            CameraConfig(
                id="cam_1",
                name=None,
                enabled=False,
                stream_url=None,
            )
        ],
        video={"rtsp_transport": "tcp"},
    )
    slots = camera_slots_from_config(config)
    assert slots[0].name == "Camera 1"
    assert slots[0].rtsp_transport == "tcp"
    assert slots[1].name == "Camera 2"
    assert slots[1].camera_id == "slot_2"


@pytest.mark.parametrize(
    "draft",
    [
        CameraDraft("cam", 1, "Camera", True, host="", path="/stream"),
        CameraDraft("cam", 1, "Camera", True, host="camera", port=0, path="/stream"),
        CameraDraft("cam", 1, "Camera", True, host="camera", port=70000, path="/stream"),
        CameraDraft("cam", 1, "Camera", True, host="camera", path=""),
        CameraDraft("cam", 1, "Camera", True, host="camera", path="/stream", password="pw"),
    ],
)
def test_camera_editor_rejects_invalid_endpoint_fields(draft: CameraDraft) -> None:
    with pytest.raises(ConfigurationError):
        validate_camera_draft(draft)


def test_camera_editor_accepts_tcp_and_ipv6() -> None:
    draft = CameraDraft(
        "cam",
        1,
        "Camera",
        True,
        host="2001:db8::20",
        port=8554,
        path="stream",
        username="user",
        password="secret",
        transport="tcp",
    )
    validated = validate_camera_draft(draft)
    assert validated.draft.transport == "tcp"
    assert validated.stream_url == "rtsp://user@[2001:db8::20]:8554/stream"


def test_repository_creates_local_config_preserves_other_sections_and_encrypts_secret(tmp_path: Path) -> None:
    example = _write_example(tmp_path)
    credentials = InMemoryCredentialStore({"huawei": "old-secret"})
    repository = CameraConfigRepository(tmp_path)
    original = yaml.safe_load(example.read_text(encoding="utf-8"))
    slot = _slots()[0]
    draft = replace(
        draft_from_slot(slot, credentials),
        host="192.168.1.21",
        password="new-secret",
    )

    result = repository.save(
        [draft],
        current_path=example,
        credentials=credentials,
    )

    assert result.path == tmp_path / "config" / "config.local.yaml"
    assert credentials.get("huawei") == "new-secret"
    saved_text = result.path.read_text(encoding="utf-8")
    assert "new-secret" not in saved_text
    assert "old-secret" not in saved_text
    saved = yaml.safe_load(saved_text)
    assert saved["windows_ui"] == original["windows_ui"]
    assert saved["custom_section"] == original["custom_section"]
    assert saved["cameras"][0]["stream_url"] == "rtsp://utente@192.168.1.21:8554/stream"
    assert saved["cameras"][1] == original["cameras"][1]


def test_repository_seeds_missing_local_config_from_example(tmp_path: Path) -> None:
    _write_example(tmp_path)
    local = tmp_path / "config" / "config.local.yaml"
    credentials = InMemoryCredentialStore()
    draft = replace(
        draft_from_slot(_slots()[0], credentials),
        name="Prima camera",
    )

    result = CameraConfigRepository(tmp_path).save(
        [draft],
        current_path=local,
        credentials=credentials,
    )

    assert result.path == local
    assert yaml.safe_load(local.read_text(encoding="utf-8"))["windows_ui"]["display_fps"] == 17


def test_repository_atomic_failure_keeps_existing_file_and_credentials(tmp_path: Path, monkeypatch) -> None:
    import app_windows.config.persistence as persistence

    example = _write_example(tmp_path)
    local = tmp_path / "config" / "config.local.yaml"
    local.write_bytes(example.read_bytes())
    before = local.read_bytes()
    credentials = InMemoryCredentialStore({"huawei": "old-secret"})
    draft = replace(
        draft_from_slot(_slots()[0], credentials),
        host="192.168.1.22",
        password="new-secret",
    )

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(persistence.os, "replace", fail_replace)
    with pytest.raises(ConfigurationError):
        CameraConfigRepository(tmp_path).save(
            [draft],
            current_path=local,
            credentials=credentials,
        )

    assert local.read_bytes() == before
    assert credentials.get("huawei") == "old-secret"
    assert not list(local.parent.glob(f".{local.name}.*.tmp"))


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is Windows-only")
def test_dpapi_store_round_trips_without_plaintext_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "config.local.secrets.json"
    store = DpapiCredentialStore(path)
    store.apply({"huawei": "secret-value"})
    assert store.get("huawei") == "secret-value"
    assert "secret-value" not in path.read_text(encoding="utf-8")
    store.apply({"huawei": None})
    assert not path.exists()


class _SpySource(VideoSource):
    def __init__(self, *, timeout: bool = False, delay_s: float = 0.0) -> None:
        self.timeout = timeout
        self.delay_s = delay_s
        self.closed = False
        self.opened = False

    def open(self) -> StreamInfo:
        self.opened = True
        return StreamInfo(
            url="rtsp://user:***@camera.local/stream",
            backend="fake",
            width=32,
            height=24,
            declared_fps=30.0,
            codec="fake",
            opened_at_utc=utc_now(),
        )

    def read(self, timeout_s: float) -> ReadResult:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.timeout:
            return ReadResult.status_result(ReadStatus.TIMEOUT, "test timeout")
        packet = FramePacket(
            frame=np.zeros((2, 2, 3), dtype=np.uint8),
            sequence=1,
            received_at_utc=utc_now(),
            received_monotonic=time.monotonic(),
            read_duration_ms=1.0,
        )
        return ReadResult.frame_result(packet)

    def reconnect(self) -> StreamInfo:
        return self.open()

    def close(self) -> None:
        self.closed = True


def test_async_connection_test_uses_source_and_always_closes(qtbot) -> None:
    source = _SpySource()
    tester = AsyncConnectionTester(
        lambda _url, _transport: source,
        read_timeout_s=0.1,
    )
    with qtbot.waitSignal(tester.finished, timeout=3000) as blocker:
        tester.start("cam", "rtsp://user:secret@camera.local/stream", "tcp")
    result = blocker.args[0]
    assert isinstance(result, ConnectionTestResult)
    assert result.success is True
    assert source.opened is True
    assert source.closed is True
    assert "secret" not in result.url


def test_async_connection_timeout_does_not_block_qt_and_closes_source(qtbot) -> None:
    source = _SpySource(timeout=True, delay_s=0.05)
    tester = AsyncConnectionTester(
        lambda _url, _transport: source,
        read_timeout_s=0.01,
    )
    ticks: list[int] = []
    from PySide6.QtCore import QTimer

    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(1)
    try:
        with qtbot.waitSignal(tester.finished, timeout=3000) as blocker:
            tester.start("cam", "rtsp://camera.local/stream", "tcp")
    finally:
        timer.stop()
    result = blocker.args[0]
    assert result.success is False
    assert "Connessione scaduta" in result.message
    assert source.closed is True
    assert ticks


def test_active_provider_is_reused_for_connection_test(qtbot) -> None:
    called = []
    tester = AsyncConnectionTester(
        lambda _url, _transport: called.append(True),
        read_timeout_s=0.1,
        existing_probe=lambda _camera: (True, True, "Connessione riuscita (provider attivo)"),
    )
    with qtbot.waitSignal(tester.finished, timeout=1000) as blocker:
        tester.start("cam", "rtsp://camera.local/stream", "tcp")
    assert blocker.args[0].success is True
    assert called == []


def test_controller_applies_only_one_camera_without_restarting_others(qtbot) -> None:
    slots = tuple(
        replace(_slots()[0], camera_id=f"cam_{index}", slot_index=index, name=f"Camera {index}")
        if index == 1
        else type(_slots()[0])(
            index,
            f"cam_{index}",
            f"Camera {index}",
            True,
            True,
            f"fake://cam_{index}/live",
        )
        for index in range(1, 4)
    )
    created: dict[str, list[FakeFrameProvider]] = {slot.camera_id: [] for slot in slots}

    def factory(slot):
        provider = FakeFrameProvider(slot, camera_index=slot.slot_index - 1, fps=30)
        created[slot.camera_id].append(provider)
        return provider

    controller = CameraMonitorController(slots, factory, display_fps=20, read_timeout_s=0.25)
    controller.start()
    try:
        qtbot.waitUntil(
            lambda: all(controller.snapshot_for(slot.camera_id).frame is not None for slot in slots),
            timeout=4000,
        )
        old_providers = dict(controller.providers)
        updated = replace(slots[0], name="Camera 1 aggiornata", stream_url="fake://cam_1/new")
        with qtbot.waitSignal(controller.camera_reconfigured, timeout=3000):
            controller.apply_camera_slot(updated)
        assert controller.slots[0].name == "Camera 1 aggiornata"
        assert controller.providers["cam_2"] is old_providers["cam_2"]
        assert controller.providers["cam_3"] is old_providers["cam_3"]
        assert controller.providers["cam_1"] is not old_providers["cam_1"]
        assert old_providers["cam_1"].snapshot().worker.thread_alive is False
        assert old_providers["cam_2"].start_calls == 1
        assert old_providers["cam_3"].start_calls == 1
    finally:
        controller.stop(timeout_s=1.0)


def test_dialog_has_six_editors_masks_password_and_cancel_preserves_file(qtbot, tmp_path: Path) -> None:
    example = _write_example(tmp_path)
    slots = _slots()
    controller = CameraMonitorController(
        slots,
        lambda slot: FakeFrameProvider(slot, camera_index=slot.slot_index - 1),
        display_fps=15,
        read_timeout_s=0.25,
    )
    credentials = InMemoryCredentialStore({"huawei": "secret"})
    dialog = CameraConfigurationDialog(
        slots,
        controller,
        config_path=example,
        repo_root=tmp_path,
        credentials=credentials,
        source_factory=fake_connection_source_factory,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    before = example.read_bytes()

    assert len(dialog.cards) == 6
    assert dialog.cards["huawei"].password_edit.echoMode().name == "Password"
    assert "secret" not in dialog.cards["huawei"].url_preview.text()
    assert dialog.cards["slot_2"].name_edit.text() == "Camera 2"

    dialog.cards["huawei"].host_edit.setText("192.168.1.30")
    dialog.reject()
    assert example.read_bytes() == before


def test_dialog_save_applies_changed_slot_when_monitor_not_started(qtbot, tmp_path: Path) -> None:
    example = _write_example(tmp_path)
    slots = _slots()
    controller = CameraMonitorController(
        slots,
        lambda slot: FakeFrameProvider(slot, camera_index=slot.slot_index - 1),
        display_fps=15,
        read_timeout_s=0.25,
    )
    dialog = CameraConfigurationDialog(
        slots,
        controller,
        config_path=example,
        repo_root=tmp_path,
        credentials=InMemoryCredentialStore({"huawei": "secret"}),
        source_factory=fake_connection_source_factory,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.cards["huawei"].name_edit.setText("Huawei modificato")
    dialog.save_and_apply()
    qtbot.waitUntil(lambda: not dialog.isVisible(), timeout=1000)

    local = tmp_path / "config" / "config.local.yaml"
    assert local.is_file()
    assert "secret" not in local.read_text(encoding="utf-8")
    assert controller.slots[0].name == "Huawei modificato"


def test_main_window_has_no_global_camera_configuration_toolbar(qtbot) -> None:
    slots = _slots()
    controller = CameraMonitorController(
        slots,
        lambda slot: FakeFrameProvider(slot, camera_index=slot.slot_index - 1),
        display_fps=15,
        read_timeout_s=0.25,
    )
    window = MainWindow(
        slots,
        controller,
        ui_settings=UiSettings(start_maximized=False),
        connection_source_factory=fake_connection_source_factory,
    )
    qtbot.addWidget(window)
    window.show()
    assert window.findChild(QToolBar, "MonitorToolbar") is None
    assert window.findChild(QToolBar) is None
    window.show_focus("huawei")
    assert window.focus_view.configuration_panel.camera_id == "huawei"
    window.close()


def test_focus_uses_only_selected_camera_and_collapsible_splitter(qtbot) -> None:
    slots = _slots()
    controller = CameraMonitorController(
        slots,
        lambda slot: FakeFrameProvider(slot, camera_index=slot.slot_index - 1),
        display_fps=15,
        read_timeout_s=0.25,
    )
    window = MainWindow(
        slots,
        controller,
        ui_settings=UiSettings(start_maximized=False),
        connection_source_factory=fake_connection_source_factory,
    )
    qtbot.addWidget(window)
    window.show()

    window.show_focus("huawei")
    panel = window.focus_view.configuration_panel
    assert panel.camera_id == "huawei"
    assert panel.card is not None
    assert panel.card.camera_id == "huawei"

    splitter = window.focus_view.findChild(QSplitter, "FocusSplitter")
    assert splitter is not None
    assert splitter.isCollapsible(0) is False
    assert splitter.isCollapsible(1) is True

    window.show_focus("slot_2")
    assert panel.camera_id == "slot_2"
    assert panel.card is not None
    assert panel.card.camera_id == "slot_2"
    window.close()


def test_focus_configuration_applies_one_valid_camera_edit_after_debounce(
    qtbot,
    tmp_path: Path,
) -> None:
    example = _write_example(tmp_path)
    slots = _slots()
    controller = CameraMonitorController(
        slots,
        lambda slot: FakeFrameProvider(slot, camera_index=slot.slot_index - 1),
        display_fps=15,
        read_timeout_s=0.25,
    )
    window = MainWindow(
        slots,
        controller,
        ui_settings=UiSettings(start_maximized=False),
        config_path=example,
        repo_root=tmp_path,
        credentials=InMemoryCredentialStore({"huawei": "secret"}),
        connection_source_factory=fake_connection_source_factory,
    )
    qtbot.addWidget(window)
    window.show()
    window.show_focus("huawei")

    panel = window.focus_view.configuration_panel
    assert panel.card is not None
    panel.card.name_edit.setText("Huawei auto applicato")
    qtbot.waitUntil(
        lambda: controller.slots[0].name == "Huawei auto applicato",
        timeout=2500,
    )

    local = tmp_path / "config" / "config.local.yaml"
    assert local.is_file()
    rendered = local.read_text(encoding="utf-8")
    assert "Huawei auto applicato" in rendered
    assert "secret" not in rendered

    original_stream = controller.slots[0].stream_url
    panel.card.host_edit.setText("")
    qtbot.wait(700)
    assert controller.slots[0].stream_url == original_stream
    window.close()

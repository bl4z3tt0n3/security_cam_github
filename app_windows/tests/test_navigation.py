from __future__ import annotations

from dataclasses import replace
import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap

from app.config import AppConfig, CameraConfig
from app.video.base import FramePacket, utc_now
from app.video.worker import CameraWorkerSnapshot, WorkerState

from app_windows.config.ui_config import UiSettings
from app_windows.models.camera_view_state import CameraSlot, CameraViewStatus, camera_slots_from_config
from app_windows.monitor_controller import CameraMonitorController
from app_windows.ui.camera_focus_view import scale_video_pixmap
from app_windows.ui.main_window import MainWindow
from app_windows.video.fake_provider import FakeFrameProvider, fake_camera_factory
from app_windows.video.frame_provider import ProviderSnapshot


def test_grid_has_six_tiles_and_focus_does_not_restart_provider(qtbot) -> None:
    config = AppConfig(
        cameras=[
            CameraConfig(
                id="cam_1",
                name="Ingresso",
                enabled=True,
                stream_url="rtsp://camera.local/live",
            )
        ]
    )
    slots = camera_slots_from_config(config)
    runtime_slots = tuple(slot.with_runtime_source(f"fake://{slot.camera_id}") for slot in slots)
    provider_factory = fake_camera_factory()
    controller = CameraMonitorController(
        runtime_slots,
        provider_factory,
        display_fps=15,
        read_timeout_s=0.25,
    )
    window = MainWindow(
        runtime_slots,
        controller,
        ui_settings=UiSettings(start_maximized=False),
        simulation=True,
    )
    qtbot.addWidget(window)
    window.show()
    controller.start()
    qtbot.waitUntil(lambda: controller.snapshot_for("cam_1").frame is not None, timeout=3000)

    assert len(window.grid.tiles) == 6
    grid_layout = window.grid.layout()
    assert grid_layout is not None
    assert [grid_layout.getItemPosition(index)[:2] for index in range(6)] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert "SIMULAZIONE" in window.windowTitle()
    assert "SIMULAZIONE" in window.statusBar().currentMessage()
    provider = controller.providers["cam_1"]
    start_calls = provider.start_calls

    qtbot.mouseClick(window.grid.tiles["cam_1"], Qt.MouseButton.LeftButton)
    assert window.focused_camera_id == "cam_1"
    assert provider.start_calls == start_calls

    qtbot.keyClick(window.focus_view, Qt.Key.Key_Escape)
    assert window.focused_camera_id is None
    assert provider.start_calls == start_calls

    window.close()


def test_focus_video_keeps_source_aspect_ratio_with_black_letterboxing(qtbot) -> None:
    slots = tuple(
        CameraSlot(index, f"cam_{index}", f"Camera {index}", False, False, None)
        for index in range(1, 7)
    )
    controller = CameraMonitorController(
        slots,
        fake_camera_factory(),
        display_fps=15,
        read_timeout_s=0.25,
    )
    window = MainWindow(
        slots,
        controller,
        ui_settings=UiSettings(start_maximized=False),
    )
    qtbot.addWidget(window)
    window.show()

    source = QPixmap(QSize(640, 360))
    scaled = scale_video_pixmap(source, QSize(400, 300))
    assert scaled.size() == QSize(400, 225)
    assert window.focus_view._video_label.hasScaledContents() is False
    assert window.focus_view._video_label.alignment() == Qt.AlignmentFlag.AlignCenter
    assert "#000000" in window.focus_view.styleSheet()
    window.close()


def test_window_close_stops_all_active_fake_workers(qtbot) -> None:
    slots = tuple(
        CameraSlot(index, f"cam_{index}", f"Camera {index}", True, True, f"fake://cam_{index}")
        for index in range(1, 7)
    )
    controller = CameraMonitorController(
        slots,
        fake_camera_factory(),
        display_fps=15,
        read_timeout_s=0.25,
    )
    window = MainWindow(
        slots,
        controller,
        ui_settings=UiSettings(start_maximized=False),
    )
    qtbot.addWidget(window)
    window.show()
    controller.start()
    qtbot.waitUntil(
        lambda: all(
            controller.snapshot_for(slot.camera_id).frame is not None for slot in slots
        ),
        timeout=5000,
    )
    providers = tuple(controller.providers.values())

    window.close()

    assert providers
    assert all(provider.snapshot().worker.thread_alive is False for provider in providers)


def test_controller_maps_one_offline_camera_without_stopping_another(qtbot) -> None:
    slots = (
        CameraSlot(1, "healthy", "Healthy", True, True, "fake://healthy"),
        CameraSlot(2, "offline", "Offline", True, True, "fake://offline"),
    )

    def provider_factory(slot: CameraSlot):
        return FakeFrameProvider(
            slot,
            camera_index=slot.slot_index - 1,
            fps=20,
            offline=slot.camera_id == "offline",
        )

    controller = CameraMonitorController(
        slots,
        provider_factory,
        display_fps=20,
        read_timeout_s=0.25,
    )
    controller.start()
    try:
        qtbot.waitUntil(
            lambda: (
                controller.snapshot_for("healthy").status is CameraViewStatus.LIVE
                and controller.snapshot_for("offline").status is CameraViewStatus.OFFLINE
            ),
            timeout=4000,
        )
        assert controller.snapshot_for("healthy").worker_snapshot.thread_alive is True
    finally:
        controller.stop(timeout_s=1.0)


def test_controller_maps_offline_to_reconnecting_to_live_from_backend_state() -> None:
    slot = CameraSlot(1, "android-huawei", "Huawei", True, True, "rtsp://phone/stream")
    controller = CameraMonitorController(
        (slot,),
        fake_camera_factory(),
        display_fps=20,
        read_timeout_s=0.25,
    )

    def worker(state: WorkerState) -> CameraWorkerSnapshot:
        return CameraWorkerSnapshot(
            camera_id=slot.camera_id,
            state=state,
            frames_received=0,
            actual_fps=0.0,
            dropped_frames=0,
            reconnect_count=1,
            successful_reconnects=0,
            failed_reconnects=1,
            queue_size=0,
            max_buffer_frames=1,
            last_received_at_utc=None,
            started_at_utc=utc_now(),
            last_error="source offline",
            thread_alive=True,
        )

    offline = controller._build_snapshot(
        slot,
        ProviderSnapshot(worker=worker(WorkerState.FAILED), stream_info=None, last_error="source offline"),
        now=time.monotonic(),
        packet=None,
    )
    reconnecting = controller._build_snapshot(
        slot,
        ProviderSnapshot(worker=worker(WorkerState.RECONNECTING), stream_info=None, last_error="source offline"),
        now=time.monotonic(),
        packet=None,
    )
    live = controller._build_snapshot(
        slot,
        ProviderSnapshot(worker=worker(WorkerState.RUNNING), stream_info=None),
        now=time.monotonic(),
        packet=FramePacket(
            frame=object(),
            sequence=1,
            received_at_utc=utc_now(),
            received_monotonic=time.monotonic(),
            read_duration_ms=0.0,
        ),
    )

    assert [offline.status, reconnecting.status, live.status] == [
        CameraViewStatus.OFFLINE,
        CameraViewStatus.RECONNECTING,
        CameraViewStatus.LIVE,
    ]


def test_controller_polling_does_not_stop_provider_during_reconnect(qtbot) -> None:
    slot = CameraSlot(1, "reconnecting", "Reconnecting", True, True, "fake://reconnecting")
    created: list[FakeFrameProvider] = []

    def provider_factory(current_slot: CameraSlot) -> FakeFrameProvider:
        provider = FakeFrameProvider(
            current_slot,
            camera_index=0,
            fps=30.0,
            fail_after_frames=1,
        )
        created.append(provider)
        return provider

    controller = CameraMonitorController(
        (slot,),
        provider_factory,
        display_fps=20,
        read_timeout_s=0.25,
    )
    controller.start()
    try:
        qtbot.waitUntil(
            lambda: created and created[0].snapshot().worker.reconnect_count > 0,
            timeout=3000,
        )
        provider = created[0]
        assert provider.snapshot().worker.thread_alive is True
        qtbot.waitUntil(
            lambda: (
                controller.snapshot_for("reconnecting").status is CameraViewStatus.LIVE
                and provider.snapshot().worker.successful_reconnects > 0
            ),
            timeout=3000,
        )
        assert provider.snapshot().worker.thread_alive is True
    finally:
        controller.stop(timeout_s=1.0)


def test_controller_does_not_create_providers_for_disabled_or_unconfigured_slots(qtbot) -> None:
    slots = (
        CameraSlot(1, "disabled", "Disabled", False, True, "fake://disabled"),
        CameraSlot(2, "missing", "Missing", True, False, None),
    )
    created: list[str] = []

    def provider_factory(slot: CameraSlot):
        created.append(slot.camera_id)
        return FakeFrameProvider(slot, camera_index=slot.slot_index - 1)

    controller = CameraMonitorController(
        slots,
        provider_factory,
        display_fps=15,
        read_timeout_s=0.25,
    )
    controller.start()
    try:
        qtbot.wait(50)
        assert created == []
        assert controller.snapshot_for("disabled").status is CameraViewStatus.DISABLED
        assert controller.snapshot_for("missing").status is CameraViewStatus.NOT_CONFIGURED
    finally:
        controller.stop(timeout_s=1.0)


def test_disabling_camera_stops_provider_during_reconnect(qtbot) -> None:
    slot = CameraSlot(1, "reconnecting", "Reconnecting", True, True, "fake://reconnecting")
    created: list[FakeFrameProvider] = []

    def provider_factory(current_slot: CameraSlot) -> FakeFrameProvider:
        provider = FakeFrameProvider(
            current_slot,
            camera_index=0,
            fps=30.0,
            fail_after_frames=1,
        )
        created.append(provider)
        return provider

    controller = CameraMonitorController(
        (slot,),
        provider_factory,
        display_fps=20,
        read_timeout_s=0.25,
    )
    controller.start()
    try:
        qtbot.waitUntil(
            lambda: created and created[0].snapshot().worker.reconnect_count > 0,
            timeout=3000,
        )
        old_provider = created[0]
        disabled = replace(slot, enabled=False)
        with qtbot.waitSignal(controller.camera_reconfigured, timeout=3000):
            controller.apply_camera_slot(disabled)

        assert controller.snapshot_for("reconnecting").status is CameraViewStatus.DISABLED
        assert "reconnecting" not in controller.providers
        assert old_provider.snapshot().worker.thread_alive is False
    finally:
        controller.stop(timeout_s=1.0)

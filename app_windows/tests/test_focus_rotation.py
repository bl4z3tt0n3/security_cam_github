from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication

from app.video.base import FramePacket, utc_now
from app_windows.config.credentials import InMemoryCredentialStore
from app_windows.config.ui_config import UiSettings
from app_windows.models.camera_display_transform import CameraDisplayTransform
from app_windows.models.camera_view_state import (
    CameraSlot,
    CameraViewSnapshot,
    CameraViewStatus,
)
from app_windows.monitor_controller import CameraMonitorController
from app_windows.ui.camera_focus_view import CameraFocusView
from app_windows.ui.main_window import MainWindow
from app_windows.video.display_transform import (
    mirror_video_pixmap,
    rotate_video_pixmap_counterclockwise,
)
from app_windows.video.fake_provider import fake_camera_factory


def test_rotate_video_pixmap_counterclockwise_rotates_the_image_left(qapp) -> None:
    image = QImage(2, 3, QImage.Format.Format_RGB32)
    image.fill(QColor("black"))
    image.setPixelColor(0, 0, QColor("red"))
    image.setPixelColor(1, 0, QColor("green"))

    rotated = rotate_video_pixmap_counterclockwise(QPixmap.fromImage(image), 90).toImage()

    assert rotated.width() == 3
    assert rotated.height() == 2
    assert rotated.pixelColor(0, 1) == QColor("red")
    assert rotated.pixelColor(0, 0) == QColor("green")


def test_mirror_video_pixmap_flips_the_image_horizontally(qapp) -> None:
    image = QImage(2, 1, QImage.Format.Format_RGB32)
    image.setPixelColor(0, 0, QColor("red"))
    image.setPixelColor(1, 0, QColor("green"))

    mirrored = mirror_video_pixmap(QPixmap.fromImage(image), True).toImage()

    assert mirrored.pixelColor(0, 0) == QColor("green")
    assert mirrored.pixelColor(1, 0) == QColor("red")


def test_focus_rotation_button_advances_by_a_quarter_turn_each_click(
    qapp,
    qtbot,
    tmp_path,
) -> None:
    slot = CameraSlot(1, "cam_1", "Camera 1", True, True, "fake://cam_1")
    controller = CameraMonitorController(
        (slot,),
        fake_camera_factory(),
        display_fps=15,
        read_timeout_s=0.25,
    )
    focus = CameraFocusView(
        controller=controller,
        config_path=tmp_path / "config.local.yaml",
        repo_root=tmp_path,
        credentials=InMemoryCredentialStore(),
    )
    qtbot.addWidget(focus)
    focus.resize(900, 600)
    focus.show()
    focus.set_snapshot(
        CameraViewSnapshot(slot, CameraViewStatus.LIVE, "Flusso attivo")
    )

    assert focus._rotation_button.isEnabled()
    assert focus._rotation_button.parentWidget() is focus._video_controls
    assert focus._mirror_button.parentWidget() is focus._video_controls
    assert focus._video_controls.parentWidget() is focus._video_surface
    assert focus.rotation_degrees == 0
    assert focus.is_mirrored is False

    image = QImage(640, 360, QImage.Format.Format_RGB32)
    image.fill(QColor("black"))
    focus._image = image
    focus._render_image()
    qapp.processEvents()
    for button in (focus._rotation_button, focus._mirror_button):
        top_widget = QApplication.widgetAt(button.mapToGlobal(button.rect().center()))
        assert top_widget is button

    for expected_rotation in (90, 180, 270, 0):
        qtbot.mouseClick(focus._rotation_button, Qt.MouseButton.LeftButton)
        assert focus.rotation_degrees == expected_rotation

    qtbot.mouseClick(focus._mirror_button, Qt.MouseButton.LeftButton)
    assert focus._mirror_button.isChecked()
    assert focus.is_mirrored is True

    qtbot.mouseClick(focus._mirror_button, Qt.MouseButton.LeftButton)
    assert focus._mirror_button.isChecked() is False
    assert focus.is_mirrored is False


def test_focus_controls_update_the_main_grid_for_only_the_selected_camera(
    qapp,
    qtbot,
    tmp_path,
) -> None:
    slots = (
        CameraSlot(1, "cam_1", "Camera 1", True, True, "fake://cam_1"),
        *(
            CameraSlot(index, f"cam_{index}", f"Camera {index}", False, False, None)
            for index in range(2, 7)
        ),
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
        config_path=tmp_path / "config.local.yaml",
        repo_root=tmp_path,
        credentials=InMemoryCredentialStore(),
    )
    qtbot.addWidget(window)
    window.resize(1200, 720)
    window.show()

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :320] = (0, 0, 255)
    frame[:, 320:] = (0, 255, 0)
    snapshot = CameraViewSnapshot(
        slots[0],
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
    grid_tile = window.grid.tiles["cam_1"]
    window.grid.set_snapshot(snapshot)
    window.focus_view.set_snapshot(snapshot)
    qapp.processEvents()

    initial = grid_tile._video_label.pixmap()
    assert initial is not None and not initial.isNull()
    initial_image = initial.toImage()
    middle_y = initial_image.height() // 2
    assert initial_image.pixelColor(initial_image.width() // 4, middle_y).red() > 200
    assert initial_image.pixelColor(initial_image.width() * 3 // 4, middle_y).green() > 200

    window._stack.setCurrentWidget(window.focus_view)
    qapp.processEvents()
    qtbot.mouseClick(window.focus_view._mirror_button, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    mirrored = grid_tile._video_label.pixmap()
    assert mirrored is not None and not mirrored.isNull()
    mirrored_image = mirrored.toImage()
    middle_y = mirrored_image.height() // 2
    assert mirrored_image.pixelColor(mirrored_image.width() // 4, middle_y).green() > 200
    assert mirrored_image.pixelColor(mirrored_image.width() * 3 // 4, middle_y).red() > 200
    assert grid_tile.display_transform == CameraDisplayTransform(mirrored=True)
    assert window.grid.tiles["cam_2"].display_transform == CameraDisplayTransform()

    qtbot.mouseClick(window.focus_view._rotation_button, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    rotated = grid_tile._video_label.pixmap()
    assert rotated is not None and not rotated.isNull()
    assert rotated.height() > rotated.width()
    assert grid_tile.display_transform == CameraDisplayTransform(90, True)
    assert window.focus_view.display_transform == grid_tile.display_transform

    window.show_grid()
    assert window.grid.isVisible()
    assert grid_tile.display_transform == CameraDisplayTransform(90, True)
    window.close()

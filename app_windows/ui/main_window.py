"""Main grid/focus window for the Windows monitor."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, Qt, Slot
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import QMainWindow, QStackedWidget

from app.logging_setup import log_event
from app_windows.config.connection_test import SourceFactory
from app_windows.config.credentials import CredentialStore, InMemoryCredentialStore
from app_windows.config.ui_config import UiSettings
from app_windows.inference.person_detection_controller import PersonDetectionController
from app_windows.inference.face_recognition_controller import FaceRecognitionController
from app_windows.models.camera_display_transform import CameraDisplayTransformStore
from app_windows.models.camera_view_state import CameraViewSnapshot, CameraSlot
from app_windows.models.person_detection_state import PersonDetectionSnapshot
from app_windows.models.face_recognition_state import FaceRecognitionSnapshot
from app_windows.monitor_controller import CameraMonitorController

from .camera_focus_view import CameraFocusView
from .camera_grid import CameraGrid
from .theme import main_window_stylesheet


class MainWindow(QMainWindow):
    def __init__(
        self,
        slots: tuple[CameraSlot, ...],
        controller: CameraMonitorController,
        *,
        ui_settings: UiSettings,
        simulation: bool = False,
        config_path: Path | None = None,
        repo_root: Path | None = None,
        credentials: CredentialStore | None = None,
        connection_source_factory: SourceFactory | None = None,
        read_timeout_s: float = 3.0,
        logger: logging.Logger | None = None,
        person_detection_controller: PersonDetectionController | None = None,
        face_recognition_controller: FaceRecognitionController | None = None,
        parent: QMainWindow | None = None,
    ) -> None:
        super().__init__(parent)
        self._logger = logger or logging.getLogger(__name__)
        self._controller = controller
        self._ui_settings = ui_settings
        self._config_path = config_path or Path("config/config.local.yaml")
        self._repo_root = repo_root or Path.cwd()
        self._credentials = credentials or InMemoryCredentialStore()
        self._connection_source_factory = connection_source_factory
        self._read_timeout_s = read_timeout_s
        self._person_detection_controller = person_detection_controller
        self._face_recognition_controller = face_recognition_controller
        self._settings = QSettings("local-security-cam", "windows-monitor")
        self._focused_camera_id: str | None = None
        self._display_transforms = CameraDisplayTransformStore(self)
        window_title = "Local Security Monitor"
        if simulation:
            window_title += " — SIMULAZIONE"
        self.setWindowTitle(window_title)
        self.setMinimumSize(900, 560)
        self.setStyleSheet(main_window_stylesheet())

        self._stack = QStackedWidget(self)
        self._grid = CameraGrid(
            slots,
            display_transforms=self._display_transforms,
            parent=self._stack,
        )
        self._focus = CameraFocusView(
            controller=controller,
            config_path=self._config_path,
            repo_root=self._repo_root,
            credentials=self._credentials,
            display_transforms=self._display_transforms,
            source_factory=self._connection_source_factory,
            read_timeout_s=self._read_timeout_s,
            logger=self._logger,
            person_detection_controller=person_detection_controller,
            face_recognition_controller=face_recognition_controller,
            parent=self._stack,
        )
        self._stack.addWidget(self._grid)
        self._stack.addWidget(self._focus)
        self.setCentralWidget(self._stack)

        self._grid.camera_selected.connect(self.show_focus)
        self._focus.back_requested.connect(self.show_grid)
        self._controller.snapshot_changed.connect(self._on_snapshot)
        self._controller.camera_reconfigured.connect(
            self._on_camera_reconfigured_for_detection
        )
        if self._person_detection_controller is not None:
            self._person_detection_controller.snapshot_changed.connect(
                self._on_person_detection_snapshot
            )
        if self._face_recognition_controller is not None:
            self._face_recognition_controller.snapshot_changed.connect(
                self._on_face_recognition_snapshot
            )
        for snapshot in self._controller.snapshots.values():
            self._grid.set_snapshot(snapshot)
        if simulation:
            self.statusBar().showMessage("SIMULAZIONE — frame sintetici, nessun flusso RTSP reale")

        if ui_settings.remember_window_geometry:
            saved_geometry = self._settings.value("window/geometry")
            if isinstance(saved_geometry, QByteArray):
                self.restoreGeometry(saved_geometry)
        if ui_settings.start_maximized:
            self.showMaximized()

    @property
    def grid(self) -> CameraGrid:
        return self._grid

    @property
    def focus_view(self) -> CameraFocusView:
        return self._focus

    @property
    def focused_camera_id(self) -> str | None:
        return self._focused_camera_id

    def start_monitoring(self) -> None:
        self._controller.start()
        if self._person_detection_controller is not None:
            self._person_detection_controller.start()
        if self._face_recognition_controller is not None:
            self._face_recognition_controller.start()
        if self._focused_camera_id is not None:
            self._set_active_detection_camera(self._focused_camera_id)
            self._set_active_face_camera(self._focused_camera_id)

    def _on_snapshot(self, camera_id: str, snapshot: object) -> None:
        if not isinstance(snapshot, CameraViewSnapshot):
            return
        self._grid.set_snapshot(snapshot)
        if self._focused_camera_id == camera_id:
            self._focus.set_snapshot(snapshot)

    def show_focus(self, camera_id: str) -> None:
        snapshot = self._controller.snapshot_for(camera_id)
        if snapshot is None:
            return
        self._focused_camera_id = camera_id
        self._focus.set_snapshot(snapshot)
        self._stack.setCurrentWidget(self._focus)
        self._focus.setFocus(Qt.FocusReason.OtherFocusReason)
        self._set_active_detection_camera(camera_id)
        self._set_active_face_camera(camera_id)
        if self._person_detection_controller is not None:
            self._focus.set_person_detection_snapshot(
                self._person_detection_controller.snapshot
            )
        if self._face_recognition_controller is not None:
            self._focus.set_face_recognition_snapshot(
                self._face_recognition_controller.snapshot
            )
        log_event(self._logger, logging.INFO, "ui_focus_opened", camera=camera_id)

    def show_grid(self) -> None:
        if self._focused_camera_id is not None:
            log_event(
                self._logger,
                logging.INFO,
                "ui_focus_closed",
                camera=self._focused_camera_id,
            )
        self._focused_camera_id = None
        self._stack.setCurrentWidget(self._grid)
        self._grid.setFocus(Qt.FocusReason.OtherFocusReason)
        if self._person_detection_controller is not None:
            self._person_detection_controller.set_active_camera(None, None)
            self._focus.set_person_detection_snapshot(None)
        if self._face_recognition_controller is not None:
            self._face_recognition_controller.set_active_camera(None, None)
            self._focus.set_face_recognition_snapshot(None)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._focus.configuration_panel.flush_pending()
        if self._person_detection_controller is not None:
            self._person_detection_controller.stop(timeout_s=1.5)
        self._controller.stop(timeout_s=1.0)
        if self._face_recognition_controller is not None:
            self._face_recognition_controller.stop(timeout_s=1.5)
        if self._ui_settings.remember_window_geometry:
            self._settings.setValue("window/geometry", self.saveGeometry())
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._focused_camera_id is not None:
            self.show_grid()
            event.accept()
            return
        super().keyPressEvent(event)

    def _set_active_detection_camera(self, camera_id: str) -> None:
        if self._person_detection_controller is None:
            return
        # Clear before changing the generation so a fast worker notification
        # cannot be overwritten by a stale snapshot from the prior camera.
        self._focus.set_person_detection_snapshot(None)
        self._person_detection_controller.set_active_camera(
            camera_id,
            self._controller.provider_for(camera_id),
        )

    def _set_active_face_camera(self, camera_id: str) -> None:
        if self._face_recognition_controller is None:
            return
        self._focus.set_face_recognition_snapshot(None)
        self._face_recognition_controller.set_active_camera(
            camera_id,
            self._controller.provider_for(camera_id),
        )

    def _on_person_detection_snapshot(self, value: object) -> None:
        if not isinstance(value, PersonDetectionSnapshot):
            return
        if self._focused_camera_id is None:
            return
        if value.camera_id in {None, self._focused_camera_id}:
            self._focus.set_person_detection_snapshot(value)
        if self._face_recognition_controller is not None:
            self._face_recognition_controller.set_person_snapshot(value)

    def _on_face_recognition_snapshot(self, value: object) -> None:
        if not isinstance(value, FaceRecognitionSnapshot):
            return
        if self._focused_camera_id is None:
            return
        if value.camera_id in {None, self._focused_camera_id}:
            self._focus.set_face_recognition_snapshot(value)

    @Slot(str, bool, str)
    def _on_camera_reconfigured_for_detection(
        self,
        camera_id: str,
        _success: bool,
        _message: str,
    ) -> None:
        if self._person_detection_controller is not None and camera_id == self._focused_camera_id:
            self._set_active_detection_camera(camera_id)
        if self._face_recognition_controller is not None and camera_id == self._focused_camera_id:
            self._set_active_face_camera(camera_id)

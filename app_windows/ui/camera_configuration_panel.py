"""Embedded, single-camera configuration panel for the focus view."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from app.config import ConfigurationError
from app.logging_setup import redact_log_text
from app.video.base import VideoSource
from app.video.factory import create_opencv_source

from app_windows.config.camera_config import (
    CameraDraft,
    draft_from_slot,
    runtime_stream_url,
    validate_camera_draft,
)
from app_windows.config.connection_test import (
    AsyncConnectionTester,
    ConnectionTestResult,
    SourceFactory,
)
from app_windows.config.credentials import CredentialStore, InMemoryCredentialStore
from app_windows.config.persistence import CameraConfigRepository
from app_windows.inference.person_detection_controller import PersonDetectionController
from app_windows.inference.face_recognition_controller import FaceRecognitionController
from app_windows.models.camera_view_state import CameraSlot
from app_windows.monitor_controller import CameraMonitorController

from .camera_configuration_dialog import CameraEditorCard
from .person_detection_panel import PersonDetectionPanel
from .face_recognition_panel import FaceRecognitionPanel
from .theme import configuration_panel_stylesheet, status_text_stylesheet


class CameraConfigurationPanel(QWidget):
    """Edit and automatically apply the configuration of one selected camera."""

    APPLY_DELAY_MS = 500
    configuration_applied = Signal(str, bool, str)

    def __init__(
        self,
        controller: CameraMonitorController,
        *,
        config_path: Path,
        repo_root: Path,
        credentials: CredentialStore | None = None,
        source_factory: SourceFactory | None = None,
        read_timeout_s: float = 3.0,
        logger: logging.Logger | None = None,
        person_detection_controller: PersonDetectionController | None = None,
        face_recognition_controller: FaceRecognitionController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("CameraConfigurationPanel")
        self.setMinimumWidth(0)

        self._controller = controller
        self._config_path = Path(config_path)
        self._repository = CameraConfigRepository(repo_root)
        self._credentials = credentials or InMemoryCredentialStore()
        self._logger = logger or logging.getLogger(__name__)
        self._camera_id: str | None = None
        self._card: CameraEditorCard | None = None
        self._pending_drafts: dict[str, CameraDraft] = {}
        self._applying_drafts: dict[str, CameraDraft] = {}
        self._person_detection_panel: PersonDetectionPanel | None = None
        self._face_recognition_panel: FaceRecognitionPanel | None = None

        self.setStyleSheet(configuration_panel_stylesheet())

        self._title_label = QLabel("Configurazione camera")
        self._title_label.setObjectName("ConfigurationPanelTitle")
        self._hint_label = QLabel(
            "Le modifiche valide vengono salvate e applicate automaticamente."
        )
        self._hint_label.setObjectName("ConfigurationPanelHint")
        self._hint_label.setWordWrap(True)
        self._summary_label = QLabel("Seleziona una camera dalla griglia.")
        self._summary_label.setObjectName("ConfigurationPanelSummary")
        self._summary_label.setWordWrap(True)

        self._content = QWidget()
        self._content.setObjectName("CameraConfigurationContent")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)

        if person_detection_controller is not None:
            self._person_detection_panel = PersonDetectionPanel(
                person_detection_controller,
                repository=self._repository,
                config_path=self._config_path,
                repo_root=repo_root,
                parent=self._content,
            )
        if face_recognition_controller is not None:
            self._face_recognition_panel = FaceRecognitionPanel(
                face_recognition_controller,
                repository=self._repository,
                config_path=self._config_path,
                repo_root=repo_root,
                parent=self._content,
            )

        self._scroll = QScrollArea()
        self._scroll.setObjectName("CameraConfigurationScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._title_label)
        layout.addWidget(self._hint_label)
        layout.addWidget(self._scroll, 1)
        layout.addWidget(self._summary_label)

        source_factory = source_factory or self._default_source_factory
        self._tester = AsyncConnectionTester(
            source_factory,
            read_timeout_s=read_timeout_s,
            existing_probe=controller.probe_existing_camera,
            logger=self._logger,
            parent=self,
        )
        self._tester.started.connect(self._on_test_started)
        self._tester.finished.connect(self._on_test_finished)
        self._controller.camera_reconfigured.connect(self._on_camera_reconfigured)

        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(self.APPLY_DELAY_MS)
        self._apply_timer.timeout.connect(self._on_apply_timer)

    @property
    def camera_id(self) -> str | None:
        return self._camera_id

    @property
    def card(self) -> CameraEditorCard | None:
        return self._card

    @property
    def person_detection_panel(self) -> PersonDetectionPanel | None:
        return self._person_detection_panel

    @property
    def face_recognition_panel(self) -> FaceRecognitionPanel | None:
        return self._face_recognition_panel

    @property
    def config_path(self) -> Path:
        return self._config_path

    def set_camera_slot(self, slot: CameraSlot) -> None:
        """Show one camera editor, preserving any valid pending application."""

        if self._camera_id == slot.camera_id and self._card is not None:
            return

        self.flush_pending()
        self._apply_timer.stop()
        self._camera_id = slot.camera_id
        self._title_label.setText(f"Configurazione — {slot.name}")
        self._summary_label.setStyleSheet(status_text_stylesheet())
        self._summary_label.setText("Le modifiche valide vengono applicate automaticamente.")

        while self._content_layout.count():
            old_item = self._content_layout.takeAt(0)
            if old_item is None:
                continue
            widget = old_item.widget()
            if (
                widget is not None
                and widget is not self._person_detection_panel
                and widget is not self._face_recognition_panel
            ):
                widget.deleteLater()

        self._card = CameraEditorCard(
            self._draft_for_slot(slot),
            parent=self._content,
        )
        self._card.test_requested.connect(self._start_test)
        self._card.changed.connect(self._on_card_changed)
        self._content_layout.addWidget(self._card)
        if self._person_detection_panel is not None:
            self._content_layout.addWidget(self._person_detection_panel)
        if self._face_recognition_panel is not None:
            self._content_layout.addWidget(self._face_recognition_panel)
        self._content_layout.addStretch(1)
        self.setAccessibleName(f"Configurazione della camera {slot.name}")

    def flush_pending(self) -> None:
        """Start a valid edit immediately before navigation or application exit."""

        if self._person_detection_panel is not None:
            self._person_detection_panel.flush_pending()
        if self._face_recognition_panel is not None:
            self._face_recognition_panel.flush_pending()

        if self._apply_timer.isActive():
            self._apply_timer.stop()
            self._on_apply_timer()
        elif self._card is not None and self._camera_id is not None and self._card.is_dirty():
            self._queue_current_draft()

    @Slot(str)
    def _on_card_changed(self, camera_id: str) -> None:
        if camera_id != self._camera_id:
            return
        self._summary_label.setStyleSheet(status_text_stylesheet())
        self._summary_label.setText("Modifica in attesa di applicazione…")
        self._apply_timer.start()

    @Slot()
    def _on_apply_timer(self) -> None:
        self._queue_current_draft()

    def _queue_current_draft(self) -> None:
        if self._card is None or self._camera_id is None:
            return
        try:
            validated = validate_camera_draft(self._card.draft())
        except Exception as exc:
            self._card.set_status(redact_log_text(exc), error=True)
            self._summary_label.setStyleSheet(status_text_stylesheet(error=True))
            self._summary_label.setText("Configurazione non valida: modifica i campi evidenziati.")
            return

        self._pending_drafts[self._camera_id] = validated.draft
        self._try_apply(self._camera_id)

    def _try_apply(self, camera_id: str) -> None:
        if camera_id in self._applying_drafts:
            return
        draft = self._pending_drafts.pop(camera_id, None)
        if draft is None:
            return

        try:
            result = self._repository.save(
                (draft,),
                current_path=self._config_path,
                credentials=self._credentials,
            )
        except Exception as exc:
            self._pending_drafts[camera_id] = draft
            self._set_error(camera_id, f"Salvataggio non riuscito: {redact_log_text(exc)}")
            return

        self._config_path = result.path
        if self._person_detection_panel is not None:
            self._person_detection_panel.set_config_path(self._config_path)
        if self._face_recognition_panel is not None:
            self._face_recognition_panel.set_config_path(self._config_path)
        value = result.values[0]
        self._applying_drafts[camera_id] = value.draft
        try:
            self._controller.apply_camera_slot(value.to_slot())
        except RuntimeError as exc:
            self._applying_drafts.pop(camera_id, None)
            self._pending_drafts[camera_id] = value.draft
            self._set_status(camera_id, f"Applicazione in attesa: {redact_log_text(exc)}")
            return
        except Exception as exc:
            self._applying_drafts.pop(camera_id, None)
            self._pending_drafts[camera_id] = value.draft
            self._set_error(camera_id, f"Applicazione non riuscita: {redact_log_text(exc)}")
            return

        self._set_status(camera_id, "Configurazione salvata; applicazione in corso…")
        if not self._controller.is_started:
            self._finish_apply(camera_id, True, "Configurazione applicata")

    @Slot(str, bool, str)
    def _on_camera_reconfigured(self, camera_id: str, success: bool, message: str) -> None:
        if camera_id not in self._applying_drafts:
            if success and camera_id in self._pending_drafts:
                self._try_apply(camera_id)
            return
        self._finish_apply(camera_id, success, message)

    def _finish_apply(self, camera_id: str, success: bool, message: str) -> None:
        if self._applying_drafts.pop(camera_id, None) is None:
            return

        if not success:
            self._set_error(camera_id, f"Applicazione non riuscita: {redact_log_text(message)}")
            self.configuration_applied.emit(camera_id, False, message)
            return

        if camera_id in self._pending_drafts:
            self._set_status(camera_id, "Applicazione della modifica più recente in corso…")
            self._try_apply(camera_id)
            return

        if camera_id == self._camera_id and self._card is not None:
            slot = next(
                (value for value in self._controller.slots if value.camera_id == camera_id),
                None,
            )
            if slot is not None:
                self._card.set_initial_draft(self._draft_for_slot(slot))
        self._set_status(camera_id, message or "Configurazione applicata")
        self.configuration_applied.emit(camera_id, True, message)

    @Slot(str)
    def _start_test(self, camera_id: str) -> None:
        if self._card is None or camera_id != self._camera_id:
            return
        try:
            validated = validate_camera_draft(self._card.draft())
            if validated.stream_url is None:
                raise ValueError("configurare un URL prima del test")
            test_url = runtime_stream_url(validated.stream_url, validated.credential_value)
            assert test_url is not None
        except Exception as exc:
            self._card.set_status(redact_log_text(exc), error=True)
            return

        self._card.set_test_running(True)
        try:
            self._tester.start(camera_id, test_url, validated.draft.transport)
        except Exception as exc:
            self._card.set_test_running(False)
            self._card.set_status(redact_log_text(exc), error=True)

    @Slot(str)
    def _on_test_started(self, camera_id: str) -> None:
        if self._card is not None and camera_id == self._camera_id:
            self._card.set_test_running(True)

    @Slot(object)
    def _on_test_finished(self, result: object) -> None:
        if not isinstance(result, ConnectionTestResult):
            return
        if self._card is None or result.camera_id != self._camera_id:
            return
        self._card.set_test_running(False)
        self._card.set_status(result.message, error=not result.success)

    def _set_status(self, camera_id: str, message: str) -> None:
        if camera_id == self._camera_id:
            self._summary_label.setStyleSheet(status_text_stylesheet())
            self._summary_label.setText(message)

    def _set_error(self, camera_id: str, message: str) -> None:
        if camera_id == self._camera_id:
            self._summary_label.setStyleSheet(status_text_stylesheet(error=True))
            self._summary_label.setText(message)

    def _draft_for_slot(self, slot: CameraSlot) -> CameraDraft:
        """Keep explicit fake-camera runs usable without treating fake URLs as RTSP."""

        try:
            return draft_from_slot(slot, self._credentials)
        except ConfigurationError:
            if slot.stream_url is None or not slot.stream_url.lower().startswith("fake://"):
                raise
            return CameraDraft(
                camera_id=slot.camera_id,
                slot_index=slot.slot_index,
                name=slot.name,
                enabled=slot.enabled,
                transport=slot.rtsp_transport,
            )

    @staticmethod
    def _default_source_factory(url: str, transport: str) -> VideoSource:
        return create_opencv_source(url, rtsp_transport=transport)

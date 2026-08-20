"""PySide6 dialog for configuring the six central camera slots."""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
import re

from PySide6.QtCore import QSignalBlocker, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.logging_setup import redact_log_text
from app.video.base import VideoSource
from app.video.factory import create_opencv_source

from app_windows.config.camera_config import (
    CameraDraft,
    build_stream_url,
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
from app_windows.models.camera_view_state import CameraSlot
from app_windows.monitor_controller import CameraMonitorController

from .theme import configuration_editor_stylesheet, status_text_stylesheet


def _safe_object_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


class CameraEditorCard(QGroupBox):
    """One camera editor card with no password value in its Qt labels."""

    test_requested = Signal(str)
    changed = Signal(str)

    def __init__(
        self,
        draft: CameraDraft,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_id = draft.camera_id
        self._initial_draft = draft
        self.setTitle(f"Camera {draft.slot_index} — {draft.name}")
        self.setObjectName(f"CameraEditor_{_safe_object_name(draft.camera_id)}")
        self.setStyleSheet(configuration_editor_stylesheet())

        self.id_label = QLabel(draft.camera_id)
        self.id_label.setObjectName("cameraIdLabel")
        self.id_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.name_edit = QLineEdit(draft.name)
        self.name_edit.setObjectName("nameEdit")
        self.name_edit.setPlaceholderText(f"Camera {draft.slot_index}")

        self.enabled_check = QCheckBox("Camera abilitata")
        self.enabled_check.setObjectName("enabledCheck")
        self.enabled_check.setChecked(draft.enabled)

        self.scheme_combo = QComboBox()
        self.scheme_combo.setObjectName("schemeCombo")
        self.scheme_combo.addItems(["rtsp", "rtsps", "http", "https"])
        self.scheme_combo.setCurrentText(draft.scheme)

        self.host_edit = QLineEdit(draft.host)
        self.host_edit.setObjectName("hostEdit")
        self.host_edit.setPlaceholderText("192.168.x.x o nome host")

        self.port_spin = QSpinBox()
        self.port_spin.setObjectName("portSpin")
        self.port_spin.setRange(0, 65535)
        self.port_spin.setSpecialValueText("automatico")
        self.port_spin.setValue(draft.port or 0)

        self.path_edit = QLineEdit(draft.path)
        self.path_edit.setObjectName("pathEdit")
        self.path_edit.setPlaceholderText("/percorso/stream")

        self.username_edit = QLineEdit(draft.username)
        self.username_edit.setObjectName("usernameEdit")
        self.username_edit.setPlaceholderText("opzionale")

        self.password_edit = QLineEdit()
        self.password_edit.setObjectName("passwordEdit")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText(
            "vuoto = conserva la password salvata"
            if draft.password_is_stored
            else "opzionale"
        )

        self.clear_password_check = QCheckBox("Rimuovi password salvata")
        self.clear_password_check.setObjectName("clearPasswordCheck")
        self.clear_password_check.setEnabled(draft.password_is_stored)

        self.transport_combo = QComboBox()
        self.transport_combo.setObjectName("transportCombo")
        self.transport_combo.addItem("TCP", "tcp")
        self.transport_combo.addItem("UDP", "udp")
        self.transport_combo.addItem("AUTO", "auto")
        transport_index = self.transport_combo.findData(draft.transport)
        self.transport_combo.setCurrentIndex(max(0, transport_index))

        self.url_preview = QLineEdit()
        self.url_preview.setObjectName("urlPreview")
        self.url_preview.setReadOnly(True)
        self.url_preview.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.url_preview.setToolTip("URL canonico redatto; la password non viene mostrata")

        self.status_label = QLabel()
        self.status_label.setObjectName("cameraConfigStatus")
        self.status_label.setWordWrap(True)

        self.test_button = QPushButton("Test connessione")
        self.test_button.setObjectName("testConnectionButton")
        self.test_button.clicked.connect(lambda: self.test_requested.emit(self.camera_id))

        form = QFormLayout()
        form.addRow("ID camera", self.id_label)
        form.addRow("Nome visualizzato", self.name_edit)
        form.addRow("Stato", self.enabled_check)
        form.addRow("Schema", self.scheme_combo)
        form.addRow("Host / IP", self.host_edit)
        form.addRow("Porta", self.port_spin)
        form.addRow("Path", self.path_edit)
        form.addRow("Username", self.username_edit)
        form.addRow("Password", self.password_edit)
        form.addRow("", self.clear_password_check)
        form.addRow("Trasporto RTSP", self.transport_combo)
        form.addRow("URL stream", self.url_preview)

        actions = QHBoxLayout()
        actions.addWidget(self.status_label, 1)
        actions.addWidget(self.test_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(actions)

        for widget in (
            self.name_edit,
            self.enabled_check,
            self.scheme_combo,
            self.host_edit,
            self.port_spin,
            self.path_edit,
            self.username_edit,
            self.password_edit,
            self.clear_password_check,
            self.transport_combo,
        ):
            if hasattr(widget, "textChanged"):
                widget.textChanged.connect(self._on_changed)  # type: ignore[attr-defined]
            if hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self._on_changed)  # type: ignore[attr-defined]
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._on_changed)  # type: ignore[attr-defined]
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._on_changed)  # type: ignore[attr-defined]
        self._refresh_preview()
        if not draft.enabled:
            self.set_status("DISABILITATA")
        elif draft.host:
            self.set_status("Configurata")
        else:
            self.set_status("NON CONFIGURATA")

    @property
    def slot_index(self) -> int:
        return self._initial_draft.slot_index

    def draft(self) -> CameraDraft:
        return CameraDraft(
            camera_id=self.camera_id,
            slot_index=self.slot_index,
            name=self.name_edit.text(),
            enabled=self.enabled_check.isChecked(),
            scheme=self.scheme_combo.currentText(),
            host=self.host_edit.text(),
            port=self.port_spin.value() or None,
            path=self.path_edit.text(),
            username=self.username_edit.text(),
            transport=str(self.transport_combo.currentData() or "tcp"),
            password=self.password_edit.text(),
            clear_password=self.clear_password_check.isChecked(),
            existing_password=self._initial_draft.existing_password,
        )

    def is_dirty(self) -> bool:
        current = self.draft()
        return current != self._initial_draft or bool(current.password) or current.clear_password

    def set_initial_draft(self, draft: CameraDraft) -> None:
        """Mark the current editor values as applied without emitting changes."""

        self._initial_draft = draft
        password_blocker = QSignalBlocker(self.password_edit)
        clear_password_blocker = QSignalBlocker(self.clear_password_check)
        self.password_edit.clear()
        self.clear_password_check.setChecked(False)
        self.clear_password_check.setEnabled(draft.password_is_stored)
        self.password_edit.setPlaceholderText(
            "vuoto = conserva la password salvata" if draft.password_is_stored else "opzionale"
        )
        del password_blocker, clear_password_blocker
        self._refresh_preview()

    def set_status(self, message: str, *, error: bool = False) -> None:
        self.status_label.setStyleSheet(status_text_stylesheet(error=error))
        self.status_label.setText(message)

    def set_test_running(self, running: bool) -> None:
        self.test_button.setEnabled(not running)
        if running:
            self.set_status("Connessione in corso…")

    @Slot()
    def _on_changed(self, *_args: object) -> None:
        self._refresh_preview()
        self.changed.emit(self.camera_id)

    def _refresh_preview(self) -> None:
        try:
            validated = validate_camera_draft(self.draft())
        except Exception as exc:
            self.url_preview.setText("URL non configurato o non valido")
            self.url_preview.setCursorPosition(0)
            self.url_preview.setToolTip(redact_log_text(exc))
            return
        runtime_url = runtime_stream_url(
            validated.stream_url,
            validated.credential_value,
        )
        if runtime_url is None:
            self.url_preview.setText("URL non configurato")
        else:
            from app.video.base import redact_url

            self.url_preview.setText(redact_url(runtime_url))
        self.url_preview.setCursorPosition(0)


class CameraConfigurationDialog(QDialog):
    """Non-modal editor for all six camera slots."""

    changes_applied = Signal(object)

    def __init__(
        self,
        slots: tuple[CameraSlot, ...],
        controller: CameraMonitorController,
        *,
        config_path: Path,
        repo_root: Path,
        credentials: CredentialStore | None = None,
        source_factory: SourceFactory | None = None,
        read_timeout_s: float = 3.0,
        logger: logging.Logger | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if len(slots) != 6:
            raise ValueError("camera configuration dialog requires exactly six slots")
        self.setWindowTitle("Configurazione camere")
        self.setMinimumSize(980, 720)
        self.setModal(False)
        self.setStyleSheet(configuration_editor_stylesheet())
        self._controller = controller
        self._config_path = Path(config_path)
        self._repository = CameraConfigRepository(repo_root)
        self._credentials = credentials or InMemoryCredentialStore()
        self._logger = logger or logging.getLogger(__name__)
        self._saving = False
        self._pending_apply: set[str] = set()
        self._apply_failures: dict[str, str] = {}
        self._cards: dict[str, CameraEditorCard] = {}

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

        content = QWidget()
        grid = QGridLayout(content)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        for index, slot in enumerate(slots):
            card = CameraEditorCard(draft_from_slot(slot, self._credentials), parent=content)
            card.test_requested.connect(self._start_test)
            self._cards[slot.camera_id] = card
            grid.addWidget(card, index // 2, index % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        scroll = QScrollArea()
        scroll.setObjectName("cameraConfigurationScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("cameraConfigurationSummary")
        self.summary_label.setWordWrap(True)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Save).setText("Salva e applica")
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Annulla")
        self.button_box.button(QDialogButtonBox.StandardButton.Save).setObjectName("saveApplyButton")
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setObjectName("cancelButton")
        self.button_box.accepted.connect(self.save_and_apply)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.button_box)

    @property
    def cards(self) -> dict[str, CameraEditorCard]:
        return dict(self._cards)

    @property
    def config_path(self) -> Path:
        return self._config_path

    @Slot(str)
    def _start_test(self, camera_id: str) -> None:
        card = self._cards[camera_id]
        try:
            validated = validate_camera_draft(card.draft())
            if validated.stream_url is None:
                raise ValueError("configurare un URL prima del test")
            test_url = runtime_stream_url(
                validated.stream_url,
                validated.credential_value,
            )
            assert test_url is not None
        except Exception as exc:
            card.set_status(redact_log_text(exc), error=True)
            return

        card.set_test_running(True)
        try:
            self._tester.start(camera_id, test_url, validated.draft.transport)
        except Exception as exc:
            card.set_test_running(False)
            card.set_status(redact_log_text(exc), error=True)

    @Slot(str)
    def _on_test_started(self, camera_id: str) -> None:
        if camera_id in self._cards:
            self._cards[camera_id].set_test_running(True)

    @Slot(object)
    def _on_test_finished(self, result: object) -> None:
        if not isinstance(result, ConnectionTestResult):
            return
        card = self._cards.get(result.camera_id)
        if card is None:
            return
        card.set_test_running(False)
        card.set_status(result.message, error=not result.success)

    @Slot()
    def save_and_apply(self) -> None:
        if self._saving:
            return
        dirty_cards = [card for card in self._cards.values() if card.is_dirty()]
        if not dirty_cards:
            self.summary_label.setText("Nessuna modifica da applicare.")
            return

        drafts: list[CameraDraft] = []
        try:
            for card in dirty_cards:
                draft = card.draft()
                validate_camera_draft(draft)
                drafts.append(draft)
            result = self._repository.save(
                drafts,
                current_path=self._config_path,
                credentials=self._credentials,
            )
        except Exception as exc:
            message = redact_log_text(exc)
            self.summary_label.setStyleSheet(status_text_stylesheet(error=True))
            self.summary_label.setText(f"Salvataggio non riuscito: {message}")
            return

        self._config_path = result.path
        self._saving = True
        self._pending_apply = {value.draft.camera_id for value in result.values}
        self._apply_failures.clear()
        self._set_buttons_enabled(False)
        self.summary_label.setStyleSheet(status_text_stylesheet())
        self.summary_label.setText("Configurazione salvata; applicazione in corso…")
        for value in result.values:
            try:
                self._controller.apply_camera_slot(value.to_slot())
                if not self._controller.is_started:
                    self._pending_apply.discard(value.draft.camera_id)
            except Exception as exc:
                self._apply_failures[value.draft.camera_id] = redact_log_text(exc)
                self._pending_apply.discard(value.draft.camera_id)
        self._finish_apply_if_ready()

    @Slot(str, bool, str)
    def _on_camera_reconfigured(self, camera_id: str, success: bool, message: str) -> None:
        if camera_id not in self._pending_apply:
            return
        self._pending_apply.discard(camera_id)
        if not success:
            self._apply_failures[camera_id] = redact_log_text(message)
        self._finish_apply_if_ready()

    def _finish_apply_if_ready(self) -> None:
        if not self._saving or self._pending_apply:
            return
        self._saving = False
        self._set_buttons_enabled(True)
        if self._apply_failures:
            details = "; ".join(
                f"{camera_id}: {message}" for camera_id, message in self._apply_failures.items()
            )
            self.summary_label.setStyleSheet(status_text_stylesheet(error=True))
            self.summary_label.setText(f"Salvataggio riuscito, applicazione parziale: {details}")
            return
        self.changes_applied.emit(tuple(self._cards))
        self.accept()

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.button_box.button(QDialogButtonBox.StandardButton.Save).setEnabled(enabled)
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setEnabled(enabled)
        for card in self._cards.values():
            card.setEnabled(enabled)

    @staticmethod
    def _default_source_factory(url: str, transport: str) -> VideoSource:
        return create_opencv_source(url, rtsp_transport=transport)

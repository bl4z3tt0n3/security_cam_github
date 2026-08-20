"""Contextual backend controls and telemetry for the selected camera."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from app.inference.prompts import normalize_prompts
from app.logging_setup import redact_log_text
from app_windows.config.persistence import CameraConfigRepository
from app_windows.inference.person_detection_controller import PersonDetectionController
from app_windows.models.person_detection_state import (
    DEFAULT_OPENVINO_MODEL,
    PersonDetectionSettings,
    PersonDetectionSnapshot,
    PersonDetectionStatus,
    SUPPORTED_OPENVINO_MODEL_NAMES,
    SUPPORTED_YOLOE_MODEL_NAMES,
)


def discover_yoloe_models(
    repo_root: Path,
    configured_model: str | None,
) -> tuple[tuple[str, str], ...]:
    """Return the supported local prompted YOLOE segmentation checkpoints."""

    root = Path(repo_root).resolve()
    candidates: dict[str, str] = {}

    configured_path: Path | None = None
    if configured_model:
        configured_path = Path(configured_model).expanduser()
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        if (
            configured_path.suffix.lower() == ".pt"
            and configured_path.name.casefold() in SUPPORTED_YOLOE_MODEL_NAMES
        ):
            display = _display_model_path(root, configured_path)
            candidates[display] = str(configured_model)

    models_root = root / "models"
    if models_root.is_dir():
        for path in sorted(models_root.rglob("*.pt")):
            if not path.is_file():
                continue
            if path.name.casefold() not in SUPPORTED_YOLOE_MODEL_NAMES:
                continue
            display = _display_model_path(root, path)
            candidates.setdefault(display, display)

    return tuple(sorted(candidates.items(), key=lambda item: item[0].casefold()))


def _display_model_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def discover_openvino_models(
    repo_root: Path,
    configured_model: str | None,
) -> tuple[tuple[str, str], ...]:
    """Return the two official YOLO26 checkpoints and local IR directories."""

    root = Path(repo_root).resolve()
    candidates: dict[str, str] = {
        f"models/{name}": f"models/{name}"
        for name in ("yolo26s.pt", "yolo26n.pt")
    }

    if configured_model:
        configured_path = Path(configured_model).expanduser()
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        if (
            configured_path.suffix.lower() == ".pt"
            and configured_path.name.casefold() in SUPPORTED_OPENVINO_MODEL_NAMES
        ) or configured_path.is_dir() or configured_path.suffix.lower() == ".xml":
            display = _display_model_path(root, configured_path)
            candidates[display] = str(configured_model)

    models_root = root / "models"
    if models_root.is_dir():
        for path in sorted(models_root.rglob("*")):
            if path.is_file() and path.suffix.lower() == ".pt":
                if path.name.casefold() not in SUPPORTED_OPENVINO_MODEL_NAMES:
                    continue
                display = _display_model_path(root, path)
                candidates.setdefault(display, display)
            elif path.is_file() and path.suffix.lower() == ".xml":
                display = _display_model_path(root, path.parent)
                candidates.setdefault(display, display)

    return tuple(sorted(candidates.items(), key=lambda item: item[0].casefold()))


# Kept as a compatibility import for callers of the first Windows panel.
discover_onnx_models = discover_yoloe_models


class PersonDetectionPanel(QGroupBox):
    """Debounced person-detection configuration with service telemetry."""

    APPLY_DELAY_MS = 500

    def __init__(
        self,
        controller: PersonDetectionController,
        *,
        repository: CameraConfigRepository,
        config_path: Path,
        repo_root: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._repository = repository
        self._config_path = Path(config_path)
        self._repo_root = Path(repo_root)
        self._settings = controller.settings

        self.setObjectName("PersonDetectionPanel")
        self.setTitle("Rilevamento persone")
        self.setStyleSheet(
            """
            QGroupBox#PersonDetectionPanel {
                color: #f4f7fb;
                background: #10283a;
                border: 1px solid #2d526b;
                border-radius: 8px;
                margin-top: 12px;
                padding: 10px 8px 8px 8px;
            }
            QGroupBox#PersonDetectionPanel::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #6ddcff;
                font-weight: 700;
            }
            QGroupBox#PersonDetectionPanel QLabel { color: #b8c7d8; }
            QGroupBox#PersonDetectionPanel QComboBox,
            QGroupBox#PersonDetectionPanel QSpinBox,
            QGroupBox#PersonDetectionPanel QLineEdit {
                color: #f4f7fb;
                background: #071522;
                border: 1px solid #31516a;
                border-radius: 5px;
                padding: 5px 7px;
                min-height: 22px;
            }
            QGroupBox#PersonDetectionPanel QComboBox:disabled,
            QGroupBox#PersonDetectionPanel QSpinBox:disabled,
            QGroupBox#PersonDetectionPanel QLineEdit:disabled {
                color: #728397;
            }
            QGroupBox#PersonDetectionPanel QCheckBox { color: #f4f7fb; }
            QLabel#PersonDetectionNote { color: #8ea7bd; font-size: 11px; }
            QLabel#PersonDetectionConfidenceValue {
                color: #6ddcff; font-weight: 700;
            }
            QLabel#PersonDetectionStatusValue { font-weight: 700; }
            QLabel#PersonDetectionTelemetryValue {
                color: #f4f7fb; font-weight: 700;
            }
            """
        )

        self._enabled_check = QCheckBox("Rilevamento persone")
        self._enabled_check.setObjectName("personDetectionEnabled")
        self._enabled_check.setChecked(self._settings.enabled)
        self._enabled_check.setAccessibleName("Abilita rilevamento persone")

        self._backend_combo = QComboBox()
        self._backend_combo.setObjectName("personDetectionBackend")
        self._backend_combo.setAccessibleName("Backend rilevamento persone")
        for label, value in (
            ("OpenVINO", "openvino"),
            ("YOLOE", "yoloe"),
            ("ONNX", "onnx"),
            ("Fake/offline", "fake"),
            ("Auto", "auto"),
        ):
            self._backend_combo.addItem(label, value)
        self._backend_combo.setCurrentIndex(
            max(0, self._backend_combo.findData(self._settings.backend))
        )

        self._model_combo = QComboBox()
        self._model_combo.setObjectName("personDetectionModel")
        self._model_combo.setAccessibleName("Modello rilevamento persone")
        self._populate_models(self._settings.model)

        self._prompt_edit = QLineEdit(", ".join(self._settings.prompts))
        self._prompt_edit.setObjectName("personDetectionPrompts")
        self._prompt_edit.setPlaceholderText("person, bottle, smartphone")
        self._prompt_edit.setAccessibleName("Categorie prompt YOLOE")
        self._prompt_edit.setToolTip("Categorie separate da virgola, massimo 20")

        self._device_combo = QComboBox()
        self._device_combo.setObjectName("personDetectionDevice")
        self._populate_devices(self._settings.device)

        self._precision_combo = QComboBox()
        self._precision_combo.setObjectName("personDetectionPrecision")
        self._precision_combo.addItem("FP16", "fp16")
        self._precision_combo.addItem("FP32", "fp32")
        self._precision_combo.setCurrentIndex(
            max(0, self._precision_combo.findData(self._settings.precision))
        )

        self._fallback_combo = QComboBox()
        self._fallback_combo.setObjectName("personDetectionFallbackDevice")
        self._fallback_combo.addItem("Nessuno", "none")
        self._fallback_combo.addItem("CPU", "cpu")
        self._fallback_combo.setCurrentIndex(
            max(0, self._fallback_combo.findData(self._settings.fallback_device))
        )

        self._image_size_spin = QSpinBox()
        self._image_size_spin.setObjectName("personDetectionImageSize")
        self._image_size_spin.setRange(32, 2048)
        self._image_size_spin.setSingleStep(32)
        self._image_size_spin.setValue(self._settings.image_size)
        self._image_size_spin.setAccessibleName("Dimensione input imgsz")

        self._confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self._confidence_slider.setObjectName("personDetectionConfidence")
        self._confidence_slider.setRange(0, 100)
        self._confidence_slider.setValue(round(self._settings.confidence_threshold * 100))
        self._confidence_slider.setTickInterval(10)
        self._confidence_slider.setAccessibleName("Soglia di confidenza")
        self._confidence_value = QLabel()
        self._confidence_value.setObjectName("PersonDetectionConfidenceValue")
        self._confidence_value.setMinimumWidth(38)
        self._confidence_value.setAlignment(Qt.AlignmentFlag.AlignRight)

        self._fps_spin = QSpinBox()
        self._fps_spin.setObjectName("personDetectionFps")
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setSuffix(" FPS")
        self._fps_spin.setValue(round(self._settings.inference_fps))
        self._fps_spin.setAccessibleName("FPS di inferenza")

        self._show_boxes_check = QCheckBox("Mostra box")
        self._show_boxes_check.setObjectName("personDetectionShowBoxes")
        self._show_boxes_check.setChecked(self._settings.show_boxes)
        self._show_boxes_check.setAccessibleName("Mostra bounding box")

        self._show_masks_check = QCheckBox("Mostra maschere")
        self._show_masks_check.setObjectName("personDetectionShowMasks")
        self._show_masks_check.setChecked(self._settings.show_masks)
        self._show_masks_check.setAccessibleName("Mostra maschere YOLOE")

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(7)
        form.addRow(self._enabled_check)
        form.addRow("Backend", self._backend_combo)
        form.addRow("Modello", self._model_combo)
        form.addRow("Categorie/prompt", self._prompt_edit)
        form.addRow("Dispositivo", self._device_combo)
        form.addRow("Precisione", self._precision_combo)
        form.addRow("Fallback device", self._fallback_combo)
        form.addRow("Dimensione input", self._image_size_spin)

        confidence_row = QGridLayout()
        confidence_row.setContentsMargins(0, 0, 0, 0)
        confidence_row.setHorizontalSpacing(8)
        confidence_row.addWidget(self._confidence_slider, 0, 0)
        confidence_row.addWidget(self._confidence_value, 0, 1)
        form.addRow("Confidenza", confidence_row)
        form.addRow("FPS inferenza", self._fps_spin)
        form.addRow(self._show_boxes_check)
        form.addRow(self._show_masks_check)

        self._prompt_note = QLabel(
            "YOLOE mantiene prompt e maschere; OpenVINO usa esclusivamente la classe person."
        )
        self._prompt_note.setObjectName("PersonDetectionNote")
        self._prompt_note.setWordWrap(True)

        self._note_label = QLabel("Identità persone non attiva")
        self._note_label.setObjectName("PersonDetectionNote")
        self._note_label.setWordWrap(True)

        telemetry = QGroupBox("Telemetria locale")
        telemetry.setObjectName("PersonDetectionTelemetry")
        telemetry_layout = QGridLayout(telemetry)
        telemetry_layout.setContentsMargins(8, 8, 8, 8)
        telemetry_layout.setHorizontalSpacing(10)
        telemetry_layout.setVerticalSpacing(6)
        self._model_value = self._telemetry_value()
        self._device_value = self._telemetry_value()
        self._latency_value = self._telemetry_value()
        self._fps_value = self._telemetry_value()
        self._detections_value = self._telemetry_value()
        self._status_value = self._telemetry_value()
        rows = (
            ("Modello", self._model_value, "Device/provider", self._device_value),
            ("Latenza", self._latency_value, "FPS medi", self._fps_value),
            ("Rilevati", self._detections_value, "Stato modello", self._status_value),
        )
        for row, (left_label, left_value, right_label, right_value) in enumerate(rows):
            telemetry_layout.addWidget(QLabel(left_label), row, 0)
            telemetry_layout.addWidget(left_value, row, 1)
            telemetry_layout.addWidget(QLabel(right_label), row, 2)
            telemetry_layout.addWidget(right_value, row, 3)
        telemetry_layout.setColumnStretch(1, 1)
        telemetry_layout.setColumnStretch(3, 1)

        self._status_detail = QLabel()
        self._status_detail.setObjectName("PersonDetectionNote")
        self._status_detail.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 2)
        layout.setSpacing(8)
        layout.addLayout(form)
        layout.addWidget(self._prompt_note)
        layout.addWidget(self._note_label)
        layout.addWidget(telemetry)
        layout.addWidget(self._status_detail)

        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(self.APPLY_DELAY_MS)
        self._apply_timer.timeout.connect(self._apply_settings)

        self._enabled_check.toggled.connect(self._on_changed)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self._model_combo.currentIndexChanged.connect(self._on_changed)
        self._prompt_edit.textChanged.connect(self._on_changed)
        self._device_combo.currentIndexChanged.connect(self._on_changed)
        self._precision_combo.currentIndexChanged.connect(self._on_changed)
        self._fallback_combo.currentIndexChanged.connect(self._on_changed)
        self._image_size_spin.valueChanged.connect(self._on_changed)
        self._confidence_slider.valueChanged.connect(self._on_confidence_changed)
        self._fps_spin.valueChanged.connect(self._on_changed)
        self._show_boxes_check.toggled.connect(self._on_changed)
        self._show_masks_check.toggled.connect(self._on_changed)
        self._controller.snapshot_changed.connect(self._on_snapshot)
        self._update_confidence_label()
        self._sync_backend_controls()
        self._on_snapshot(self._controller.snapshot)

    def _telemetry_value(self) -> QLabel:
        value = QLabel("—")
        value.setObjectName("PersonDetectionTelemetryValue")
        value.setWordWrap(True)
        return value

    def _populate_models(self, configured_model: str | None) -> None:
        self._model_combo.clear()
        backend = self._effective_backend()
        if backend == "fake":
            self._model_combo.addItem("Nessun modello (fake/offline)", None)
            self._model_combo.setEnabled(False)
            return

        models = (
            discover_openvino_models(self._repo_root, configured_model)
            if backend == "openvino"
            else discover_yoloe_models(self._repo_root, configured_model)
        )
        for display, value in models:
            path = Path(value)
            absolute = path if path.is_absolute() else self._repo_root / path
            missing = not absolute.is_file() and not absolute.is_dir()
            if backend == "openvino" and Path(value).name.casefold() in {
                "yolo26s.pt",
                "yolo26n.pt",
            }:
                label = f"{display} (download al primo uso)" if missing else display
            else:
                label = f"{display} (mancante)" if missing else display
            self._model_combo.addItem(label, value)

        if not models:
            self._model_combo.addItem(
                "Nessun modello compatibile disponibile",
                None,
            )
            self._model_combo.setEnabled(False)
            return

        self._model_combo.setEnabled(True)
        index = self._model_combo.findData(configured_model)
        if index >= 0:
            self._model_combo.setCurrentIndex(index)
        else:
            self._model_combo.setCurrentIndex(0)

    def _effective_backend(self) -> str:
        selected = str(self._backend_combo.currentData() or self._settings.backend)
        if selected != "auto":
            return selected
        model = (self._settings.model or "").casefold()
        if model.endswith(".onnx"):
            return "onnx"
        if model.endswith(("yolo26s.pt", "yolo26n.pt", "_openvino_model", ".xml")):
            return "openvino"
        return "yoloe"

    def _populate_devices(self, configured_device: str) -> None:
        self._device_combo.clear()
        backend = self._effective_backend()
        if backend == "openvino":
            devices = (("Auto", "auto"), ("CPU", "cpu"), ("GPU", "gpu"))
        elif backend == "fake":
            devices = (("Auto", "auto"), ("CPU", "cpu"))
        else:
            devices = (("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda"))
        for label, value in devices:
            self._device_combo.addItem(label, value)
        index = self._device_combo.findData(configured_device)
        self._device_combo.setCurrentIndex(max(0, index))

    def _sync_backend_controls(self) -> None:
        backend = self._effective_backend()
        is_openvino = backend == "openvino"
        is_fake = backend == "fake"
        if is_openvino:
            self._prompt_edit.setText("person")
            self._prompt_edit.setEnabled(False)
            self._show_masks_check.setChecked(False)
            self._show_masks_check.setEnabled(False)
            self._precision_combo.setEnabled(True)
            self._fallback_combo.setEnabled(True)
        else:
            self._prompt_edit.setEnabled(not is_fake)
            self._show_masks_check.setEnabled(not is_fake)
            self._precision_combo.setEnabled(False)
            self._fallback_combo.setEnabled(False)
        self._model_combo.setEnabled(not is_fake and self._model_combo.count() > 0)
        self._prompt_note.setText(
            "OpenVINO: solo la classe person; maschere disabilitate."
            if is_openvino
            else "Categorie separate da virgola, massimo 20. Le maschere sono disponibili con YOLOE segmentation."
        )

    @Slot(int)
    def _on_backend_changed(self, _index: int) -> None:
        self._populate_devices(str(self._backend_combo.currentData() or "auto"))
        self._populate_models(self._settings.model)
        self._sync_backend_controls()
        self._apply_timer.start()

    def set_config_path(self, path: Path) -> None:
        self._config_path = Path(path)

    def flush_pending(self) -> None:
        if self._apply_timer.isActive():
            self._apply_timer.stop()
            self._apply_settings()

    @Slot()
    def _on_changed(self, *_args: object) -> None:
        self._apply_timer.start()

    @Slot(int)
    def _on_confidence_changed(self, _value: int) -> None:
        self._update_confidence_label()
        self._apply_timer.start()

    def _update_confidence_label(self) -> None:
        self._confidence_value.setText(f"{self._confidence_slider.value()}%")

    def _read_settings(self) -> PersonDetectionSettings:
        model = self._model_combo.currentData() or self._settings.model
        backend = str(self._backend_combo.currentData() or self._settings.backend)
        prompts = normalize_prompts(self._prompt_edit.text())
        if self._effective_backend() == "openvino":
            prompts = ("person",)
            model = model or DEFAULT_OPENVINO_MODEL
        if backend == "fake":
            model = None
        return PersonDetectionSettings(
            enabled=self._enabled_check.isChecked(),
            backend=backend,
            model=str(model) if model else None,
            confidence_threshold=self._confidence_slider.value() / 100.0,
            inference_fps=float(self._fps_spin.value()),
            device=str(self._device_combo.currentData() or "auto"),
            precision=str(self._precision_combo.currentData() or "fp16"),
            fallback_device=str(self._fallback_combo.currentData() or "none"),
            image_size=int(self._image_size_spin.value()),
            classes=prompts,
            show_boxes=self._show_boxes_check.isChecked(),
            prompts=prompts,
            show_masks=self._show_masks_check.isChecked(),
        )

    @Slot()
    def _apply_settings(self) -> None:
        try:
            settings = self._read_settings()
            self._config_path = self._repository.save_person_detection(
                settings,
                current_path=self._config_path,
            )
        except Exception as exc:
            self._status_detail.setStyleSheet("color: #ff8b8b;")
            self._status_detail.setText(f"Salvataggio AI non riuscito: {redact_log_text(exc)}")
            return

        self._settings = settings
        self._status_detail.setStyleSheet("color: #b8c7d8;")
        self._status_detail.setText("Configurazione salvata; applicazione backend in corso…")
        self._sync_backend_controls()
        self._controller.update_settings(settings)

    @Slot(object)
    def _on_snapshot(self, value: object) -> None:
        if not isinstance(value, PersonDetectionSnapshot):
            return
        snapshot = value
        self._model_value.setText(snapshot.model_name)
        if snapshot.actual_device:
            provider = snapshot.provider or "provider n/d"
            backend = snapshot.backend or snapshot.settings.backend
            precision = snapshot.precision or snapshot.settings.precision
            verification = "verificato" if snapshot.device_verified else "candidato"
            self._device_value.setText(
                f"{backend} · {snapshot.actual_device.upper()} · {provider} · {precision} · {verification}"
            )
        else:
            self._device_value.setText(
                f"{snapshot.backend or snapshot.settings.backend} · "
                f"{snapshot.requested_device.upper()}"
            )
        self._latency_value.setText(
            "—" if snapshot.latency_ms is None else f"{snapshot.latency_ms:.1f} ms"
        )
        self._fps_value.setText(
            "—" if snapshot.inference_fps is None else f"{snapshot.inference_fps:.1f} FPS"
        )
        self._detections_value.setText(str(snapshot.detection_count))
        self._status_value.setText(snapshot.status.label)
        self._status_value.setStyleSheet(
            f"color: {self._status_color(snapshot.status)}; font-weight: 700;"
        )
        self._status_detail.setText(snapshot.message)
        if snapshot.status in {
            PersonDetectionStatus.ERROR,
            PersonDetectionStatus.MODEL_MISSING,
        }:
            self._status_detail.setStyleSheet("color: #ff8b8b;")
        else:
            self._status_detail.setStyleSheet("color: #b8c7d8;")

    @staticmethod
    def _status_color(status: PersonDetectionStatus) -> str:
        return {
            PersonDetectionStatus.DISABLED: "#9aa8b8",
            PersonDetectionStatus.LOADING: "#e8c66d",
            PersonDetectionStatus.READY: "#6ddcff",
            PersonDetectionStatus.RUNNING: "#74e09b",
            PersonDetectionStatus.ERROR: "#ff7777",
            PersonDetectionStatus.MODEL_MISSING: "#ff9e6d",
        }[status]

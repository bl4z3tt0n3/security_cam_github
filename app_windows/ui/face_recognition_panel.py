"""Essential PySide6 controls for the opt-in face pipeline."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from app.logging_setup import redact_log_text
from app_windows.config.persistence import CameraConfigRepository
from app_windows.inference.face_recognition_controller import FaceRecognitionController
from app_windows.models.face_recognition_state import (
    FaceRecognitionSettings,
    FaceRecognitionSnapshot,
    FaceRecognitionStatus,
)


class FaceRecognitionPanel(QGroupBox):
    """Keep face settings separate from person detection settings."""

    APPLY_DELAY_MS = 500

    def __init__(
        self,
        controller: FaceRecognitionController,
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
        self._detector_capabilities: list[dict] = []
        self.setObjectName("FaceRecognitionPanel")
        self.setTitle("Face detection → recognition")

        self._detection_enabled = QCheckBox("Abilita face detection")
        self._detection_enabled.setObjectName("faceDetectionEnabled")
        self._detector_combo = QComboBox()
        self._detector_combo.setObjectName("faceDetectorModel")
        self._detector_backend = QComboBox()
        self._detector_backend.setObjectName("faceDetectorBackend")
        self._detector_device = QComboBox()
        self._detector_device.setObjectName("faceDetectorDevice")
        self._detector_path = QLineEdit()
        self._detector_path.setObjectName("faceDetectorPath")
        self._detector_path.setPlaceholderText("models/face_detection/…")
        self._detector_threshold = QDoubleSpinBox()
        self._detector_threshold.setObjectName("faceDetectorThreshold")
        self._detector_threshold.setRange(0.0, 1.0)
        self._detector_threshold.setSingleStep(0.05)
        self._detector_threshold.setDecimals(2)
        self._detector_fps = QSpinBox()
        self._detector_fps.setObjectName("faceDetectorFps")
        self._detector_fps.setRange(1, 60)
        self._detector_fps.setSuffix(" FPS")

        self._landmarks_enabled = QCheckBox("Usa landmarker 0009")
        self._landmarks_enabled.setObjectName("faceLandmarksEnabled")
        self._landmarker_path = QLineEdit()
        self._landmarker_path.setObjectName("faceLandmarkerPath")
        self._landmarker_path.setPlaceholderText("models/face_landmarks/…")
        self._landmarker_device = QComboBox()
        self._landmarker_device.setObjectName("faceLandmarkerDevice")

        self._recognition_enabled = QCheckBox("Abilita face recognition")
        self._recognition_enabled.setObjectName("faceRecognitionEnabled")
        self._recognizer_combo = QComboBox()
        self._recognizer_combo.setObjectName("faceRecognizerModel")
        self._recognizer_backend = QComboBox()
        self._recognizer_backend.setObjectName("faceRecognizerBackend")
        self._recognizer_device = QComboBox()
        self._recognizer_device.setObjectName("faceRecognizerDevice")
        self._recognizer_path = QLineEdit()
        self._recognizer_path.setObjectName("faceRecognizerPath")
        self._recognizer_path.setPlaceholderText("models/face_embedding/…")
        self._recognizer_threshold = QDoubleSpinBox()
        self._recognizer_threshold.setObjectName("faceRecognizerThreshold")
        self._recognizer_threshold.setRange(0.0, 1.0)
        self._recognizer_threshold.setSingleStep(0.01)
        self._recognizer_threshold.setDecimals(3)
        self._recognizer_fps = QSpinBox()
        self._recognizer_fps.setObjectName("faceRecognizerFps")
        self._recognizer_fps.setRange(1, 30)
        self._recognizer_fps.setSuffix(" FPS")
        self._confirmations = QSpinBox()
        self._confirmations.setObjectName("faceRecognitionConfirmations")
        self._confirmations.setRange(1, 20)
        self._confirmation_window = QDoubleSpinBox()
        self._confirmation_window.setObjectName("faceRecognitionConfirmationWindow")
        self._confirmation_window.setRange(0.1, 300.0)
        self._confirmation_window.setSuffix(" s")

        self._status = QLabel()
        self._status.setObjectName("FaceRecognitionStatus")
        self._telemetry = QLabel()
        self._telemetry.setObjectName("FaceRecognitionTelemetry")
        self._telemetry.setWordWrap(True)
        self._gallery = QLabel("Gallery: —")
        self._gallery.setObjectName("FaceGallerySummary")
        self._gallery.setWordWrap(True)

        self._populate_capabilities()
        self._load_settings(self._settings)

        form = QFormLayout()
        form.addRow(self._detection_enabled)
        form.addRow("Detector", self._detector_combo)
        form.addRow("Backend detector", self._detector_backend)
        form.addRow("Device detector", self._detector_device)
        form.addRow("Modello detector", self._detector_path)
        form.addRow("Soglia detector", self._detector_threshold)
        form.addRow("Sampling face", self._detector_fps)
        form.addRow(self._landmarks_enabled)
        form.addRow("Modello landmark", self._landmarker_path)
        form.addRow("Device landmark", self._landmarker_device)
        form.addRow(self._recognition_enabled)
        form.addRow("Recognizer", self._recognizer_combo)
        form.addRow("Backend recognizer", self._recognizer_backend)
        form.addRow("Device recognizer", self._recognizer_device)
        form.addRow("Modello recognizer", self._recognizer_path)
        form.addRow("Soglia modello", self._recognizer_threshold)
        form.addRow("Sampling recognition", self._recognizer_fps)
        form.addRow("Conferme minime", self._confirmations)
        form.addRow("Finestra conferma", self._confirmation_window)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._status)
        layout.addWidget(self._telemetry)
        layout.addWidget(self._gallery)

        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(self.APPLY_DELAY_MS)
        self._apply_timer.timeout.connect(self._apply_settings)
        for widget in (
            self._detection_enabled,
            self._landmarks_enabled,
            self._recognition_enabled,
        ):
            widget.toggled.connect(self._schedule_apply)
        for widget in (
            self._detector_backend,
            self._detector_device,
            self._landmarker_device,
            self._recognizer_combo,
            self._recognizer_backend,
            self._recognizer_device,
        ):
            widget.currentIndexChanged.connect(self._schedule_apply)
        self._detector_combo.currentIndexChanged.connect(self._detector_selection_changed)
        for widget in (
            self._detector_path,
            self._landmarker_path,
            self._recognizer_path,
        ):
            widget.textChanged.connect(self._schedule_apply)
        for widget in (
            self._detector_threshold,
            self._detector_fps,
            self._recognizer_threshold,
            self._recognizer_fps,
            self._confirmations,
            self._confirmation_window,
        ):
            widget.valueChanged.connect(self._schedule_apply)
        controller.snapshot_changed.connect(self._on_snapshot)
        controller.gallery_changed.connect(self._on_gallery)
        self._on_snapshot(controller.snapshot)
        self._on_gallery(controller.gallery)

    def _populate_capabilities(self) -> None:
        rows = self._controller.refresh_capabilities()
        detector_rows = [row for row in rows if row["component"] == "face_detection"]
        self._detector_capabilities = detector_rows
        recognizer_rows = [row for row in rows if row["component"] == "recognition"]
        landmarker_rows = [row for row in rows if row["component"] == "face_landmarks"]
        self._fill_model_combo(self._detector_combo, detector_rows, self._settings.detector_id)
        self._fill_model_combo(self._recognizer_combo, recognizer_rows, self._settings.recognizer_id)
        detector_model = self._detector_combo.currentData() or self._settings.detector_id
        recognizer_model = self._recognizer_combo.currentData() or self._settings.recognizer_id
        self._fill_backend(
            self._detector_backend,
            detector_rows,
            detector_model,
            self._settings.detector_backend,
        )
        self._fill_backend(
            self._recognizer_backend,
            recognizer_rows,
            recognizer_model,
            self._settings.recognizer_backend,
        )
        self._fill_devices(
            self._detector_device,
            detector_rows,
            detector_model,
            self._settings.detector_backend,
            self._settings.detector_device,
        )
        self._fill_devices(
            self._recognizer_device,
            recognizer_rows,
            recognizer_model,
            self._settings.recognizer_backend,
            self._settings.recognizer_device,
        )
        self._fill_devices(
            self._landmarker_device,
            landmarker_rows,
            self._settings.landmarker_id,
            "openvino",
            self._settings.landmarker_device,
        )

    @staticmethod
    def _fill_model_combo(combo: QComboBox, rows: list[dict], configured: str | None) -> None:
        combo.clear()
        values = sorted({str(row["model_id"]) for row in rows if row.get("available")})
        if configured and configured not in values:
            values.insert(0, configured)
        for value in values:
            combo.addItem(value, value)
        if configured:
            index = combo.findData(configured)
            if index >= 0:
                combo.setCurrentIndex(index)

    @staticmethod
    def _fill_devices(
        combo: QComboBox,
        rows: list[dict],
        model_id: str | None,
        backend: str,
        configured: str,
    ) -> None:
        combo.clear()
        values = {
            str(row["device"])
            for row in rows
            if row.get("available")
            and (not model_id or str(row["model_id"]) == model_id)
            and (backend == "auto" or str(row["backend"]) == backend)
        }
        if not values:
            values.add(configured)
        for value in sorted(values):
            combo.addItem(value.upper(), value)
        index = combo.findData(configured)
        combo.setCurrentIndex(max(index, 0))

    @staticmethod
    def _fill_backend(
        combo: QComboBox,
        rows: list[dict],
        model_id: str | None,
        configured: str,
    ) -> None:
        combo.clear()
        values = {
            str(row["backend"])
            for row in rows
            if row.get("available") and (not model_id or str(row["model_id"]) == model_id)
        }
        values.add("auto")
        values.add(configured)
        for value in sorted(values):
            combo.addItem(value, value)
        index = combo.findData(configured)
        combo.setCurrentIndex(max(index, 0))

    @Slot()
    def _detector_selection_changed(self, *_args: object) -> None:
        model_id = self._detector_combo.currentData()
        if not model_id:
            self._schedule_apply()
            return

        capability = next(
            (
                row
                for row in self._detector_capabilities
                if row.get("available")
                and str(row.get("model_id")) == str(model_id)
            ),
            None,
        )
        if capability is None:
            self._schedule_apply()
            return

        backend = str(capability.get("backend") or "auto")
        compatible_devices = sorted(
            {
                str(row["device"])
                for row in self._detector_capabilities
                if row.get("available")
                and str(row.get("model_id")) == str(model_id)
                and str(row.get("backend")) == backend
            }
        )
        current_device = str(self._detector_device.currentData() or "auto")
        selected_device = (
            current_device
            if current_device in compatible_devices
            else "auto"
            if "auto" in compatible_devices
            else compatible_devices[0]
            if compatible_devices
            else "auto"
        )

        widgets = (
            self._detector_path,
            self._detector_backend,
            self._detector_device,
        )
        blocked = [(widget, widget.blockSignals(True)) for widget in widgets]
        try:
            model_path = str(capability.get("model_path") or "").strip()
            if model_path:
                self._detector_path.setText(model_path)
            self._fill_backend(
                self._detector_backend,
                self._detector_capabilities,
                str(model_id),
                backend,
            )
            self._fill_devices(
                self._detector_device,
                self._detector_capabilities,
                str(model_id),
                backend,
                selected_device,
            )
        finally:
            for widget, was_blocked in blocked:
                widget.blockSignals(was_blocked)
        self._schedule_apply()

    def _load_settings(self, settings: FaceRecognitionSettings) -> None:
        self._detection_enabled.setChecked(settings.face_detection_enabled)
        self._landmarks_enabled.setChecked(settings.landmarks_enabled)
        self._recognition_enabled.setChecked(settings.recognition_enabled)
        self._detector_path.setText(settings.detector_model or "")
        self._landmarker_path.setText(settings.landmarker_model or "")
        self._recognizer_path.setText(settings.recognizer_model or "")
        self._detector_threshold.setValue(settings.detector_confidence_threshold)
        self._detector_fps.setValue(round(settings.detector_inference_fps))
        self._recognizer_threshold.setValue(settings.recognition_threshold or 0.0)
        self._recognizer_fps.setValue(round(settings.recognition_inference_fps))
        self._confirmations.setValue(settings.min_confirmations)
        self._confirmation_window.setValue(settings.confirmation_window_seconds)

    def set_config_path(self, path: Path) -> None:
        self._config_path = Path(path)

    def flush_pending(self) -> None:
        if self._apply_timer.isActive():
            self._apply_timer.stop()
            self._apply_settings()

    @Slot()
    def _schedule_apply(self, *_args: object) -> None:
        self._apply_timer.start()

    def _read_settings(self) -> FaceRecognitionSettings:
        detector_model = self._detector_path.text().strip() or None
        recognizer_model = self._recognizer_path.text().strip() or None
        return FaceRecognitionSettings(
            face_detection_enabled=self._detection_enabled.isChecked(),
            detector_id=self._detector_combo.currentData() or None,
            detector_backend=self._detector_backend.currentData() or "auto",
            detector_model=detector_model,
            detector_device=self._detector_device.currentData() or "auto",
            detector_confidence_threshold=self._detector_threshold.value(),
            detector_inference_fps=float(self._detector_fps.value()),
            landmarks_enabled=self._landmarks_enabled.isChecked(),
            landmarker_model=self._landmarker_path.text().strip() or None,
            landmarker_device=self._landmarker_device.currentData() or "auto",
            recognition_enabled=self._recognition_enabled.isChecked(),
            recognizer_id=self._recognizer_combo.currentData() or None,
            recognizer_backend=self._recognizer_backend.currentData() or "auto",
            recognizer_model=recognizer_model,
            recognizer_device=self._recognizer_device.currentData() or "auto",
            recognition_threshold=self._recognizer_threshold.value() or None,
            recognition_inference_fps=float(self._recognizer_fps.value()),
            min_confirmations=self._confirmations.value(),
            confirmation_window_seconds=self._confirmation_window.value(),
            show_face_boxes=True,
            show_landmarks=True,
        )

    @Slot()
    def _apply_settings(self) -> None:
        try:
            settings = self._read_settings()
            self._config_path = self._repository.save_face_analysis(
                settings,
                current_path=self._config_path,
            )
            self._settings = settings
            self._controller.update_settings(settings)
            self._status.setText("Configurazione face salvata; applicazione in corso…")
        except Exception as exc:
            self._status.setText(f"Configurazione face non valida: {redact_log_text(exc)}")

    @Slot(object)
    def _on_snapshot(self, value: object) -> None:
        if not isinstance(value, FaceRecognitionSnapshot):
            return
        detector_device = value.actual_detector_device or value.requested_detector_device or "n/d"
        recognizer_device = value.actual_recognizer_device or value.requested_recognizer_device or "n/d"
        self._status.setText(
            f"{value.status.label}: {value.message} · detector {detector_device} · "
            f"recognizer {recognizer_device}"
        )
        telemetry = value.telemetry
        if telemetry:
            self._telemetry.setText(
                "face {faces_detected} · reject {faces_rejected} · embed {embeddings_generated} · "
                "known {known_recognitions} · unknown {unknown_recognitions} · "
                "landmark {face_landmark_inference_ms} ms · align {alignment_inference_ms} ms".format(
                    **{key: telemetry.get(key, "—") for key in (
                        "faces_detected", "faces_rejected", "embeddings_generated",
                        "known_recognitions", "unknown_recognitions",
                        "face_landmark_inference_ms", "alignment_inference_ms",
                    )}
                )
            )

    @Slot(object)
    def _on_gallery(self, value: object) -> None:
        persons = getattr(value, "persons", ())
        recognizer = getattr(value, "recognizer_id", None) or "nessun recognizer"
        self._gallery.setText(f"Gallery {recognizer}: {len(persons)} persone")


__all__ = ["FaceRecognitionPanel"]

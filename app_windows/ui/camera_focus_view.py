"""Large view for one selected camera, reusing the same snapshot stream."""

from __future__ import annotations

import logging
from pathlib import Path
import time

from PySide6.QtCore import (
    QPointF,
    QRectF,
    QSignalBlocker,
    QSize,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QPainter,
    QPen,
    QPolygonF,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app_windows.config.connection_test import SourceFactory
from app_windows.config.credentials import CredentialStore
from app_windows.inference.person_detection_controller import PersonDetectionController
from app_windows.inference.face_recognition_controller import FaceRecognitionController
from app_windows.models.camera_display_transform import (
    CameraDisplayTransform,
    CameraDisplayTransformStore,
)
from app_windows.models.camera_view_state import CameraViewSnapshot
from app_windows.models.person_detection_state import (
    PersonDetectionSnapshot,
    PersonDetectionStatus,
)
from app_windows.models.face_recognition_state import (
    FaceRecognitionSnapshot,
    FaceRecognitionStatus,
)
from app_windows.monitor_controller import CameraMonitorController
from app_windows.video.display_transform import transform_video_pixmap
from app_windows.video.frame_converter import decoded_frame_size, frame_to_qimage
from app_windows.video.frame_geometry import keep_aspect_ratio_rect
from app_windows.video.detection_geometry import (
    map_detection_bbox_to_widget,
    map_detection_polygon_to_widget,
)

from .camera_configuration_panel import CameraConfigurationPanel
from .theme import focus_view_stylesheet


def scale_video_pixmap(pixmap: QPixmap, target_size: QSize) -> QPixmap:
    """Fit a source frame without cropping or distorting its aspect ratio."""

    if pixmap.isNull() or target_size.isEmpty():
        return QPixmap()
    rect = keep_aspect_ratio_rect(
        pixmap.width(),
        pixmap.height(),
        target_size.width(),
        target_size.height(),
    )
    return pixmap.scaled(
        QSize(rect.width, rect.height),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class CameraFocusView(QWidget):
    back_requested = Signal()

    def __init__(
        self,
        *,
        controller: CameraMonitorController,
        config_path: Path,
        repo_root: Path,
        credentials: CredentialStore,
        display_transforms: CameraDisplayTransformStore | None = None,
        source_factory: SourceFactory | None = None,
        read_timeout_s: float = 3.0,
        logger: logging.Logger | None = None,
        person_detection_controller: PersonDetectionController | None = None,
        face_recognition_controller: FaceRecognitionController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._logger = logger or logging.getLogger(__name__)
        self._last_geometry_log: tuple[object, ...] | None = None
        self._image: QImage | None = None
        self._snapshot: CameraViewSnapshot | None = None
        self._person_detection_snapshot: PersonDetectionSnapshot | None = None
        self._face_recognition_snapshot: FaceRecognitionSnapshot | None = None
        self._display_transforms = (
            display_transforms
            if display_transforms is not None
            else CameraDisplayTransformStore(self)
        )
        self._display_transforms.transform_changed.connect(
            self._on_display_transform_changed
        )
        self.setObjectName("CameraFocusView")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet(focus_view_stylesheet())

        self._back_button = QPushButton("← Torna alle 6 camere")
        self._back_button.setObjectName("BackButton")
        self._back_button.clicked.connect(self.back_requested.emit)
        self._title_label = QLabel()
        self._title_label.setObjectName("FocusTitle")
        self._status_label = QLabel()
        self._status_label.setObjectName("FocusMeta")
        header = QHBoxLayout()
        header.addWidget(self._back_button)
        header.addWidget(self._title_label, 1)
        header.addWidget(self._status_label)

        self._video_label = QLabel("Seleziona una camera")
        self._video_label.setObjectName("FocusVideo")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._video_label.setScaledContents(False)
        self._video_label.setMinimumSize(320, 240)

        self._rotation_button = QPushButton("↺ Ruota 90°")
        self._rotation_button.setObjectName("RotateButton")
        self._rotation_button.setToolTip(
            "Ruota il video di 90° in senso antiorario"
        )
        self._rotation_button.setAccessibleName(
            "Ruota video di 90 gradi in senso antiorario"
        )
        self._rotation_button.setEnabled(False)
        self._rotation_button.clicked.connect(self._rotate_video_counterclockwise)

        self._mirror_button = QPushButton("↔ Specchia")
        self._mirror_button.setObjectName("MirrorButton")
        self._mirror_button.setCheckable(True)
        self._mirror_button.setToolTip("Specchia orizzontalmente il video")
        self._mirror_button.setAccessibleName("Specchia il video orizzontalmente")
        self._mirror_button.setEnabled(False)
        self._mirror_button.toggled.connect(self._set_video_mirrored)

        self._video_surface = QWidget()
        self._video_surface.setObjectName("FocusVideoSurface")
        self._video_controls = QWidget(self._video_surface)
        video_controls_layout = QHBoxLayout(self._video_controls)
        video_controls_layout.setContentsMargins(0, 0, 0, 0)
        video_controls_layout.setSpacing(8)
        video_controls_layout.addWidget(self._rotation_button)
        video_controls_layout.addWidget(self._mirror_button)

        video_surface_layout = QGridLayout(self._video_surface)
        video_surface_layout.setContentsMargins(8, 8, 8, 8)
        video_surface_layout.addWidget(self._video_label, 0, 0)
        video_surface_layout.addWidget(
            self._video_controls,
            0,
            0,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
        )
        self._video_controls.raise_()

        self._meta_label = QLabel()
        self._meta_label.setObjectName("FocusMeta")

        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(8)
        video_layout.addLayout(header)
        video_layout.addWidget(self._video_surface, 1)
        video_layout.addWidget(self._meta_label)

        self._configuration_panel = CameraConfigurationPanel(
            controller,
            config_path=config_path,
            repo_root=repo_root,
            credentials=credentials,
            source_factory=source_factory,
            read_timeout_s=read_timeout_s,
            logger=logger,
            person_detection_controller=person_detection_controller,
            face_recognition_controller=face_recognition_controller,
            parent=self,
        )
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("FocusSplitter")
        self._splitter.addWidget(video_container)
        self._splitter.addWidget(self._configuration_panel)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, True)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([760, 260])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)
        layout.addWidget(self._splitter)

    @property
    def snapshot(self) -> CameraViewSnapshot | None:
        return self._snapshot

    @property
    def configuration_panel(self) -> CameraConfigurationPanel:
        return self._configuration_panel

    @property
    def display_transform(self) -> CameraDisplayTransform:
        if self._snapshot is None:
            return CameraDisplayTransform()
        return self._display_transforms.transform_for(self._snapshot.slot.camera_id)

    @property
    def rotation_degrees(self) -> int:
        """Current counterclockwise rotation for the selected camera."""

        return self.display_transform.rotation_degrees

    @property
    def is_mirrored(self) -> bool:
        """Whether the selected camera is horizontally mirrored."""

        return self.display_transform.mirrored

    @property
    def person_detection_snapshot(self) -> PersonDetectionSnapshot | None:
        return self._person_detection_snapshot

    def set_snapshot(self, snapshot: CameraViewSnapshot) -> None:
        previous_camera_id = self._snapshot.slot.camera_id if self._snapshot is not None else None
        self._snapshot = snapshot
        self._title_label.setText(snapshot.slot.name)
        self._status_label.setText(snapshot.status.label)
        self._rotation_button.setEnabled(True)
        self._mirror_button.setEnabled(True)
        self._sync_mirror_button(self.is_mirrored)
        if previous_camera_id != snapshot.slot.camera_id:
            self._person_detection_snapshot = None
            self._face_recognition_snapshot = None
            self._configuration_panel.set_camera_slot(snapshot.slot)
        if snapshot.frame is not None:
            try:
                self._image = frame_to_qimage(snapshot.frame.frame)
                self._render_image()
            except (TypeError, ValueError):
                self._image = None
                self._video_label.setText("Frame non valido")
        elif self._image is None:
            self._video_label.setText(snapshot.message)

        details = [snapshot.message]
        if snapshot.frame is not None:
            try:
                width, height = decoded_frame_size(snapshot.frame.frame)
                details.append(f"decoded {width}x{height}")
            except (TypeError, ValueError):
                details.append("decoded n/d")
        elif snapshot.stream_info is not None:
            info = snapshot.stream_info
            if info.width and info.height:
                details.append(f"{info.width}×{info.height}")
            if info.codec:
                details.append(info.codec)
        if snapshot.last_frame_age_s is not None:
            details.append(f"ultimo frame {snapshot.last_frame_age_s:.1f}s fa")
        if snapshot.display_fps is not None:
            details.append(f"UI {snapshot.display_fps:.0f} FPS")
        if snapshot.worker_snapshot is not None:
            details.append(f"drop {snapshot.worker_snapshot.dropped_frames}")
        self._meta_label.setText(" • ".join(details))
        self.setAccessibleName(f"Vista ingrandita {snapshot.slot.name}")

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._render_image()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.back_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _render_image(self) -> None:
        if self._image is None or self._video_label.size().isEmpty():
            return
        pixmap = QPixmap.fromImage(self._image)
        display_transform = self.display_transform
        transformed = transform_video_pixmap(pixmap, display_transform)
        target = self._video_label.size()
        rect = keep_aspect_ratio_rect(
            transformed.width(),
            transformed.height(),
            target.width(),
            target.height(),
        )
        geometry_log = (
            pixmap.width(),
            pixmap.height(),
            transformed.width(),
            transformed.height(),
            target.width(),
            target.height(),
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            display_transform.rotation_degrees,
            display_transform.mirrored,
        )
        if geometry_log != self._last_geometry_log:
            self._last_geometry_log = geometry_log
            self._logger.info(
                "WINDOWS_FRAME decoded=%sx%s effective=%sx%s widget=%sx%s "
                "rendered_rect=%s,%s,%sx%s aspect_preserved=true "
                "manual_override_rotation=%s manual_override_mirrored=%s",
                pixmap.width(),
                pixmap.height(),
                transformed.width(),
                transformed.height(),
                target.width(),
                target.height(),
                rect.x,
                rect.y,
                rect.width,
                rect.height,
                display_transform.rotation_degrees,
                display_transform.mirrored,
            )
        rendered = scale_video_pixmap(transformed, self._video_label.size())
        if self._should_draw_detections():
            rendered = self._draw_detections(rendered, display_transform)
        if self._should_draw_face_overlays():
            rendered = self._draw_face_overlays(rendered, display_transform)
        self._video_label.setPixmap(rendered)
        self._video_controls.raise_()

    def set_person_detection_snapshot(self, snapshot: PersonDetectionSnapshot | None) -> None:
        """Attach service-derived detections to the currently focused video."""

        if snapshot is not None and self._snapshot is not None:
            if snapshot.camera_id not in {None, self._snapshot.slot.camera_id}:
                return
        self._person_detection_snapshot = snapshot
        self._render_image()

    def set_face_recognition_snapshot(self, snapshot: FaceRecognitionSnapshot | None) -> None:
        """Attach frame-space face/recognition overlays to the focused video."""

        if snapshot is not None and self._snapshot is not None:
            if snapshot.camera_id not in {None, self._snapshot.slot.camera_id}:
                return
        self._face_recognition_snapshot = snapshot
        self._render_image()

    def _should_draw_face_overlays(self) -> bool:
        value = self._face_recognition_snapshot
        if (
            value is None
            or self._snapshot is None
            or self._snapshot.frame is None
            or value.camera_id != self._snapshot.slot.camera_id
            or value.status is not FaceRecognitionStatus.RUNNING
            or not value.overlays
            or value.frame_sequence != self._snapshot.frame.sequence
        ):
            return False
        return True

    def _should_draw_detections(self) -> bool:
        detection = self._person_detection_snapshot
        if (
            detection is None
            or self._snapshot is None
            or detection.camera_id != self._snapshot.slot.camera_id
            or detection.status is not PersonDetectionStatus.RUNNING
            or not (detection.settings.show_boxes or detection.settings.show_masks)
            or not detection.detections
            or detection.result_monotonic is None
            or detection.source_width is None
            or detection.source_height is None
        ):
            return False
        max_age = max(1.0, 2.0 / detection.settings.inference_fps)
        if detection.is_obsolete or time.monotonic() - detection.result_monotonic > max_age:
            return False
        return (
            detection.source_width == self._image.width()
            and detection.source_height == self._image.height()
        )

    def _draw_detections(
        self,
        pixmap: QPixmap,
        display_transform: CameraDisplayTransform,
    ) -> QPixmap:
        detection = self._person_detection_snapshot
        if (
            detection is None
            or detection.source_width is None
            or detection.source_height is None
            or pixmap.isNull()
        ):
            return pixmap

        annotated = QPixmap(pixmap)
        painter = QPainter(annotated)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        pen = QPen(QColor("#72ed92"))
        pen.setWidth(2)
        for detection_item in detection.detections:
            if detection.settings.show_masks and detection_item.mask_polygon:
                polygon = map_detection_polygon_to_widget(
                    detection_item.mask_polygon,
                    detection.source_width,
                    detection.source_height,
                    annotated.width(),
                    annotated.height(),
                    display_transform,
                )
                painter.setPen(QPen(QColor("#57e6a1"), 1))
                painter.setBrush(QBrush(QColor(87, 230, 161, 70)))
                painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in polygon]))

            if not detection.settings.show_boxes:
                continue
            box = map_detection_bbox_to_widget(
                detection_item.bbox,
                detection.source_width,
                detection.source_height,
                annotated.width(),
                annotated.height(),
                display_transform,
            )
            rect = QRectF(box.x, box.y, box.width, box.height)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            painter.drawRect(rect)
            category = (
                "Persona"
                if detection_item.label.casefold() == "person"
                else detection_item.label
            )
            label = f"{category} {detection_item.confidence:.2f}"
            label_rect = painter.boundingRect(
                rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                label,
            )
            label_rect.moveTop(max(0.0, rect.top() - label_rect.height()))
            painter.fillRect(label_rect, QBrush(QColor(8, 31, 24, 220)))
            painter.setPen(QColor("#d9ffe2"))
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                label,
            )
        painter.end()
        return annotated

    def _draw_face_overlays(
        self,
        pixmap: QPixmap,
        display_transform: CameraDisplayTransform,
    ) -> QPixmap:
        value = self._face_recognition_snapshot
        if value is None or self._image is None or pixmap.isNull():
            return pixmap
        annotated = QPixmap(pixmap)
        painter = QPainter(annotated)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        for overlay in value.overlays:
            box = map_detection_bbox_to_widget(
                overlay.bbox,
                self._image.width(),
                self._image.height(),
                annotated.width(),
                annotated.height(),
                display_transform,
            )
            known = overlay.recognition_status == "known"
            color = QColor("#65e6ff" if known else "#ffd166")
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = QRectF(box.x, box.y, box.width, box.height)
            painter.drawRect(rect)
            if overlay.landmarks:
                points = map_detection_polygon_to_widget(
                    overlay.landmarks,
                    self._image.width(),
                    self._image.height(),
                    annotated.width(),
                    annotated.height(),
                    display_transform,
                )
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color, 1))
                for x, y in points:
                    painter.drawEllipse(QPointF(x, y), 2.5, 2.5)
            label = f"track {overlay.track_id} · {overlay.recognition_status.upper()}"
            if overlay.person_name:
                label += f" · {overlay.person_name}"
            if overlay.score is not None:
                label += f" {overlay.score:.2f}"
            label_rect = painter.boundingRect(
                rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                label,
            )
            label_rect.moveTop(min(annotated.height() - label_rect.height(), rect.bottom()))
            painter.fillRect(label_rect, QBrush(QColor(8, 26, 35, 220)))
            painter.setPen(QColor("#e5fbff"))
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                label,
            )
        painter.end()
        return annotated

    def _rotate_video_counterclockwise(self) -> None:
        if self._snapshot is None:
            return
        self._display_transforms.rotate_counterclockwise(self._snapshot.slot.camera_id)

    def _set_video_mirrored(self, mirrored: bool) -> None:
        if self._snapshot is None:
            return
        self._display_transforms.set_mirrored(self._snapshot.slot.camera_id, mirrored)

    @Slot(str, int, bool)
    def _on_display_transform_changed(
        self,
        camera_id: str,
        _rotation_degrees: int,
        mirrored: bool,
    ) -> None:
        if self._snapshot is None or self._snapshot.slot.camera_id != camera_id:
            return
        self._sync_mirror_button(mirrored)
        self._render_image()

    def _sync_mirror_button(self, mirrored: bool) -> None:
        blocker = QSignalBlocker(self._mirror_button)
        self._mirror_button.setChecked(mirrored)
        del blocker

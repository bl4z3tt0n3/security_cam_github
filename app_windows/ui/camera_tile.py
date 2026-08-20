"""Clickable video tile with independent camera status overlay."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QVBoxLayout

from app_windows.models.camera_display_transform import (
    CameraDisplayTransform,
    CameraDisplayTransformStore,
)
from app_windows.models.camera_view_state import CameraViewSnapshot, CameraViewStatus
from app_windows.video.display_transform import transform_video_pixmap
from app_windows.video.frame_converter import frame_to_qimage

from .theme import camera_tile_stylesheet, status_badge_stylesheet, status_color


class CameraTile(QFrame):
    clicked = Signal(str)

    def __init__(
        self,
        camera_id: str,
        *,
        display_transforms: CameraDisplayTransformStore | None = None,
        parent: QFrame | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self._image: QImage | None = None
        self._snapshot: CameraViewSnapshot | None = None
        self._display_transforms = (
            display_transforms
            if display_transforms is not None
            else CameraDisplayTransformStore(self)
        )
        self._display_transforms.transform_changed.connect(
            self._on_display_transform_changed
        )
        self.setObjectName("CameraTile")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(220, 140)
        self.setStyleSheet(camera_tile_stylesheet())

        self._video_label = QLabel("Nessun frame")
        self._video_label.setObjectName("CameraVideo")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._video_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._name_label = QLabel()
        self._name_label.setObjectName("CameraName")
        self._name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._status_label = QLabel()
        self._status_label.setObjectName("CameraStatus")
        self._status_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._message_label = QLabel()
        self._message_label.setObjectName("CameraMessage")
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message_label.setWordWrap(True)
        self._message_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        surface = QFrame()
        surface.setObjectName("CameraSurface")
        surface.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        surface_layout = QGridLayout(surface)
        surface_layout.setContentsMargins(0, 0, 0, 0)
        surface_layout.addWidget(self._video_label, 0, 0)
        surface_layout.addWidget(
            self._name_label,
            0,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
        )
        surface_layout.addWidget(
            self._status_label,
            0,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )
        surface_layout.addWidget(self._message_label, 0, 0, Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(surface)

    @property
    def snapshot(self) -> CameraViewSnapshot | None:
        return self._snapshot

    @property
    def display_transform(self) -> CameraDisplayTransform:
        return self._display_transforms.transform_for(self.camera_id)

    def set_snapshot(self, snapshot: CameraViewSnapshot) -> None:
        self._snapshot = snapshot
        self._name_label.setText(snapshot.slot.name)
        self._status_label.setText(snapshot.status.label)
        self._status_label.setStyleSheet(status_badge_stylesheet(snapshot.status.name))
        self._message_label.setText(snapshot.message)
        self._message_label.setVisible(snapshot.status is not CameraViewStatus.LIVE)
        if snapshot.frame is not None:
            try:
                self._image = frame_to_qimage(snapshot.frame.frame)
                self._message_label.setVisible(snapshot.status is not CameraViewStatus.LIVE)
                self._render_image()
            except (TypeError, ValueError):
                self._image = None
                self._message_label.setText("Frame non valido")
                self._message_label.setVisible(True)
        elif self._image is None:
            self._video_label.clear()
            self._video_label.setText(snapshot.message)

        self.setAccessibleName(f"{snapshot.slot.name}, {snapshot.status.label}")
        self.setToolTip(f"{snapshot.slot.name} — {snapshot.status.label}")

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._render_image()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[name-defined]
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.camera_id)
            self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit(self.camera_id)
            event.accept()
            return
        super().keyPressEvent(event)

    def _render_image(self) -> None:
        if self._image is None or self._video_label.size().isEmpty():
            return
        pixmap = transform_video_pixmap(
            QPixmap.fromImage(self._image),
            self.display_transform,
        )
        scaled = pixmap.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    @Slot(str, int, bool)
    def _on_display_transform_changed(
        self,
        camera_id: str,
        _rotation_degrees: int,
        _mirrored: bool,
    ) -> None:
        if camera_id == self.camera_id:
            self._render_image()

    @staticmethod
    def _status_color(status: CameraViewStatus) -> str:
        return status_color(status.name)

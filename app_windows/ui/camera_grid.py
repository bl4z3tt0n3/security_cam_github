"""Three-by-two camera grid."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QWidget

from app_windows.models.camera_display_transform import CameraDisplayTransformStore
from app_windows.models.camera_view_state import CameraSlot, CameraViewSnapshot

from .camera_tile import CameraTile


class CameraGrid(QWidget):
    camera_selected = Signal(str)

    def __init__(
        self,
        slots: tuple[CameraSlot, ...],
        *,
        display_transforms: CameraDisplayTransformStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if len(slots) != 6:
            raise ValueError("camera grid requires exactly six slots")
        self._display_transforms = (
            display_transforms
            if display_transforms is not None
            else CameraDisplayTransformStore(self)
        )
        self._tiles: dict[str, CameraTile] = {}
        layout = QGridLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        for index, slot in enumerate(slots):
            tile = CameraTile(
                slot.camera_id,
                display_transforms=self._display_transforms,
                parent=self,
            )
            tile.clicked.connect(self.camera_selected.emit)
            self._tiles[slot.camera_id] = tile
            layout.addWidget(tile, index // 3, index % 3)
            layout.setColumnStretch(index % 3, 1)
            layout.setRowStretch(index // 3, 1)

    @property
    def tiles(self) -> dict[str, CameraTile]:
        return dict(self._tiles)

    def set_snapshot(self, snapshot: CameraViewSnapshot) -> None:
        tile = self._tiles.get(snapshot.slot.camera_id)
        if tile is not None:
            tile.set_snapshot(snapshot)

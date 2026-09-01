"""Pure data models for six independent camera tiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from app.config import AppConfig, is_placeholder_url
from app.video.base import FramePacket, StreamInfo
from app.video.worker import CameraWorkerSnapshot


class CameraViewStatus(str, Enum):
    CONNECTING = "CONNECTING"
    LIVE = "LIVE"
    OFFLINE = "OFFLINE"
    RECONNECTING = "RECONNECTING"
    DISABLED = "DISABLED"
    ERROR = "ERROR"
    NOT_CONFIGURED = "NOT_CONFIGURED"

    @property
    def label(self) -> str:
        return {
            CameraViewStatus.CONNECTING: "CONNESSIONE",
            CameraViewStatus.LIVE: "LIVE",
            CameraViewStatus.OFFLINE: "OFFLINE",
            CameraViewStatus.RECONNECTING: "RICONNESSIONE",
            CameraViewStatus.DISABLED: "DISABILITATA",
            CameraViewStatus.ERROR: "ERRORE",
            CameraViewStatus.NOT_CONFIGURED: "NON CONFIGURATA",
        }[self]


@dataclass(frozen=True)
class CameraSlot:
    """One logical grid position, including unconfigured positions."""

    slot_index: int
    camera_id: str
    name: str
    enabled: bool
    configured: bool
    stream_url: str | None = None
    rtsp_transport: str = "tcp"

    def with_runtime_source(self, stream_url: str) -> "CameraSlot":
        return replace(self, configured=True, enabled=True, stream_url=stream_url)


@dataclass(frozen=True)
class CameraViewSnapshot:
    """Immutable point-in-time state sent from the controller to the widgets."""

    slot: CameraSlot
    status: CameraViewStatus
    message: str
    frame: FramePacket | None = None
    stream_info: StreamInfo | None = None
    worker_snapshot: CameraWorkerSnapshot | None = None
    last_frame_age_s: float | None = None
    display_fps: float | None = None
    hardware_acceleration: str | None = None


def camera_slots_from_config(config: AppConfig, *, slot_count: int = 6) -> tuple[CameraSlot, ...]:
    """Normalize any central configuration into exactly ``slot_count`` UI slots."""

    if slot_count < 1:
        raise ValueError("slot_count must be greater than zero")

    configured_cameras = tuple(config.cameras[:slot_count])
    used_ids = {camera.id for camera in configured_cameras}
    slots: list[CameraSlot] = []
    for index in range(1, slot_count + 1):
        camera = configured_cameras[index - 1] if index <= len(configured_cameras) else None
        if camera is not None:
            camera_name = camera.name or f"Camera {index}"
            slots.append(
                CameraSlot(
                    slot_index=index,
                    camera_id=camera.id,
                    name=camera_name,
                    enabled=camera.enabled,
                    configured=not is_placeholder_url(camera.stream_url),
                    stream_url=camera.stream_url,
                    rtsp_transport=camera.rtsp_transport or config.video.rtsp_transport,
                )
            )
            continue

        fallback_id = f"slot_{index}"
        while fallback_id in used_ids:
            fallback_id = f"{fallback_id}_ui"
        used_ids.add(fallback_id)
        slots.append(
            CameraSlot(
                slot_index=index,
                camera_id=fallback_id,
                name=f"Camera {index}",
                enabled=False,
                configured=False,
                rtsp_transport=config.video.rtsp_transport,
            )
        )
    return tuple(slots)

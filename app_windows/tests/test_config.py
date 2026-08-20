from __future__ import annotations

import logging

from app.config import AppConfig, CameraConfig

from app_windows.config.ui_config import UiSettings
from app_windows.models.camera_view_state import (
    CameraViewStatus,
    camera_slots_from_config,
)


def test_camera_slots_always_contain_six_logical_positions() -> None:
    config = AppConfig(
        cameras=[
            CameraConfig(
                id="huawei",
                name="Ingresso",
                enabled=True,
                stream_url="rtsp://camera.local/live",
            )
        ]
    )

    slots = camera_slots_from_config(config)

    assert len(slots) == 6
    assert slots[0].name == "Ingresso"
    assert slots[0].configured is True
    assert slots[0].enabled is True
    assert slots[1].name == "Camera 2"
    assert slots[1].configured is False
    assert slots[1].enabled is False


def test_placeholder_and_disabled_camera_are_not_started() -> None:
    config = AppConfig(
        cameras=[
            CameraConfig(
                id="placeholder",
                enabled=True,
                stream_url="${CAMERA_URL}",
            ),
            CameraConfig(
                id="disabled",
                name="Cortile",
                enabled=False,
                stream_url="rtsp://camera.local/cortile",
            ),
        ]
    )

    slots = camera_slots_from_config(config)

    assert slots[0].configured is False
    assert slots[1].configured is True
    assert slots[1].enabled is False
    assert UiSettings.from_app_config(config).display_fps == 15.0
    assert CameraViewStatus.NOT_CONFIGURED.label == "NON CONFIGURATA"
    assert CameraViewStatus.DISABLED.label == "DISABILITATA"


def test_windows_ui_configuration_is_independent_from_inference_fps() -> None:
    config = AppConfig.model_validate(
        {
            "inference": {"person_detection_fps": 2},
            "windows_ui": {
                "display_fps": 17,
                "start_maximized": False,
                "remember_window_geometry": False,
            },
        }
    )

    settings = UiSettings.from_app_config(config)

    assert settings.display_fps == 17
    assert settings.start_maximized is False
    assert settings.remember_window_geometry is False
    assert config.inference.person_detection_fps == 2


def test_windows_provider_factory_forwards_central_rtsp_configuration(monkeypatch) -> None:
    import app_windows.main as windows_main

    config = AppConfig.model_validate(
        {
            "cameras": [
                {
                    "id": "huawei_p30",
                    "name": "Huawei P30 Pro",
                    "enabled": True,
                    "stream_url": "rtsp://user:secret@camera.local/live",
                }
            ],
            "video": {
                "rtsp_transport": "tcp",
                "open_timeout_seconds": 7,
                "read_timeout_seconds": 4,
                "max_reconnect_attempts": 0,
                "max_buffer_frames": 1,
            },
        }
    )
    captured: dict[str, object] = {}

    def fake_source(url: str, *, video, rtsp_transport, logger):
        captured["url"] = url
        captured["video"] = video
        captured["rtsp_transport"] = rtsp_transport
        captured["logger"] = logger
        return object()

    class StubProvider:
        def __init__(self, camera_id: str, source: object, **kwargs: object) -> None:
            captured["camera_id"] = camera_id
            captured["source"] = source
            captured["provider_kwargs"] = kwargs

    monkeypatch.setattr(windows_main, "create_opencv_source", fake_source)
    monkeypatch.setattr(windows_main, "BackendFrameProvider", StubProvider)

    slot = camera_slots_from_config(config)[0]
    factory = windows_main._build_provider_factory(
        config=config,
        fake_mode=False,
        fake_offline_camera=None,
        fake_reconnect_camera=None,
        logger=logging.getLogger("windows-wiring-test"),
    )
    provider = factory(slot)

    assert isinstance(provider, StubProvider)
    assert captured["url"] == "rtsp://user:secret@camera.local/live"
    assert captured["video"] is config.video
    assert captured["video"].rtsp_transport == "tcp"
    assert captured["rtsp_transport"] == "tcp"
    assert captured["provider_kwargs"]["read_timeout_s"] == 4
    assert captured["provider_kwargs"]["max_reconnect_attempts"] == 0
    assert captured["provider_kwargs"]["max_buffer_frames"] == 1

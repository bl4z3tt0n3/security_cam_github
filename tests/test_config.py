from __future__ import annotations

from pathlib import Path
import warnings

import pytest

from app.config import (
    AppConfig,
    ConfigurationError,
    expand_environment_variables,
    load_config,
    validate_stream_url,
)


def test_environment_variables_are_expanded_and_unresolved_placeholders_preserved() -> None:
    assert expand_environment_variables(
        "url=${CAMERA_URL};missing=${MISSING}",
        {"CAMERA_URL": "rtsp://camera/stream"},
    ) == "url=rtsp://camera/stream;missing=${MISSING}"


def test_example_configuration_loads() -> None:
    config = load_config(Path("config/config.example.yaml"))
    assert isinstance(config, AppConfig)
    assert [camera.id for camera in config.cameras] == [
        "huawei_p30",
        "cam_2",
        "cam_3",
        "cam_4",
        "cam_5",
        "cam_6",
    ]
    assert config.cameras[0].name == "Huawei P30 Pro"
    assert config.cameras[0].enabled is True
    assert all(camera.enabled is False for camera in config.cameras[1:])
    assert config.cameras[0].id == "huawei_p30"
    assert config.video.rtsp_transport == "tcp"
    assert config.video.max_reconnect_attempts == 0
    assert config.video.max_buffer_frames == 1
    assert config.video.hardware_acceleration == "mfx"
    assert config.inference.person_detection_fps == 3
    assert config.person_detection.inference_fps == 3
    assert config.person_detection.enabled is False
    assert config.person_detection.backend == "openvino"
    assert config.person_detection.model == "models/yolo26s.pt"
    assert config.person_detection.prompts == ["person"]
    assert config.person_detection.classes == ["person"]
    assert config.person_detection.confidence_threshold == pytest.approx(0.45)
    assert config.person_detection.precision == "fp16"
    assert config.person_detection.device == "gpu"
    assert config.person_detection.fallback_device == "cpu"
    assert config.person_detection.image_size == 640
    assert config.person_detection.openvino_performance_mode == "latency"
    assert config.person_detection.openvino_num_streams == 1
    assert config.person_detection.openvino_num_requests == 1
    assert config.person_detection.max_process_ram_mb == 6144
    assert config.hardware_optimization.enabled is True
    assert config.hardware_optimization.profile == "intel_iris_xe"
    assert config.motion_detection.enabled is False
    assert config.motion_detection.pixel_threshold == 25
    assert config.motion_detection.min_changed_fraction == pytest.approx(0.01)
    assert config.motion_detection.resize_width == 320
    assert config.motion_detection.warmup_frames == 1
    assert config.events.save_snapshot is True
    assert config.events.known_person_cooldown_seconds == 30
    assert config.events.unknown_person_cooldown_seconds == 15
    assert config.recording.enabled is False
    assert config.windows_ui.display_fps == 15
    assert config.windows_ui.start_maximized is True


def test_canonical_inference_fps_takes_precedence_over_legacy_key() -> None:
    legacy_only = AppConfig.model_validate(
        {"person_detection": {"inference_fps": 5}}
    )
    assert legacy_only.inference.person_detection_fps == 5
    assert legacy_only.person_detection.inference_fps == 5

    both_keys = AppConfig.model_validate(
        {
            "inference": {"person_detection_fps": 3},
            "person_detection": {"inference_fps": 5},
        }
    )
    assert both_keys.inference.person_detection_fps == 3
    assert both_keys.person_detection.inference_fps == 3


def test_person_detection_prompts_are_normalized_and_allow_multiple_categories() -> None:
    config = AppConfig.model_validate(
        {"person_detection": {"prompts": " person, dog, PERSON "}}
    )

    assert config.person_detection.prompts == ["person", "dog"]


def test_person_detection_rejects_empty_prompt_categories() -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate({"person_detection": {"prompts": []}})


def test_person_detection_accepts_imgsz_alias_and_restricts_openvino_classes() -> None:
    config = AppConfig.model_validate(
        {
            "person_detection": {
                "backend": "openvino",
                "model": "models/yolo26s.pt",
                "imgsz": 512,
                "classes": ["PERSON"],
            }
        }
    )
    assert config.person_detection.image_size == 512
    assert config.person_detection.imgsz == 512
    assert config.person_detection.classes == ["person"]

    with pytest.raises(ValueError, match="classes"):
        AppConfig.model_validate(
            {"person_detection": {"backend": "openvino", "classes": ["person", "car"]}}
        )


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), "invalid"])
def test_invalid_canonical_inference_fps_is_rejected(fps: object) -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate({"inference": {"person_detection_fps": fps}})


def test_duplicate_camera_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="camera ids must be unique"):
        AppConfig.model_validate(
            {
                "cameras": [
                    {"id": "same", "stream_url": "rtsp://one/stream"},
                    {"id": "same", "stream_url": "rtsp://two/stream"},
                ]
            }
        )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("rtsp://192.168.1.20:8554/live", "rtsp://192.168.1.20:8554/live"),
        ("http://camera.local/mjpeg", "http://camera.local/mjpeg"),
    ],
)
def test_stream_url_validation(url: str, expected: str) -> None:
    assert validate_stream_url(url) == expected


@pytest.mark.parametrize("url", [None, "", "${CAMERA_HUAWEI_URL}", "ftp://camera/stream"])
def test_invalid_or_unconfigured_stream_url_is_rejected(url: str | None) -> None:
    with pytest.raises(ConfigurationError):
        validate_stream_url(url)


def test_example_configuration_contains_face_quality_settings() -> None:
    config = load_config("config/config.example.yaml")
    assert config.face_quality.min_width == 80
    assert config.face_quality.min_height == 80
    assert config.face_quality.blur_threshold == pytest.approx(40)
    assert config.face_quality.min_brightness == pytest.approx(30)
    assert config.face_quality.max_brightness == pytest.approx(225)
    assert config.face_quality.min_confidence == pytest.approx(0.5)
    assert config.face_detection.model == (
        "models/face_detection/scrfd_2.5g_kps/scrfd_2.5g_bnkps.onnx"
    )
    assert config.face_detection.device == "cpu"
    assert config.recognition.model == (
        "models/face_embedding/face-reidentification-retail-0095/"
        "face-reidentification-retail-0095.xml"
    )
    assert config.recognition.model_version == "1"
    assert config.recognition.device == "cpu"


def test_face_quality_rejects_reversed_brightness_range() -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {"face_quality": {"min_brightness": 230, "max_brightness": 20}}
        )


def test_enabled_face_stage_requires_concrete_model_before_factory() -> None:
    with pytest.raises(ValueError, match="face_detection.model is required"):
        AppConfig.model_validate({"face_detection": {"enabled": True}})


def test_legacy_face_backend_and_model_keys_warn_and_migrate() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        config = AppConfig.model_validate(
            {
                "face_detection": {
                    "enabled": False,
                    "backend_id": "onnxruntime",
                    "model_id": "scrfd_2.5g_kps",
                }
            }
        )
    assert config.face_detection.backend == "onnxruntime"
    assert config.face_detection.detector_id == "scrfd_2.5g_kps"
    assert config.face_detection.model == (
        "models/face_detection/scrfd_2.5g_kps/scrfd_2.5g_bnkps.onnx"
    )
    assert any("deprecated keys" in str(item.message) for item in captured)


def test_recognition_requires_face_detection() -> None:
    with pytest.raises(ValueError, match="requires face_detection.enabled"):
        AppConfig.model_validate(
            {
                "face_detection": {"enabled": False},
                "recognition": {
                    "enabled": True,
                    "model": "models/face.onnx",
                },
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"motion_detection": {"pixel_threshold": 0}},
        {"motion_detection": {"pixel_threshold": 256}},
        {"motion_detection": {"min_changed_fraction": -0.1}},
        {"motion_detection": {"min_changed_fraction": 1.1}},
        {"motion_detection": {"resize_width": 0}},
        {"motion_detection": {"warmup_frames": 0}},
    ],
)
def test_invalid_motion_detection_configuration_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate(payload)

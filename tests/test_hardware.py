from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.config import AppConfig
import app.hardware as hardware_module
from app.hardware import (
    adaptive_person_profile,
    ensure_process_memory_budget,
    resolve_cpu_thread_budget,
)


def _cameras(count: int) -> list[dict[str, object]]:
    return [
        {
            "id": f"cam_{index}",
            "enabled": index <= count,
            "stream_url": f"rtsp://camera-{index}.local/stream",
        }
        for index in range(1, 7)
    ]


def test_adaptive_iris_xe_person_profiles_scale_with_camera_count() -> None:
    one = adaptive_person_profile(1)
    two = adaptive_person_profile(2)
    four = adaptive_person_profile(4)
    six = adaptive_person_profile(6)

    assert (one.model, one.image_size, one.inference_fps, one.num_streams) == (
        "models/yolo26s.pt",
        640,
        3.0,
        1,
    )
    assert (two.model, two.inference_fps, two.num_streams) == (
        "models/yolo26s.pt",
        2.5,
        2,
    )
    assert (four.model, four.image_size, four.inference_fps) == (
        "models/yolo26s.pt",
        640,
        2.0,
    )
    assert (six.model, six.image_size, six.inference_fps, six.num_streams) == (
        "models/yolo26n.pt",
        512,
        2.0,
        2,
    )


def test_intel_profile_applies_gpu_person_cpu_face_and_memory_bounds() -> None:
    config = AppConfig.model_validate(
        {
            "cameras": _cameras(1),
            "hardware_optimization": {
                "enabled": True,
                "profile": "intel_iris_xe",
            },
        }
    )

    assert config.video.max_buffer_frames == 1
    assert config.video.hardware_acceleration == "auto"
    assert config.person_detection.backend == "openvino"
    assert config.person_detection.device == "gpu"
    assert config.person_detection.precision == "fp16"
    assert config.person_detection.model == "models/yolo26s.pt"
    assert config.person_detection.image_size == 640
    assert config.inference.person_detection_fps == 3.0
    assert config.person_detection.openvino_num_streams == 1
    assert config.person_detection.openvino_num_requests == 1
    assert config.person_detection.max_process_ram_mb == 6144
    assert config.face_detection.device == "cpu"
    assert config.face_landmarks.device == "cpu"
    assert config.recognition.device == "cpu"


def test_intel_profile_switches_six_cameras_to_small_512_model() -> None:
    config = AppConfig.model_validate(
        {
            "cameras": _cameras(6),
            "hardware_optimization": {
                "enabled": True,
                "profile": "intel_iris_xe",
                "gpu_streams": 2,
                "gpu_num_requests": 2,
            },
        }
    )

    assert config.person_detection.model == "models/yolo26n.pt"
    assert config.person_detection.image_size == 512
    assert config.inference.person_detection_fps == 2.0
    assert config.person_detection.openvino_performance_mode == "throughput"
    assert config.person_detection.openvino_num_streams == 2
    assert config.person_detection.openvino_num_requests == 2


def test_cpu_thread_budget_honors_explicit_value_and_bounds_auto() -> None:
    assert resolve_cpu_thread_budget(3) == 3
    automatic = resolve_cpu_thread_budget(0)
    assert 2 <= automatic <= 8

def test_memory_budget_reserves_ram_for_integrated_gpu(monkeypatch) -> None:
    monkeypatch.setattr(hardware_module, "process_rss_mb", lambda: 256.0)
    monkeypatch.setattr(
        hardware_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=900 * 1024 * 1024),
    )

    with pytest.raises(MemoryError, match="integrated-GPU profile reserves"):
        ensure_process_memory_budget(6144, stage="test model")


def test_memory_budget_allows_load_with_process_and_system_headroom(monkeypatch) -> None:
    monkeypatch.setattr(hardware_module, "process_rss_mb", lambda: 512.0)
    monkeypatch.setattr(
        hardware_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=8 * 1024 * 1024 * 1024),
    )

    ensure_process_memory_budget(6144, stage="test model")


from __future__ import annotations

from pathlib import Path

import yaml

from scripts.benchmark_intel_hardware import (
    _apply_recommendation,
    _decode_performance_delta,
    _model_performance_delta,
    _recommend_decode,
)


def test_decode_recommendation_prefers_verified_hardware_with_cpu_gain() -> None:
    recommendation = _recommend_decode(
        [
            {
                "requested": "none",
                "actual": "none",
                "decoded_fps": 30.0,
                "process_cpu_percent": 80.0,
            },
            {
                "requested": "mfx",
                "actual": "mfx",
                "decoded_fps": 30.5,
                "process_cpu_percent": 48.0,
            },
            {
                "requested": "d3d11",
                "actual": "d3d11",
                "decoded_fps": 29.9,
                "process_cpu_percent": 55.0,
            },
        ]
    )

    assert recommendation["requested"] == "mfx"
    assert recommendation["source"] == "measured"


def test_decode_recommendation_rejects_unverified_hardware_fallback() -> None:
    recommendation = _recommend_decode(
        [
            {
                "requested": "none",
                "actual": "none",
                "decoded_fps": 30.0,
                "process_cpu_percent": 70.0,
            },
            {
                "requested": "mfx",
                "actual": "none",
                "decoded_fps": 30.0,
                "process_cpu_percent": 50.0,
            },
        ]
    )

    assert recommendation["requested"] == "none"


def test_apply_recommendation_writes_local_profile_without_destroying_other_sections(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    source = config_dir / "config.example.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "cameras": [{"id": "cam_1", "enabled": True, "stream_url": "rtsp://camera/live"}],
                "person_detection": {"enabled": True, "confidence_threshold": 0.47},
                "inference": {"person_detection_fps": 2},
                "video": {"rtsp_transport": "tcp"},
                "hardware_optimization": {
                    "enabled": True,
                    "profile": "intel_iris_xe",
                    "adaptive_person_detection": True,
                },
                "custom_section": {"preserve": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    target = _apply_recommendation(
        source,
        count=4,
        model_recommendation={
            "source": "measured",
            "candidate": {
                "model": "models/yolo26s.pt",
                "image_size": 640,
                "performance_mode": "throughput",
                "num_streams": 2,
                "num_requests": 2,
            },
        },
        decode_recommendation={"requested": "d3d11", "source": "measured"},
    )

    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert target == config_dir / "config.local.yaml"
    assert document["hardware_optimization"]["adaptive_person_detection"] is False
    assert document["hardware_optimization"]["decode_acceleration"] == "d3d11"
    assert document["person_detection"]["model"] == "models/yolo26s.pt"
    assert document["person_detection"]["image_size"] == 640
    assert document["person_detection"]["confidence_threshold"] == 0.47
    assert document["inference"]["person_detection_fps"] == 2.0
    assert document["video"]["hardware_acceleration"] == "d3d11"
    assert document["custom_section"] == {"preserve": True}

def test_model_performance_delta_reports_throughput_gain() -> None:
    baseline_candidate = {
        "name": "yolo26s-640-1stream",
        "model": "models/yolo26s.pt",
        "image_size": 640,
        "performance_mode": "latency",
        "num_streams": 1,
        "num_requests": 1,
    }
    selected_candidate = {
        "name": "yolo26s-640-2stream",
        "model": "models/yolo26s.pt",
        "image_size": 640,
        "performance_mode": "throughput",
        "num_streams": 2,
        "num_requests": 2,
    }
    result = _model_performance_delta(
        [
            {
                "candidate": baseline_candidate,
                "raw_openvino": {"throughput_fps": 10.0},
                "end_to_end": {"mean_ms": 80.0},
            },
            {
                "candidate": selected_candidate,
                "raw_openvino": {"throughput_fps": 15.0},
                "end_to_end": {"mean_ms": 85.0},
            },
        ],
        {"candidate": selected_candidate},
    )

    assert result is not None
    assert result["aggregate_throughput_gain_percent"] == 50.0
    assert result["aggregate_throughput_multiplier"] == 1.5
    assert result["single_call_latency_change_percent"] == 6.2


def test_decode_performance_delta_reports_cpu_reduction() -> None:
    result = _decode_performance_delta(
        [
            {
                "requested": "none",
                "actual": "none",
                "decoded_fps": 30.0,
                "process_cpu_percent": 80.0,
            },
            {
                "requested": "mfx",
                "actual": "mfx",
                "decoded_fps": 30.0,
                "process_cpu_percent": 48.0,
            },
        ],
        {"requested": "mfx"},
    )

    assert result is not None
    assert result["process_cpu_reduction_percent"] == 40.0
    assert result["decoded_fps_change_percent"] == 0.0


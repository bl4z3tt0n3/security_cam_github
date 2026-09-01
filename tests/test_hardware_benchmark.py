from __future__ import annotations

from pathlib import Path

import yaml

from scripts.benchmark_intel_hardware import (
    _apply_recommendation,
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

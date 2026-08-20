from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_scalability import run_scalability


def test_fake_scalability_reports_ordered_levels_one_to_six() -> None:
    report = run_scalability(
        mode="fake",
        max_cameras=6,
        duration=0.03,
        warmup=0.0,
        scenario="two_persons",
    )

    assert [level.camera_count for level in report.levels] == [1, 2, 3, 4, 5, 6]
    assert all(level.status == "measured" for level in report.levels)
    assert all(
        camera.face_pipeline_concurrency is None
        for level in report.levels
        for camera in level.cameras
    )
    json.dumps(report.to_dict())


def test_real_scalability_reports_unavailable_without_concrete_stream_or_model() -> None:
    report = run_scalability(
        mode="real",
        max_cameras=2,
        duration=0.01,
        warmup=0.0,
        scenario="none",
        config=Path("config/config.example.yaml"),
    )

    assert [level.status for level in report.levels] == ["unavailable", "unavailable"]
    assert all(level.reason for level in report.levels)

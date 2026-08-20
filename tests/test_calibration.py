from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.face import CalibrationError, CalibrationSample, calibrate_scores, read_score_samples


def test_read_csv_and_calculate_far_frr(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text(
        "label,score\n"
        "genuine,0.90\n"
        "genuine,0.80\n"
        "impostor,0.20\n"
        "impostor,0.40\n",
        encoding="utf-8",
    )

    samples = read_score_samples(path)
    report = calibrate_scores(samples, target_far=0.0)

    assert len(samples) == 4
    assert report.genuine.count == 2
    assert report.impostor.count == 2
    assert report.target_far_selection is not None
    assert report.target_far_selection.far == pytest.approx(0.0)
    assert report.target_far_selection.frr == pytest.approx(0.0)


def test_read_jsonl_and_require_both_classes(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {"label": "genuine", "score": 0.9},
                {"label": "impostor", "score": 0.1},
            )
        ),
        encoding="utf-8",
    )

    samples = read_score_samples(path)
    assert [sample.label for sample in samples] == ["genuine", "impostor"]

    only_genuine = tmp_path / "only-genuine.csv"
    only_genuine.write_text("label,score\ngenuine,0.9\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="both genuine and impostor"):
        read_score_samples(only_genuine)


@pytest.mark.parametrize(
    "content",
    [
        "label,score\ngenuine,not-a-number\nimpostor,0.1\n",
        "label,score\ngenuine,1.2\nimpostor,0.1\n",
        "kind,score\ngenuine,0.9\nimpostor,0.1\n",
    ],
    ids=["bad-score", "out-of-range", "missing-column"],
)
def test_invalid_calibration_input_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CalibrationError):
        read_score_samples(path)


def test_calibration_does_not_modify_configuration(tmp_path: Path) -> None:
    config = tmp_path / "config.local.yaml"
    config.write_text("recognition:\n  threshold: null\n", encoding="utf-8")
    before = config.read_bytes()
    samples = (
        CalibrationSample("genuine", 0.9),
        CalibrationSample("impostor", 0.1),
    )

    calibrate_scores(samples, target_far=0.0)
    assert config.read_bytes() == before

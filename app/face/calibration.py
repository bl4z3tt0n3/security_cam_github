"""Offline recognition-score calibration helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Literal


CalibrationLabel = Literal["genuine", "impostor"]


class CalibrationError(ValueError):
    """Raised when a score calibration input is invalid or incomplete."""


@dataclass(frozen=True)
class CalibrationSample:
    label: CalibrationLabel
    score: float
    row_number: int | None = None


@dataclass(frozen=True)
class ScoreDistribution:
    count: int
    minimum: float
    maximum: float
    mean: float
    p50: float
    p95: float

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
        }


@dataclass(frozen=True)
class ThresholdEvaluation:
    threshold: float
    false_accepts: int
    false_rejects: int
    genuine_total: int
    impostor_total: int
    far: float
    frr: float

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "false_accepts": self.false_accepts,
            "false_rejects": self.false_rejects,
            "genuine_total": self.genuine_total,
            "impostor_total": self.impostor_total,
            "far": self.far,
            "frr": self.frr,
        }


@dataclass(frozen=True)
class CalibrationReport:
    genuine: ScoreDistribution
    impostor: ScoreDistribution
    thresholds: tuple[ThresholdEvaluation, ...]
    eer_candidate: ThresholdEvaluation
    target_far: float | None
    target_far_selection: ThresholdEvaluation | None

    def to_dict(self) -> dict[str, object]:
        return {
            "genuine": self.genuine.to_dict(),
            "impostor": self.impostor.to_dict(),
            "thresholds": [item.to_dict() for item in self.thresholds],
            "eer_candidate": self.eer_candidate.to_dict(),
            "target_far": self.target_far,
            "target_far_selection": (
                self.target_far_selection.to_dict()
                if self.target_far_selection is not None
                else None
            ),
        }


def _validate_label(value: object, *, row_number: int | None = None) -> CalibrationLabel:
    normalized = str(value).strip().lower()
    if normalized not in {"genuine", "impostor"}:
        suffix = f" at row {row_number}" if row_number is not None else ""
        raise CalibrationError(
            f"label must be 'genuine' or 'impostor'{suffix}; got {value!r}"
        )
    return normalized  # type: ignore[return-value]


def _validate_score(value: object, *, row_number: int | None = None) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        suffix = f" at row {row_number}" if row_number is not None else ""
        raise CalibrationError(f"score must be numeric{suffix}") from exc
    if not math.isfinite(score) or not -1 <= score <= 1:
        suffix = f" at row {row_number}" if row_number is not None else ""
        raise CalibrationError(f"score must be finite and between -1 and 1{suffix}")
    return score


def _sample_from_mapping(mapping: object, *, row_number: int) -> CalibrationSample:
    if not isinstance(mapping, dict):
        raise CalibrationError(f"each score record must be an object at row {row_number}")
    if "label" not in mapping or "score" not in mapping:
        raise CalibrationError(f"record requires label and score at row {row_number}")
    return CalibrationSample(
        label=_validate_label(mapping["label"], row_number=row_number),
        score=_validate_score(mapping["score"], row_number=row_number),
        row_number=row_number,
    )


def read_score_samples(path: Path | str, *, input_format: str = "auto") -> tuple[CalibrationSample, ...]:
    """Read CSV or JSONL score records and require both calibration classes."""

    source = Path(path)
    if not source.is_file():
        raise CalibrationError(f"score input not found: {source}")
    normalized_format = input_format.lower()
    if normalized_format == "auto":
        normalized_format = "jsonl" if source.suffix.lower() in {".jsonl", ".ndjson"} else "csv"
    if normalized_format not in {"csv", "jsonl"}:
        raise CalibrationError("input format must be auto, csv, or jsonl")

    try:
        if normalized_format == "csv":
            with source.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "label" not in reader.fieldnames or "score" not in reader.fieldnames:
                    raise CalibrationError("CSV input requires label and score columns")
                samples = tuple(
                    _sample_from_mapping(row, row_number=index)
                    for index, row in enumerate(reader, start=2)
                )
        else:
            records: list[CalibrationSample] = []
            for index, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CalibrationError(f"invalid JSONL record at row {index}") from exc
                records.append(_sample_from_mapping(payload, row_number=index))
            samples = tuple(records)
    except OSError as exc:
        raise CalibrationError(f"cannot read score input {source}: {exc}") from exc

    if not samples:
        raise CalibrationError("score input contains no records")
    labels = {sample.label for sample in samples}
    if labels != {"genuine", "impostor"}:
        raise CalibrationError("score input must contain both genuine and impostor records")
    return samples


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> ScoreDistribution:
    if not values:
        raise CalibrationError("cannot summarize an empty score distribution")
    return ScoreDistribution(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=sum(values) / len(values),
        p50=_percentile(values, 0.50),
        p95=_percentile(values, 0.95),
    )


def evaluate_threshold(
    samples: tuple[CalibrationSample, ...],
    threshold: float,
) -> ThresholdEvaluation:
    if not math.isfinite(threshold):
        raise CalibrationError("threshold must be finite")
    genuine = [sample.score for sample in samples if sample.label == "genuine"]
    impostor = [sample.score for sample in samples if sample.label == "impostor"]
    if not genuine or not impostor:
        raise CalibrationError("both genuine and impostor scores are required")
    false_accepts = sum(score >= threshold for score in impostor)
    false_rejects = sum(score < threshold for score in genuine)
    return ThresholdEvaluation(
        threshold=threshold,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        genuine_total=len(genuine),
        impostor_total=len(impostor),
        far=false_accepts / len(impostor),
        frr=false_rejects / len(genuine),
    )


def calibrate_scores(
    samples: tuple[CalibrationSample, ...],
    *,
    target_far: float | None = None,
) -> CalibrationReport:
    """Return distributions and threshold trade-offs without changing config."""

    if not samples:
        raise CalibrationError("at least one score sample is required")
    labels = {sample.label for sample in samples}
    if labels != {"genuine", "impostor"}:
        raise CalibrationError("both genuine and impostor scores are required")
    if target_far is not None and (not math.isfinite(target_far) or not 0 <= target_far <= 1):
        raise CalibrationError("target_far must be finite and between 0 and 1")

    candidates = sorted({sample.score for sample in samples})
    candidates.append(math.nextafter(candidates[-1], math.inf))
    evaluations = tuple(evaluate_threshold(samples, threshold) for threshold in candidates)
    eer_candidate = min(
        evaluations,
        key=lambda item: (abs(item.far - item.frr), item.frr, item.threshold),
    )
    selection = None
    if target_far is not None:
        feasible = [item for item in evaluations if item.far <= target_far]
        if feasible:
            selection = min(feasible, key=lambda item: (item.threshold, item.frr))

    genuine_values = [sample.score for sample in samples if sample.label == "genuine"]
    impostor_values = [sample.score for sample in samples if sample.label == "impostor"]
    return CalibrationReport(
        genuine=_distribution(genuine_values),
        impostor=_distribution(impostor_values),
        thresholds=evaluations,
        eer_candidate=eer_candidate,
        target_far=target_far,
        target_far_selection=selection,
    )

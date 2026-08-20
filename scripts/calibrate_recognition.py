"""Calibrate recognition thresholds from pre-collected genuine/impostor scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from app.face import CalibrationError, calibrate_scores, read_score_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a local recognition threshold from CSV or JSONL scores."
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV or JSONL score file.")
    parser.add_argument(
        "--format",
        choices=("auto", "csv", "jsonl"),
        default="auto",
        dest="input_format",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=None,
        help="Optional maximum false-accept rate used to select a threshold.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    return parser.parse_args()


def _print_human(report: object) -> None:
    payload = report.to_dict()
    genuine = payload["genuine"]
    impostor = payload["impostor"]
    eer = payload["eer_candidate"]
    selected = payload["target_far_selection"]
    print("RECOGNITION CALIBRATION")
    print(
        f"genuine: n={genuine['count']} min={genuine['minimum']:.4f} "
        f"mean={genuine['mean']:.4f} p95={genuine['p95']:.4f} max={genuine['maximum']:.4f}"
    )
    print(
        f"impostor: n={impostor['count']} min={impostor['minimum']:.4f} "
        f"mean={impostor['mean']:.4f} p95={impostor['p95']:.4f} max={impostor['maximum']:.4f}"
    )
    print(
        f"EER-like candidate: threshold={eer['threshold']:.6f} "
        f"FAR={eer['far']:.4f} FRR={eer['frr']:.4f}"
    )
    if selected is None:
        print("target FAR selection: n/d")
    else:
        print(
            f"target FAR selection: threshold={selected['threshold']:.6f} "
            f"FAR={selected['far']:.4f} FRR={selected['frr']:.4f}"
        )
    print("warning: threshold is a dataset/model/camera-specific suggestion; config was not changed")


def main() -> int:
    args = parse_args()
    try:
        samples = read_score_samples(args.input, input_format=args.input_format)
        report = calibrate_scores(samples, target_far=args.target_far)
        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        if args.output is not None:
            args.output.write_text(payload + "\n", encoding="utf-8")
        if args.json:
            print(payload)
        else:
            _print_human(report)
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"CALIBRATION ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

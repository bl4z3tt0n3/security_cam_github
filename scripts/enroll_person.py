"""Enroll one known person from a local directory of reference images."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from app.config import AppConfig, ConfigurationError, load_config
from app.face import (
    EnrollmentError,
    EnrollmentService,
    FaceEmbeddingError,
    FaceDetectorError,
    FaceQualityEvaluator,
    PersonStorageError,
    PersonStore,
    SimilarityFaceAligner,
    create_face_detector,
    create_face_embedder,
    create_face_landmarker,
)
from app.face.registry import recognizer_spec

from scripts._common import REPO_ROOT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enroll a person locally from multiple face reference images."
    )
    parser.add_argument("--name", required=True, help="Display name for the enrolled person.")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing local reference images.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "config.local.yaml",
        help="Local YAML configuration (default: config/config.local.yaml).",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=None,
        help="Override face_detection.model and enable the local face detector.",
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=None,
        help="Override recognition.model and enable the local face embedder.",
    )
    parser.add_argument("--detector-id", default=None, help="Registered detector id override.")
    parser.add_argument("--recognizer-id", default=None, help="Registered recognizer id override.")
    parser.add_argument(
        "--landmarker-model",
        type=Path,
        default=None,
        help="Override face_landmarks.model for detectors without native landmarks.",
    )
    parser.add_argument(
        "--persons-dir",
        type=Path,
        default=None,
        help="Override storage.persons_dir.",
    )
    parser.add_argument("--person-id", default=None, help="Optional filesystem-safe person id.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing person record.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Root used to resolve relative models and storage paths.",
    )
    return parser


def _resolve_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _load_enrollment_config(args: argparse.Namespace) -> tuple[AppConfig, Path]:
    config_path = args.config if args.config.is_absolute() else args.project_root / args.config
    if config_path.is_file():
        return load_config(config_path), config_path
    if args.face_model is None or args.embedding_model is None:
        raise ConfigurationError(
            f"configuration not found: {config_path}; provide --config or both model overrides"
        )
    return AppConfig.model_validate({}), config_path


def _configure_models(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    face = config.face_detection.model_copy(deep=True)
    landmarks = config.face_landmarks.model_copy(deep=True)
    recognition = config.recognition.model_copy(deep=True)
    if args.detector_id is not None:
        face.detector_id = str(args.detector_id)
    if args.recognizer_id is not None:
        recognition.recognizer_id = str(args.recognizer_id)
    if args.face_model is not None:
        face.model = str(args.face_model)
        face.enabled = True
    if args.embedding_model is not None:
        recognition.model = str(args.embedding_model)
        recognition.enabled = True
    if args.landmarker_model is not None:
        landmarks.model = str(args.landmarker_model)
        landmarks.enabled = True
    if not face.enabled:
        raise ConfigurationError(
            "face detection is disabled; enable face_detection or pass --face-model"
        )
    if not recognition.enabled:
        raise ConfigurationError(
            "recognition model is disabled; enable recognition or pass --embedding-model"
        )
    return config.model_copy(
        update={
            "face_detection": face,
            "face_landmarks": landmarks,
            "recognition": recognition,
        }
    )


def _print_report(report: object) -> None:
    print(f"name: {getattr(report, 'name')}")
    print(f"processed images: {getattr(report, 'processed_count')}")
    print(f"accepted images: {getattr(report, 'accepted_count')}")
    print(f"rejected images: {getattr(report, 'rejected_count')}")
    for result in getattr(report, "images"):
        if result.accepted:
            print(f"  ACCEPTED {result.image_name}")
        else:
            print(f"  REJECTED {result.image_name}: {', '.join(result.reasons)}")
    record = getattr(report, "record")
    if record is not None:
        print(f"person id: {record.person_id}")
        print(f"stored directory: {record.directory}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    try:
        config, _ = _load_enrollment_config(args)
        config = _configure_models(config, args)
        model_root = project_root
        detector = create_face_detector(config.face_detection, model_root=model_root)
        embedder = create_face_embedder(config.recognition, model_root=model_root)
        landmarker = create_face_landmarker(config.face_landmarks, model_root=model_root)
        recognizer_id = config.recognition.recognizer_id
        if not recognizer_id:
            raise ConfigurationError(
                "recognition.recognizer_id is required for landmark-based enrollment"
            )
        aligner = SimilarityFaceAligner(recognizer_spec(recognizer_id).alignment_template)
        persons_dir = (
            args.persons_dir
            if args.persons_dir is not None
            else config.storage.persons_dir
        )
        store = PersonStore(
            _resolve_path(project_root, persons_dir),
            scope=Path(recognizer_id) / embedder.metadata.fingerprint,
        )
        evaluator = FaceQualityEvaluator(
            min_width=config.face_quality.min_width,
            min_height=config.face_quality.min_height,
            blur_threshold=config.face_quality.blur_threshold,
            min_brightness=config.face_quality.min_brightness,
            max_brightness=config.face_quality.max_brightness,
            min_confidence=config.face_quality.min_confidence,
        )
        service = EnrollmentService(
            detector,
            embedder,
            store,
            aligner=aligner,
            landmarker=landmarker,
            evaluator=evaluator,
        )
        report = service.enroll(
            args.name,
            args.images,
            person_id=args.person_id,
            overwrite=args.overwrite,
        )
        _print_report(report)
        if report.record is None:
            print("ERROR: no valid face embedding was produced; no person was stored.")
            return 1
        return 0
    except (
        ConfigurationError,
        EnrollmentError,
        FaceDetectorError,
        FaceEmbeddingError,
        PersonStorageError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

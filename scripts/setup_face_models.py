"""Explicit face-model setup and offline verification.

The application never calls this module implicitly.  Network access is only
used when ``--download`` is supplied by the operator. Conversion is likewise
an explicit operator action and is never performed by runtime code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from app.face.registry import (  # noqa: E402
    FACE_DETECTOR_SPECS,
    LANDMARKER_SPEC,
    RECOGNIZER_SPECS,
    FaceModelSpec,
    RecognizerSpec,
    model_path,
)
from app.face.capabilities import face_capability_matrix  # noqa: E402


MANIFEST_NAME = "face_models.manifest.json"


@dataclass(frozen=True)
class DownloadSpec:
    urls: tuple[str, ...]
    kind: str = "files"
    archive_member: str | None = None
    archive_sha256: str | None = None
    version: str = "unversioned"
    artifact_sha256: tuple[str, ...] = ()
    source_sha256: str | None = None


DOWNLOADS: dict[str, DownloadSpec] = {
    "scrfd_2.5g_kps": DownloadSpec(
        (
            "https://huggingface.co/hsuyabc/scrfd_2.5g_bnkps.onnx/resolve/"
            "169a587dd965f3981d358dfe6844c6449a97bedc/scrfd_2.5g_bnkps.onnx",
        ),
        version="hf-169a587dd965f3981d358dfe6844c6449a97bedc",
        artifact_sha256=(
            "bc24bb349491481c3ca793cf89306723162c280cb284c5a5e49df3760bf5c2ce",
        ),
    ),
    "yunet_2023mar": DownloadSpec(
        (
            "https://raw.githubusercontent.com/opencv/opencv_zoo/"
            "47534e27c9851bb1128ccc0102f1145e27f23f98/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ),
        version="opencv_zoo-47534e27c9851bb1128ccc0102f1145e27f23f98",
        artifact_sha256=(
            "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        ),
    ),
    "face_detection_0205": DownloadSpec(
        (
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/face-detection-0205/FP32/face-detection-0205.xml",
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/face-detection-0205/FP32/face-detection-0205.bin",
        ),
        version="openvino-omz-2023.0-FP32",
        artifact_sha256=(
            "7adb0d4c9af5152ce2abee74aafeb0be2daafea74b8494aea7a4508010ff8eab",
            "e748cf53a2cfa2cb8d453cc702c9f0267fa5a4726192f4769802c8179b0c2255",
        ),
    ),
    "landmarks-regression-retail-0009": DownloadSpec(
        (
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml",
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.bin",
        ),
        version="openvino-omz-2023.0-FP16",
        artifact_sha256=(
            "1d7e43d3060e8f0328932ae9eeced07ff216eac80f10fd07b58c52fbaf69af58",
            "a7285eec8cb20a50c0ed567c426db6c7b8753c51b47f798f8ef5d8e0fa12d3c5",
        ),
    ),
    "face-reidentification-retail-0095": DownloadSpec(
        (
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml",
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.bin",
        ),
        version="openvino-omz-2023.0-FP16",
        artifact_sha256=(
            "ce53d2c9c08c0bd1c1660fb8a5b6d0e3e4ec19eb92f1036d2d83a85e83082dce",
            "241229ca3d206321868d46ce74a3c0b06c49cea58db7dc70b2e842ff287545d1",
        ),
    ),
    "arcface-resnet50-webface600k": DownloadSpec(
        (
            "https://github.com/deepinsight/insightface/releases/download/"
            "v0.7/buffalo_l.zip",
        ),
        kind="archive",
        archive_member="w600k_r50.onnx",
        archive_sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46eb28218a7b0ca2f",
        version="insightface-v0.7-buffalo_l",
        artifact_sha256=(
            "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        ),
    ),
    "facenet-20180402-vggface2": DownloadSpec(
        (
            "https://drive.google.com/uc?export=download&id="
            "1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-",
        ),
        kind="checkpoint",
        version="20180402-114759",
        source_sha256="669f37d3954b72b53121ff0638ca5bb8b14309bfe4767fe6bd851eee75f1f0de",
        artifact_sha256=(
            "2210d2d69f9cf1675af1bb5987b01d8dde282df9af86a6a5a3b85e47b7ab9989",
        ),
    ),
}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec(model_id: str) -> FaceModelSpec | RecognizerSpec:
    for candidate in (*FACE_DETECTOR_SPECS, LANDMARKER_SPEC, *RECOGNIZER_SPECS):
        candidate_id = candidate.model_id if isinstance(candidate, FaceModelSpec) else candidate.recognizer_id
        if candidate_id == model_id:
            return candidate
    raise ValueError(f"unknown face model: {model_id}")


def _paths(spec: FaceModelSpec | RecognizerSpec, root: Path) -> tuple[Path, ...]:
    primary = model_path(spec, root)
    if primary.suffix.lower() == ".xml":
        return (primary, primary.with_suffix(".bin"))
    return (primary,)


def _inspect(model_id: str, root: Path) -> tuple[bool, str]:
    spec = _spec(model_id)
    paths = _paths(spec, root)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        return False, "missing: " + ", ".join(str(path) for path in missing)
    download = DOWNLOADS.get(model_id)
    if download is not None and download.artifact_sha256:
        if len(download.artifact_sha256) != len(paths):
            return False, "pinned SHA-256 count does not match artifact count"
        mismatches = [
            f"{path.name}={_sha256(path)} expected={expected}"
            for path, expected in zip(paths, download.artifact_sha256)
            if _sha256(path) != expected
        ]
        if mismatches:
            return False, "SHA-256 mismatch: " + "; ".join(mismatches)
    digests = ", ".join(f"{path.name}={_sha256(path)}" for path in paths)
    if any(path.suffix.lower() == ".onnx" for path in paths):
        try:
            import cv2

            network = cv2.dnn.readNetFromONNX(str(next(path for path in paths if path.suffix.lower() == ".onnx")))
            if network.empty():
                return False, "invalid ONNX graph"
        except Exception as exc:
            return False, f"ONNX inspection failed: {exc}"
    return True, f"ready; {digests}"


def _expected_digests(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--sha256 deve usare MODEL_ID=DIGEST")
        model_id, digest = value.split("=", 1)
        model_id = model_id.strip()
        digest = digest.strip().lower()
        if not model_id or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"digest SHA-256 non valido per {model_id or '<vuoto>'}")
        parsed[model_id] = digest
    return parsed


def _manifest_path(root: Path, value: Path | None) -> Path:
    path = value or (root / "scripts" / MANIFEST_NAME)
    return path if path.is_absolute() else root / path


def _model_contract(model_id: str, spec: FaceModelSpec | RecognizerSpec) -> dict[str, object]:
    """Return the checked-in I/O and preprocessing contract for the manifest."""

    if isinstance(spec, RecognizerSpec):
        precision = "FP16" if spec.backend == "openvino" else "FP32"
        contract = {
            "precision": precision,
            "input": {
                "layout": "NCHW",
                "shape": ["N", 3, spec.input_height, spec.input_width],
                "color_order": spec.color_order,
                "normalization": spec.normalization,
            },
            "output": {
                "shape": ["N", spec.embedding_dimension],
                "type": "embedding",
                "dimension": spec.embedding_dimension,
            },
            "alignment_template": {
                "width": spec.alignment_template.width,
                "height": spec.alignment_template.height,
                "points": [list(point) for point in spec.alignment_template.points.points],
            },
        }
        if spec.recognizer_id == "facenet-20180402-vggface2":
            contract["export"] = {
                "script": "scripts/export_facenet_onnx.py",
                "source_file": "20180402-114759.pb",
                "source_sha256": "bf2c12f31880aaa865fa5a9c168dcbd619f7a40b1633f6446d416fac2421ab99",
                "phase_train": False,
                "inputs_as_nchw": True,
                "canonicalized_for_reproducibility": True,
            }
        return contract
    contracts: dict[str, dict[str, object]] = {
        "scrfd_2.5g_kps": {
            "precision": "FP32",
            "input": {"layout": "NCHW", "shape": ["N", 3, "H", "W"], "color_order": "BGR"},
            "output": {"type": "boxes_scores_landmarks", "tensors": 9},
        },
        "yunet_2023mar": {
            "precision": "FP32",
            "input": {"layout": "runtime-set", "shape": [1, 3, "H", "W"], "color_order": "BGR"},
            "output": {
                "type": "rows",
                "columns": ["x", "y", "w", "h", "score", "landmark_5x2"],
            },
        },
        "face_detection_0205": {
            "precision": "FP32",
            "input": {"layout": "NCHW", "shape": [1, 3, 416, 416], "color_order": "BGR"},
            "output": {"type": "detection_out", "shape": [1, 1, "N", 7]},
        },
        "landmarks-regression-retail-0009": {
            "precision": "FP16",
            "input": {"layout": "NCHW", "shape": [1, 3, 48, 48], "color_order": "BGR"},
            "output": {"type": "landmarks_5x2_normalized", "shape": [1, 10]},
        },
    }
    return contracts.get(model_id, {})


def _write_manifest(root: Path, path: Path, model_ids: list[str]) -> None:
    """Write a local, reproducible inventory after explicit verification."""

    entries: dict[str, object] = {}
    for model_id in model_ids:
        spec = _spec(model_id)
        download = DOWNLOADS.get(model_id)
        artifact_paths = _paths(spec, root)
        expected = download.artifact_sha256 if download is not None else ()
        artifacts = []
        for index, artifact in enumerate(artifact_paths):
            actual = _sha256(artifact) if artifact.is_file() else None
            expected_digest = expected[index] if index < len(expected) else None
            artifacts.append(
                {
                    "path": str(artifact.relative_to(root)).replace("\\", "/"),
                    "sha256": actual,
                    "expected_sha256": expected_digest,
                    "sha256_verified": (
                        actual is not None
                        and (expected_digest is None or actual == expected_digest)
                    ),
                }
            )
        source_artifacts = []
        if download is not None:
            if download.archive_sha256:
                source_artifacts.append(
                    {
                        "kind": "archive",
                        "url": download.urls[0],
                        "sha256": download.archive_sha256,
                        "member": download.archive_member,
                    }
                )
            elif download.source_sha256:
                source_artifacts.append(
                    {
                        "kind": "checkpoint",
                        "file": "20180402-114759.zip",
                        "url": download.urls[0],
                        "sha256": download.source_sha256,
                    }
                )
        entries[model_id] = {
            "version": download.version if download is not None else "unversioned",
            "source": (
                download.urls[0]
                if download is not None
                else spec.source
            ),
            "license": spec.license,
            "backend": spec.backend,
            "devices": list(spec.devices),
            "artifacts": artifacts,
            "source_artifacts": source_artifacts,
            "contract": _model_contract(model_id, spec),
        }
    document = {"schema_version": 2, "manifest_version": "2026-08-18", "models": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_sha256(model_id: str, root: Path, expected: str) -> tuple[bool, str]:
    spec = _spec(model_id)
    paths = _paths(spec, root)
    actual = [_sha256(path) for path in paths]
    if any(value is None for value in actual):
        return False, "artifact missing"
    # A paired IR is verified as two named files so XML/BIN cannot be swapped
    # accidentally while retaining a single opaque model scope in the app.
    combined = hashlib.sha256()
    for path in paths:
        combined.update(path.name.encode("utf-8"))
        combined.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                combined.update(chunk)
    digest = combined.hexdigest()
    return digest == expected, f"sha256={digest} expected={expected}"


def _download(model_id: str, root: Path, *, force: bool = False) -> str | None:
    download = DOWNLOADS.get(model_id)
    if download is None:
        raise ValueError(
            f"no pinned downloader for {model_id}; provide the concrete artifact locally and rerun verification"
        )
    spec = _spec(model_id)
    targets = _paths(spec, root)
    target_dir = targets[0].parent
    target_dir.mkdir(parents=True, exist_ok=True)
    if download.kind == "checkpoint":
        checkpoint_dir = root / ".test-tmp" / "face-model-checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        target = checkpoint_dir / "20180402-114759.zip"
        if target.is_file() and not force:
            return str(target)
        return _download_url(download.urls[0], target)
    if download.kind == "archive":
        if not download.archive_member:
            raise ValueError(f"archive member is missing for {model_id}")
        archive = target_dir / (download.archive_member.rsplit("/", 1)[-1] + ".zip.part")
        archive_final = archive.with_suffix("")
        if not archive_final.is_file() or force:
            _download_url(download.urls[0], archive_final)
        if download.archive_sha256 and _sha256(archive_final) != download.archive_sha256:
            raise ValueError(f"archive SHA-256 mismatch for {model_id}")
        target = targets[0]
        if target.is_file() and not force:
            return None
        temporary = target.with_name(target.name + ".part")
        with zipfile.ZipFile(archive_final) as archive_handle:
            try:
                with archive_handle.open(download.archive_member) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            except KeyError as exc:
                raise ValueError(
                    f"archive member not found for {model_id}: {download.archive_member}"
                ) from exc
        temporary.replace(target)
        return None
    urls = download.urls
    if len(urls) != len(targets):
        raise ValueError(f"pinned source count does not match artifact count for {model_id}")
    for url, target in zip(urls, targets):
        if target.is_file() and not force:
            continue
        _download_url(url, target)
    return None


def _download_url(url: str, target: Path) -> str:
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "LocalSecurityCam/face-setup"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    partial.replace(target)
    return str(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download or verify explicit local face models.")
    parser.add_argument("--model", action="append", dest="models", help="Model id; repeat for multiple models.")
    parser.add_argument("--all", action="store_true", help="Inspect every registered model.")
    parser.add_argument("--download", action="store_true", help="Explicitly download pinned artifacts.")
    parser.add_argument(
        "--sha256",
        action="append",
        default=[],
        metavar="MODEL_ID=DIGEST",
        help="Verify an operator-supplied SHA-256 fingerprint; repeatable.",
    )
    parser.add_argument(
        "--show-license",
        action="store_true",
        help="Print the registered source/license declaration without downloading.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run the real local capability probe after inspection.",
    )
    parser.add_argument(
        "--convert-onnx",
        type=Path,
        default=None,
        metavar="SOURCE_ONNX",
        help="Explicitly convert one local ONNX file to OpenVINO IR.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing local artifact after explicit operator confirmation.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write a versioned local artifact manifest after inspection.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest output path; defaults to scripts/face_models.manifest.json.",
    )
    parser.add_argument(
        "--output-xml",
        type=Path,
        default=None,
        metavar="OUTPUT_XML",
        help="Destination XML for --convert-onnx (writes XML and paired BIN).",
    )
    parser.add_argument("--root", type=Path, default=SCRIPT_ROOT, help="Project root for relative model paths.")
    args = parser.parse_args(argv)
    if args.convert_onnx is not None:
        if args.output_xml is None:
            parser.error("--convert-onnx richiede --output-xml")
        source = args.convert_onnx.expanduser().resolve()
        output = args.output_xml.expanduser().resolve()
        if not source.is_file():
            parser.error(f"ONNX non trovato: {source}")
        try:
            from openvino import convert_model, save_model

            converted = convert_model(str(source))
            output.parent.mkdir(parents=True, exist_ok=True)
            save_model(converted, str(output))
        except Exception as exc:
            print(f"NOT READY conversion: {type(exc).__name__}: {exc}")
            return 1
        print(f"READY conversion: {source.name} -> {output}")
        return 0
    model_ids = args.models or []
    if args.all or not model_ids:
        model_ids = [spec.model_id for spec in FACE_DETECTOR_SPECS]
        model_ids += [LANDMARKER_SPEC.model_id]
        model_ids += [spec.recognizer_id for spec in RECOGNIZER_SPECS]
    root = args.root.expanduser().resolve()
    expected_digests = _expected_digests(args.sha256)
    exit_code = 0
    for model_id in model_ids:
        try:
            spec = _spec(model_id)
            if args.show_license:
                print(
                    f"LICENSE {model_id}: source={spec.source}; license={spec.license}"
                )
            if args.download:
                checkpoint = _download(model_id, root, force=args.force)
                if checkpoint is not None:
                    print(f"CHECKPOINT {model_id}: {checkpoint}")
            ok, message = _inspect(model_id, root)
            if ok and model_id in expected_digests:
                ok, digest_message = _verify_sha256(model_id, root, expected_digests[model_id])
                message = f"{message}; {digest_message}"
            elif model_id in expected_digests:
                ok = False
                message = f"{message}; sha256 not checked because artifact is unavailable"
        except Exception as exc:
            ok, message = False, f"{type(exc).__name__}: {exc}"
        print(f"{'READY' if ok else 'NOT READY'} {model_id}: {message}")
        if not ok:
            exit_code = 1
    if args.probe:
        for row in face_capability_matrix(root):
            if row.model_id not in model_ids:
                continue
            print(
                f"PROBE {row.component} {row.model_id}/{row.backend}/{row.device}: "
                f"{'READY' if row.available else 'NOT READY'}; {row.reason}"
            )
            if not row.available:
                exit_code = 1
    if args.write_manifest:
        _write_manifest(root, _manifest_path(root, args.manifest), model_ids)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

"""Canonical local face-model registry and model fingerprints."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Literal

from .alignment import AlignmentTemplate, ARC_FACE_TEMPLATE, FACENET_TEMPLATE, RETAIL_0095_TEMPLATE


FaceBackend = Literal["onnxruntime", "openvino", "opencv_dnn", "fake"]


@dataclass(frozen=True)
class FaceModelSpec:
    model_id: str
    display_name: str
    backend: FaceBackend
    relative_path: str
    devices: tuple[str, ...]
    landmarks: bool
    source: str
    license: str


@dataclass(frozen=True)
class RecognizerSpec:
    recognizer_id: str
    display_name: str
    backend: Literal["onnxruntime", "openvino", "fake"]
    relative_path: str
    devices: tuple[str, ...]
    input_width: int
    input_height: int
    color_order: Literal["RGB", "BGR"]
    normalization: str
    embedding_dimension: int
    alignment_template: AlignmentTemplate
    source: str
    license: str
    default_threshold: float | None = None

    def with_path(self, path: str | Path) -> "RecognizerSpec":
        return replace(self, relative_path=str(path))


FACE_DETECTOR_SPECS: tuple[FaceModelSpec, ...] = (
    FaceModelSpec(
        "scrfd_2.5g_kps",
        "SCRFD 2.5G KPS",
        "onnxruntime",
        "models/face_detection/scrfd_2.5g_kps/scrfd_2.5g_bnkps.onnx",
        ("auto", "cpu", "cuda"),
        True,
        "InsightFace SCRFD model zoo",
        "InsightFace model terms; verify before redistribution",
    ),
    FaceModelSpec(
        "face_detection_0205",
        "Intel face-detection-0205",
        "openvino",
        "models/face_detection/face_detection_0205_fp32/face-detection-0205.xml",
        ("auto", "cpu", "gpu", "npu"),
        False,
        "Intel Open Model Zoo",
        "Intel Open Model Zoo model license",
    ),
    FaceModelSpec(
        "yunet_2023mar",
        "YuNet 2023mar",
        "opencv_dnn",
        "models/face_detection/yunet_2023mar/face_detection_yunet_2023mar.onnx",
        ("auto", "cpu"),
        True,
        "OpenCV Zoo",
        "MIT / OpenCV Zoo terms",
    ),
)


LANDMARKER_SPEC = FaceModelSpec(
    "landmarks-regression-retail-0009",
    "OpenVINO five-point landmark regressor",
    "openvino",
    "models/face_landmarks/landmarks-regression-retail-0009/landmarks-regression-retail-0009.xml",
    ("auto", "cpu", "gpu", "npu"),
    True,
    "Intel Open Model Zoo",
    "Intel Open Model Zoo model license",
)


RECOGNIZER_SPECS: tuple[RecognizerSpec, ...] = (
    RecognizerSpec(
        "face-reidentification-retail-0095",
        "OpenVINO face-reidentification-retail-0095",
        "openvino",
        "models/face_embedding/face-reidentification-retail-0095/face-reidentification-retail-0095.xml",
        ("auto", "cpu", "gpu", "npu"),
        128,
        128,
        "BGR",
        "openvino_raw_bgr",
        256,
        RETAIL_0095_TEMPLATE,
        "Intel Open Model Zoo",
        "Apache-2.0 / Open Model Zoo terms",
    ),
    RecognizerSpec(
        "facenet-20180402-vggface2",
        "FaceNet Inception-ResNet-v1 VGGFace2",
        "onnxruntime",
        "models/face_embedding/facenet-20180402-vggface2.onnx",
        ("auto", "cpu", "cuda"),
        160,
        160,
        "RGB",
        "facenet_fixed_standardization",
        512,
        FACENET_TEMPLATE,
        "davidsandberg/facenet checkpoint 20180402-114759",
        "MIT code; checkpoint/dataset attribution and terms must be retained",
    ),
    RecognizerSpec(
        "arcface-resnet50-webface600k",
        "ArcFace ResNet50 WebFace600K",
        "onnxruntime",
        "models/face_embedding/arcface-resnet50-webface600k.onnx",
        ("auto", "cpu", "cuda"),
        112,
        112,
        "BGR",
        "arcface_127.5_127.5",
        512,
        ARC_FACE_TEMPLATE,
        "InsightFace model zoo / WebFace600K",
        "InsightFace model terms; local private/non-commercial use only",
    ),
)


DETECTOR_ID_ALIASES = {
    "face-detection-0205": "face_detection_0205",
}


def detector_spec(model_id: str) -> FaceModelSpec:
    normalized = str(model_id).strip().lower()
    normalized = DETECTOR_ID_ALIASES.get(normalized, normalized)
    for spec in FACE_DETECTOR_SPECS:
        if spec.model_id == normalized:
            return spec
    raise ValueError(f"unsupported face detector model: {model_id}")


def recognizer_spec(recognizer_id: str) -> RecognizerSpec:
    normalized = str(recognizer_id).strip().lower()
    for spec in RECOGNIZER_SPECS:
        if spec.recognizer_id == normalized:
            return spec
    raise ValueError(f"unsupported face recognizer model: {recognizer_id}")


def model_path(spec: FaceModelSpec | RecognizerSpec, model_root: Path | None = None) -> Path:
    path = Path(spec.relative_path).expanduser()
    if path.is_absolute():
        return path
    return (model_root or Path.cwd()) / path


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str | None:
    """Hash one model artifact, including the BIN paired with an OpenVINO XML."""

    primary = Path(path)
    paths = (primary, primary.with_suffix(".bin")) if primary.suffix.lower() == ".xml" else (primary,)
    if any(not candidate.is_file() for candidate in paths):
        return None
    digest = hashlib.sha256()
    for candidate in paths:
        digest.update(candidate.name.encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(
    spec: RecognizerSpec,
    path: Path,
    *,
    actual_device: str | None = None,
) -> str:
    """Return a stable scope key without exposing the full local path."""

    digest = artifact_sha256(path) or "missing"
    device = actual_device or "requested"
    value = f"{spec.recognizer_id}:{digest}:{spec.embedding_dimension}:{device}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


__all__ = [
    "FACE_DETECTOR_SPECS",
    "DETECTOR_ID_ALIASES",
    "LANDMARKER_SPEC",
    "RECOGNIZER_SPECS",
    "FaceModelSpec",
    "RecognizerSpec",
    "detector_spec",
    "artifact_sha256",
    "file_sha256",
    "model_fingerprint",
    "model_path",
    "recognizer_spec",
]

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.face import (
    FaceEmbeddingError,
    FakeEmbedder,
    OnnxFaceEmbedder,
)


class FakeValue:
    def __init__(self, name: str, shape: list[object]) -> None:
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.feed: dict[str, np.ndarray] | None = None

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def get_inputs(self) -> list[FakeValue]:
        return [FakeValue("images", [1, 3, 112, 112])]

    def get_outputs(self) -> list[FakeValue]:
        return [FakeValue("embedding", [1, 4])]

    def run(self, output_names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        del output_names
        self.feed = feed
        return [self.output]


def test_fake_embedder_returns_normalized_embedding() -> None:
    embedder = FakeEmbedder(embedding_dimension=3, callback=lambda image: [3, 4, 0])
    result = embedder.embed(np.zeros((20, 20, 3), dtype=np.uint8))
    assert result.dtype == np.float32
    assert np.allclose(result, [0.6, 0.8, 0.0])
    assert embedder.metadata.embedding_dimension == 3
    assert embedder.calls == 1


def test_fake_embedder_rejects_wrong_dimension() -> None:
    embedder = FakeEmbedder(embedding_dimension=3, callback=lambda image: [1, 2])
    with pytest.raises(FaceEmbeddingError, match="dimension"):
        embedder.embed(np.zeros((20, 20, 3), dtype=np.uint8))


def test_onnx_embedder_preprocesses_and_records_metadata(tmp_path: Path) -> None:
    session = FakeSession(np.array([[1, 2, 3, 4]], dtype=np.float32))
    model_path = tmp_path / "face_embedder.onnx"
    embedder = OnnxFaceEmbedder(
        model_path,
        model_version="2026-08",
        device="cpu",
        session=session,
    )
    result = embedder.embed(np.full((160, 120, 3), 128, dtype=np.uint8))
    assert result.shape == (4,)
    assert np.isclose(np.linalg.norm(result), 1.0)
    assert session.feed is not None
    assert session.feed["images"].shape == (1, 3, 112, 112)
    assert embedder.metadata.model_id == "face_embedder"
    assert embedder.metadata.model_version == "2026-08"
    assert embedder.metadata.model_sha256 is None
    assert embedder.provider_used == "CPUExecutionProvider"


def test_onnx_embedder_rejects_output_dimension_mismatch(tmp_path: Path) -> None:
    session = FakeSession(np.array([[1, 2, 3]], dtype=np.float32))
    embedder = OnnxFaceEmbedder(tmp_path / "model.onnx", session=session)
    with pytest.raises(FaceEmbeddingError, match="dimension"):
        embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8))

def test_onnx_embedder_cpu_session_uses_bounded_threads(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "embedder.onnx"
    model.write_bytes(b"onnx")
    observed: dict[str, object] = {}

    class Options:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    def factory(path: str, **kwargs: object) -> FakeSession:
        observed["path"] = path
        observed.update(kwargs)
        return FakeSession(np.array([[1, 2, 3, 4]], dtype=np.float32))

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=factory,
        SessionOptions=Options,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    embedder = OnnxFaceEmbedder(
        model,
        device="cpu",
        cpu_threads=4,
        max_process_ram_mb=6144,
    )

    options = observed["sess_options"]
    assert options.intra_op_num_threads == 4
    assert options.inter_op_num_threads == 1
    assert observed["providers"] == ["CPUExecutionProvider"]
    assert embedder.provider_used == "CPUExecutionProvider"


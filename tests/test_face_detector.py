from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.face import OnnxFaceDetector


class FakeValue:
    def __init__(self, name: str, shape: list[object]) -> None:
        self.name = name
        self.shape = shape


class FakeSession:
    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def get_inputs(self) -> list[FakeValue]:
        return [FakeValue("images", [1, 3, 640, 640])]

    def run(self, output_names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        del output_names
        assert feed["images"].shape == (1, 3, 640, 640)
        return [
            np.array(
                [
                    [64, 128, 320, 512, 0.9],
                    [0, 0, 100, 100, 0.2],
                ],
                dtype=np.float32,
            )
        ]


def test_onnx_face_detector_scales_and_filters_detections(tmp_path: Path) -> None:
    detector = OnnxFaceDetector(
        tmp_path / "face_detector.onnx",
        confidence_threshold=0.5,
        device="cpu",
        session=FakeSession(),
    )
    detections = detector.detect(np.zeros((320, 640, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].bbox == (64.0, 64.0, 320.0, 256.0)
    assert detections[0].confidence == pytest.approx(0.9)

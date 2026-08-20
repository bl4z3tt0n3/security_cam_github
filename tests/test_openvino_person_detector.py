from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.inference import OpenVINOPersonDetector, PersonDetectionError


class FakeCore:
    def __init__(self, devices: tuple[str, ...] = ("CPU", "GPU.0")) -> None:
        self.available_devices = list(devices)


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.asarray(
            [
                [-3.0, 1.0, 15.0, 12.0],
                [1.0, 1.0, 8.0, 8.0],
                [0.0, 0.0, 4.0, 4.0],
                [0.0, 0.0, np.nan, 4.0],
            ],
            dtype=np.float32,
        )
        self.conf = np.asarray([0.91, 0.44, 0.99, 0.99], dtype=np.float32)
        self.cls = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


class FakeRuntimeModel:
    task = "detect"

    def __init__(self) -> None:
        self.predict_calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[object]:
        self.predict_calls.append(kwargs)
        return [SimpleNamespace(boxes=FakeBoxes())]


class FakeExportModel:
    task = "detect"

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.export_calls: list[dict[str, object]] = []

    def export(self, **kwargs: object) -> str:
        self.export_calls.append(kwargs)
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.cache / "yolo26s.xml").write_text("xml", encoding="utf-8")
        (self.cache / "yolo26s.bin").write_bytes(b"bin")
        (self.cache / "metadata.yaml").write_text("task: detect\n", encoding="utf-8")
        return str(self.cache)


def _factory_for(cache: Path, created: list[object]):
    def factory(path: str, **_kwargs: object) -> object:
        if Path(path).resolve() == cache.parent.joinpath("yolo26s.pt").resolve():
            model = FakeExportModel(cache)
        else:
            model = FakeRuntimeModel()
        created.append(model)
        return model

    return factory


def _write_valid_cache(cache: Path, *, precision: str = "fp16", image_size: int = 640) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "model.xml").write_text("xml", encoding="utf-8")
    (cache / "model.bin").write_bytes(b"bin")
    (cache / "metadata.yaml").write_text("task: detect\n", encoding="utf-8")
    (cache / ".person_detector.json").write_text(
        json.dumps(
            {
                "format": "openvino",
                "source_checkpoint": "yolo26s.pt",
                "task": "detect",
                "precision": precision,
                "image_size": image_size,
            }
        ),
        encoding="utf-8",
    )


def test_openvino_filters_persons_clips_boxes_and_preserves_timestamp(tmp_path: Path) -> None:
    checkpoint = tmp_path / "models" / "yolo26s.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    cache = checkpoint.with_name("yolo26s_openvino_model")
    _write_valid_cache(cache)
    created: list[object] = []

    detector = OpenVINOPersonDetector(
        checkpoint,
        confidence_threshold=0.45,
        device="gpu",
        model_root=tmp_path,
        core_factory=FakeCore,
        yolo_factory=_factory_for(cache, created),
        execution_devices_reader=lambda _model: ("GPU.0",),
    )
    timestamp = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)

    detections = detector.detect(np.zeros((10, 12, 3), dtype=np.uint8), timestamp)

    assert len(detections) == 1
    assert detections[0].bbox == (0.0, 1.0, 12.0, 10.0)
    assert detections[0].timestamp == timestamp
    assert detector.backend == "openvino"
    assert detector.device_used == "GPU.0"
    assert detector.provider_used == "OpenVINO/GPU.0"
    assert detector.device_verified is True
    runtime = next(item for item in created if isinstance(item, FakeRuntimeModel))
    assert runtime.predict_calls[0]["classes"] == [0]
    assert runtime.predict_calls[0]["device"] == "intel:GPU.0"


def test_openvino_cache_is_reused_without_export(tmp_path: Path) -> None:
    checkpoint = tmp_path / "models" / "yolo26s.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    cache = checkpoint.with_name("yolo26s_openvino_model")
    _write_valid_cache(cache)
    created: list[object] = []

    OpenVINOPersonDetector(
        checkpoint,
        model_root=tmp_path,
        core_factory=FakeCore,
        yolo_factory=_factory_for(cache, created),
    )

    assert created == []


def test_openvino_export_uses_quantize_16_and_writes_marker(tmp_path: Path) -> None:
    checkpoint = tmp_path / "models" / "yolo26s.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    cache = checkpoint.with_name("yolo26s_openvino_model")
    created: list[object] = []

    detector = OpenVINOPersonDetector(
        checkpoint,
        precision="fp16",
        image_size=512,
        model_root=tmp_path,
        core_factory=FakeCore,
        yolo_factory=_factory_for(cache, created),
    )

    exported = next(item for item in created if isinstance(item, FakeExportModel))
    assert exported.export_calls[0]["quantize"] == 16
    assert exported.export_calls[0]["imgsz"] == 512
    marker = json.loads((cache / ".person_detector.json").read_text(encoding="utf-8"))
    assert marker["precision"] == "fp16"
    assert marker["image_size"] == 512
    assert detector.cache_path == cache


def test_openvino_allows_only_official_missing_checkpoint_download(tmp_path: Path) -> None:
    target = tmp_path / "models" / "yolo26s.pt"
    cache = target.with_name("yolo26s_openvino_model")
    created: list[object] = []

    def download(path: Path) -> Path:
        path.write_bytes(b"downloaded")
        return path

    OpenVINOPersonDetector(
        target,
        model_root=tmp_path,
        core_factory=FakeCore,
        yolo_factory=_factory_for(cache, created),
        download_fn=download,
    )
    assert target.read_bytes() == b"downloaded"

    with pytest.raises(PersonDetectionError, match="restricted"):
        OpenVINOPersonDetector(
            tmp_path / "models" / "custom.pt",
            model_root=tmp_path,
            core_factory=FakeCore,
            yolo_factory=lambda *_args, **_kwargs: None,
        )


def test_openvino_gpu_falls_back_only_when_configured(tmp_path: Path) -> None:
    checkpoint = tmp_path / "models" / "yolo26s.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    cache = checkpoint.with_name("yolo26s_openvino_model")
    _write_valid_cache(cache)

    detector = OpenVINOPersonDetector(
        checkpoint,
        device="gpu",
        fallback_device="cpu",
        model_root=tmp_path,
        core_factory=lambda: FakeCore(("CPU",)),
        yolo_factory=_factory_for(cache, []),
    )
    assert detector.device_used == "CPU"
    assert detector.fallback_reason is not None

    with pytest.raises(PersonDetectionError, match="no GPU"):
        OpenVINOPersonDetector(
            checkpoint,
            device="gpu",
            fallback_device="none",
            model_root=tmp_path,
            core_factory=lambda: FakeCore(("CPU",)),
            yolo_factory=_factory_for(cache, []),
        )


def test_openvino_runtime_gpu_failure_retries_on_cpu_and_verifies_cpu(tmp_path: Path) -> None:
    checkpoint = tmp_path / "models" / "yolo26s.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    cache = checkpoint.with_name("yolo26s_openvino_model")
    _write_valid_cache(cache)

    class DeviceAwareModel(FakeRuntimeModel):
        def __init__(self) -> None:
            super().__init__()
            self.last_device = ""

        def predict(self, **kwargs: object) -> list[object]:
            self.last_device = str(kwargs["device"])
            if "GPU" in self.last_device:
                raise RuntimeError("synthetic GPU compile failure")
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=np.empty((0, 4), dtype=np.float32),
                        conf=np.empty((0,), dtype=np.float32),
                        cls=np.empty((0,), dtype=np.float32),
                    )
                )
            ]

    created: list[DeviceAwareModel] = []

    def factory(path: str, **_kwargs: object) -> object:
        assert Path(path).resolve() == cache.resolve()
        model = DeviceAwareModel()
        created.append(model)
        return model

    def devices(model: DeviceAwareModel) -> tuple[str, ...]:
        return ("GPU.0",) if "GPU" in model.last_device else ("CPU",)

    detector = OpenVINOPersonDetector(
        checkpoint,
        device="gpu",
        fallback_device="cpu",
        model_root=tmp_path,
        core_factory=FakeCore,
        yolo_factory=factory,
        execution_devices_reader=devices,
    )

    assert detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)) == []
    assert detector.device_used == "CPU"
    assert detector.execution_devices == ("CPU",)
    assert detector.device_verified is True
    assert detector.fallback_reason == "synthetic GPU compile failure"
    assert len(created) == 2

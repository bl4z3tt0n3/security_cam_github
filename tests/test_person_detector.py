from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
import time

import numpy as np
import pytest

from app.config import PersonDetectionConfig
import app.inference.factory as factory_module
from app.inference import (
    DisabledPersonDetector,
    FakePersonDetector,
    OnnxPersonDetector,
    PersonDetection,
    PersonDetectionError,
    YoloEPersonDetector,
    create_person_detector,
)
from app.video.buffer import LatestFrameBuffer
from app.video.fake_source import FakeVideoSource
from app.video.sampler import FrameSampler
from app.video.worker import CameraWorker, WorkerState


class FakeInput:
    name = "images"
    shape = [1, 3, 4, 4]


class FakeSession:
    def __init__(self, output: object, providers: list[str] | None = None) -> None:
        self.output = output
        self.providers = providers or ["CPUExecutionProvider"]
        self.run_calls = 0

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def get_providers(self) -> list[str]:
        return self.providers

    def run(self, output_names: object, inputs: object) -> list[object]:
        del output_names, inputs
        self.run_calls += 1
        if isinstance(self.output, BaseException):
            raise self.output
        return [self.output]


class FakeYoloBoxes:
    def __init__(
        self,
        xyxy: object,
        confidence: object,
        classes: object,
    ) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(confidence, dtype=np.float32)
        self.cls = np.asarray(classes, dtype=np.float32)


class FakeYoloResult:
    names = {0: "person", 1: "dog"}

    def __init__(self, boxes: FakeYoloBoxes) -> None:
        self.boxes = boxes


class FakeYoloModel:
    def __init__(self, result: FakeYoloResult, *, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.classes: list[list[str]] = []
        self.predict_calls: list[dict[str, object]] = []

    def set_classes(self, classes: list[str]) -> None:
        self.classes.append(list(classes))

    def predict(self, **kwargs: object) -> list[FakeYoloResult]:
        self.predict_calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return [self.result]


def make_detection(
    bbox: tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0),
    confidence: float = 0.9,
) -> PersonDetection:
    return PersonDetection(
        bbox=bbox,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc),
    )


def make_onnx_detector(model_path: Path, output: object, **kwargs: object) -> tuple[OnnxPersonDetector, FakeSession]:
    session = FakeSession(output, providers=kwargs.pop("providers", None))
    detector = OnnxPersonDetector(model_path, session=session, **kwargs)
    return detector, session


def make_yoloe_detector(
    result: FakeYoloResult,
    **kwargs: object,
) -> tuple[YoloEPersonDetector, FakeYoloModel, list[str]]:
    model = FakeYoloModel(result)
    created: list[str] = []

    def factory(model_spec: str) -> FakeYoloModel:
        created.append(model_spec)
        return model

    detector = YoloEPersonDetector(
        "yoloe-26n-seg.pt",
        model_factory=factory,
        cuda_available=lambda: False,
        **kwargs,
    )
    return detector, model, created


def test_disabled_detector_does_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnexpectedYoloEConstruction:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("YOLOE must not be constructed while disabled")

    monkeypatch.setattr(factory_module, "YoloEPersonDetector", UnexpectedYoloEConstruction)
    detector = create_person_detector(
        PersonDetectionConfig(
            enabled=False,
            backend="yoloe",
            model="yoloe-26n-seg.pt",
        ),
        model_root=Path("does-not-exist-root"),
    )

    assert isinstance(detector, DisabledPersonDetector)
    assert detector.detect(np.zeros((2, 2, 3), dtype=np.uint8)) == []
    assert detector.device_used == "disabled"


def test_factory_auto_routes_legacy_onnx_and_yoloe_models(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[tuple[str, object, dict[str, object]]] = []

    class SentinelDetector:
        def __init__(self, model: object, **kwargs: object) -> None:
            created.append((type(self).__name__, model, kwargs))

    class SentinelOnnx(SentinelDetector):
        pass

    class SentinelYoloE(SentinelDetector):
        pass

    monkeypatch.setattr(factory_module, "OnnxPersonDetector", SentinelOnnx)
    monkeypatch.setattr(factory_module, "YoloEPersonDetector", SentinelYoloE)
    root = Path("model-root")
    onnx = create_person_detector(
        PersonDetectionConfig(enabled=True, backend="auto", model="legacy.onnx"),
        model_root=root,
    )
    yoloe = create_person_detector(
        PersonDetectionConfig(enabled=True, backend="auto", model="yoloe-26n-seg.pt"),
        model_root=root,
    )
    configured_default = create_person_detector(
        PersonDetectionConfig(enabled=True),
        model_root=root,
    )

    assert isinstance(onnx, SentinelOnnx)
    assert isinstance(yoloe, SentinelYoloE)
    assert isinstance(configured_default, SentinelYoloE)
    assert created[0][1] == root / "legacy.onnx"
    assert created[1][1] == "yoloe-26n-seg.pt"
    assert Path(str(created[2][1])) == root / "models" / "yoloe-26n-seg.pt"


def test_factory_rejects_onnx_when_explicitly_configured_for_yoloe() -> None:
    with pytest.raises(PersonDetectionError, match="backend=onnx"):
        create_person_detector(
            PersonDetectionConfig(enabled=True, backend="yoloe", model="legacy.onnx")
        )


def test_fake_detector_is_deterministic_and_records_calls() -> None:
    expected = [make_detection()]
    detector = FakePersonDetector(expected)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    assert detector.detect(frame) == expected
    assert detector.detect(frame) == expected
    assert detector.calls == 2
    assert detector.device_used == "fake"


def test_factory_fake_backend_does_not_require_a_model() -> None:
    detector = create_person_detector(
        PersonDetectionConfig(enabled=True, backend="fake", model=None)
    )
    assert isinstance(detector, FakePersonDetector)
    assert detector.detect(np.zeros((2, 2, 3), dtype=np.uint8)) == []


def test_fake_detector_supports_multiple_people_and_errors() -> None:
    expected = [make_detection((0.0, 0.0, 2.0, 2.0)), make_detection((2.0, 2.0, 4.0, 4.0))]
    detector = FakePersonDetector(expected)
    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == expected
    assert len(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))) == 2

    broken = FakePersonDetector(error=RuntimeError("synthetic detector error"))
    with pytest.raises(RuntimeError, match="synthetic detector error"):
        broken.detect(np.zeros((4, 4, 3), dtype=np.uint8))


def test_yoloe_loads_once_sets_person_prompt_and_filters_results() -> None:
    result = FakeYoloResult(
        FakeYoloBoxes(
            xyxy=[
                [0, 0, 8, 8],
                [1, 1, 9, 9],
                [2, 2, 5, 5],
                [-1, -2, 20, 30],
            ],
            confidence=[0.90, 0.80, 0.49, 0.50],
            classes=[0, 1, 0, 0],
        )
    )
    detector, model, created = make_yoloe_detector(result)
    timestamp = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)

    detections = detector.detect(np.zeros((10, 12, 3), dtype=np.uint8), timestamp)
    second = detector.detect(np.zeros((10, 12, 3), dtype=np.uint8), timestamp)

    assert created == ["yoloe-26n-seg.pt"]
    assert model.classes == [["person"]]
    assert len(model.predict_calls) == 2
    assert model.predict_calls[0]["conf"] == pytest.approx(0.5)
    assert model.predict_calls[0]["device"] == "cpu"
    assert model.predict_calls[0]["imgsz"] == 640
    assert len(detections) == len(second) == 2
    assert detections[0].bbox == (0.0, 0.0, 8.0, 8.0)
    assert detections[1].bbox == (0.0, 0.0, 12.0, 10.0)
    assert all(item.timestamp == timestamp for item in detections)
    assert detector.device_used == "cpu"
    assert detector.device_verified is True


def test_yoloe_supports_empty_output_and_rejects_invalid_timestamp() -> None:
    result = FakeYoloResult(FakeYoloBoxes([], [], []))
    detector, _, _ = make_yoloe_detector(result)

    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []
    with pytest.raises(PersonDetectionError, match="timezone-aware"):
        detector.detect(
            np.zeros((4, 4, 3), dtype=np.uint8),
            datetime(2026, 8, 15, 12, 0),
        )


def test_yoloe_rejects_invalid_frames_and_confidence_values() -> None:
    result = FakeYoloResult(
        FakeYoloBoxes(
            [[0, 0, 2, 2], [0, 0, 2, 2]],
            [1.1, 0.9],
            [0, 0],
        )
    )
    detector, _, _ = make_yoloe_detector(result)

    assert len(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))) == 1
    with pytest.raises(PersonDetectionError, match="shape HxWx3"):
        detector.detect(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(PersonDetectionError, match="finite pixels"):
        detector.detect(np.full((4, 4, 3), np.nan, dtype=np.float32))


def test_yoloe_wraps_model_load_and_inference_errors() -> None:
    with pytest.raises(PersonDetectionError, match="load/download"):
        YoloEPersonDetector(
            "yoloe-26n-seg.pt",
            model_factory=lambda _: (_ for _ in ()).throw(RuntimeError("load failed")),
            cuda_available=lambda: False,
        )

    result = FakeYoloResult(FakeYoloBoxes([], [], []))
    model = FakeYoloModel(result, error=RuntimeError("inference failed"))
    detector = YoloEPersonDetector(
        "yoloe-26n-seg.pt",
        model_factory=lambda _: model,
        cuda_available=lambda: False,
    )
    with pytest.raises(PersonDetectionError, match="inference failed"):
        detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))


def test_yoloe_auto_retries_once_on_cpu_after_cuda_failure() -> None:
    result = FakeYoloResult(
        FakeYoloBoxes([[0, 0, 2, 2]], [0.9], [0])
    )

    class CudaFailsModel(FakeYoloModel):
        def predict(self, **kwargs: object) -> list[FakeYoloResult]:
            self.predict_calls.append(dict(kwargs))
            if kwargs["device"] == "cuda:0":
                raise RuntimeError("CUDA kernel unavailable")
            return [self.result]

    model = CudaFailsModel(result)
    detector = YoloEPersonDetector(
        "yoloe-26n-seg.pt",
        model_factory=lambda _: model,
        cuda_available=lambda: True,
    )

    detections = detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert [call["device"] for call in model.predict_calls] == ["cuda:0", "cpu"]
    assert detector.device_used == "cpu"
    assert detector.device_verified is True


def test_yoloe_explicit_cuda_requires_available_cuda() -> None:
    with pytest.raises(PersonDetectionError, match="torch.cuda.is_available"):
        YoloEPersonDetector(
            "yoloe-26n-seg.pt",
            model_factory=lambda _: FakeYoloModel(FakeYoloResult(FakeYoloBoxes([], [], []))),
            device="cuda",
            cuda_available=lambda: False,
        )


def test_onnx_detector_filters_persons_and_threshold() -> None:
    output = np.array(
        [
            [0, 0, 2, 2, 0.90, 0],
            [0, 0, 3, 3, 0.80, 1],
            [0, 0, 1, 1, 0.49, 0],
            [1, 1, 3, 3, 0.50, 0],
        ],
        dtype=np.float32,
    )[None, ...]
    detector, session = make_onnx_detector(
        Path("models/fake-yolo26.onnx"), output, confidence_threshold=0.5
    )

    detections = detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert session.run_calls == 1
    assert len(detections) == 2
    assert detections[0].bbox == (0.0, 0.0, 2.0, 2.0)
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[1].bbox == (1.0, 1.0, 3.0, 3.0)
    assert detections[0].timestamp.tzinfo is not None
    assert detector.device_used == "cpu"
    assert detector.provider_used == "CPUExecutionProvider"


def test_onnx_detector_supports_multiple_people_and_empty_output() -> None:
    output = np.array(
        [
            [0, 0, 2, 2, 0.8, 0],
            [1, 1, 4, 4, 0.7, 0],
        ],
        dtype=np.float32,
    )[None, ...]
    detector, _ = make_onnx_detector(Path("models/fake-yolo26.onnx"), output)
    detections = detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(detections) == 2

    empty_detector, _ = make_onnx_detector(
        Path("models/fake-yolo26.onnx"), np.empty((1, 0, 6), dtype=np.float32)
    )
    assert empty_detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []


def test_onnx_detector_maps_and_clips_bbox_after_letterbox() -> None:
    output = np.array([[[0, 1, 4, 3, 0.9, 0]]], dtype=np.float32)
    detector, _ = make_onnx_detector(Path("models/fake-yolo26.onnx"), output)

    detections = detector.detect(np.zeros((2, 4, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].bbox == (0.0, 0.0, 4.0, 2.0)


def test_onnx_detector_rejects_missing_model() -> None:
    with pytest.raises(PersonDetectionError, match="model not found"):
        OnnxPersonDetector(Path("models/__missing_test_yolo26.onnx"))


def test_onnx_detector_wraps_inference_errors() -> None:
    detector, _ = make_onnx_detector(
        Path("models/fake-yolo26.onnx"), RuntimeError("inference exploded")
    )

    with pytest.raises(PersonDetectionError, match="inference failed"):
        detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))


def test_onnx_detector_rejects_incompatible_output() -> None:
    detector, _ = make_onnx_detector(
        Path("models/fake-yolo26.onnx"), np.zeros((1, 3, 7), dtype=np.float32)
    )

    with pytest.raises(PersonDetectionError, match=r"shape \(batch, detections, 6\)"):
        detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))


def test_auto_and_cpu_device_selection_use_cpu_provider() -> None:
    output = np.empty((1, 0, 6), dtype=np.float32)
    auto, _ = make_onnx_detector(Path("models/fake-yolo26.onnx"), output, device="auto")
    cpu, _ = make_onnx_detector(Path("models/fake-yolo26.onnx"), output, device="cpu")

    assert auto.device_used == "cpu"
    assert cpu.device_used == "cpu"


def test_cuda_device_requires_cuda_provider() -> None:
    with pytest.raises(PersonDetectionError, match="CUDAExecutionProvider"):
        make_onnx_detector(
            Path("models/fake-yolo26.onnx"),
            np.empty((1, 0, 6), dtype=np.float32),
            device="cuda",
        )


def wait_until(predicate: Callable[[], bool], timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def test_detector_receives_only_frames_consumed_from_sampler() -> None:
    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    source = FakeVideoSource([frame], fps=20.0, read_delay_s=0.01)
    worker = CameraWorker(
        "sampled-camera",
        source,
        read_timeout_s=0.1,
        reconnect_delay_s=0,
        max_buffer_frames=1,
    )
    sampler = FrameSampler(worker, target_fps=2.0, input_wait_timeout_s=0.02)
    detector = FakePersonDetector()

    worker.start()
    assert worker.wait_for_state(WorkerState.RUNNING, 1.0) is WorkerState.RUNNING
    sampler.start()
    consumed_packets = 0
    try:
        deadline = time.monotonic() + 1.5
        while consumed_packets < 2 and time.monotonic() < deadline:
            packet = sampler.get_latest(timeout_s=0.05)
            if packet is not None:
                consumed_packets += 1
                detector.detect(packet.frame)
    finally:
        sampler.stop(timeout_s=0.5)
        worker.stop(timeout_s=0.5)

    assert detector.calls == consumed_packets == 2
    assert sampler.snapshot().frames_sampled >= detector.calls
    assert worker.snapshot().frames_received > detector.calls

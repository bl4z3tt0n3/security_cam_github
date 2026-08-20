from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import pytest

from app.face import (
    ARC_FACE_TEMPLATE,
    RETAIL_0095_TEMPLATE,
    FaceAnalysisService,
    FaceDetection,
    FaceLandmark5,
    FaceMatcher,
    FaceQualityEvaluator,
    FakeEmbedder,
    FakeFaceDetector,
    OpenVINOLandmarksRegressor,
    OpenVINOFaceDetector0205,
    PersonStore,
    PersonStorageError,
    ScrfdFaceDetector,
    SimilarityFaceAligner,
    TrackRecognitionConfirmer,
    YuNetFaceDetector,
)
from app.tracking import CameraState, CameraTrackingPipeline, IoUGreedyTracker, Track


def _quality_image(value: int = 128) -> np.ndarray:
    image = np.full((160, 160, 3), value, dtype=np.uint8)
    for y in range(0, 140, 10):
        for x in range(0, 140, 10):
            if (x // 10 + y // 10) % 2:
                image[y : y + 10, x : x + 10] = min(255, value + 35)
    return image


def _track(track_id: int = 1, bbox: tuple[float, float, float, float] = (0, 0, 160, 160)) -> Track:
    now = datetime.now(timezone.utc)
    return Track(track_id, bbox, 0.95, now, now)


def test_face_landmark_order_and_similarity_alignment() -> None:
    source = FaceLandmark5(((20, 20), (80, 20), (50, 50), (30, 80), (70, 80)))
    detection = FaceDetection((10, 10, 90, 90), 0.95, landmarks=source)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    aligned = SimilarityFaceAligner(RETAIL_0095_TEMPLATE).align(image, detection)
    assert aligned.shape == (128, 128, 3)
    assert FaceLandmark5(source.points).translated(2, 3).points[0] == (22.0, 23.0)
    with pytest.raises(ValueError):
        FaceLandmark5(((1, 2),) * 4)  # type: ignore[arg-type]


class _SessionPort:
    def __init__(self, name: str, shape: list[object]) -> None:
        self.name = name
        self.any_name = name
        self.shape = shape


class _ScrfdSession:
    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def get_inputs(self) -> list[_SessionPort]:
        return [_SessionPort("input", [1, 3, 64, 64])]

    def run(self, _names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert feed["input"].shape == (1, 3, 64, 64)
        sizes = (64, 16, 4)
        scores = [np.zeros((size, 1), dtype=np.float32) for size in sizes]
        boxes = [np.zeros((size, 4), dtype=np.float32) for size in sizes]
        keypoints = [np.zeros((size, 10), dtype=np.float32) for size in sizes]
        # stride-8 grid position (x=4,y=4), with a full-frame box.
        scores[0][36, 0] = 0.95
        boxes[0][36] = (4, 4, 4, 4)
        keypoints[0][36] = (
            -1.5, -1.5, 1.5, -1.5, 0, 0, -1, 2, 1, 2
        )
        return [*scores, *boxes, *keypoints]


def test_scrfd_decodes_nine_outputs_and_landmarks() -> None:
    detector = ScrfdFaceDetector(
        Path("missing.onnx"),
        device="cpu",
        session=_ScrfdSession(),
        input_size=(64, 64),
    )
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].bbox == pytest.approx((0, 0, 64, 64))
    assert detections[0].landmarks is not None
    assert detections[0].landmarks.points[0] == pytest.approx((20, 20))


def test_capability_probe_uses_single_sample_for_dynamic_onnx_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.face.capabilities import _probe_onnx

    class _ProbeSession:
        def __init__(self) -> None:
            self.feed_shape: tuple[int, ...] | None = None

        def get_inputs(self) -> list[_SessionPort]:
            return [_SessionPort("input", [None, 3, 112, 112])]

        def run(self, _names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            self.feed_shape = tuple(feed["input"].shape)
            return [np.zeros((1, 512), dtype=np.float32)]

    session = _ProbeSession()
    fake_onnxruntime = types.SimpleNamespace(
        get_available_providers=lambda: ("CPUExecutionProvider",),
        InferenceSession=lambda *_args, **_kwargs: session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

    ready, reason, actual = _probe_onnx(
        Path("dynamic.onnx"),
        "cpu",
        expected_output_dimension=512,
    )

    assert ready is True
    assert reason == "I/O probe passed"
    assert actual == "cpu"
    assert session.feed_shape == (1, 3, 112, 112)


def test_auto_recognizer_resolution_uses_onnx_contract_not_filename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.face.model_resolution import resolve_recognizer

    fake_onnxruntime = types.SimpleNamespace(
        get_available_providers=lambda: ("CPUExecutionProvider",),
        InferenceSession=lambda *_args, **_kwargs: types.SimpleNamespace(
            get_inputs=lambda: [_SessionPort("input", [None, 3, 160, 160])],
            get_outputs=lambda: [_SessionPort("embeddings", [None, 512])],
        ),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

    resolution, spec = resolve_recognizer(
        model_id="face-reidentification-retail-0095",
        requested_backend="auto",
        path=Path("face_embedder.onnx"),
    )
    assert resolution.model_id == "facenet-20180402-vggface2"
    assert spec.embedding_dimension == 512


def test_openvino_rejects_onnx_without_deriving_a_bin_path() -> None:
    from app.face import FaceEmbeddingError, OpenVINOFaceEmbedder

    with pytest.raises(FaceEmbeddingError, match=r"\.xml"):
        OpenVINOFaceEmbedder(Path("face_embedder.onnx"))


class _YuNetFake:
    def setScoreThreshold(self, value: float) -> None:
        assert value == pytest.approx(0.5)

    def setInputSize(self, value: tuple[int, int]) -> None:
        assert value == (100, 80)

    def detect(self, image: np.ndarray):
        assert image.shape == (80, 100, 3)
        row = [10, 12, 40, 42, 20, 25, 40, 25, 30, 32, 30, 20, 38, 32, 0.9]
        return 0, np.asarray([row], dtype=np.float32)


def test_yunet_returns_box_and_fixed_landmark_order() -> None:
    detector = YuNetFaceDetector(Path("missing.onnx"), detector=_YuNetFake())
    detections = detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert detections[0].bbox == pytest.approx((10, 12, 50, 54))
    assert detections[0].landmarks is not None
    assert detections[0].landmarks.points == (
        pytest.approx((20, 25)),
        pytest.approx((40, 25)),
        pytest.approx((30, 32)),
        pytest.approx((30, 20)),
        pytest.approx((38, 32)),
    )


class _CompiledModel:
    def __init__(self, values: dict[str, np.ndarray], input_shape=(1, 3, 48, 48)) -> None:
        self.inputs = [_SessionPort("image", list(input_shape))]
        self.outputs = [_SessionPort("output", [1, 10])]
        self._values = values

    def get_property(self, name: str):
        assert name == "EXECUTION_DEVICES"
        return ["CPU"]

    def __call__(self, _inputs):
        return self._values


def test_openvino_0205_and_landmarker_decode_fake_compiled_outputs() -> None:
    detector = OpenVINOFaceDetector0205(
        Path("missing.xml"),
        device="cpu",
        compiled_model=_CompiledModel(
            {
                "boxes": np.asarray([[10, 12, 50, 54, 0.9]], dtype=np.float32),
                "labels": np.asarray([0], dtype=np.int64),
            },
            input_shape=(1, 3, 416, 416),
        ),
    )
    detections = detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert detections[0].bbox == pytest.approx((2.4038, 2.3077, 12.0192, 10.3846), rel=1e-3)

    landmarker = OpenVINOLandmarksRegressor(
        Path("missing.xml"),
        device="cpu",
        compiled_model=_CompiledModel(
            {"output": np.asarray([[0.25, 0.25, 0.75, 0.25, 0.5, 0.5, 0.3, 0.75, 0.7, 0.75]], dtype=np.float32)},
        ),
    )
    points = landmarker.landmark(
        np.zeros((80, 100, 3), dtype=np.uint8),
        FaceDetection((10, 10, 50, 50), 0.9),
    )
    assert points is not None
    assert points.points[0] == pytest.approx((20, 20))
    assert points.points[-1] == pytest.approx((38, 40))


def test_service_associates_face_to_person_and_matches_256d(tmp_path: Path) -> None:
    landmarks = FaceLandmark5(((30, 30), (100, 30), (65, 65), (40, 105), (90, 105)))
    face = FaceDetection((20, 20, 120, 120), 0.95, landmarks=landmarks)
    detector = FakeFaceDetector([face])
    embedder = FakeEmbedder(embedding_dimension=256, model_id="retail-0095")
    store = PersonStore(tmp_path / "persons")
    store.save(
        name="Mario",
        embeddings=np.ones((2, 256), dtype=np.float32),
        model=embedder.metadata,
    )
    matcher = FaceMatcher(embedder, store, threshold=0.5)
    service = FaceAnalysisService(
        "cam-1",
        detector,
        aligner=SimilarityFaceAligner(RETAIL_0095_TEMPLATE),
        matcher=matcher,
        evaluator=FaceQualityEvaluator(
            min_width=40,
            min_height=40,
            blur_threshold=0,
            min_brightness=0,
            max_brightness=255,
        ),
    )
    result = service.process(
        _quality_image(),
        state=CameraState.TRACKING,
        tracks=[_track()],
    )
    assert len(result.recognitions) == 1
    assert result.recognitions[0].status == "known"
    assert result.results[0].decisions[0].frame_bbox == pytest.approx((20, 20, 120, 120))
    assert result.results[0].decisions[0].aligned_face is not None


def test_recognition_failure_keeps_detector_decision_alive(tmp_path: Path) -> None:
    detector = FakeFaceDetector(
        [FaceDetection((20, 20, 120, 120), 0.95, landmarks=FaceLandmark5(((30, 30), (100, 30), (65, 65), (40, 105), (90, 105))))]
    )
    embedder = FakeEmbedder(embedding_dimension=256, model_id="retail-0095")
    store = PersonStore(tmp_path / "persons")
    store.save(name="Mario", embeddings=np.ones((1, 256), dtype=np.float32), model=embedder.metadata)
    matcher = FaceMatcher(embedder, store, threshold=0.5)

    def fail_match(_image: np.ndarray):
        raise RuntimeError("recognizer intentionally unavailable")

    matcher.match = fail_match  # type: ignore[method-assign]
    service = FaceAnalysisService(
        "cam-1",
        detector,
        aligner=SimilarityFaceAligner(RETAIL_0095_TEMPLATE),
        matcher=matcher,
        evaluator=FaceQualityEvaluator(
            min_width=40,
            min_height=40,
            blur_threshold=0,
            min_brightness=0,
            max_brightness=255,
        ),
    )
    result = service.analyze_track(_quality_image(), _track())
    assert len(result.decisions) == 1
    assert result.best_recognition is None
    assert service.last_recognition_error is not None


def test_orchestrator_keeps_known_candidate_unknown_until_confirmation(tmp_path: Path) -> None:
    landmarks = FaceLandmark5(((30, 30), (100, 30), (65, 65), (40, 105), (90, 105)))
    detector = FakeFaceDetector(
        [FaceDetection((20, 20, 120, 120), 0.95, landmarks=landmarks)]
    )
    embedder = FakeEmbedder(embedding_dimension=256, model_id="retail-0095")
    store = PersonStore(tmp_path / "persons")
    store.save(name="Mario", embeddings=np.ones((1, 256), dtype=np.float32), model=embedder.metadata)
    matcher = FaceMatcher(embedder, store, threshold=0.5)
    service = FaceAnalysisService(
        "cam-1",
        detector,
        aligner=SimilarityFaceAligner(RETAIL_0095_TEMPLATE),
        matcher=matcher,
        evaluator=FaceQualityEvaluator(
            min_width=40,
            min_height=40,
            blur_threshold=0,
            min_brightness=0,
            max_brightness=255,
        ),
    )
    from app.face import FaceRecognitionOrchestrator
    from app.inference import PersonDetection

    clock_value = [0.0]
    orchestrator = FaceRecognitionOrchestrator(service, face_fps=1.0, clock=lambda: clock_value[0])
    pipeline = CameraTrackingPipeline(
        "cam-1",
        tracker=IoUGreedyTracker(max_missed_samples=0),
        recognition_confirmer=TrackRecognitionConfirmer(
            min_confirmations=2,
            camera_id="cam-1",
        ),
    )
    person = PersonDetection((0, 0, 160, 160), 0.9, datetime.now(timezone.utc))
    update = pipeline.update([person])
    first = orchestrator.process(_quality_image(), update, pipeline)
    assert first.analysis.recognitions[0].status == "known"
    assert first.final_for_track(1) is not None
    assert first.final_for_track(1).status == "unknown"  # type: ignore[union-attr]
    clock_value[0] = 2.0
    second = orchestrator.process(_quality_image(), update, pipeline)
    assert second.final_for_track(1).status == "known"  # type: ignore[union-attr]


def test_matcher_accepts_512d_and_rejects_cross_fingerprint(tmp_path: Path) -> None:
    first = FakeEmbedder(embedding_dimension=512, model_id="arcface")
    second = FakeEmbedder(embedding_dimension=512, model_id="arcface", model_version="2")
    store = PersonStore(tmp_path / "persons")
    store.save(name="A", embeddings=np.ones((1, 512), dtype=np.float32), model=first.metadata)
    assert FaceMatcher(first, store, threshold=0.8).match(np.zeros((8, 8, 3), dtype=np.uint8)).status == "known"
    with pytest.raises(ValueError, match="incompatible"):
        FaceMatcher(second, store, threshold=0.8)


def test_gallery_scope_is_nested_and_path_safe(tmp_path: Path) -> None:
    model = FakeEmbedder(embedding_dimension=256, model_id="retail-0095").metadata
    first = PersonStore(tmp_path / "persons", scope=Path("retail-0095") / "fingerprint-a")
    second = PersonStore(tmp_path / "persons", scope=Path("retail-0095") / "fingerprint-b")
    first.save(name="Mario", embeddings=np.ones((1, 256), dtype=np.float32), model=model)
    second.save(name="Mario", embeddings=np.ones((1, 256), dtype=np.float32), model=model)
    assert first.load_all(expected_model=model)[0].directory == (
        tmp_path / "persons" / "retail-0095" / "fingerprint-a" / "mario"
    )
    assert second.load_all(expected_model=model)[0].directory != first.load_all(expected_model=model)[0].directory
    with pytest.raises(PersonStorageError):
        PersonStore(tmp_path / "persons", scope="../escape")


def test_orchestrator_runs_only_on_due_active_track_and_temporally_confirms() -> None:
    clock_value = [0.0]
    detector = FakeFaceDetector([FaceDetection((10, 10, 100, 100), 0.95)])
    service = FaceAnalysisService(
        "cam-1",
        detector,
        evaluator=FaceQualityEvaluator(
            min_width=40,
            min_height=40,
            blur_threshold=0,
            min_brightness=0,
            max_brightness=255,
        ),
    )
    from app.face import FaceRecognitionOrchestrator

    orchestrator = FaceRecognitionOrchestrator(service, face_fps=1.0, clock=lambda: clock_value[0])
    pipeline = CameraTrackingPipeline(
        "cam-1",
        tracker=IoUGreedyTracker(max_missed_samples=0),
        recognition_confirmer=TrackRecognitionConfirmer(min_confirmations=2, camera_id="cam-1"),
    )
    from app.inference import PersonDetection

    person = PersonDetection((0, 0, 140, 140), 0.9, datetime.now(timezone.utc))
    update = pipeline.update([person])
    first = orchestrator.process(_quality_image(), update, pipeline)
    assert first.skipped is False
    assert detector.calls == 1
    clock_value[0] = 2.0
    # The face clock can be due again without a second person-detection sample.
    second = orchestrator.process(_quality_image(), update, pipeline)
    assert detector.calls == 2
    assert second.confirmations == ()  # no matcher means no temporal recognition

from datetime import datetime, timezone

import numpy as np

from app.face import (
    FaceAnalysisService,
    FaceDetection,
    FaceQualityEvaluator,
    FaceQualityReason,
    FakeFaceDetector,
    crop_frame,
)
from app.tracking import CameraState, Track


def quality_image(value: int = 128) -> np.ndarray:
    image = np.full((140, 140, 3), value, dtype=np.uint8)
    for y in range(0, 120, 10):
        for x in range(0, 120, 10):
            if (x // 10 + y // 10) % 2:
                image[y : y + 10, x : x + 10] = min(255, value + 35)
    return image


def evaluator() -> FaceQualityEvaluator:
    return FaceQualityEvaluator(
        min_width=80,
        min_height=80,
        blur_threshold=40,
        min_brightness=30,
        max_brightness=225,
        min_confidence=0.5,
    )


def evaluate(**kwargs: object):
    detection = FaceDetection(kwargs.pop("bbox", (10, 10, 100, 100)), kwargs.pop("confidence", 0.9))
    return evaluator().evaluate(kwargs.pop("image", quality_image()), detection, **kwargs)


def test_face_too_small_is_rejected() -> None:
    result = evaluate(bbox=(10, 10, 50, 90))
    assert not result.accepted
    assert FaceQualityReason.TOO_SMALL in result.reasons


def test_face_blurry_is_rejected() -> None:
    image = np.full((140, 140, 3), 128, dtype=np.uint8)
    result = evaluate(image=image)
    assert not result.accepted
    assert FaceQualityReason.BLURRY in result.reasons


def test_face_too_dark_is_rejected() -> None:
    result = evaluate(image=quality_image(10))
    assert not result.accepted
    assert FaceQualityReason.TOO_DARK in result.reasons


def test_face_too_bright_is_rejected() -> None:
    result = evaluate(image=quality_image(245))
    assert not result.accepted
    assert FaceQualityReason.TOO_BRIGHT in result.reasons


def test_face_partial_bbox_is_rejected() -> None:
    result = evaluate(partial_bbox=True)
    assert not result.accepted
    assert FaceQualityReason.PARTIAL_BBOX in result.reasons


def test_low_confidence_is_rejected() -> None:
    result = evaluate(confidence=0.49)
    assert not result.accepted
    assert FaceQualityReason.LOW_CONFIDENCE in result.reasons


def test_valid_face_is_accepted() -> None:
    result = evaluate()
    assert result.accepted
    assert result.reasons == ()
    assert result.blur_score >= 40


def test_crop_clips_bbox_and_reports_partial_frame() -> None:
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    cropped = crop_frame(frame, (-2, 2, 8, 14))
    assert cropped is not None
    assert cropped.image.shape == (8, 8, 3)
    assert cropped.bbox == (0.0, 2.0, 8.0, 10.0)
    assert cropped.was_partial


def make_track(track_id: int, bbox: tuple[float, float, float, float]) -> Track:
    timestamp = datetime.now(timezone.utc)
    return Track(track_id, bbox, 0.9, timestamp, timestamp)


def test_service_does_not_call_detector_without_person() -> None:
    detector = FakeFaceDetector([FaceDetection((0, 0, 100, 100), 0.9)])
    service = FaceAnalysisService("camera-1", detector, evaluator=evaluator())
    result = service.process(
        quality_image(), state=CameraState.PERSON_SCAN, tracks=()
    )
    assert result.skipped
    assert detector.calls == 0


def test_service_isolated_between_two_cameras() -> None:
    detector_one = FakeFaceDetector([FaceDetection((10, 10, 100, 100), 0.9)])
    detector_three = FakeFaceDetector([FaceDetection((10, 10, 100, 100), 0.9)])
    service_one = FaceAnalysisService("camera-1", detector_one, evaluator=evaluator())
    service_three = FaceAnalysisService("camera-3", detector_three, evaluator=evaluator())
    frame = quality_image()
    service_one.process(frame, state=CameraState.TRACKING, tracks=[make_track(1, (0, 0, 140, 140))])
    service_three.process(frame, state=CameraState.PERSON_SCAN, tracks=[])
    assert detector_one.calls == 1
    assert detector_three.calls == 0


def test_service_only_analyzes_active_tracks_and_returns_quality_reason() -> None:
    detector = FakeFaceDetector([FaceDetection((10, 10, 40, 40), 0.9)])
    service = FaceAnalysisService("camera-1", detector, evaluator=evaluator())
    result = service.process(
        quality_image(),
        state=CameraState.TRACKING,
        tracks=[make_track(7, (0, 0, 140, 140))],
    )
    assert detector.calls == 1
    assert result.results[0].track_id == 7
    assert FaceQualityReason.TOO_SMALL in result.results[0].decisions[0].quality.reasons


def test_service_translates_face_bbox_to_original_frame() -> None:
    detector = FakeFaceDetector([FaceDetection((10, 20, 70, 80), 0.9)])
    service = FaceAnalysisService("camera-1", detector, evaluator=evaluator())
    result = service.process(
        quality_image(),
        state=CameraState.TRACKING,
        tracks=[make_track(3, (20, 30, 130, 130))],
    )
    assert result.results[0].decisions[0].frame_bbox == (30.0, 50.0, 90.0, 110.0)

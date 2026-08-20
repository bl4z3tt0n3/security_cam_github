"""Offline demonstration of selective face analysis with synthetic data."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.face import FaceAnalysisService, FaceDetection, FakeFaceDetector
from app.inference import PersonDetection
from app.tracking import CameraState, CameraTrackingPipeline


def main() -> int:
    frame = np.full((160, 160, 3), 128, dtype=np.uint8)
    for y in range(0, 160, 10):
        for x in range(0, 160, 10):
            if (x // 10 + y // 10) % 2:
                frame[y : y + 10, x : x + 10] = 163
    face_detector = FakeFaceDetector([FaceDetection((20, 20, 120, 120), 0.95)])
    service = FaceAnalysisService("demo-camera", face_detector)
    pipeline = CameraTrackingPipeline("demo-camera")
    update = pipeline.update(
        [
            PersonDetection(
                bbox=(0, 0, 160, 160),
                confidence=0.95,
                timestamp=datetime.now(timezone.utc),
            )
        ]
    )
    result = service.process(frame, state=CameraState.TRACKING, tracks=update.active_tracks)
    print(f"camera: {result.camera_id}")
    print(f"face detector calls: {face_detector.calls}")
    for tracked in result.results:
        for decision in tracked.decisions:
            print(
                f"track {tracked.track_id}: accepted={decision.quality.accepted} "
                f"reasons={[reason.value for reason in decision.quality.reasons]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

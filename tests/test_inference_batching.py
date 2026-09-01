from __future__ import annotations

from datetime import datetime, timezone
import threading

import numpy as np

from app.inference import BatchingPersonDetector, PersonDetection, PersonDetector


class BatchCapableDetector(PersonDetector):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    @property
    def supports_batch_inference(self) -> bool:
        return True

    @property
    def preferred_batch_size(self) -> int:
        return 2

    def detect(self, frame: np.ndarray, timestamp: datetime | None = None) -> list[PersonDetection]:
        del frame, timestamp
        raise AssertionError("batch wrapper should use detect_batch")

    def detect_batch(
        self,
        frames: list[np.ndarray],
        timestamps: list[datetime | None] | None = None,
    ) -> list[list[PersonDetection]]:
        self.batch_sizes.append(len(frames))
        stamps = timestamps or [None] * len(frames)
        return [
            [
                PersonDetection(
                    bbox=(0.0, 0.0, 4.0, 4.0),
                    confidence=0.9,
                    timestamp=stamp or datetime.now(timezone.utc),
                )
            ]
            for stamp in stamps
        ]


def test_batching_detector_groups_concurrent_camera_requests() -> None:
    backend = BatchCapableDetector()
    detector = BatchingPersonDetector(
        backend,
        max_batch_size=2,
        batch_window_ms=50,
    )
    barrier = threading.Barrier(3)
    results: list[list[PersonDetection]] = []

    def run(value: int) -> None:
        barrier.wait()
        results.append(
            detector.detect(np.full((8, 8, 3), value, dtype=np.uint8))
        )

    threads = [threading.Thread(target=run, args=(value,)) for value in (1, 2)]
    try:
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=1.0)
        assert all(not thread.is_alive() for thread in threads)
        assert backend.batch_sizes == [2]
        assert len(results) == 2
        assert all(len(result) == 1 for result in results)
    finally:
        detector.close()

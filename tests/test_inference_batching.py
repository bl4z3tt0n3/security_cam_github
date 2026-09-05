from __future__ import annotations

from datetime import datetime, timezone
import threading
import time

import numpy as np
import pytest

from app.inference import BatchingPersonDetector, PersonDetection, PersonDetector


class BatchCapableDetector(PersonDetector):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.closed = False

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

    def close(self) -> None:
        self.closed = True


class BlockingBatchDetector(BatchCapableDetector):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.close_calls = 0

    def detect_batch(
        self,
        frames: list[np.ndarray],
        timestamps: list[datetime | None] | None = None,
    ) -> list[list[PersonDetection]]:
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test did not release backend")
        return super().detect_batch(frames, timestamps)

    def close(self) -> None:
        self.close_calls += 1
        super().close()


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


def test_batching_detector_shutdown_releases_accepted_work_without_closing_in_use_backend() -> None:
    backend = BlockingBatchDetector()
    detector = BatchingPersonDetector(
        backend,
        max_batch_size=2,
        batch_window_ms=0,
    )
    result: list[list[PersonDetection]] = []
    errors: list[BaseException] = []

    def call_detect() -> None:
        try:
            result.append(detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    caller = threading.Thread(target=call_detect)
    closer = threading.Thread(target=detector.close)
    caller.start()
    assert backend.started.wait(timeout=1.0)

    closer.start()
    time.sleep(0.05)
    assert backend.closed is False
    with pytest.raises(RuntimeError, match="closing or closed"):
        detector.detect(np.ones((8, 8, 3), dtype=np.uint8))

    backend.release.set()
    caller.join(timeout=1.0)
    closer.join(timeout=1.0)
    assert not caller.is_alive()
    assert not closer.is_alive()
    assert errors == []
    assert len(result) == 1
    assert backend.closed is True
    assert backend.close_calls == 1

    # Repeated shutdown is idempotent and cannot double-close the backend.
    detector.close()
    assert backend.close_calls == 1

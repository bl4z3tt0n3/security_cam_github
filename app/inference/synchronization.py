"""Concurrency boundaries for shared local inference adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import queue
import threading
import time
from typing import Any, TypeVar

from .base import PersonDetection, PersonDetector


ResultT = TypeVar("ResultT")


class InferenceGate:
    """Serialize calls into a model instance shared by camera runtimes.

    ONNX Runtime sessions are commonly safe to use concurrently, but the
    replaceable detector/embedder contracts do not require that guarantee.
    The gate gives the application one conservative, explicit policy without
    making camera state or metrics shared.
    """

    def __init__(self, max_parallel: int = 1) -> None:
        if isinstance(max_parallel, bool) or not isinstance(max_parallel, int) or max_parallel < 1:
            raise ValueError("max_parallel must be a positive integer")
        self._max_parallel = max_parallel
        self._guard: Any = (
            threading.RLock()
            if max_parallel == 1
            else threading.BoundedSemaphore(max_parallel)
        )

    @property
    def max_parallel(self) -> int:
        return self._max_parallel

    def run(
        self,
        operation: Callable[..., ResultT],
        *args: Any,
        **kwargs: Any,
    ) -> ResultT:
        """Run one model operation while holding the shared inference lock."""

        with self._guard:
            return operation(*args, **kwargs)


@dataclass
class _BatchJob:
    frame: Any
    timestamp: datetime | None
    completed: threading.Event = field(default_factory=threading.Event)
    result: list[PersonDetection] | None = None
    error: BaseException | None = None


class BatchingPersonDetector(PersonDetector):
    """Micro-batch concurrent camera requests into one backend inference call."""

    def __init__(
        self,
        detector: PersonDetector,
        *,
        max_batch_size: int | None = None,
        batch_window_ms: float = 5.0,
    ) -> None:
        if not detector.supports_batch_inference:
            raise ValueError("detector does not support batched inference")
        preferred = max(1, int(detector.preferred_batch_size))
        self._detector = detector
        self._max_batch_size = min(
            preferred,
            max(1, int(max_batch_size)) if max_batch_size is not None else preferred,
        )
        if self._max_batch_size < 2:
            raise ValueError("batching requires max_batch_size >= 2")
        if batch_window_ms < 0:
            raise ValueError("batch_window_ms cannot be negative")
        self._batch_window_s = float(batch_window_ms) / 1000.0
        self._queue: queue.Queue[_BatchJob | None] = queue.Queue()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="person-inference-batcher",
            daemon=True,
        )
        self._thread.start()

    @property
    def detector(self) -> PersonDetector:
        return self._detector

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def device_used(self) -> str:
        return self._detector.device_used

    @property
    def provider_used(self) -> str:
        return self._detector.provider_used

    @property
    def backend(self) -> str:
        return self._detector.backend

    @property
    def device_verified(self) -> bool:
        return self._detector.device_verified

    @property
    def supports_concurrent_inference(self) -> bool:
        # Concurrent callers only enqueue work; one worker owns the backend.
        return True

    def detect(
        self,
        frame: Any,
        timestamp: datetime | None = None,
    ) -> list[PersonDetection]:
        if self._closed.is_set():
            raise RuntimeError("batched person detector is closed")
        job = _BatchJob(frame=frame, timestamp=timestamp)
        self._queue.put(job)
        job.completed.wait()
        if job.error is not None:
            raise job.error
        return job.result or []

    def _run(self) -> None:
        while not self._closed.is_set():
            first = self._queue.get()
            if first is None:
                break
            jobs = [first]
            deadline = time.monotonic() + self._batch_window_s
            while len(jobs) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    candidate = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if candidate is None:
                    self._closed.set()
                    break
                jobs.append(candidate)
            try:
                results = self._detector.detect_batch(
                    [job.frame for job in jobs],
                    [job.timestamp for job in jobs],
                )
                if len(results) != len(jobs):
                    raise RuntimeError(
                        "batched detector returned a different number of results than inputs"
                    )
                for job, result in zip(jobs, results, strict=True):
                    job.result = result
            except BaseException as exc:
                for job in jobs:
                    job.error = exc
            finally:
                for job in jobs:
                    job.completed.set()

        # Release any callers queued during shutdown instead of leaving camera
        # threads blocked indefinitely.
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job.error = RuntimeError("batched person detector stopped")
                job.completed.set()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)
        self._thread.join(timeout=2.0)
        self._detector.close()


class SynchronizedPersonDetector(PersonDetector):
    """Delegate person detection through a shared :class:`InferenceGate`."""

    def __init__(
        self,
        detector: PersonDetector,
        gate: InferenceGate,
    ) -> None:
        if detector is None:
            raise ValueError("detector is required")
        if gate is None:
            raise ValueError("gate is required")
        self._detector = detector
        self._gate = gate

    @property
    def detector(self) -> PersonDetector:
        return self._detector

    @property
    def gate(self) -> InferenceGate:
        return self._gate

    @property
    def device_used(self) -> str:
        return self._detector.device_used

    @property
    def provider_used(self) -> str:
        return self._detector.provider_used

    @property
    def backend(self) -> str:
        return self._detector.backend

    @property
    def device_verified(self) -> bool:
        return self._detector.device_verified

    @property
    def supports_concurrent_inference(self) -> bool:
        return self._detector.supports_concurrent_inference

    def detect(
        self,
        frame: Any,
        timestamp: datetime | None = None,
    ) -> list[PersonDetection]:
        return self._gate.run(self._detector.detect, frame, timestamp)

    def close(self) -> None:
        self._detector.close()

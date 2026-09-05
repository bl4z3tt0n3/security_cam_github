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
    """Micro-batch concurrent camera requests into one backend inference call.

    Job acceptance and the OPEN -> CLOSING transition share the same lock.  A
    job accepted while OPEN is therefore guaranteed to be visible to the sole
    worker before shutdown can complete.  The worker, not ``close()``, owns
    backend destruction so a join timeout can never close a backend that is
    still executing an inference.
    """

    _OPEN = "open"
    _CLOSING = "closing"
    _CLOSED = "closed"

    def __init__(
        self,
        detector: PersonDetector,
        *,
        max_batch_size: int | None = None,
        batch_window_ms: float = 5.0,
        max_pending_jobs: int | None = None,
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
        if max_pending_jobs is None:
            max_pending_jobs = max(8, self._max_batch_size * 4)
        if (
            isinstance(max_pending_jobs, bool)
            or not isinstance(max_pending_jobs, int)
            or max_pending_jobs < self._max_batch_size
        ):
            raise ValueError("max_pending_jobs must be an integer >= max_batch_size")
        self._batch_window_s = float(batch_window_ms) / 1000.0
        self._queue: queue.Queue[_BatchJob] = queue.Queue(maxsize=max_pending_jobs)
        self._state_lock = threading.Lock()
        self._state = self._OPEN
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
    def max_pending_jobs(self) -> int:
        return self._queue.maxsize

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
        job = _BatchJob(frame=frame, timestamp=timestamp)
        # The state check and queue admission are one atomic lifecycle action.
        # close() cannot transition to CLOSING between them.
        with self._state_lock:
            if self._state != self._OPEN:
                raise RuntimeError("batched person detector is closing or closed")
            try:
                self._queue.put_nowait(job)
            except queue.Full as exc:
                raise RuntimeError("batched person detector queue is full") from exc
        job.completed.wait()
        if job.error is not None:
            raise job.error
        return job.result or []

    def _state_is_open(self) -> bool:
        with self._state_lock:
            return self._state == self._OPEN

    def _run(self) -> None:
        try:
            while True:
                try:
                    first = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if not self._state_is_open():
                        break
                    continue

                jobs = [first]
                deadline = time.monotonic() + self._batch_window_s
                while len(jobs) < self._max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        jobs.append(self._queue.get(timeout=remaining))
                    except queue.Empty:
                        break

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

                # Once closing starts no new jobs can be admitted.  Drain every
                # job that was accepted before the transition, then exit.
                if not self._state_is_open() and self._queue.empty():
                    break
        finally:
            # Defensive release for jobs left behind if the worker itself
            # fails unexpectedly.  No accepted caller is allowed to wait forever.
            while True:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    break
                job.error = RuntimeError("batched person detector stopped")
                job.completed.set()
            try:
                self._detector.close()
            finally:
                with self._state_lock:
                    self._state = self._CLOSED

    def close(self) -> None:
        with self._state_lock:
            if self._state == self._CLOSED:
                return
            if self._state == self._OPEN:
                self._state = self._CLOSING
        if self._thread is threading.current_thread():
            return
        # A timeout deliberately does not close the backend here.  The worker
        # owns it and will close it after the current inference and accepted
        # queue drain complete.
        self._thread.join(timeout=2.0)


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

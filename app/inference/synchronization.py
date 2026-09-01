"""Concurrency boundaries for shared local inference adapters."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import threading
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

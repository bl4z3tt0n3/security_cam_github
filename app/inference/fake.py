"""Deterministic person detector used by offline tests and development."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime

import numpy as np

from .base import PersonDetection, PersonDetector


class FakePersonDetector(PersonDetector):
    """Return predefined detections without loading a model."""

    def __init__(
        self,
        detections: Iterable[PersonDetection] = (),
        *,
        callback: Callable[[np.ndarray], Iterable[PersonDetection]] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if callback is not None and not callable(callback):
            raise ValueError("callback must be callable")
        if error is not None and not isinstance(error, BaseException):
            raise ValueError("error must be an exception")
        self._detections = tuple(detections)
        self._callback = callback
        self._error = error
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def device_used(self) -> str:
        return "fake"

    @property
    def provider_used(self) -> str:
        return "fake"

    @property
    def backend(self) -> str:
        return "fake"

    @property
    def device_verified(self) -> bool:
        return True

    def detect(
        self,
        frame: np.ndarray,
        timestamp: datetime | None = None,
    ) -> list[PersonDetection]:
        self._calls += 1
        if self._error is not None:
            raise self._error
        if self._callback is not None:
            return list(self._callback(frame))
        return list(self._detections)

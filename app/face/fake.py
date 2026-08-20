"""Deterministic face detector for tests and offline demos."""

from collections.abc import Callable, Iterable

import numpy as np

from .base import FaceDetection, FaceDetector


class FakeFaceDetector(FaceDetector):
    def __init__(
        self,
        detections: Iterable[FaceDetection] = (),
        *,
        callback: Callable[[np.ndarray], Iterable[FaceDetection]] | None = None,
    ) -> None:
        if callback is not None and not callable(callback):
            raise ValueError("callback must be callable")
        self._detections = tuple(detections)
        self._callback = callback
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        self._calls += 1
        if self._callback is not None:
            return list(self._callback(frame))
        return list(self._detections)

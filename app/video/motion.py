"""Optional low-cost frame-difference motion detection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class MotionDetectionError(ValueError):
    """Raised when a frame cannot be evaluated by the motion detector."""


@dataclass(frozen=True)
class MotionDecision:
    """One motion decision and the measured changed-pixel fraction."""

    motion_detected: bool
    changed_fraction: float
    warmup: bool = False


class MotionDetector:
    """Compare consecutive low-resolution grayscale frames.

    The detector deliberately owns state for one camera only.  The first frame
    and all configured warm-up frames are treated as motion so a newly opened or
    reconnected stream cannot suppress its first person-detection opportunity.
    """

    def __init__(
        self,
        *,
        pixel_threshold: int = 25,
        min_changed_fraction: float = 0.01,
        resize_width: int = 320,
        warmup_frames: int = 1,
    ) -> None:
        if isinstance(pixel_threshold, bool) or not 1 <= int(pixel_threshold) <= 255:
            raise ValueError("pixel_threshold must be an integer between 1 and 255")
        if not math.isfinite(float(min_changed_fraction)) or not 0 <= float(min_changed_fraction) <= 1:
            raise ValueError("min_changed_fraction must be finite and between 0 and 1")
        if isinstance(resize_width, bool) or not 1 <= int(resize_width) <= 2048:
            raise ValueError("resize_width must be between 1 and 2048")
        if isinstance(warmup_frames, bool) or not 1 <= int(warmup_frames) <= 30:
            raise ValueError("warmup_frames must be between 1 and 30")

        self.pixel_threshold = int(pixel_threshold)
        self.min_changed_fraction = float(min_changed_fraction)
        self.resize_width = int(resize_width)
        self.warmup_frames = int(warmup_frames)
        self._previous: np.ndarray | None = None
        self._warmup_remaining = self.warmup_frames

    @property
    def initialized(self) -> bool:
        return self._previous is not None and self._warmup_remaining == 0

    def reset(self) -> None:
        """Forget the previous frame, for a new stream or a reconnect."""

        self._previous = None
        self._warmup_remaining = self.warmup_frames

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise MotionDetectionError("motion frame must have shape HxWx3")
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise MotionDetectionError("motion frame cannot be empty")
        if image.dtype == np.uint8:
            return image
        return np.clip(image, 0, 255).astype(np.uint8)

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        image = self._validate_frame(frame)
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - OpenCV is a project dependency.
            raise MotionDetectionError("OpenCV is required for motion detection") from exc

        height, width = image.shape[:2]
        target_width = min(self.resize_width, width)
        target_height = max(1, round(height * target_width / width))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def detect(self, frame: np.ndarray) -> MotionDecision:
        """Evaluate one frame and update the per-camera comparison state."""

        current = self._prepare(frame)
        previous = self._previous
        self._previous = current

        if previous is None or previous.shape != current.shape or self._warmup_remaining > 0:
            self._warmup_remaining = max(0, self._warmup_remaining - 1)
            return MotionDecision(True, 1.0, warmup=True)

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - guarded by _prepare.
            raise MotionDetectionError("OpenCV is required for motion detection") from exc

        difference = cv2.absdiff(previous, current)
        changed = difference > self.pixel_threshold
        changed_fraction = float(np.count_nonzero(changed) / changed.size)
        return MotionDecision(
            motion_detected=changed_fraction >= self.min_changed_fraction,
            changed_fraction=changed_fraction,
            warmup=False,
        )

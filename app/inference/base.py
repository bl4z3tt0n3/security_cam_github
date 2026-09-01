"""Stable contracts shared by real and fake person detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Final

import numpy as np


BBox = tuple[float, float, float, float]
"""Bounding box in ``(x1, y1, x2, y2)`` pixel coordinates."""

MaskPolygon = tuple[tuple[float, float], ...]
"""One instance mask polygon in source-frame pixel coordinates."""


class PersonDetectionError(RuntimeError):
    """Raised when a person detector cannot load or execute its model."""


@dataclass(frozen=True)
class PersonDetection:
    """One local detection associated with the frame being processed.

    The original contract represented only class-0 people.  Optional class
    and mask metadata keeps that contract compatible while allowing YOLOE to
    return the prompt category selected by the user.
    """

    bbox: BBox
    confidence: float
    timestamp: datetime
    class_id: int = 0
    label: str = "person"
    mask_polygon: MaskPolygon | None = None

    def __post_init__(self) -> None:
        if len(self.bbox) != 4 or not all(math.isfinite(value) for value in self.bbox):
            raise ValueError("bbox must contain four finite values")
        x1, y1, x2, y2 = self.bbox
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox coordinates must be ordered as x1,y1,x2,y2")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be a finite value between 0 and 1")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError("class_id must be a non-negative integer")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if self.mask_polygon is not None:
            if len(self.mask_polygon) < 3:
                raise ValueError("mask_polygon must contain at least three points")
            if not all(
                len(point) == 2 and all(math.isfinite(float(value)) for value in point)
                for point in self.mask_polygon
            ):
                raise ValueError("mask_polygon must contain finite x,y points")


class PersonDetector(ABC):
    """Replaceable detector contract independent of preview or stream code."""

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        timestamp: datetime | None = None,
    ) -> list[PersonDetection]:
        """Detect people in one sampled BGR frame."""

    @property
    def device_used(self) -> str:
        """Return the logical device used by this detector."""

        return "unknown"

    @property
    def provider_used(self) -> str:
        """Return the concrete inference provider, when available."""

        return "unknown"

    @property
    def backend(self) -> str:
        """Return the adapter backend name."""

        return "unknown"

    @property
    def device_verified(self) -> bool:
        """Return whether at least one real inference verified the device."""

        return False

    @property
    def supports_concurrent_inference(self) -> bool:
        """Whether one detector instance can execute independent calls concurrently."""

        return False

    def close(self) -> None:
        """Release optional model resources."""

        return None


class DisabledPersonDetector(PersonDetector):
    """No-op detector used when person detection is explicitly disabled."""

    device_used: Final[str] = "disabled"
    provider_used: Final[str] = "disabled"
    backend: Final[str] = "disabled"

    def detect(
        self,
        frame: np.ndarray,
        timestamp: datetime | None = None,
    ) -> list[PersonDetection]:
        del frame, timestamp
        return []


def utc_now() -> datetime:
    """Return an aware UTC timestamp for detection results."""

    return datetime.now(timezone.utc)

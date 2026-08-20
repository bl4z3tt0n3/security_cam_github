"""Video source abstractions and stream diagnostics."""

from .base import (
    FramePacket,
    ReadResult,
    ReadStatus,
    StreamInfo,
    VideoSource,
    VideoSourceError,
    redact_url,
)
from .sampler import FrameSampler, FrameSamplerSnapshot
from .motion import MotionDecision, MotionDetectionError, MotionDetector

__all__ = [
    "FramePacket",
    "ReadResult",
    "ReadStatus",
    "StreamInfo",
    "VideoSource",
    "VideoSourceError",
    "redact_url",
    "FrameSampler",
    "FrameSamplerSnapshot",
    "MotionDecision",
    "MotionDetectionError",
    "MotionDetector",
]

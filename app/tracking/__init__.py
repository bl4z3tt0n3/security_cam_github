"""Replaceable person tracking and per-camera state management."""

from .models import (
    CameraTrackingUpdate,
    CameraState,
    FaceAnalysisHook,
    FaceAnalysisOutcome,
    EventPublisherLike,
    InvalidStateTransitionError,
    RecognitionConfirmationLike,
    RecognitionResultLike,
    StateTransition,
    Track,
    TrackingUpdate,
    TrackRecognitionConfirmerLike,
)
from .pipeline import CameraTrackingPipeline
from .state_machine import CameraStateMachine
from .tracker import IoUGreedyTracker, PersonTracker

__all__ = [
    "CameraState",
    "CameraStateMachine",
    "CameraTrackingPipeline",
    "CameraTrackingUpdate",
    "EventPublisherLike",
    "FaceAnalysisHook",
    "FaceAnalysisOutcome",
    "InvalidStateTransitionError",
    "RecognitionConfirmationLike",
    "RecognitionResultLike",
    "IoUGreedyTracker",
    "PersonTracker",
    "StateTransition",
    "Track",
    "TrackRecognitionConfirmerLike",
    "TrackingUpdate",
]

"""Temporal confirmation of recognition results for independent tracks."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from .matcher import RecognitionResult


@dataclass(frozen=True)
class RecognitionConfirmation:
    """Result of adding one recognition observation to a track."""

    result: RecognitionResult
    confirmed: bool
    consecutive_count: int
    stable_result: RecognitionResult | None


@dataclass
class _TrackState:
    pending_key: tuple[str, str | None]
    consecutive_count: int
    stable_result: RecognitionResult | None = None
    last_observed: float = 0.0


class TrackRecognitionConfirmer:
    """Keep recognition confirmation state isolated by track.

    A candidate identity must be observed consecutively ``min_confirmations``
    times.  A conflicting candidate resets only that track's pending streak;
    an already confirmed identity remains stable until the replacement is
    itself confirmed.
    """

    def __init__(
        self,
        min_confirmations: int = 2,
        *,
        camera_id: str = "",
        confirmation_window_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(min_confirmations, bool)
            or not isinstance(min_confirmations, int)
            or min_confirmations < 1
        ):
            raise ValueError("min_confirmations must be a positive integer")
        self.min_confirmations = min_confirmations
        normalized_camera = str(camera_id).strip()
        if confirmation_window_seconds is not None and confirmation_window_seconds <= 0:
            raise ValueError("confirmation_window_seconds must be positive")
        self.camera_id = normalized_camera
        self.confirmation_window_seconds = confirmation_window_seconds
        self._clock = clock
        self._states: dict[tuple[str, int], _TrackState] = {}

    @staticmethod
    def _key(result: RecognitionResult) -> tuple[str, str | None]:
        return result.status, result.person_id if result.status == "known" else None

    def observe(self, track_id: int, result: RecognitionResult) -> RecognitionConfirmation:
        """Add one result and report whether it is now temporally confirmed."""

        now = float(self._clock())
        key_id = (self.camera_id, int(track_id))
        key = self._key(result)
        state = self._states.get(key_id)
        if (
            state is None
            or state.pending_key != key
            or (
                self.confirmation_window_seconds is not None
                and now - state.last_observed > self.confirmation_window_seconds
            )
        ):
            state = _TrackState(
                pending_key=key,
                consecutive_count=0,
                stable_result=None if state is None else state.stable_result,
                last_observed=now,
            )
            self._states[key_id] = state
        state.consecutive_count += 1
        state.last_observed = now
        confirmed = state.consecutive_count >= self.min_confirmations
        if confirmed:
            state.stable_result = result
        return RecognitionConfirmation(
            result=result,
            confirmed=confirmed,
            consecutive_count=state.consecutive_count,
            stable_result=state.stable_result,
        )

    def forget(self, track_id: int) -> None:
        """Discard all pending and confirmed state for a closed track."""

        self._states.pop((self.camera_id, int(track_id)), None)

    reset = forget

    def count(self, track_id: int) -> int:
        """Return the current consecutive count for a track."""

        state = self._states.get((self.camera_id, int(track_id)))
        return 0 if state is None else state.consecutive_count

    def stable_result(self, track_id: int) -> RecognitionResult | None:
        """Return the last temporally confirmed result for a track."""

        state = self._states.get((self.camera_id, int(track_id)))
        return None if state is None else state.stable_result

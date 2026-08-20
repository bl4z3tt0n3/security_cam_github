"""Per-camera state machine layered over tracker lifecycle updates."""

from __future__ import annotations

from .models import (
    CameraState,
    FaceAnalysisHook,
    FaceAnalysisOutcome,
    InvalidStateTransitionError,
    StateTransition,
    Track,
    TrackingUpdate,
)


class CameraStateMachine:
    """Aggregate state for one camera with explicit future face-analysis hooks."""

    _ALLOWED_TRANSITIONS: dict[CameraState, frozenset[CameraState]] = {
        CameraState.PERSON_SCAN: frozenset({CameraState.PERSON_DETECTED}),
        CameraState.PERSON_DETECTED: frozenset({CameraState.TRACKING}),
        CameraState.TRACKING: frozenset(
            {CameraState.PERSON_SCAN, CameraState.PERSON_DETECTED, CameraState.FACE_ANALYSIS}
        ),
        CameraState.FACE_ANALYSIS: frozenset(
            {
                CameraState.KNOWN,
                CameraState.UNKNOWN,
                CameraState.TRACKING,
                CameraState.PERSON_SCAN,
            }
        ),
        CameraState.KNOWN: frozenset(
            {
                CameraState.COOLDOWN,
                CameraState.TRACKING,
                CameraState.PERSON_SCAN,
                CameraState.FACE_ANALYSIS,
            }
        ),
        CameraState.UNKNOWN: frozenset(
            {
                CameraState.COOLDOWN,
                CameraState.TRACKING,
                CameraState.PERSON_SCAN,
                CameraState.FACE_ANALYSIS,
            }
        ),
        CameraState.COOLDOWN: frozenset({CameraState.TRACKING, CameraState.PERSON_SCAN}),
    }

    def __init__(self) -> None:
        self._state = CameraState.PERSON_SCAN
        self._active_tracks: dict[int, Track] = {}
        self._analysis_track_id: int | None = None
        self._analysis_attempts: dict[int, int] = {}
        self._outcomes: dict[int, FaceAnalysisOutcome] = {}
        self._transition_history: list[StateTransition] = []

    @property
    def state(self) -> CameraState:
        return self._state

    @property
    def active_tracks(self) -> tuple[Track, ...]:
        return tuple(self._active_tracks[track_id] for track_id in sorted(self._active_tracks))

    @property
    def analysis_track_id(self) -> int | None:
        return self._analysis_track_id

    @property
    def transition_history(self) -> tuple[StateTransition, ...]:
        return tuple(self._transition_history)

    def reset(self, *, reason: str = "camera session reset") -> None:
        """Drop temporal face state when the camera session is no longer continuous."""

        previous = self._state
        self._state = CameraState.PERSON_SCAN
        self._active_tracks.clear()
        self._analysis_track_id = None
        self._analysis_attempts.clear()
        self._outcomes.clear()
        if previous is not CameraState.PERSON_SCAN:
            self._transition_history.append(
                StateTransition(
                    previous=previous,
                    current=CameraState.PERSON_SCAN,
                    reason=reason,
                )
            )

    def observe(self, update: TrackingUpdate) -> tuple[StateTransition, ...]:
        """Apply tracker output and return only transitions caused by this sample."""

        self._active_tracks = {track.track_id: track for track in update.active_tracks}
        for lost_track in update.lost_tracks:
            self._analysis_attempts.pop(lost_track.track_id, None)
            self._outcomes.pop(lost_track.track_id, None)
            if self._analysis_track_id == lost_track.track_id:
                self._analysis_track_id = None
        transitions_before = len(self._transition_history)

        if self._state in {
            CameraState.FACE_ANALYSIS,
            CameraState.KNOWN,
            CameraState.UNKNOWN,
            CameraState.COOLDOWN,
        }:
            if self._analysis_track_id not in self._active_tracks:
                self._analysis_track_id = None
                self._set_observed_state(
                    reason="analysis track closed",
                    has_active_tracks=bool(self._active_tracks),
                )
            return tuple(self._transition_history[transitions_before:])

        if update.active_tracks:
            if update.new_tracks:
                self._transition(CameraState.PERSON_DETECTED, "new person detection")
                self._transition(CameraState.TRACKING, "track created")
            else:
                self._transition(CameraState.TRACKING, "active person track")
        else:
            self._transition(CameraState.PERSON_SCAN, "no active person tracks")

        return tuple(self._transition_history[transitions_before:])

    def begin_face_analysis(self, track_id: int) -> Track:
        """Select an active track for a future face-analysis operation."""

        track = self._require_active_track(track_id)
        self._require_state(
            (
                CameraState.TRACKING,
                CameraState.FACE_ANALYSIS,
                CameraState.KNOWN,
                CameraState.UNKNOWN,
            ),
            "begin face analysis",
        )
        self._analysis_track_id = track_id
        self._analysis_attempts[track_id] = self._analysis_attempts.get(track_id, 0) + 1
        self._transition(CameraState.FACE_ANALYSIS, f"face analysis requested for track {track_id}")
        return track

    def analyze_current_track(self, hook: FaceAnalysisHook) -> FaceAnalysisOutcome | None:
        """Invoke a future hook; ``None`` leaves the camera in ``FACE_ANALYSIS``."""

        if not callable(getattr(hook, "analyze", None)):
            raise TypeError("face analysis hook must provide callable analyze(track)")
        self._require_state(CameraState.FACE_ANALYSIS, "run face analysis")
        if self._analysis_track_id is None:
            raise InvalidStateTransitionError("face analysis has no selected track")
        track = self._require_active_track(self._analysis_track_id)
        outcome = hook.analyze(track)
        if outcome is None:
            return None
        return self.complete_face_analysis(track.track_id, outcome)

    def complete_face_analysis(
        self,
        track_id: int,
        outcome: FaceAnalysisOutcome | str,
    ) -> FaceAnalysisOutcome:
        """Apply a future face-analysis result without performing recognition here."""

        self._require_state(CameraState.FACE_ANALYSIS, "complete face analysis")
        self._require_selected_track(track_id)
        self._require_active_track(track_id)
        try:
            normalized = FaceAnalysisOutcome(outcome)
        except ValueError as exc:
            raise ValueError(f"unsupported face analysis outcome: {outcome!r}") from exc
        self._outcomes[track_id] = normalized
        self._transition(
            CameraState.KNOWN if normalized is FaceAnalysisOutcome.KNOWN else CameraState.UNKNOWN,
            f"face analysis completed for track {track_id}",
        )
        return normalized

    def apply_confirmed_face_analysis(
        self,
        track_id: int,
        outcome: FaceAnalysisOutcome | str,
        *,
        confirmed: bool,
    ) -> FaceAnalysisOutcome | None:
        """Apply a recognition result only after temporal confirmation.

        Returning ``None`` for an unconfirmed observation makes it impossible
        for a caller to accidentally move the state machine on a single noisy
        frame.
        """

        if not confirmed:
            return None
        return self.complete_face_analysis(track_id, outcome)

    def start_cooldown(self, track_id: int | None = None) -> None:
        """Enter cooldown after a known or unknown face-analysis result."""

        self._require_state(
            (CameraState.KNOWN, CameraState.UNKNOWN),
            "start cooldown",
        )
        selected = self._analysis_track_id if track_id is None else track_id
        if selected is None:
            raise InvalidStateTransitionError("cooldown has no selected track")
        self._require_selected_track(selected)
        self._require_active_track(selected)
        self._transition(CameraState.COOLDOWN, f"cooldown started for track {selected}")

    def finish_cooldown(self) -> None:
        """Return to tracking or scanning after a caller-managed cooldown interval."""

        self._require_state(CameraState.COOLDOWN, "finish cooldown")
        self._analysis_track_id = None
        target = CameraState.TRACKING if self._active_tracks else CameraState.PERSON_SCAN
        self._transition(target, "cooldown finished")

    def analysis_attempts(self, track_id: int) -> int:
        """Return how many face-analysis attempts were requested for a track."""

        return self._analysis_attempts.get(track_id, 0)

    def face_analysis_outcome(self, track_id: int) -> FaceAnalysisOutcome | None:
        """Return the last explicit outcome for a track, if any."""

        return self._outcomes.get(track_id)

    def _set_observed_state(self, *, reason: str, has_active_tracks: bool) -> None:
        target = CameraState.TRACKING if has_active_tracks else CameraState.PERSON_SCAN
        if self._state is target:
            return
        self._transition(target, reason)

    def _transition(self, target: CameraState, reason: str) -> None:
        if target is self._state:
            return
        allowed = self._ALLOWED_TRANSITIONS[self._state]
        if target not in allowed:
            raise InvalidStateTransitionError(
                f"invalid transition {self._state.value} -> {target.value}: {reason}"
            )
        transition = StateTransition(previous=self._state, current=target, reason=reason)
        self._state = target
        self._transition_history.append(transition)

    def _require_state(
        self,
        expected: CameraState | tuple[CameraState, ...],
        action: str,
    ) -> None:
        expected_states = (expected,) if isinstance(expected, CameraState) else expected
        if self._state not in expected_states:
            expected_text = ", ".join(state.value for state in expected_states)
            raise InvalidStateTransitionError(
                f"cannot {action} from {self._state.value}; expected {expected_text}"
            )

    def _require_active_track(self, track_id: int) -> Track:
        try:
            return self._active_tracks[track_id]
        except KeyError as exc:
            raise InvalidStateTransitionError(f"track {track_id} is not active") from exc

    def _require_selected_track(self, track_id: int) -> None:
        if self._analysis_track_id != track_id:
            raise InvalidStateTransitionError(
                f"track {track_id} is not the selected face-analysis track"
            )

"""Small replaceable tracker based on greedy IoU and center-distance matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from collections.abc import Sequence

from app.inference.base import BBox, PersonDetection

from .models import Track, TrackingUpdate, as_detection_sequence, detection_timestamp


class PersonTracker(ABC):
    """Contract for associating consecutive person-detection samples."""

    @abstractmethod
    def update(self, detections: Sequence[PersonDetection]) -> TrackingUpdate:
        """Associate one detection sample and return the lifecycle changes."""

    @property
    @abstractmethod
    def active_tracks(self) -> tuple[Track, ...]:
        """Return currently live tracks in deterministic ID order."""


class IoUGreedyTracker(PersonTracker):
    """One-to-one tracker with IoU matching and a center-distance fallback."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.30,
        max_center_distance_px: float = 100.0,
        max_missed_samples: int = 3,
    ) -> None:
        if not math.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be finite and between 0 and 1")
        if not math.isfinite(max_center_distance_px) or max_center_distance_px <= 0:
            raise ValueError("max_center_distance_px must be finite and greater than zero")
        if isinstance(max_missed_samples, bool) or not isinstance(max_missed_samples, int):
            raise ValueError("max_missed_samples must be an integer")
        if max_missed_samples < 0:
            raise ValueError("max_missed_samples cannot be negative")

        self._iou_threshold = float(iou_threshold)
        self._max_center_distance_px = float(max_center_distance_px)
        self._max_missed_samples = max_missed_samples
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1

    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold

    @property
    def max_center_distance_px(self) -> float:
        return self._max_center_distance_px

    @property
    def max_missed_samples(self) -> int:
        return self._max_missed_samples

    @property
    def active_tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks[track_id] for track_id in sorted(self._tracks))

    def reset(self) -> None:
        """Invalidate every active track after a camera restart or reconnect."""

        self._tracks.clear()

    def update(self, detections: Sequence[PersonDetection]) -> TrackingUpdate:
        current_detections = as_detection_sequence(detections)
        previous_tracks = dict(self._tracks)
        matches = self._associate(previous_tracks, current_detections)
        matched_track_ids = {track_id for track_id, _ in matches}
        matched_detection_indexes = {index for _, index in matches}

        updated_tracks: list[Track] = []
        for track_id, detection_index in matches:
            previous = previous_tracks[track_id]
            detection = current_detections[detection_index]
            track = Track(
                track_id=track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                first_seen_at=previous.first_seen_at,
                last_seen_at=detection_timestamp(detection),
            )
            self._tracks[track_id] = track
            updated_tracks.append(track)

        lost_tracks: list[Track] = []
        for track_id, previous in previous_tracks.items():
            if track_id in matched_track_ids:
                continue
            missed_samples = previous.missed_samples + 1
            if missed_samples <= self._max_missed_samples:
                self._tracks[track_id] = Track(
                    track_id=previous.track_id,
                    bbox=previous.bbox,
                    confidence=previous.confidence,
                    first_seen_at=previous.first_seen_at,
                    last_seen_at=previous.last_seen_at,
                    missed_samples=missed_samples,
                )
            else:
                lost_tracks.append(previous)
                self._tracks.pop(track_id, None)

        new_tracks: list[Track] = []
        for detection_index, detection in enumerate(current_detections):
            if detection_index in matched_detection_indexes:
                continue
            track = Track(
                track_id=self._next_track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                first_seen_at=detection_timestamp(detection),
                last_seen_at=detection_timestamp(detection),
            )
            self._next_track_id += 1
            self._tracks[track.track_id] = track
            new_tracks.append(track)

        return TrackingUpdate(
            active_tracks=self.active_tracks,
            new_tracks=tuple(sorted(new_tracks, key=lambda track: track.track_id)),
            updated_tracks=tuple(sorted(updated_tracks, key=lambda track: track.track_id)),
            lost_tracks=tuple(sorted(lost_tracks, key=lambda track: track.track_id)),
        )

    def _associate(
        self,
        tracks: dict[int, Track],
        detections: tuple[PersonDetection, ...],
    ) -> tuple[tuple[int, int], ...]:
        if not tracks or not detections:
            return ()

        unmatched_track_ids = set(tracks)
        unmatched_detection_indexes = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        iou_candidates: list[tuple[float, float, int, int]] = []
        for track_id, track in tracks.items():
            for detection_index, detection in enumerate(detections):
                overlap = intersection_over_union(track.bbox, detection.bbox)
                if overlap >= self._iou_threshold:
                    distance = center_distance(track.bbox, detection.bbox)
                    iou_candidates.append((overlap, distance, track_id, detection_index))

        for _, _, track_id, detection_index in sorted(
            iou_candidates,
            key=lambda item: (-item[0], item[1], item[2], item[3]),
        ):
            if track_id not in unmatched_track_ids or detection_index not in unmatched_detection_indexes:
                continue
            matches.append((track_id, detection_index))
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indexes.remove(detection_index)

        distance_candidates: list[tuple[float, float, int, int]] = []
        for track_id in unmatched_track_ids:
            for detection_index in unmatched_detection_indexes:
                detection = detections[detection_index]
                distance = center_distance(tracks[track_id].bbox, detection.bbox)
                if distance <= self._max_center_distance_px:
                    overlap = intersection_over_union(tracks[track_id].bbox, detection.bbox)
                    distance_candidates.append((distance, overlap, track_id, detection_index))

        for _, _, track_id, detection_index in sorted(
            distance_candidates,
            key=lambda item: (item[0], -item[1], item[2], item[3]),
        ):
            if track_id not in unmatched_track_ids or detection_index not in unmatched_detection_indexes:
                continue
            matches.append((track_id, detection_index))
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indexes.remove(detection_index)

        return tuple(matches)


def intersection_over_union(first: BBox, second: BBox) -> float:
    """Calculate IoU for two ``(x1, y1, x2, y2)`` boxes."""

    first_area = _area(first)
    second_area = _area(second)
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = _area((left, top, right, bottom))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def center_distance(first: BBox, second: BBox) -> float:
    """Calculate Euclidean distance between two box centers."""

    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    return math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.bounding_box import BoundingBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou(b1: BoundingBox, b2: BoundingBox) -> float:
    xl = max(b1.left, b2.left)
    yt = max(b1.top,  b2.top)
    xr = min(b1.right, b2.right)
    yb = min(b1.bottom, b2.bottom)
    if xr < xl or yb < yt:
        return 0.0
    inter = (xr - xl) * (yb - yt)
    a1 = (b1.right - b1.left) * (b1.bottom - b1.top)
    a2 = (b2.right - b2.left) * (b2.bottom - b2.top)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _extract_crop(img: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1 = max(0, int(bbox.left)),  max(0, int(bbox.top))
    x2, y2 = min(w, int(bbox.right)), min(h, int(bbox.bottom))
    return img[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackObservation:
    frame_idx: int
    bbox: BoundingBox
    img: np.ndarray
    timestamp: float | None = None


@dataclass
class Track:
    track_id: int
    history: list[TrackObservation] = field(default_factory=list)
    # Internal state — valid while the track is active or lost, not meaningful after finalize()
    current_bbox: BoundingBox | None = None
    last_seen: int = -1


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class SingleCameraTracker(Protocol):
    """Interface that every single-camera tracker must satisfy."""

    def update(
        self,
        detections: list[BoundingBox],
        frame_img: np.ndarray,
        frame_idx: int,
        timestamp: float | None = None,
    ) -> None:
        """Process one frame: match detections to existing tracks and create new ones."""
        ...

    def finalize(self) -> list[Track]:
        """Flush all active/lost tracks and return the complete list of tracks."""
        ...


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _BaseTracker:
    def __init__(self, max_age: int = 5, iou_threshold: float = 0.45) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._active: dict[int, Track] = {}
        self._lost:   dict[int, Track] = {}
        self._finished: list[Track] = []

    # -- helpers --

    def _observe(
        self,
        bbox: BoundingBox,
        frame_img: np.ndarray,
        frame_idx: int,
        timestamp: float | None,
    ) -> TrackObservation:
        return TrackObservation(
            frame_idx=frame_idx,
            bbox=bbox,
            img=_extract_crop(frame_img, bbox),
            timestamp=timestamp,
        )

    def _attach(
        self,
        track: Track,
        det: BoundingBox,
        frame_img: np.ndarray,
        frame_idx: int,
        timestamp: float | None,
    ) -> None:
        """Append an observation to an existing track and update its state."""
        track.history.append(self._observe(det, frame_img, frame_idx, timestamp))
        track.current_bbox = det
        track.last_seen = frame_idx

    def _spawn(
        self,
        det: BoundingBox,
        frame_img: np.ndarray,
        frame_idx: int,
        timestamp: float | None,
    ) -> Track:
        obs = self._observe(det, frame_img, frame_idx, timestamp)
        track = Track(track_id=self._next_id, history=[obs], current_bbox=det, last_seen=frame_idx)
        self._next_id += 1
        return track

    def _expire_lost(self, frame_idx: int) -> None:
        expired = [tid for tid, t in self._lost.items() if frame_idx - t.last_seen > self.max_age]
        for tid in expired:
            self._finished.append(self._lost.pop(tid))

    def finalize(self) -> list[Track]:
        for track in (*self._active.values(), *self._lost.values()):
            self._finished.append(track)
        self._active.clear()
        self._lost.clear()
        return self._finished


# ---------------------------------------------------------------------------
# Max-Overlap (greedy) tracker
# ---------------------------------------------------------------------------

class MaxOverlapTracker(_BaseTracker):
    """Greedy IoU matching: each track claims its best unmatched detection."""

    def update(
        self,
        detections: list[BoundingBox],
        frame_img: np.ndarray,
        frame_idx: int,
        timestamp: float | None = None,
    ) -> None:
        unmatched = list(detections)
        new_active: dict[int, Track] = {}

        # 1. Match active tracks greedily
        for tid, track in self._active.items():
            best_iou, best_idx = 0.0, -1
            for i, det in enumerate(unmatched):
                iou = _iou(track.current_bbox, det)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_iou > self.iou_threshold:
                det = unmatched.pop(best_idx)
                self._attach(track, det, frame_img, frame_idx, timestamp)
                new_active[tid] = track
            else:
                self._lost[tid] = track

        # 2. Try to recover lost tracks with remaining detections
        self._expire_lost(frame_idx)
        still_unmatched: list[BoundingBox] = []
        for det in unmatched:
            best_iou, best_tid = 0.0, -1
            for tid, track in self._lost.items():
                iou = _iou(track.current_bbox, det)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_iou > self.iou_threshold:
                track = self._lost.pop(best_tid)
                self._attach(track, det, frame_img, frame_idx, timestamp)
                new_active[best_tid] = track
            else:
                still_unmatched.append(det)

        # 3. Spawn new tracks for genuinely unmatched detections
        for det in still_unmatched:
            track = self._spawn(det, frame_img, frame_idx, timestamp)
            new_active[track.track_id] = track

        self._active = new_active


# ---------------------------------------------------------------------------
# SORT tracker (Hungarian algorithm)
# ---------------------------------------------------------------------------

class SORTTracker(_BaseTracker):
    """Optimal IoU matching via the Hungarian algorithm (SORT-style)."""

    def update(
        self,
        detections: list[BoundingBox],
        frame_img: np.ndarray,
        frame_idx: int,
        timestamp: float | None = None,
    ) -> None:
        self._expire_lost(frame_idx)

        # Build the combined pool: active + eligible lost tracks
        pool_ids:    list[tuple[int, str]] = []
        pool_bboxes: list[BoundingBox]     = []
        for tid, track in self._active.items():
            pool_ids.append((tid, 'active'))
            pool_bboxes.append(track.current_bbox)
        for tid, track in self._lost.items():
            pool_ids.append((tid, 'lost'))
            pool_bboxes.append(track.current_bbox)

        new_active: dict[int, Track] = {}
        unmatched_det_indices = set(range(len(detections)))

        if pool_bboxes and detections:
            iou_matrix = np.array(
                [[_iou(tb, db) for db in detections] for tb in pool_bboxes],
                dtype=np.float32,
            )
            row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] <= self.iou_threshold:
                    continue
                unmatched_det_indices.discard(c)
                tid, status = pool_ids[r]
                track = self._active.pop(tid) if status == 'active' else self._lost.pop(tid)
                self._attach(track, detections[c], frame_img, frame_idx, timestamp)
                new_active[tid] = track

        # Remaining active tracks become lost
        for tid, track in self._active.items():
            self._lost[tid] = track

        # Spawn new tracks for unmatched detections
        for i in unmatched_det_indices:
            track = self._spawn(detections[i], frame_img, frame_idx, timestamp)
            new_active[track.track_id] = track

        self._active = new_active

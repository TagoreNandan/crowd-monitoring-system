"""Lightweight, IoU-based multi-person tracker.

Provides object persistence (stable track IDs and center-point trajectories)
over continuous video frames for detected person bounding boxes (`PersonDetection`).

This module is self-contained and optional: it wraps `PersonDetection` objects
without altering dense DM-Count crowd estimation or detector availability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .detection import PersonDetection


@dataclass
class TrackedPerson:
    """A tracked person box with persistent track_id and trajectory history."""

    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    center_xy: Tuple[int, int]
    history: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def box_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = float(box1_area + box2_area - inter_area)
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class PersonTracker:
    """Tracks detected persons across consecutive frames using IoU matching.

    Parameters
    ----------
    max_disappeared : int
        Maximum number of consecutive frames a track can be missing before being deleted.
    min_iou : float
        Minimum IoU threshold required to associate a detection with an existing track.
    max_history : int
        Maximum number of past centroid locations to retain per track for trajectory display.
    """

    def __init__(self, max_disappeared: int = 10, min_iou: float = 0.3, max_history: int = 20):
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        self.max_history = max_history

        self._next_id = 1
        self._tracks: dict[int, TrackedPerson] = {}
        self._disappeared: dict[int, int] = {}

    def reset(self) -> None:
        """Reset all active tracks and reset ID sequence."""
        self._next_id = 1
        self._tracks.clear()
        self._disappeared.clear()

    def update(self, detections: List[PersonDetection]) -> List[TrackedPerson]:
        """Update tracker with new detections for the current frame.

        Returns list of active `TrackedPerson` instances.
        """
        if not detections:
            # Mark all existing tracks as disappeared
            to_delete = []
            for track_id in self._tracks:
                self._disappeared[track_id] += 1
                if self._disappeared[track_id] > self.max_disappeared:
                    to_delete.append(track_id)
            for track_id in to_delete:
                del self._tracks[track_id]
                del self._disappeared[track_id]
            return list(self._tracks.values())

        det_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
        det_centers = [((d.x1 + d.x2) // 2, (d.y1 + d.y2) // 2) for d in detections]

        if not self._tracks:
            # Initialize tracks for all detections
            for det, box, center in zip(detections, det_boxes, det_centers):
                track_id = self._next_id
                self._next_id += 1
                tp = TrackedPerson(
                    track_id=track_id,
                    x1=det.x1,
                    y1=det.y1,
                    x2=det.x2,
                    y2=det.y2,
                    confidence=det.confidence,
                    center_xy=center,
                    history=[center],
                )
                self._tracks[track_id] = tp
                self._disappeared[track_id] = 0
            return list(self._tracks.values())

        # Match existing tracks with new detections using greedy IoU matching
        track_ids = list(self._tracks.keys())
        track_boxes = [self._tracks[tid].box_xyxy for tid in track_ids]

        iou_matrix = []
        for t_box in track_boxes:
            row = [compute_iou(t_box, d_box) for d_box in det_boxes]
            iou_matrix.append(row)

        matched_tracks = set()
        matched_dets = set()

        # Sort all potential (track_idx, det_idx) pairs by descending IoU score
        candidates = []
        for t_idx in range(len(track_ids)):
            for d_idx in range(len(detections)):
                score = iou_matrix[t_idx][d_idx]
                if score >= self.min_iou:
                    candidates.append((score, t_idx, d_idx))
        candidates.sort(key=lambda x: x[0], reverse=True)

        for score, t_idx, d_idx in candidates:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue

            tid = track_ids[t_idx]
            det = detections[d_idx]
            center = det_centers[d_idx]

            tp = self._tracks[tid]
            tp.x1, tp.y1, tp.x2, tp.y2 = det.x1, det.y1, det.x2, det.y2
            tp.confidence = det.confidence
            tp.center_xy = center
            tp.history.append(center)
            if len(tp.history) > self.max_history:
                tp.history.pop(0)

            self._disappeared[tid] = 0
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        # Handle unmatched existing tracks
        to_delete = []
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_tracks:
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self.max_disappeared:
                    to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]
            del self._disappeared[tid]

        # Handle unmatched new detections (create new tracks)
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                track_id = self._next_id
                self._next_id += 1
                center = det_centers[d_idx]
                tp = TrackedPerson(
                    track_id=track_id,
                    x1=det.x1,
                    y1=det.y1,
                    x2=det.x2,
                    y2=det.y2,
                    confidence=det.confidence,
                    center_xy=center,
                    history=[center],
                )
                self._tracks[track_id] = tp
                self._disappeared[track_id] = 0

        return list(self._tracks.values())

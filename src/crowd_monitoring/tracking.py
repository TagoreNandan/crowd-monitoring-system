"""Lightweight, IoU-based multi-person tracker.

Provides object persistence (stable track IDs and center-point trajectories)
over continuous video frames for detected person bounding boxes (`PersonDetection`).

This module is self-contained and optional: it wraps `PersonDetection` objects
without altering dense DM-Count crowd estimation or detector availability.

Ego-motion compensation: on a panning/drifting drone every person appears to
move by the camera's own motion, which pushes frame-to-frame IoU below
threshold and churns track IDs. `EgoMotionEstimator` measures that motion
from the *background* only (person boxes masked out) as a 2x3 partial-affine
matrix (translation + rotation + uniform scale), and `PersonTracker.update`
warps existing tracks by it before IoU matching, so matching only has to
absorb people's own movement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detection import PersonDetection

logger = logging.getLogger(__name__)


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


class EgoMotionEstimator:
    """Estimates camera (drone) motion between consecutive frames.

    Each call: Shi-Tomasi corners on the current frame with person boxes
    masked out -> pyramidal Lucas-Kanade flow back into the previous frame ->
    RANSAC partial-affine fit. The result is the prev -> current transform in
    full-resolution pixel coordinates, or None when it can't be trusted (first
    frame, featureless sky/water, too few inliers, or an implausible jump such
    as a failover switch to a different video). None means "assume static",
    i.e. exactly the tracker's uncompensated behaviour.

    Work is done on a grayscale copy downscaled to `max_side` (a few ms per
    frame on CPU); only the translation needs rescaling back to full res.
    """

    def __init__(
        self,
        max_side: int = 640,
        max_corners: int = 300,
        quality_level: float = 0.01,
        min_distance: int = 8,
        min_inliers: int = 15,
        min_inlier_ratio: float = 0.5,
        max_fb_error_px: float = 1.0,
        box_padding_px: int = 4,
        max_translation_frac: float = 0.25,
        max_scale_change: float = 0.2,
    ):
        self.max_side = max_side
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_fb_error_px = max_fb_error_px
        self.box_padding_px = box_padding_px
        self.max_translation_frac = max_translation_frac
        self.max_scale_change = max_scale_change
        self._prev_gray: Optional[np.ndarray] = None
        self.last_inliers = 0

    def reset(self) -> None:
        self._prev_gray = None

    def estimate(
        self, frame_bgr: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]] = ()
    ) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / float(max(h, w)))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        if scale < 1.0:
            gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

        prev, self._prev_gray = self._prev_gray, gray
        self.last_inliers = 0
        if prev is None or prev.shape != gray.shape:
            return None

        gh, gw = gray.shape[:2]
        mask = np.full((gh, gw), 255, dtype=np.uint8)
        pad = self.box_padding_px
        for x1, y1, x2, y2 in boxes:
            mx1, my1 = max(0, int(x1 * scale) - pad), max(0, int(y1 * scale) - pad)
            mx2, my2 = min(gw, int(x2 * scale) + pad), min(gh, int(y2 * scale) + pad)
            if mx2 > mx1 and my2 > my1:
                mask[my1:my2, mx1:mx2] = 0

        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=self.quality_level,
            minDistance=self.min_distance, mask=mask,
        )
        if pts is None or len(pts) < self.min_inliers:
            return None

        lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        prev_pts, status, _err = cv2.calcOpticalFlowPyrLK(gray, prev, pts, None, **lk)
        if prev_pts is None or status is None:
            return None
        # Forward-backward check: flow each match back into the current frame
        # and keep only points that return to where they started. Between
        # unrelated frames (a failover cut, a scene change) LK still "finds"
        # matches, but they don't survive the round trip.
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, prev_pts, None, **lk)
        if back_pts is None or back_status is None:
            return None
        fb_err = np.linalg.norm((pts - back_pts).reshape(-1, 2), axis=1)
        ok = (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1) & (fb_err < self.max_fb_error_px)
        n_tracked = int(ok.sum())
        if n_tracked < self.min_inliers:
            return None

        # Fit current -> previous (the direction the flow was measured in),
        # then invert to get the prev -> current warp for existing tracks.
        m_curr_to_prev, inliers = cv2.estimateAffinePartial2D(
            pts[ok], prev_pts[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if m_curr_to_prev is None or inliers is None:
            return None
        self.last_inliers = int(inliers.sum())
        if self.last_inliers < max(self.min_inliers, self.min_inlier_ratio * n_tracked):
            return None

        m = cv2.invertAffineTransform(m_curr_to_prev)
        sim_scale = float(np.hypot(m[0, 0], m[1, 0]))
        if abs(sim_scale - 1.0) > self.max_scale_change:
            return None
        if abs(m[0, 2]) > self.max_translation_frac * gw or abs(m[1, 2]) > self.max_translation_frac * gh:
            return None
        m = m.astype(np.float64)
        m[:, 2] /= scale  # downscaled -> full-resolution translation
        return m


def _warp_points(points: Sequence[Tuple[float, float]], motion: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.transform(arr, motion).reshape(-1, 2)


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

    def apply_motion(self, motion: np.ndarray) -> None:
        """Warp every existing track (box, centre, trajectory) by a 2x3
        prev -> current camera-motion matrix so it lines up with the new frame."""
        for tp in self._tracks.values():
            corners = _warp_points(
                [(tp.x1, tp.y1), (tp.x2, tp.y1), (tp.x1, tp.y2), (tp.x2, tp.y2)], motion
            )
            tp.x1, tp.y1 = (int(round(v)) for v in corners.min(axis=0))
            tp.x2, tp.y2 = (int(round(v)) for v in corners.max(axis=0))
            cx, cy = _warp_points([tp.center_xy], motion)[0]
            tp.center_xy = (int(round(cx)), int(round(cy)))
            if tp.history:
                tp.history = [(int(round(x)), int(round(y))) for x, y in _warp_points(tp.history, motion)]

    def update(
        self, detections: List[PersonDetection], motion: Optional[np.ndarray] = None
    ) -> List[TrackedPerson]:
        """Update tracker with new detections for the current frame.

        `motion` is an optional 2x3 prev -> current camera-motion matrix (see
        `EgoMotionEstimator`); tracks are warped by it before IoU matching.

        Returns list of active `TrackedPerson` instances.
        """
        if motion is not None and self._tracks:
            self.apply_motion(motion)

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

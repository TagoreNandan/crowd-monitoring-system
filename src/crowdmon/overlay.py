"""Professional on-frame overlay: person bounding boxes + a compact stats
panel, drawn on a copy of the frame.

Purely a rendering helper -- it takes already-computed values (a `FrameResult`
`frame_bgr` and `PersonDetection`s, plus small bits of caller-side bookkeeping
like elapsed time/frame index that the video job runner already tracks) and
draws on a copy of the frame. It knows nothing about jobs, WebSockets, or the
video loop that calls it, and it never mutates its input.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

import cv2
import numpy as np

from .detection import PersonDetection

_BOX_COLOR = (0, 220, 0)
_TEXT_FG = (255, 255, 255)
_TEXT_BG = (0, 0, 0)


def draw_person_boxes(frame_bgr: np.ndarray, detections: Iterable[Any]) -> np.ndarray:
    """Draw one rectangle + confidence/track label per real detection or tracked person.
    Returns a new array -- `frame_bgr` itself is never modified."""
    out = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, 2)

        if hasattr(det, "track_id"):
            label = f"ID #{det.track_id} ({det.confidence * 100:.0f}%)"
            if hasattr(det, "history") and len(det.history) > 1:
                pts = np.array(det.history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
        else:
            label = f"{det.confidence * 100:.0f}%"

        label_y = y1 - 6 if y1 - 6 > 10 else y1 + 14
        cv2.putText(out, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _BOX_COLOR, 1, cv2.LINE_AA)
    return out


def _put_outlined_line(frame_bgr: np.ndarray, text: str, org: tuple, scale: float = 0.55) -> None:
    """Black outline + white fill, the same legible-on-any-background
    technique already used by `scripts/run_video.py`'s CLI overlay."""
    cv2.putText(frame_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _TEXT_BG, 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _TEXT_FG, 1, cv2.LINE_AA)


def draw_stats_panel(
    frame_bgr: np.ndarray,
    *,
    frame_number: int,
    total_frames: Optional[int],
    elapsed_s: float,
    visible_count: Optional[int],
    estimated_count: float,
    peak_count: float,
    fps: Optional[float],
    latency_ms: float,
    density: float,
    cluster_count: int,
    largest_cluster: int,
    risk_status: str,
    risk_score: float,
) -> np.ndarray:
    """Draw the fixed top-left stats panel. Returns a new array; `frame_bgr`
    is never mutated. `visible_count`/`fps` are `None` when the person
    detector (visible_count) or the frame-rate estimate (fps, first frame)
    isn't available yet -- rendered as "n/a" rather than a fake 0."""
    out = frame_bgr.copy()

    total_frames_text = str(total_frames) if total_frames is not None else "?"
    visible_text = str(visible_count) if visible_count is not None else "n/a"
    fps_text = f"{fps:.1f}" if fps is not None else "n/a"

    lines = [
        f"Time: {time.strftime('%H:%M:%S')}  Elapsed: {elapsed_s:.1f}s",
        f"Frame: {frame_number}/{total_frames_text}",
        f"Visible: {visible_text}   Estimated: {estimated_count:.0f}   Max Estimated: {peak_count:.0f}",
        f"FPS: {fps_text}   Latency: {latency_ms:.0f} ms",
        f"Density: {density:.2f}   Clusters: {cluster_count}   Largest: {largest_cluster}",
        f"Risk: {risk_status} ({risk_score:.0f})",
    ]

    y = 24
    for line in lines:
        _put_outlined_line(out, line, (8, y))
        y += 22
    return out


def render_video_overlay(
    frame_bgr: np.ndarray,
    detections: Iterable[PersonDetection],
    **stats_kwargs,
) -> np.ndarray:
    """Boxes first, then the stats panel on top -- the one annotated frame
    the video job runner computes per processed frame. Callers must write
    (or discard) this frame immediately rather than collecting it into a
    list: nothing here buffers frames across calls."""
    annotated = draw_person_boxes(frame_bgr, detections)
    return draw_stats_panel(annotated, **stats_kwargs)

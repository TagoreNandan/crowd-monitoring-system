"""Professional on-frame overlay: person bounding boxes, density heatmaps, and a compact stats
panel, drawn on a copy of the frame.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

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
    h, w = frame_bgr.shape[:2]
    ref_dim = min(h, w)
    factor = max(0.5, ref_dim / 720.0)

    box_thickness = max(1, int(round(2 * factor)))
    poly_thickness = max(1, int(round(2 * factor)))
    font_scale = max(0.35, 0.45 * factor)
    label_thickness = max(1, int(round(1 * factor)))

    for det in detections:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, box_thickness)

        track_id = getattr(det, "track_id", None)
        if track_id is not None:
            label = f"ID #{track_id} ({det.confidence * 100:.0f}%)"
            if hasattr(det, "history") and len(det.history) > 1:
                pts = np.array(det.history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], isClosed=False, color=(0, 255, 255), thickness=poly_thickness)
        else:
            label = f"{det.confidence * 100:.0f}%"

        offset_above = max(4, int(round(6 * factor)))
        offset_below = max(12, int(round(14 * factor)))
        min_top = max(8, int(round(10 * factor)))
        label_y = y1 - offset_above if y1 - offset_above > min_top else y1 + offset_below
        cv2.putText(out, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, _BOX_COLOR, label_thickness, cv2.LINE_AA)
    return out


def draw_density_heatmap_overlay(
    frame_bgr: np.ndarray,
    density_map: Optional[np.ndarray],
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend a color-mapped density heatmap overlay onto frame_bgr."""
    if density_map is None or density_map.size == 0:
        return frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    d_min, d_max = float(density_map.min()), float(density_map.max())
    if d_max > d_min:
        norm_map = (density_map - d_min) / (d_max - d_min)
    else:
        norm_map = np.zeros_like(density_map)
    heatmap_u8 = (norm_map * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    if heatmap_color.shape[:2] != (h, w):
        heatmap_color = cv2.resize(heatmap_color, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap_color, alpha, 0)


def _put_outlined_line(
    frame_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.55,
    outline_thickness: int = 3,
    text_thickness: int = 1,
) -> None:
    """Black outline + white fill for high-legibility stats overlays."""
    cv2.putText(frame_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _TEXT_BG, outline_thickness, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _TEXT_FG, text_thickness, cv2.LINE_AA)


def draw_stats_panel(
    frame_bgr: np.ndarray,
    *,
    frame_number: int,
    total_frames: Optional[int],
    video_time_s: float,
    total_duration_s: Optional[float] = None,
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
    """Draw the fixed top-left stats panel.

    `video_time_s` is the VIDEO's own playback position (frame_number /
    source_fps) -- NOT wall-clock processing time. Every caller must pass
    the exact same value here as it stores/broadcasts as
    `VideoFrameMetrics.video_time_s`, so the burned-in overlay and the
    UI/API can never show a different "Time" for the same frame (a real bug
    this parameter name/contract exists specifically to prevent).
    """
    out = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    ref_dim = min(h, w)
    factor = max(0.5, ref_dim / 720.0)

    font_scale = max(0.40, 0.55 * factor)
    outline_thickness = max(2, int(round(3 * factor)))
    text_thickness = max(1, int(round(1.5 * factor)))

    start_x = max(8, int(round(12 * factor)))
    start_y = max(20, int(round(28 * factor)))
    y_step = max(18, int(round(26 * factor)))

    total_frames_text = str(total_frames) if total_frames is not None else "?"
    total_duration_text = f"{total_duration_s:.1f}s" if total_duration_s is not None else "?"
    visible_text = str(visible_count) if visible_count is not None else "n/a"
    fps_text = f"{fps:.1f}" if fps is not None else "n/a"

    lines = [
        f"Time: {video_time_s:.1f}s / {total_duration_text}",
        f"Frame: {frame_number} / {total_frames_text}",
        f"Visible People: {visible_text}    Estimated Crowd (periodic): {estimated_count:.0f}",
        f"Peak Estimated: {peak_count:.0f}    FPS: {fps_text}    Latency: {latency_ms:.0f}ms",
        f"Density: {density:.2f}    Clusters: {cluster_count} (largest {largest_cluster})",
        f"Risk: {risk_status} ({risk_score:.0f}/100)",
    ]

    y = start_y
    for line in lines:
        _put_outlined_line(
            out,
            line,
            (start_x, y),
            scale=font_scale,
            outline_thickness=outline_thickness,
            text_thickness=text_thickness,
        )
        y += y_step
    return out


def render_video_overlay(
    frame_bgr: np.ndarray,
    detections: Iterable[PersonDetection],
    **stats_kwargs,
) -> np.ndarray:
    """Boxes first, then the stats panel on top."""
    annotated = draw_person_boxes(frame_bgr, detections)
    return draw_stats_panel(annotated, **stats_kwargs)

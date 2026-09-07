"""Unit tests for `crowdmon.overlay`: box drawing and the stats panel, the
"professional frame overlay" the video job runner composes per frame. Pure
pixel-drawing functions -- no model, no video file, no job machinery.
"""

from __future__ import annotations

import numpy as np

from crowdmon.detection import PersonDetection
from crowdmon.overlay import draw_person_boxes, draw_stats_panel, render_video_overlay

_STATS_KWARGS = dict(
    frame_number=5,
    total_frames=100,
    elapsed_s=1.23,
    visible_count=3,
    estimated_count=12.5,
    peak_count=20.0,
    fps=24.0,
    latency_ms=41.0,
    density=0.85,
    cluster_count=2,
    largest_cluster=6,
    risk_status="CAUTION",
    risk_score=35.0,
)


def _blank_frame(h=120, w=160) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_person_boxes_returns_new_array_and_does_not_mutate_input():
    frame = _blank_frame()
    original = frame.copy()
    detections = [PersonDetection(x1=10, y1=10, x2=50, y2=90, confidence=0.87)]

    out = draw_person_boxes(frame, detections)

    assert out.shape == frame.shape
    assert np.array_equal(frame, original)  # input untouched
    assert not np.array_equal(out, frame)  # something was actually drawn


def test_draw_person_boxes_with_no_detections_leaves_frame_unchanged():
    frame = _blank_frame()

    out = draw_person_boxes(frame, [])

    assert np.array_equal(out, frame)


def test_draw_stats_panel_renders_without_error_and_does_not_mutate_input():
    frame = _blank_frame()
    original = frame.copy()

    out = draw_stats_panel(frame, **_STATS_KWARGS)

    assert out.shape == frame.shape
    assert np.array_equal(frame, original)
    assert not np.array_equal(out, frame)


def test_draw_stats_panel_handles_unavailable_visible_count_and_fps():
    frame = _blank_frame()
    kwargs = dict(_STATS_KWARGS, visible_count=None, fps=None, total_frames=None)

    out = draw_stats_panel(frame, **kwargs)  # must not raise

    assert out.shape == frame.shape


def test_render_video_overlay_draws_both_boxes_and_panel():
    frame = _blank_frame()
    detections = [PersonDetection(x1=5, y1=5, x2=40, y2=60, confidence=0.7)]

    boxes_only = draw_person_boxes(frame, detections)
    panel_only = draw_stats_panel(frame, **_STATS_KWARGS)
    combined = render_video_overlay(frame, detections, **_STATS_KWARGS)

    assert combined.shape == frame.shape
    # The combined frame differs from either single-layer rendering alone --
    # both the boxes and the panel are actually present, not just one.
    assert not np.array_equal(combined, boxes_only)
    assert not np.array_equal(combined, panel_only)

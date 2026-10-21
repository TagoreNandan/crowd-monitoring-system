"""Ego-motion compensation: background-only camera-motion estimate + tracker warp."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from crowd_monitoring.detection import PersonDetection
from crowd_monitoring.tracking import EgoMotionEstimator, PersonTracker

W, H = 960, 540


def _scene(seed: int = 3) -> np.ndarray:
    """Large textured 'ground' we can crop at different offsets to fake a pan."""
    rng = np.random.default_rng(seed)
    base = (rng.random((H + 400, W + 400)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def _view(scene: np.ndarray, ox: int, oy: int) -> np.ndarray:
    return scene[200 + oy : 200 + oy + H, 200 + ox : 200 + ox + W].copy()


def test_estimates_pure_pan_translation_in_full_res_pixels():
    scene = _scene()
    est = EgoMotionEstimator()
    assert est.estimate(_view(scene, 0, 0)) is None  # first frame: nothing to compare
    # Camera moves right/down by (30, 12) => content shifts by (-30, -12).
    m = est.estimate(_view(scene, 30, 12))
    assert m is not None and m.shape == (2, 3)
    assert m[0, 2] == pytest.approx(-30, abs=1.0)
    assert m[1, 2] == pytest.approx(-12, abs=1.0)
    assert np.allclose(m[:, :2], np.eye(2), atol=0.02)


def test_estimates_small_rotation():
    scene = _scene()
    frame = _view(scene, 0, 0)
    rot = cv2.getRotationMatrix2D((W / 2, H / 2), 2.0, 1.0)
    rotated = cv2.warpAffine(frame, rot, (W, H), borderMode=cv2.BORDER_REFLECT)
    est = EgoMotionEstimator()
    est.estimate(frame)
    m = est.estimate(rotated)
    assert m is not None
    assert np.allclose(m, rot, atol=0.03 * np.abs(rot).max() + 1.0)


def test_person_motion_is_masked_out_of_the_estimate():
    """A large 'person' moving against a static background must not be read as camera motion."""
    scene = _scene()
    f0, f1 = _view(scene, 0, 0), _view(scene, 0, 0)
    blob = (np.random.default_rng(9).random((200, 120, 3)) * 255).astype(np.uint8)
    f0[100:300, 100:220] = blob
    f1[100:300, 160:280] = blob  # moved 60px right
    est = EgoMotionEstimator()
    est.estimate(f0, [(100, 100, 220, 300)])
    m = est.estimate(f1, [(160, 100, 280, 300)])
    assert m is not None
    assert abs(m[0, 2]) < 1.0 and abs(m[1, 2]) < 1.0


def test_featureless_frames_return_none():
    flat = np.full((H, W, 3), 128, np.uint8)
    est = EgoMotionEstimator()
    est.estimate(flat)
    assert est.estimate(flat) is None


def test_implausible_jump_rejected():
    est = EgoMotionEstimator()
    est.estimate(_view(_scene(1), 0, 0))
    assert est.estimate(_view(_scene(2), 0, 0)) is None  # e.g. failover cut to another video


def test_tracker_keeps_id_through_fast_pan_only_with_compensation():
    # 60px-wide person, camera pans 40px/frame: raw IoU ~0.2 < min_iou 0.3.
    pan = np.array([[1.0, 0.0, -40.0], [0.0, 1.0, 0.0]])
    frames = [PersonDetection(x1=400 - 40 * i, y1=200, x2=460 - 40 * i, y2=320, confidence=0.9) for i in range(4)]

    plain = PersonTracker()
    ids_plain = {plain.update([d])[-1].track_id for d in frames}
    assert len(ids_plain) > 1  # uncompensated tracker churns IDs

    comp = PersonTracker()
    comp.update([frames[0]])
    for d in frames[1:]:
        tracks = comp.update([d], motion=pan)
    assert [t.track_id for t in tracks] == [1]
    # trajectory history is expressed in current-frame coordinates
    assert tracks[0].history[-1] == tracks[0].center_xy
    assert len({x for x, _ in tracks[0].history}) == 1


def test_apply_motion_warps_coasting_tracks():
    tracker = PersonTracker()
    tracker.update([PersonDetection(x1=100, y1=100, x2=160, y2=220, confidence=0.9)])
    (t,) = tracker.update([], motion=np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -10.0]]))
    assert t.box_xyxy == (125, 90, 185, 210)
    assert t.center_xy == (155, 150)

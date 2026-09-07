import pytest

from crowdmon.detection import PersonDetection
from crowdmon.tracking import PersonTracker, TrackedPerson, compute_iou


def test_compute_iou():
    box1 = (0, 0, 10, 10)
    box2 = (0, 0, 10, 10)
    assert compute_iou(box1, box2) == 1.0

    box3 = (20, 20, 30, 30)
    assert compute_iou(box1, box3) == 0.0

    box4 = (5, 0, 15, 10)
    assert abs(compute_iou(box1, box4) - 0.3333) < 0.01


def test_tracker_assigns_track_ids_and_updates_history():
    tracker = PersonTracker(max_disappeared=3, min_iou=0.3)

    det1 = [PersonDetection(x1=10, y1=10, x2=50, y2=50, confidence=0.9)]
    tracks1 = tracker.update(det1)

    assert len(tracks1) == 1
    assert tracks1[0].track_id == 1
    assert tracks1[0].center_xy == (30, 30)
    assert tracks1[0].history == [(30, 30)]

    # Frame 2: slightly moved box
    det2 = [PersonDetection(x1=12, y1=12, x2=52, y2=52, confidence=0.85)]
    tracks2 = tracker.update(det2)

    assert len(tracks2) == 1
    assert tracks2[0].track_id == 1  # persistent ID
    assert tracks2[0].center_xy == (32, 32)
    assert tracks2[0].history == [(30, 30), (32, 32)]


def test_tracker_removes_disappeared_tracks():
    tracker = PersonTracker(max_disappeared=2, min_iou=0.3)

    det1 = [PersonDetection(x1=10, y1=10, x2=50, y2=50, confidence=0.9)]
    tracker.update(det1)

    # 2 frames with no detections
    tracker.update([])
    tracker.update([])

    # 3rd empty frame -> track deleted
    tracks = tracker.update([])
    assert len(tracks) == 0


def test_tracker_reset():
    tracker = PersonTracker()
    tracker.update([PersonDetection(x1=10, y1=10, x2=50, y2=50, confidence=0.9)])
    tracker.reset()

    tracks = tracker.update([PersonDetection(x1=10, y1=10, x2=50, y2=50, confidence=0.9)])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1

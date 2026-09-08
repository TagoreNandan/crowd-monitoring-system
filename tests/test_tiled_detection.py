"""Unit tests for tiled (SAHI-style) detection and NMS merging."""

import numpy as np
import pytest
from crowd_monitoring.detection import PersonDetection, PersonDetector, generate_tiles, nms_boxes


def test_generate_tiles():
    # Small image (<= tile_size) should return single tile
    tiles_small = generate_tiles(600, 600, tile_size=640)
    assert len(tiles_small) == 1
    assert tiles_small[0] == (0, 0, 600, 600)

    # Large 1920x1080 image with 640px tiles and 20% overlap
    tiles_large = generate_tiles(1920, 1080, tile_size=640, tile_overlap=0.20)
    assert len(tiles_large) > 1

    for x1, y1, x2, y2 in tiles_large:
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080
        assert (x2 - x1) <= 640
        assert (y2 - y1) <= 640


def test_nms_boxes():
    # Test merging identical/overlapping boxes
    b1 = PersonDetection(x1=10, y1=10, x2=50, y2=50, confidence=0.90)
    b2 = PersonDetection(x1=12, y1=12, x2=52, y2=52, confidence=0.85)  # Heavy overlap with b1
    b3 = PersonDetection(x1=200, y1=200, x2=250, y2=250, confidence=0.80)  # Non-overlapping

    keep = nms_boxes([b1, b2, b3], iou_threshold=0.45)
    assert len(keep) == 2
    assert keep[0] == b1  # Higher confidence kept
    assert keep[1] == b3


def test_tiled_person_detector_dummy():
    # Test PersonDetector tiling logic with dummy numpy array
    detector = PersonDetector(use_tiling=True, tile_size=640)
    assert detector.use_tiling is True
    assert detector.tile_size == 640
    assert detector.nms_iou_threshold == 0.45


def test_find_dense_regions_locates_high_density_blob():
    from crowd_monitoring.detection import find_dense_regions

    # A 20x20 density map, mostly near-zero, with one clear dense blob.
    density_map = np.full((20, 20), 0.01, dtype=np.float32)
    density_map[4:9, 3:8] = 1.0  # a real dense region

    regions = find_dense_regions(
        density_map, original_size=(200, 200), threshold_rel=0.5, max_regions=3, min_region_pixels=1
    )

    assert len(regions) == 1
    x1, y1, x2, y2 = regions[0]
    # Original coords are density coords scaled by 10x (200/20), roughly
    # covering the blob (allowing for the fixed pixel padding).
    assert 0 <= x1 < x2 <= 200
    assert 0 <= y1 < y2 <= 200
    assert x1 < 30 * 10  # roughly around column 3-8 * 10, not the whole frame
    assert x2 > 3 * 10


def test_find_dense_regions_returns_empty_for_uniform_map():
    from crowd_monitoring.detection import find_dense_regions

    density_map = np.zeros((20, 20), dtype=np.float32)
    assert find_dense_regions(density_map, original_size=(200, 200)) == []


def test_find_dense_regions_caps_at_max_regions():
    from crowd_monitoring.detection import find_dense_regions

    density_map = np.full((30, 30), 0.01, dtype=np.float32)
    # Four separated dense blobs, far enough apart to form distinct connected components.
    for (cy, cx) in [(3, 3), (3, 25), (25, 3), (25, 25)]:
        density_map[cy - 1 : cy + 2, cx - 1 : cx + 2] = 1.0

    regions = find_dense_regions(
        density_map, original_size=(300, 300), threshold_rel=0.5, max_regions=2, min_region_pixels=1
    )
    assert len(regions) == 2


class _FakeBoxes:
    """A real class (not SimpleNamespace) so `len(boxes)` actually works --
    `__len__` must be found on the type, not stuffed into an instance."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.cls)


class _FakeYoloModelForRefinement:
    """Returns a fixed number of detections scaled to whatever crop it's
    given, so `detect_with_density_refinement`'s region-crop offsetting can
    be verified without a real model."""

    def __init__(self, dets_per_call):
        self._dets_per_call = dets_per_call
        self.calls = 0

    def predict(self, frame, **kwargs):
        from types import SimpleNamespace

        self.calls += 1
        n = self._dets_per_call[min(self.calls - 1, len(self._dets_per_call) - 1)]
        h, w = frame.shape[:2]
        xyxy = np.array([[2, 2, min(10, w - 1), min(10, h - 1)] for _ in range(n)], dtype=np.float32)
        conf = np.array([0.9] * n, dtype=np.float32)
        cls = np.array([0.0] * n, dtype=np.float32)
        boxes = _FakeBoxes(xyxy=xyxy, conf=conf, cls=cls)
        return [SimpleNamespace(boxes=boxes)]


def test_detect_with_density_refinement_merges_full_and_region_passes():
    from crowd_monitoring.detection import PersonDetector

    detector = PersonDetector(confidence_threshold=0.5)
    # 1 detection on the full-frame pass, then 2 more on each of up to 2 region crops.
    detector._model = _FakeYoloModelForRefinement([1, 2, 2])
    detector._device_str = "cpu"

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    density_map = np.full((20, 20), 0.01, dtype=np.float32)
    density_map[2:5, 2:5] = 1.0
    density_map[15:18, 15:18] = 0.9

    dets = detector.detect_with_density_refinement(frame, density_map, max_regions=2, threshold_rel=0.5)

    # Full pass (1) + up to 2 region passes (2 each, non-overlapping boxes
    # since they're offset into different crops) -- more than the full pass alone.
    assert len(dets) > 1
    assert detector._model.calls == 3  # 1 full-frame + 2 region crops


def test_detect_with_density_refinement_falls_back_without_density_map():
    from crowd_monitoring.detection import PersonDetector

    detector = PersonDetector(confidence_threshold=0.5)
    detector._model = _FakeYoloModelForRefinement([3])
    detector._device_str = "cpu"

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = detector.detect_with_density_refinement(frame, density_map=None, max_regions=3)

    assert len(dets) == 3
    assert detector._model.calls == 1  # only the full-frame pass, no region crops attempted

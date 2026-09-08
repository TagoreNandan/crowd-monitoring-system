import numpy as np

from crowd_monitoring.localization import extract_points


def test_extract_points_finds_distinct_peaks():
    d = np.zeros((50, 50), dtype=np.float32)
    d[10, 10] = 5.0
    d[40, 40] = 5.0
    result = extract_points(d, original_size=(50, 50))
    assert len(result.points_xy) == 2
    xs = sorted(result.points_xy[:, 0])
    assert xs[0] < 20 and xs[1] > 30


def test_extract_points_empty_map():
    d = np.zeros((20, 20), dtype=np.float32)
    result = extract_points(d, original_size=(20, 20))
    assert len(result.points_xy) == 0


def test_extract_points_scales_to_original_size():
    d = np.zeros((10, 10), dtype=np.float32)
    d[5, 5] = 1.0
    result = extract_points(d, original_size=(100, 100))
    assert len(result.points_xy) == 1
    x, y = result.points_xy[0]
    assert 40 <= x <= 60 and 40 <= y <= 60


def test_threshold_filters_low_peaks():
    d = np.zeros((30, 30), dtype=np.float32)
    d[5, 5] = 1.0
    d[20, 20] = 0.01  # far below threshold_rel of the max
    result = extract_points(d, original_size=(30, 30), threshold_rel=0.1)
    assert len(result.points_xy) == 1

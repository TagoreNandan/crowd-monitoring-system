import numpy as np

from crowdmon.heatmap import colorize_density, normalize_density, overlay_heatmap


def _fake_density_map():
    d = np.zeros((32, 40), dtype=np.float32)
    d[10:14, 10:14] = 1.0
    d[20:24, 28:32] = 2.0
    return d


def test_normalize_density_range():
    d = _fake_density_map()
    norm = normalize_density(d)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0 + 1e-6


def test_normalize_constant_map_returns_zeros():
    d = np.ones((5, 5), dtype=np.float32)
    norm = normalize_density(d)
    assert np.allclose(norm, 0)


def test_colorize_density_shape_and_resize():
    d = _fake_density_map()
    heat = colorize_density(d, output_size=(64, 80))
    assert heat.shape == (64, 80, 3)
    assert heat.dtype == np.uint8


def test_overlay_matches_frame_size():
    d = _fake_density_map()
    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    overlay = overlay_heatmap(frame, d)
    assert overlay.shape == frame.shape

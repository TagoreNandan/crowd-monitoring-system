"""Real model load + real forward pass -- no mocked inference."""

import numpy as np

from crowd_monitoring.inference import CrowdCounter


def test_real_inference_on_synthetic_image(counter, sample_image):
    result = counter.infer(sample_image)

    assert result.count >= 0.0
    assert result.density_map.ndim == 2
    assert result.density_map.dtype == np.float32
    assert result.latency_ms > 0
    assert result.model_name == "DM-Count"
    assert result.device == "cpu"


def test_density_map_sum_equals_count(counter, sample_image):
    result = counter.infer(sample_image)
    assert abs(float(result.density_map.sum()) - result.count) < 1e-3


def test_resize_downscales_large_images():
    # Needs its own instance (non-default max_side), so it can't share the
    # session-scoped `counter` fixture -- still downloads-if-missing / skips
    # offline rather than hard-requiring pre-cached weights.
    import pytest

    try:
        counter = CrowdCounter(model_name="DM-Count", model_weights="SHA", device="cpu", max_side=200, download=True)
    except Exception as exc:
        pytest.skip(f"Could not obtain DM-Count/SHA pretrained weights (offline runner?): {exc}")

    big = (np.random.default_rng(1).random((900, 1400, 3)) * 255).astype(np.uint8)
    result = counter.infer(big, resize=True)
    assert max(result.input_size) <= 200
    assert result.original_size == (900, 1400)

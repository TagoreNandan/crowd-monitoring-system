"""Real model load + real forward pass -- no mocked inference."""

import numpy as np

from crowdmon.inference import CrowdCounter


def test_real_inference_on_synthetic_image(sample_image):
    counter = CrowdCounter(model_name="DM-Count", model_weights="SHA", device="cpu", download=False)
    result = counter.infer(sample_image)

    assert result.count >= 0.0
    assert result.density_map.ndim == 2
    assert result.density_map.dtype == np.float32
    assert result.latency_ms > 0
    assert result.model_name == "DM-Count"
    assert result.device == "cpu"


def test_density_map_sum_equals_count(sample_image):
    counter = CrowdCounter(model_name="DM-Count", model_weights="SHA", device="cpu", download=False)
    result = counter.infer(sample_image)
    assert abs(float(result.density_map.sum()) - result.count) < 1e-3


def test_resize_downscales_large_images():
    counter = CrowdCounter(model_name="DM-Count", model_weights="SHA", device="cpu", max_side=200, download=False)
    big = (np.random.default_rng(1).random((900, 1400, 3)) * 255).astype(np.uint8)
    result = counter.infer(big, resize=True)
    assert max(result.input_size) <= 200
    assert result.original_size == (900, 1400)

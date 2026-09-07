import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="session")
def sample_image():
    """A small synthetic RGB-ish test image (deterministic noise pattern)."""
    rng = np.random.default_rng(0)
    img = (rng.random((256, 320, 3)) * 255).astype(np.uint8)
    return img


@pytest.fixture(scope="session")
def pipeline():
    """Real pipeline using the already-cached DM-Count/SHA weights, forced to CPU
    for deterministic, dependency-free test runs."""
    from crowdmon.pipeline import CrowdAnalysisPipeline

    return CrowdAnalysisPipeline(model_name="DM-Count", model_weights="SHA", device="cpu", download=False)

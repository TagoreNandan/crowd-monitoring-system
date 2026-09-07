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
def counter():
    """Real CrowdCounter (DM-Count/SHA, CPU) for tests that exercise the model
    directly rather than through the full pipeline. Same download-if-missing /
    skip-if-offline behavior as the `pipeline` fixture below."""
    from crowdmon.inference import CrowdCounter

    try:
        return CrowdCounter(model_name="DM-Count", model_weights="SHA", device="cpu", download=True)
    except Exception as exc:
        pytest.skip(f"Could not obtain DM-Count/SHA pretrained weights (offline runner?): {exc}")


@pytest.fixture(scope="session")
def pipeline():
    """Real pipeline using DM-Count/SHA weights, forced to CPU for deterministic,
    hardware-independent test runs. Downloads the checkpoint if it isn't already
    cached locally (e.g. a fresh checkout/CI runner); skips these integration
    tests rather than hard-failing the whole suite if that download can't
    succeed (no network access from the runner)."""
    from crowdmon.pipeline import CrowdAnalysisPipeline

    try:
        return CrowdAnalysisPipeline(model_name="DM-Count", model_weights="SHA", device="cpu", download=True)
    except Exception as exc:
        pytest.skip(f"Could not obtain DM-Count/SHA pretrained weights (offline runner?): {exc}")

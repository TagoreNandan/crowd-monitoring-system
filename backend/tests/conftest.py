import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Force CPU + the already-cached QNRF weights so these tests are
# deterministic and never require a network download, mirroring the
# `pipeline`/`counter` fixtures in tests/conftest.py.
os.environ.setdefault("CROWDMON_DEVICE", "cpu")
os.environ.setdefault("CROWDMON_MODEL", "DM-Count")
os.environ.setdefault("CROWDMON_WEIGHTS", "QNRF")


@pytest.fixture(scope="session")
def client():
    from backend.app.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    img = (np.random.default_rng(3).random((80, 100, 3)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()

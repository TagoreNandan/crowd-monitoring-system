import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASHBOARD_DIR))


@pytest.fixture(scope="module")
def client():
    import app as dashboard_app  # imports dashboard/app.py

    dashboard_app.app.config["TESTING"] = True
    with dashboard_app.app.test_client() as c:
        yield c


def _fake_jpeg_bytes():
    img = (np.random.default_rng(6).random((80, 100, 3)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Crowd Monitoring" in resp.data


def test_status_endpoint(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "state" in data
    assert data["state"] in ("ONLINE", "DEGRADED", "OFFLINE")


def test_analyze_image_end_to_end(client):
    data = {"image": (io.BytesIO(_fake_jpeg_bytes()), "test.jpg")}
    resp = client.post("/analyze/image", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Estimated count" in resp.data or b"count" in resp.data.lower()


def test_api_analyze_json(client):
    data = {"image": (io.BytesIO(_fake_jpeg_bytes()), "test.jpg")}
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "count" in payload
    assert "risk" in payload
    assert payload["risk"]["status"] in ("SAFE", "CAUTION", "WARNING", "CRITICAL")


def test_api_analyze_missing_image_returns_400(client):
    resp = client.post("/api/analyze", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400

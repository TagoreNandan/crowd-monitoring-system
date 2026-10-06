"""Unit tests for Live Multi-Stream Analytics Router."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def user_token():
    res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert res.status_code == 200
    return res.json()["access_token"]


def test_live_analytics_unauthorized():
    res = client.get("/api/analytics/live")
    assert res.status_code == 401


def test_live_analytics_authorized(user_token):
    res = client.get("/api/analytics/live", headers={"Authorization": f"Bearer {user_token}"})
    assert res.status_code == 200
    data = res.json()
    assert "total_crowd" in data
    assert "total_visible" in data
    assert "fleet_risk_status" in data
    assert "streams" in data
    assert "zones" in data
    assert "timeline" in data
    assert "recent_alerts" in data
    assert isinstance(data["streams"], list)
    assert isinstance(data["zones"], list)
    assert isinstance(data["timeline"], list)

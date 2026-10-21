"""Unit tests for Live Multi-Stream Analytics Router."""


def test_live_analytics_allows_anonymous_polling(anon_client):
    # The fleet dashboard polls this endpoint with optional auth (see
    # get_current_user_optional in routers/analytics.py).
    res = anon_client.get("/api/analytics/live")
    assert res.status_code == 200


def test_live_analytics_authorized(client):
    res = client.get("/api/analytics/live")
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

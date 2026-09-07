def test_status_reports_real_state(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] in ("ONLINE", "DEGRADED", "OFFLINE")
    assert data["model_name"] == "DM-Count"
    assert "fps" in data
    assert "frames_processed" in data

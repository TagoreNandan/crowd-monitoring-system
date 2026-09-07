def test_config_reflects_resolved_pipeline(client):
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()

    assert data["model"]["name"] == "DM-Count"
    assert data["model"]["weights"] == "QNRF"
    assert data["model"]["device_resolved"] is True
    assert data["model"]["device"] == "cpu"

    assert "eps_px" in data["clustering"]
    assert data["risk"]["count_caution"] == 30
    assert data["risk"]["weight_count"] == 0.35

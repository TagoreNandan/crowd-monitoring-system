import pytest
from starlette.testclient import TestClient


def test_feeds_endpoints(client: TestClient):
    # List initial feeds
    res = client.get("/api/feeds")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # Start sample feed
    payload = {"source_type": "sample", "loop": False}
    res = client.post("/api/feeds", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert "feed_id" in data
    feed_id = data["feed_id"]

    # Get feed detail
    res_get = client.get(f"/api/feeds/{feed_id}")
    assert res_get.status_code == 200
    assert res_get.json()["feed_id"] == feed_id

    # Stop feed
    res_stop = client.post(f"/api/feeds/{feed_id}/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] in ("stopped", "completed")

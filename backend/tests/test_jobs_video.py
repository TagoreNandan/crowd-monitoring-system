def test_create_video_job_returns_pending_record(client):
    files = {"video": ("clip.mp4", b"fake video bytes for interface testing", "video/mp4")}
    resp = client.post("/api/jobs/video", files=files)
    assert resp.status_code == 202
    data = resp.json()

    assert data["job_type"] == "video"
    assert data["status"] == "pending"
    assert data["result"] is None
    assert data["progress"] == 0.0
    assert "not implemented" in data["message"]

    lookup = client.get(f"/api/jobs/{data['job_id']}")
    assert lookup.status_code == 200
    assert lookup.json()["job_id"] == data["job_id"]


def test_create_video_job_empty_upload_returns_400(client):
    files = {"video": ("empty.mp4", b"", "video/mp4")}
    resp = client.post("/api/jobs/video", files=files)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_upload"

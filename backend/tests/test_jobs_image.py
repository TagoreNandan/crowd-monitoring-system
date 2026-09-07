def test_create_image_job_end_to_end(client, sample_jpeg_bytes):
    files = {"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
    resp = client.post("/api/jobs/image", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["job_type"] == "image"
    assert data["status"] == "completed"
    assert data["completed_at"] is not None
    assert data["result"]["count"] >= 0.0
    assert data["result"]["model_name"] == "DM-Count"
    assert data["result"]["risk"]["status"] in ("SAFE", "CAUTION", "WARNING", "CRITICAL")
    assert "n_clusters" in data["result"]["clusters"]


def test_job_lookup_after_image_analysis(client, sample_jpeg_bytes):
    files = {"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
    created = client.post("/api/jobs/image", files=files).json()

    resp = client.get(f"/api/jobs/{created['job_id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == created["job_id"]
    assert data["status"] == "completed"
    assert data["result"]["count"] == created["result"]["count"]


def test_create_image_job_invalid_upload_returns_400(client):
    files = {"image": ("bad.jpg", b"not a real image", "image/jpeg")}
    resp = client.post("/api/jobs/image", files=files)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "invalid_upload"


def test_create_image_job_missing_file_returns_422(client):
    resp = client.post("/api/jobs/image", files={})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "validation_error"


def test_get_unknown_job_returns_404(client):
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "job_not_found"

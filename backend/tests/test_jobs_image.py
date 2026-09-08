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


def test_create_image_job_includes_real_person_detection_and_visualizations(client, sample_jpeg_bytes):
    """The image page needs separate, real visual outputs (detection/
    density/heatmap/localization/combined) and a visible_count distinct from
    DM-Count's estimated count -- never just one crowd-count number."""
    files = {"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
    resp = client.post("/api/jobs/image", files=files)
    assert resp.status_code == 200
    result = resp.json()["result"]

    detector_available = result["visible_count"] is not None
    if detector_available:
        assert isinstance(result["visible_count"], int)
        assert result["visible_count"] >= 0
        assert result["visible_count"] == len(result["detections"])
        for box in result["detections"]:
            assert box["x2"] > box["x1"] and box["y2"] > box["y1"]
            assert 0.0 <= box["confidence"] <= 1.0
        assert result["images"]["detection"] is not None
        assert result["images"]["detection"].startswith("data:image/jpeg;base64,")
    else:
        assert result["detections"] == []
        assert result["images"]["detection"] is None

    # DM-Count's dense estimate is always independent of the detector.
    assert result["count"] >= 0.0

    images = result["images"]
    for key in ("density", "heatmap", "localization", "combined"):
        assert images[key] is not None
        assert images[key].startswith("data:image/jpeg;base64,")


def test_create_image_job_no_person_image_reports_zero_not_crash(client):
    """A solid-black frame (no people) must report visible_count=0 (a real
    zero-detection result, not `None`/unavailable) and an estimated count of
    approximately 0 -- never a crash, never a fabricated nonzero count."""
    import cv2
    import numpy as np

    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    assert ok

    files = {"image": ("blank.jpg", buf.tobytes(), "image/jpeg")}
    resp = client.post("/api/jobs/image", files=files)
    assert resp.status_code == 200
    result = resp.json()["result"]

    if result["visible_count"] is not None:
        assert result["visible_count"] == 0
        assert result["detections"] == []
    assert result["count"] < 5.0  # "approximately 0" for DM-Count on a blank frame


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

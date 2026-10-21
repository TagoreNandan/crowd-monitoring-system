import time

import pytest
from starlette.testclient import TestClient


def test_feeds_endpoints(client: TestClient, sample_video_path):
    # List initial feeds
    res = client.get("/api/feeds")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # Start sample feed
    payload = {"source_type": "sample", "source_uri": sample_video_path, "loop": False}
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


def test_feed_websocket_carries_real_analytics_and_annotated_frame(client: TestClient, sample_video_path):
    """A successful WebSocket connection alone is not a successful real-time
    feed: the browser needs actual per-frame analytics (visible_count
    distinct from estimated_count) and an actual annotated JPEG frame to
    render, not just a connection ack.

    The fast (detector) and slow (DM-Count) loops broadcast independently
    (see backend/app/routers/feeds.py) -- a single message may carry only
    one loop's fields, so this merges messages the same way the real
    frontend does (FeedView.tsx) before asserting on the merged result.
    """
    res = client.post("/api/feeds", json={"source_type": "sample", "source_uri": sample_video_path, "loop": True})
    assert res.status_code == 201
    feed_id = res.json()["feed_id"]

    try:
        with client.websocket_connect(f"/ws/feeds/{feed_id}") as ws:
            ack = ws.receive_json()
            assert ack["type"] == "connection.ack"

            status = ws.receive_json()
            assert status["type"] == "feed.status"

            merged: dict = {}
            fast_messages = 0
            density_messages = 0
            for _ in range(60):
                msg = ws.receive_json()
                if msg["type"] != "feed.progress":
                    continue
                data = msg["data"]
                merged.update(data)
                if "frame_jpeg" in data:
                    fast_messages += 1
                if "heatmap_jpeg" in data:
                    density_messages += 1
                if fast_messages >= 1 and density_messages >= 1:
                    break

            assert fast_messages >= 1, "no fast-path (detector) message arrived"
            assert density_messages >= 1, "no density-path (DM-Count) message arrived"

            # Fast-path fields
            assert "visible_count" in merged  # present (possibly null), never silently omitted
            assert merged["frame_jpeg"].startswith("data:image/jpeg;base64,")

            # Density-path fields
            assert "estimated_count" in merged
            assert "risk_status" in merged
            assert "density" in merged
            assert merged["heatmap_jpeg"].startswith("data:image/jpeg;base64,")
            assert merged["combined_jpeg"].startswith("data:image/jpeg;base64,")

            # visible_count (detector) and estimated_count (DM-Count) are
            # never conflated -- both present, independently sourced.
            assert merged["estimated_count"] != merged.get("visible_count")
    finally:
        client.post(f"/api/feeds/{feed_id}/stop")


def test_feed_fast_path_delivers_multiple_frames_while_density_pass_runs(client: TestClient, sample_video_path):
    """This is the actual regression test for the "realtime visibly stalls"
    bug: the fast (detector) path must keep delivering new frames while a
    single slow DM-Count pass is in flight, not block on it. Asserts on
    real frame_number progression and wall-clock timing, not mocks."""
    res = client.post("/api/feeds", json={"source_type": "sample", "source_uri": sample_video_path, "loop": True})
    assert res.status_code == 201
    feed_id = res.json()["feed_id"]

    try:
        with client.websocket_connect(f"/ws/feeds/{feed_id}") as ws:
            ws.receive_json()  # ack
            ws.receive_json()  # feed.status

            frame_numbers = []
            timestamps = []
            deadline = time.time() + 15.0
            while time.time() < deadline and len(frame_numbers) < 6:
                msg = ws.receive_json()
                if msg["type"] != "feed.progress":
                    continue
                data = msg["data"]
                if "frame_jpeg" in data:
                    frame_numbers.append(data["frame_number"])
                    timestamps.append(time.time())

            assert len(frame_numbers) >= 6, f"fast path only delivered {len(frame_numbers)} frames in 15s"
            # Strictly increasing (or looping back around) frame numbers --
            # i.e. real, distinct frames, not the same frame re-sent.
            assert len(set(frame_numbers)) > 1

            # No single inter-frame gap should look like a multi-second
            # stall behind a blocking DM-Count call.
            gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
            assert max(gaps) < 2.0, f"fast-path frame delivery stalled: gaps={gaps}"
    finally:
        client.post(f"/api/feeds/{feed_id}/stop")


def test_upload_feed_endpoint(client: TestClient, tmp_path):
    import cv2
    import numpy as np

    # Create a small valid test mp4 video file
    video_file = tmp_path / "test_feed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_file), fourcc, 10.0, (160, 120))
    for _ in range(5):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with open(video_file, "rb") as f:
        res = client.post("/api/feeds/upload", files={"video": ("test_feed.mp4", f, "video/mp4")}, data={"loop": "false"})

    assert res.status_code == 201
    data = res.json()
    assert "feed_id" in data
    feed_id = data["feed_id"]

    res_get = client.get(f"/api/feeds/{feed_id}")
    assert res_get.status_code == 200

    res_stop = client.post(f"/api/feeds/{feed_id}/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] in ("stopped", "completed")


def test_upload_4k_feed_websocket_delivers_scaled_frames(client: TestClient, tmp_path):
    """Regression test: uploading a 4K frame (> max_side) must correctly downscale
    and construct TrackedPerson objects with center_xy without raising TypeError."""
    import cv2
    import numpy as np

    video_file = tmp_path / "test_4k_feed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_file), fourcc, 10.0, (3840, 2160))
    for _ in range(5):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with open(video_file, "rb") as f:
        res = client.post("/api/feeds/upload", files={"video": ("test_4k_feed.mp4", f, "video/mp4")}, data={"loop": "false"})

    assert res.status_code == 201
    feed_id = res.json()["feed_id"]

    try:
        with client.websocket_connect(f"/ws/feeds/{feed_id}") as ws:
            ws.receive_json()  # ack
            ws.receive_json()  # status
            progress_recvd = False
            for _ in range(20):
                msg = ws.receive_json()
                if msg["type"] == "feed.progress":
                    progress_recvd = True
                    assert "frame_jpeg" in msg["data"]
                    break
            assert progress_recvd, "4K video feed did not deliver feed.progress"
    finally:
        client.post(f"/api/feeds/{feed_id}/stop")

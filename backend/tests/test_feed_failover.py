"""End-to-end drone failover through the real /api/feeds + /ws/feeds path."""

from __future__ import annotations

import time

from backend.app.routers.feeds import _wrap_with_failover
from crowd_monitoring.streaming import FailoverFrameSource, FileVideoSource, RTSPSource

# Nothing listens on port 1, so FFmpeg's connect fails fast: a drone that is
# offline when the feed is started.
DEAD_DRONE_URL = "rtsp://127.0.0.1:1/drone"


def test_wrap_with_failover_only_wraps_live_network_streams(sample_video_path):
    live = _wrap_with_failover(RTSPSource(DEAD_DRONE_URL), sample_video_path, None)
    assert isinstance(live, FailoverFrameSource)

    vod = RTSPSource("https://bucket.s3.amazonaws.com/clip.mp4")
    assert _wrap_with_failover(vod, sample_video_path, None) is vod

    local = FileVideoSource(sample_video_path)
    assert _wrap_with_failover(local, sample_video_path, None) is local

    # Explicit '' disables failover for that feed.
    plain = RTSPSource(DEAD_DRONE_URL)
    assert _wrap_with_failover(plain, "", None) is plain


def test_dead_drone_feed_plays_backup_and_reports_backup_active(client, sample_video_path):
    res = client.post(
        "/api/feeds",
        json={"source_type": "rtsp", "source_uri": DEAD_DRONE_URL, "fallback_uri": sample_video_path},
    )
    assert res.status_code == 201, res.text
    body = res.json()
    feed_id = body["feed_id"]
    assert body["failover_enabled"] is True
    assert body["source_mode"] == "auto"

    try:
        statuses = set()
        frames = 0
        with client.websocket_connect(f"/ws/feeds/{feed_id}") as ws:
            deadline = time.time() + 15.0
            while time.time() < deadline and not (frames >= 3 and "BACKUP_ACTIVE" in statuses):
                msg = ws.receive_json()
                if msg["type"] != "feed.progress":
                    continue
                if "source_status" in msg["data"]:
                    statuses.add(msg["data"]["source_status"])
                if "frame_jpeg" in msg["data"]:
                    frames += 1
        assert "BACKUP_ACTIVE" in statuses
        assert frames >= 3, "backup archive frames were not streamed"

        detail = client.get(f"/api/feeds/{feed_id}").json()
        assert detail["source_status"] == "BACKUP_ACTIVE"
        assert detail["status"] == "running"
    finally:
        client.post(f"/api/feeds/{feed_id}/stop")


def test_manual_source_override_endpoint(client, sample_video_path):
    res = client.post(
        "/api/feeds",
        json={"source_type": "rtsp", "source_uri": DEAD_DRONE_URL, "fallback_uri": sample_video_path},
    )
    feed_id = res.json()["feed_id"]
    try:
        forced = client.post(f"/api/feeds/{feed_id}/source", json={"mode": "backup"})
        assert forced.status_code == 200
        assert forced.json()["source_mode"] == "backup"

        back = client.post(f"/api/feeds/{feed_id}/source", json={"mode": "auto"})
        assert back.json()["source_mode"] == "auto"

        assert client.post(f"/api/feeds/{feed_id}/source", json={"mode": "nope"}).status_code == 422
    finally:
        client.post(f"/api/feeds/{feed_id}/stop")


def test_source_override_rejected_for_feed_without_failover(client, sample_video_path):
    res = client.post("/api/feeds", json={"source_type": "sample", "source_uri": sample_video_path})
    feed_id = res.json()["feed_id"]
    try:
        assert res.json()["failover_enabled"] is False
        resp = client.post(f"/api/feeds/{feed_id}/source", json={"mode": "backup"})
        assert resp.status_code == 409
    finally:
        client.post(f"/api/feeds/{feed_id}/stop")


def test_drone_fallback_map_lookup():
    from backend.app.routers.feeds import DRONE_FALLBACK_VIDEOS, drone_fallback_uri

    assert set(DRONE_FALLBACK_VIDEOS) == {f"drone_{i}" for i in range(1, 7)}
    assert drone_fallback_uri("DRONE_3") == DRONE_FALLBACK_VIDEOS["drone_3"]
    assert drone_fallback_uri(None, "drone_6") == DRONE_FALLBACK_VIDEOS["drone_6"]
    assert drone_fallback_uri("CAM-DEMO-01", "feed_ab12cd34") is None


def test_presigned_s3_video_signs_bare_keys_and_uris(monkeypatch):
    from backend.app.routers import feeds

    calls = []
    monkeypatch.setattr(
        feeds.storage, "presigned_url",
        lambda key, bucket, **kw: calls.append((bucket, key)) or f"https://{bucket}.s3/{key}?X-Amz-Signature=s",
    )
    url, desc = feeds.presigned_s3_video("s3://flight-bucket/failsafe/backup_2.mp4")
    assert url.startswith("https://flight-bucket.s3/failsafe/backup_2.mp4?")
    assert desc == "s3://flight-bucket/failsafe/backup_2.mp4"
    assert calls == [("flight-bucket", "failsafe/backup_2.mp4")]


def test_mapped_drone_feed_uses_its_own_looping_backup(client, sample_video_path, monkeypatch):
    from backend.app.routers import feeds

    monkeypatch.setitem(feeds.DRONE_FALLBACK_VIDEOS, "drone_4", sample_video_path)
    res = client.post(
        "/api/feeds",
        json={"source_type": "rtsp", "source_uri": DEAD_DRONE_URL, "device_id": "drone_4", "device_name": "Drone 4"},
    )
    assert res.status_code == 201, res.text
    feed_id = res.json()["feed_id"]
    try:
        with feeds._feeds_lock:
            source = feeds._active_feeds[feed_id].source
        assert isinstance(source, FailoverFrameSource)
        deadline = time.time() + 5.0
        while time.time() < deadline and source._backup is None:
            time.sleep(0.05)
        assert isinstance(source._backup, FileVideoSource)
        assert source._backup.loop is True
        assert str(source._backup._path) == sample_video_path
    finally:
        client.post(f"/api/feeds/{feed_id}/stop")


def test_feed_websocket_frames_are_throttled_to_max_fps(client, sample_video_path, monkeypatch):
    from backend.app.settings import get_settings

    monkeypatch.setattr(get_settings(), "feed_ws_max_fps", 5.0)
    res = client.post("/api/feeds", json={"source_type": "sample", "source_uri": sample_video_path, "loop": True})
    feed_id = res.json()["feed_id"]
    try:
        stamps = []
        with client.websocket_connect(f"/ws/feeds/{feed_id}") as ws:
            deadline = time.time() + 4.0
            while time.time() < deadline:
                msg = ws.receive_json()
                if msg["type"] == "feed.progress" and "frame_jpeg" in msg["data"]:
                    stamps.append(msg["ts"])  # server-side send time
        assert len(stamps) >= 5
        window = stamps[-1] - stamps[0]
        assert (len(stamps) - 1) / window <= 5.0 * 1.15
    finally:
        client.post(f"/api/feeds/{feed_id}/stop")


def test_local_video_outside_project_directory_is_allowed(tmp_path):
    from crowd_monitoring.streaming.frame_source import FileVideoSource

    from backend.app.routers.feeds import _resolve_source

    video = tmp_path / "Downloads" / "video.mp4"
    video.parent.mkdir()
    video.write_bytes(b"")
    source = _resolve_source("video", str(video))
    assert isinstance(source, FileVideoSource)
    assert source._path == video

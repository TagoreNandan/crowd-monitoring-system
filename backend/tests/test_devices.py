"""Unit tests for Device Management and Multi-Stream Onboarding."""

import uuid

import pytest


@pytest.fixture
def user_token(make_user_token):
    _, token = make_user_token("analyst")
    return token


def test_list_devices_authenticated(client, user_token):
    res = client.get("/api/devices", headers={"Authorization": f"Bearer {user_token}"})
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_device_creation_forbidden_for_regular_user(client, user_token):
    dev_id = f"CAM-TEST-{uuid.uuid4().hex[:4].upper()}"
    res = client.post(
        "/api/devices",
        json={
            "device_id": dev_id,
            "name": "Unauthorized Camera",
            "location": "Sector 4",
            "stream_type": "RTSP",
            "stream_url": "rtsp://10.0.0.1/live",
        },
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert res.status_code == 403


def test_device_crud_lifecycle_admin(client):
    dev_id = f"CAM-TEST-{uuid.uuid4().hex[:4].upper()}"

    # 1. Create Device
    create_res = client.post(
        "/api/devices",
        json={
            "device_id": dev_id,
            "name": "Gate 1 Surveillance",
            "description": "Main entrance outdoor optical camera",
            "location": "North Entrance",
            "stream_type": "VIDEO_FILE",
            "stream_url": "dashboard/static/uploads/sample.mp4",
            "status": "ONLINE",
            "resolution": "1920x1080",
            "fps": 30.0,
        },
    )
    assert create_res.status_code == 201
    data = create_res.json()
    assert data["device_id"] == dev_id
    assert data["name"] == "Gate 1 Surveillance"
    assert data["status"] == "ONLINE"

    # 2. Duplicate device_id should fail
    dup_res = client.post(
        "/api/devices",
        json={
            "device_id": dev_id,
            "name": "Duplicate Camera",
            "location": "North Entrance",
            "stream_type": "VIDEO_FILE",
            "stream_url": "sample.mp4",
        },
    )
    assert dup_res.status_code == 400

    # 3. Retrieve Single Device
    get_res = client.get(f"/api/devices/{dev_id}")
    assert get_res.status_code == 200
    assert get_res.json()["device_id"] == dev_id

    # 4. Update Device
    update_res = client.put(
        f"/api/devices/{dev_id}",
        json={"name": "Gate 1 Updated Camera", "status": "STANDBY"},
    )
    assert update_res.status_code == 200
    assert update_res.json()["name"] == "Gate 1 Updated Camera"
    assert update_res.json()["status"] == "STANDBY"

    # 5. Delete Device
    del_res = client.delete(f"/api/devices/{dev_id}")
    assert del_res.status_code == 200

    # 6. Verify Deletion
    not_found = client.get(f"/api/devices/{dev_id}")
    assert not_found.status_code == 404


def test_probe_treats_local_and_url_mp4s_as_non_s3(monkeypatch, tmp_path):
    from backend.app.routers import devices

    opened = []

    class _Cap:
        def __init__(self, source):
            opened.append(source)

        def set(self, *a):
            return True

        def isOpened(self):
            return False

        def release(self):
            pass

    def _no_s3(*a, **k):
        raise AssertionError("must not presign an S3 key")

    monkeypatch.setattr(devices.cv2, "VideoCapture", _Cap)
    monkeypatch.setattr(devices.storage, "presigned_url", _no_s3)
    monkeypatch.setattr(devices.storage, "is_s3_enabled", lambda: True)

    local = tmp_path / "Founder Downloads" / "video.mp4"
    local.parent.mkdir()
    local.write_bytes(b"")
    devices._probe_stream_sync(str(local), "VIDEO_FILE")
    devices._probe_stream_sync("rtsp://192.168.1.20:8554/live.mp4", "RTSP")
    assert opened == [str(local), "rtsp://192.168.1.20:8554/live.mp4"]

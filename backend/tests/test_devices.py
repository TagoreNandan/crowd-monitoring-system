"""Unit tests for Device Management and Multi-Stream Onboarding."""

import uuid
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def admin_token():
    res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert res.status_code == 200
    return res.json()["access_token"]


@pytest.fixture(scope="module")
def user_token():
    uname = f"operator_{uuid.uuid4().hex[:6]}"
    res = client.post(
        "/api/auth/register",
        json={
            "username": uname,
            "email": f"{uname}@test.local",
            "password": "pass123456",
            "full_name": "Test Operator",
        },
    )
    assert res.status_code == 201
    login_res = client.post(
        "/api/auth/login",
        json={"username": uname, "password": "pass123456"},
    )
    assert login_res.status_code == 200
    return login_res.json()["access_token"]


def test_list_devices_authenticated(user_token):
    res = client.get("/api/devices", headers={"Authorization": f"Bearer {user_token}"})
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_device_creation_forbidden_for_regular_user(user_token):
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


def test_device_crud_lifecycle_admin(admin_token):
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
        headers={"Authorization": f"Bearer {admin_token}"},
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
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert dup_res.status_code == 400

    # 3. Retrieve Single Device
    get_res = client.get(
        f"/api/devices/{dev_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_res.status_code == 200
    assert get_res.json()["device_id"] == dev_id

    # 4. Update Device
    update_res = client.put(
        f"/api/devices/{dev_id}",
        json={"name": "Gate 1 Updated Camera", "status": "STANDBY"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert update_res.status_code == 200
    assert update_res.json()["name"] == "Gate 1 Updated Camera"
    assert update_res.json()["status"] == "STANDBY"

    # 5. Delete Device
    del_res = client.delete(
        f"/api/devices/{dev_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert del_res.status_code == 200

    # 6. Verify Deletion
    not_found = client.get(
        f"/api/devices/{dev_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert not_found.status_code == 404

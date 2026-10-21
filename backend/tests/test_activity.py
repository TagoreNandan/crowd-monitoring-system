"""Tests for role-scoped activity history and output download metadata."""

from __future__ import annotations

import json
import uuid

from backend.app.auth.security import create_access_token
from backend.app.db.models import ActivityLog, User
from backend.app.db.session import SessionLocal


def _create_user(role: str) -> User:
    username = f"activity_{role}_{uuid.uuid4().hex[:8]}"
    db = SessionLocal()
    try:
        user = User(
            username=username,
            email=f"{username}@example.com",
            full_name=username,
            hashed_password="unused",
            role=role,
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def _token(user: User) -> str:
    return create_access_token(data={"sub": user.username, "role": user.role, "id": user.id})


def _add_activity(user: User, **values) -> ActivityLog:
    db = SessionLocal()
    try:
        entry = ActivityLog(
            user_id=user.id,
            username=user.username,
            action=values.pop("action", "IMAGE_ANALYZED"),
            category=values.pop("category", "INFERENCE"),
            details=values.pop("details", "test activity"),
            severity=values.pop("severity", "INFO"),
            **values,
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry
    finally:
        db.close()


def test_analyst_sees_only_own_rows(client):
    analyst = _create_user("analyst")
    other = _create_user("analyst")
    own = _add_activity(analyst)
    _add_activity(other)

    response = client.get(
        "/api/activity",
        headers={"Authorization": f"Bearer {_token(analyst)}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert [item["id"] for item in data["items"]] == [own.id]
    assert "output_keys" not in data["items"][0]


def test_admin_sees_all_rows(client):
    admin = _create_user("admin")
    analyst = _create_user("analyst")
    own = _add_activity(admin)
    other = _add_activity(analyst)

    response = client.get(
        "/api/activity",
        headers={"Authorization": f"Bearer {_token(admin)}"},
    )

    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["items"]}
    assert own.id in ids
    assert other.id in ids


def test_analyst_gets_404_for_another_users_row(client):
    analyst = _create_user("analyst")
    other = _create_user("analyst")
    entry = _add_activity(other)

    response = client.get(
        f"/api/activity/{entry.id}/download?view=detection",
        headers={"Authorization": f"Bearer {_token(analyst)}"},
    )

    assert response.status_code == 404


def test_s3_download_returns_presigned_url_info(monkeypatch, client):
    analyst = _create_user("analyst")
    entry = _add_activity(
        analyst,
        job_id="historic-image",
        job_type="IMAGE",
        output_keys=json.dumps({"detection": "jobs/historic-image/image/detection.jpg"}),
        storage_backend="s3",
    )
    monkeypatch.setattr(
        "backend.app.routers.activity.storage.get_presigned_url_info",
        lambda key, response_disposition=None: {
            "url": "https://signed.example/output.jpg",
            "expires_in": 300,
            "expires_at": 1234567890,
        },
    )
    monkeypatch.setattr("backend.app.routers.activity.storage.object_exists", lambda key: True)

    response = client.get(
        f"/api/activity/{entry.id}/download?view=detection",
        headers={"Authorization": f"Bearer {_token(analyst)}"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "mode": "s3",
        "url": "https://signed.example/output.jpg",
        "expires_in": 300,
        "expires_at": 1234567890,
        "filename": "detection.jpg",
        "view": "detection",
    }


def test_activity_sql_pagination_and_download_filter(client):
    admin = _create_user("admin")
    output_values = [
        {"detection": "jobs/pagination-1/detection.jpg"},
        {"detection": "jobs/pagination-2/detection.jpg"},
    ]
    for output_keys in output_values:
        _add_activity(
            admin,
            action="PAGINATION_TEST",
            output_keys=json.dumps(output_keys),
            storage_backend="s3",
        )
    _add_activity(admin, action="PAGINATION_TEST")

    headers = {"Authorization": f"Bearer {_token(admin)}"}
    response = client.get(
        f"/api/activity?username={admin.username}&action=PAGINATION_TEST&has_download=true&limit=1&offset=1",
        headers=headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["items"]) == 1
    assert data["items"][0]["has_download"] is True

    no_output_response = client.get(
        f"/api/activity?username={admin.username}&action=PAGINATION_TEST&has_download=false",
        headers=headers,
    )
    assert no_output_response.status_code == 200
    assert no_output_response.json()["total"] == 1


def test_s3_download_returns_410_when_object_is_deleted(monkeypatch, client):
    analyst = _create_user("analyst")
    entry = _add_activity(
        analyst,
        job_id="deleted-output",
        job_type="IMAGE",
        output_keys=json.dumps({"detection": "jobs/deleted-output/detection.jpg"}),
        storage_backend="s3",
    )
    monkeypatch.setattr("backend.app.routers.activity.storage.object_exists", lambda key: False)

    response = client.get(
        f"/api/activity/{entry.id}/download?view=detection",
        headers={"Authorization": f"Bearer {_token(analyst)}"},
    )

    assert response.status_code == 410
    assert response.json()["error"]["message"] == "The requested output is no longer available"


def test_s3_download_returns_503_for_storage_errors(monkeypatch, client):
    analyst = _create_user("analyst")
    entry = _add_activity(
        analyst,
        job_id="unavailable-output",
        job_type="IMAGE",
        output_keys=json.dumps({"detection": "jobs/unavailable-output/detection.jpg"}),
        storage_backend="s3",
    )

    def raise_storage_error(key):
        raise RuntimeError("S3 unavailable")

    monkeypatch.setattr("backend.app.routers.activity.storage.object_exists", raise_storage_error)

    response = client.get(
        f"/api/activity/{entry.id}/download?view=detection",
        headers={"Authorization": f"Bearer {_token(analyst)}"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "The output storage service is temporarily unavailable"


def test_rows_without_outputs_return_404(client):
    analyst = _create_user("analyst")
    entry = _add_activity(analyst)

    response = client.get(
        f"/api/activity/{entry.id}/download?view=detection",
        headers={"Authorization": f"Bearer {_token(analyst)}"},
    )

    assert response.status_code == 404


def test_activity_requires_authentication(anon_client):
    response = anon_client.get("/api/activity")
    assert response.status_code == 401

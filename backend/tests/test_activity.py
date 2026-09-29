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
        lambda key: {
            "url": "https://signed.example/output.jpg",
            "expires_in": 300,
            "expires_at": 1234567890,
        },
    )

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


def test_rows_without_outputs_return_404(client):
    analyst = _create_user("analyst")
    entry = _add_activity(analyst)

    response = client.get(
        f"/api/activity/{entry.id}/download?view=detection",
        headers={"Authorization": f"Bearer {_token(analyst)}"},
    )

    assert response.status_code == 404


def test_activity_requires_authentication(client):
    response = client.get("/api/activity")
    assert response.status_code == 401

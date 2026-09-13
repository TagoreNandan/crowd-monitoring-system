"""Unit tests for user management, authentication, and admin activity monitoring."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_auth_login_admin():
    response = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["role"] == "admin"
    assert data["username"] == "admin"


def test_auth_login_invalid_password():
    response = client.post("/api/auth/login", json={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 401
    assert "Invalid username or password" in response.json()["detail"]


def test_auth_register_and_login_new_user():
    import uuid
    uname = f"test_{uuid.uuid4().hex[:6]}"
    reg_response = client.post(
        "/api/auth/register",
        json={
            "username": uname,
            "email": f"{uname}@example.com",
            "password": "securepassword123",
            "full_name": "Test User",
        },
    )
    assert reg_response.status_code == 201
    user_data = reg_response.json()
    assert user_data["username"] == uname
    assert user_data["role"] == "user"

    # Login with new user
    login_response = client.post(
        "/api/auth/login",
        json={"username": uname, "password": "securepassword123"},
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    # Test /api/auth/me
    me_response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 200
    assert me_response.json()["username"] == uname


def test_admin_endpoints_require_admin():
    # Login as regular analyst
    analyst_login = client.post("/api/auth/login", json={"username": "analyst", "password": "analyst123"})
    assert analyst_login.status_code == 200
    user_token = analyst_login.json()["access_token"]

    # Regular user attempting admin endpoint should get 403 Forbidden
    resp = client.get("/api/admin/users", headers={"Authorization": f"Bearer {user_token}"})
    assert resp.status_code == 403

    # Login as admin
    admin_login = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    admin_token = admin_login.json()["access_token"]

    # Admin accessing /api/admin/users
    users_resp = client.get("/api/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert users_resp.status_code == 200
    users = users_resp.json()
    assert any(u["username"] == "admin" for u in users)

    # Admin accessing /api/admin/activities
    act_resp = client.get("/api/admin/activities", headers={"Authorization": f"Bearer {admin_token}"})
    assert act_resp.status_code == 200
    activities = act_resp.json()
    assert len(activities) > 0
    assert any("LOGIN" in a["action"] or "SYSTEM" in a["action"] for a in activities)

    # Admin accessing /api/admin/stats
    stats_resp = client.get("/api/admin/stats", headers={"Authorization": f"Bearer {admin_token}"})
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_users"] >= 2
    assert stats["admin_users"] >= 1

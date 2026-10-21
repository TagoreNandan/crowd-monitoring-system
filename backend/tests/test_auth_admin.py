"""Unit tests for user management, authentication, and admin activity monitoring."""


def test_auth_login_admin(anon_client):
    response = anon_client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["role"] == "admin"
    assert data["username"] == "admin"


def test_auth_login_invalid_password(anon_client):
    response = anon_client.post("/api/auth/login", json={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Invalid username or password"


def test_public_registration_is_disabled(anon_client):
    # Self-service sign-up was removed; accounts are provisioned by admins.
    response = anon_client.post(
        "/api/auth/register",
        json={"username": "nobody", "email": "nobody@example.com", "password": "x" * 12},
    )
    assert response.status_code in (404, 405)


def test_admin_created_user_can_login_and_read_profile(client, make_user_token):
    uname, token = make_user_token("analyst", password="securepassword123")

    me_response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 200
    assert me_response.json()["username"] == uname
    assert me_response.json()["role"] == "analyst"


def test_admin_endpoints_require_admin(anon_client):
    # Login as regular analyst
    analyst_login = anon_client.post("/api/auth/login", json={"username": "analyst", "password": "analyst123"})
    assert analyst_login.status_code == 200
    user_token = analyst_login.json()["access_token"]

    # Regular user attempting admin endpoint should get 403 Forbidden
    resp = anon_client.get("/api/admin/users", headers={"Authorization": f"Bearer {user_token}"})
    assert resp.status_code == 403

    # Login as admin
    admin_login = anon_client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    admin_token = admin_login.json()["access_token"]

    # Admin accessing /api/admin/users
    users_resp = anon_client.get("/api/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert users_resp.status_code == 200
    users = users_resp.json()
    assert any(u["username"] == "admin" for u in users)


def test_protected_endpoint_rejects_missing_token(anon_client):
    response = anon_client.get("/api/auth/me")
    assert response.status_code == 401

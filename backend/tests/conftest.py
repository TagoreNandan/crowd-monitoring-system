import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Force CPU + the already-cached QNRF weights so these tests are
# deterministic and never require a network download, mirroring the
# `pipeline`/`counter` fixtures in tests/conftest.py.
os.environ.setdefault("CROWD_MONITORING_DEVICE", "cpu")
os.environ.setdefault("CROWD_MONITORING_MODEL", "DM-Count")
os.environ.setdefault("CROWD_MONITORING_WEIGHTS", "QNRF")

# Isolate the suite from the developer's data/cms.db. This must run before
# backend.app.db.session is imported, because the engine is bound at import
# time. Credentials are pinned so the seeded admin is always known.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="cms-test-db-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TEST_DB_DIR) / 'cms-test.db'}"
os.environ["CROWD_MONITORING_BACKEND_ENVIRONMENT"] = "test"
os.environ["CROWD_MONITORING_BACKEND_MEDIAMTX_ENABLED"] = "false"
os.environ["CROWD_MONITORING_BACKEND_ADMIN_USERNAME"] = "admin"
os.environ["CROWD_MONITORING_BACKEND_ADMIN_PASSWORD"] = "admin123"
# Never touch real AWS from tests. storage.is_s3_enabled() treats a non-empty
# S3_BUCKET_NAME (with a built-in default) as "enabled", so blank it explicitly;
# tests that exercise S3 paths monkeypatch storage directly.
os.environ["STORAGE_BACKEND"] = "local"
os.environ["S3_BUCKET_NAME"] = ""

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


@pytest.fixture(scope="session")
def app():
    from backend.app.main import create_app

    return create_app()


@pytest.fixture(scope="session")
def _lifespan_client(app):
    # Entering the context runs the lifespan: SQLite schema + seeded admin,
    # orphan-recording recovery, and the pipeline warm-up.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def admin_token(_lifespan_client) -> str:
    res = _lifespan_client.post(
        "/api/auth/login", json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}
    )
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


@pytest.fixture
def client(_lifespan_client, admin_token):
    """Lifespan-initialised client authenticated as the seeded admin.

    Per-request `headers={"Authorization": ...}` still override the default,
    so tests can act as other users.
    """
    _lifespan_client.headers["Authorization"] = f"Bearer {admin_token}"
    yield _lifespan_client
    _lifespan_client.headers["Authorization"] = f"Bearer {admin_token}"


# Alias matching the name used in the auth-migration notes.
auth_client = client


@pytest.fixture
def anon_client(_lifespan_client):
    """Same lifespan client with no default Authorization header."""
    saved = _lifespan_client.headers.pop("Authorization", None)
    yield _lifespan_client
    if saved is not None:
        _lifespan_client.headers["Authorization"] = saved


@pytest.fixture
def make_user_token(client):
    """Create a user through the admin API and return a bearer token for it."""
    import uuid

    def _make(role: str = "analyst", password: str = "pass123456") -> tuple[str, str]:
        username = f"{role}_{uuid.uuid4().hex[:8]}"
        res = client.post(
            "/api/admin/users",
            json={
                "username": username,
                "email": f"{username}@example.com",
                "password": password,
                "full_name": f"Test {role}",
                "role": role,
            },
        )
        assert res.status_code == 201, res.text
        login = client.post("/api/auth/login", json={"username": username, "password": password})
        assert login.status_code == 200, login.text
        return username, login.json()["access_token"]

    return _make


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    img = (np.random.default_rng(3).random((80, 100, 3)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="session")
def sample_video_path(tmp_path_factory) -> str:
    """Synthetic clip standing in for examples/sample_crowd.mp4, which the
    repo deliberately does not ship (see examples/README.md). Lives under the
    system temp dir, which _resolve_source allows.

    Frames are a jittered grid of dark head-sized blobs: DM-Count returns a
    small non-zero density on it, whereas plain noise yields exactly 0.0."""
    video_file = tmp_path_factory.mktemp("sample_feed") / "sample_crowd.avi"
    # Lossless PNG-in-AVI: mp4v artefacts smear the blobs and DM-Count drops to 0.
    writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter_fourcc(*"png "), 15.0, (320, 240))
    rng = np.random.default_rng(7)
    for i in range(45):
        frame = np.full((240, 320, 3), 170, dtype=np.uint8)
        for y in range(12, 240, 18):
            for x in range(10, 320, 16):
                jx, jy = (int(v) for v in rng.integers(-3, 4, size=2))
                cv2.circle(frame, (x + (i % 4) + jx, y + jy), 9, (40, 40, 40), -1)
        writer.write(frame)
    writer.release()
    return str(video_file)

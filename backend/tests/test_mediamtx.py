"""MediaMTX child-process lifecycle and repo-anchored path helpers."""

import os
import socket
import stat
import sys

import pytest

from backend.app import mediamtx
from backend.app.mediamtx import MediaMTXServer, find_mediamtx_binary
from backend.app.paths import REPO_ROOT, find_local_file, is_within


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(mediamtx, "DATA_DIR", tmp_path)
    if os.name == "nt":
        path = tmp_path / "mediamtx.cmd"
        path.write_text(f'@"{sys.executable}" -c "import time; time.sleep(60)"\n')
    else:
        path = tmp_path / "mediamtx"
        path.write_text(f"#!/bin/sh\nexec '{sys.executable}' -c 'import time; time.sleep(60)'\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_binary_name_is_os_specific():
    assert mediamtx.BINARY_NAME == ("mediamtx.exe" if sys.platform == "win32" else "mediamtx")


def test_explicit_path_that_does_not_exist_is_not_found(tmp_path):
    assert find_mediamtx_binary(str(tmp_path / "nope")) is None


def test_start_and_stop_child_process(fake_binary, tmp_path):
    server = MediaMTXServer(str(fake_binary), rtmp_port=_free_port())
    server.log_path = tmp_path / "mediamtx.log"
    server.start()
    try:
        assert server.running
        server.check_started()
    finally:
        server.stop(timeout=5)
    assert not server.running


def test_skips_launch_when_rtmp_port_already_served(fake_binary):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        server = MediaMTXServer(str(fake_binary), rtmp_port=listener.getsockname()[1])
        server.start()
        assert not server.running
        server.stop()


def test_missing_binary_does_not_raise(tmp_path):
    server = MediaMTXServer(str(tmp_path / "missing"), rtmp_port=_free_port())
    server.start()
    assert not server.running
    server.stop()


def test_find_local_file_resolves_repo_relative_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # works regardless of the launch directory
    assert find_local_file("backend/app/paths.py") == REPO_ROOT / "backend" / "app" / "paths.py"
    assert find_local_file("failsafe/backup_1.mp4") is None
    assert find_local_file("rtmp://127.0.0.1/live/drone_1") is None


def test_is_within_rejects_sibling_prefix(tmp_path):
    root = tmp_path / "repo"
    sibling = tmp_path / "repo2"
    root.mkdir()
    sibling.mkdir()
    assert is_within(root / "a.mp4", root)
    assert not is_within(sibling / "a.mp4", root)
    assert not is_within(root / ".." / "repo2" / "a.mp4", root)

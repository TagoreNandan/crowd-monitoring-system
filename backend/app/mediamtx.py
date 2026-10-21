"""Run the MediaMTX RTMP/RTSP server as a child of the API process.

Drones publish to ``rtmp://<host>:1935/<path>``; the backend reads the same
streams back. Launching MediaMTX from the FastAPI lifespan means one command
(``uvicorn backend.app.main:app``) brings up the whole stack, on Windows
(``mediamtx.exe``) and macOS/Linux (``./mediamtx``) alike.

Startup never blocks or fails on MediaMTX: if the binary is missing, or a
MediaMTX is already listening on the RTMP port (e.g. one left running by
hand), the API still boots and this just logs what it did.
"""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import IO, Optional

from .paths import DATA_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
BINARY_NAME = "mediamtx.exe" if IS_WINDOWS else "mediamtx"


def find_mediamtx_binary(explicit_path: str = "") -> Optional[Path]:
    """Explicit setting first, then the repo root (``./mediamtx``), a
    ``mediamtx/`` folder in the repo (the extracted release archive), and
    finally the system PATH."""
    if explicit_path:
        path = Path(explicit_path).expanduser()
        return path if path.is_file() else None
    for candidate in (REPO_ROOT / BINARY_NAME, REPO_ROOT / "mediamtx" / BINARY_NAME):
        if candidate.is_file():
            return candidate
    on_path = shutil.which(BINARY_NAME)
    return Path(on_path) if on_path else None


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


class MediaMTXServer:
    def __init__(self, explicit_path: str = "", rtmp_port: int = 1935):
        self._explicit_path = explicit_path
        self._rtmp_port = rtmp_port
        self._process: Optional[subprocess.Popen] = None
        self._log_file: Optional[IO[bytes]] = None
        self.log_path = DATA_DIR / "mediamtx.log"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if _port_in_use(self._rtmp_port):
            logger.info("RTMP port %d already in use; using the MediaMTX already running.", self._rtmp_port)
            return
        binary = find_mediamtx_binary(self._explicit_path)
        if binary is None:
            logger.warning(
                "MediaMTX binary '%s' not found (repo root, ./mediamtx/, PATH%s); RTMP ingest disabled. "
                "Download it from https://github.com/bluenviron/mediamtx/releases.",
                BINARY_NAME, f", {self._explicit_path}" if self._explicit_path else "",
            )
            return

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "ab")
        kwargs: dict = {}
        if IS_WINDOWS:
            # No console window, and no inherited Ctrl+C: shutdown is ours.
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        # cwd = the binary's folder so MediaMTX picks up the mediamtx.yml shipped next to it.
        self._process = subprocess.Popen(
            [str(binary)],
            cwd=str(binary.parent),
            stdin=subprocess.DEVNULL,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            **kwargs,
        )
        logger.info("Started MediaMTX (pid %d) from %s; log: %s", self._process.pid, binary, self.log_path)

    def check_started(self) -> None:
        """Log if the child died right after launch (bad config, port clash)."""
        if self._process is not None and self._process.poll() is not None:
            logger.error(
                "MediaMTX exited immediately with code %s; see %s", self._process.returncode, self.log_path
            )

    def stop(self, timeout: float = 5.0) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("MediaMTX did not exit within %.0fs; killing it.", timeout)
                process.kill()
                process.wait(timeout=timeout)
            logger.info("MediaMTX stopped.")
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

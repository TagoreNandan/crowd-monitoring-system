"""Upload handling helpers shared by the job routers.

Videos are streamed to a temp file in chunks rather than read fully into
memory -- required so a large upload doesn't blow up process memory. Images
are small enough that the routers read them fully (bounded by
`max_image_upload_mb`) and hand raw bytes to `decode_image_bytes` directly.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

from .errors import InvalidUploadError

UPLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "crowd_monitoring_backend_uploads"


def decode_image_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise InvalidUploadError("Could not decode the uploaded file as an image.")
    return frame


async def save_upload_to_tempfile(upload: UploadFile, max_bytes: int, chunk_size: int = 1024 * 1024) -> Path:
    """Stream `upload` to a new temp file under `UPLOAD_TMP_DIR`, never
    holding more than one chunk in memory. Raises `InvalidUploadError` (and
    removes the partial file) if it's empty or exceeds `max_bytes`."""
    UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "").suffix or ".bin"
    fd, tmp_name = tempfile.mkstemp(prefix="crowd_monitoring_upload_", suffix=suffix, dir=str(UPLOAD_TMP_DIR))
    tmp_path = Path(tmp_name)
    written = 0
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise InvalidUploadError(
                        f"Uploaded file exceeds the maximum allowed size ({max_bytes / (1024 * 1024):.0f} MB)."
                    )
                f.write(chunk)
    except InvalidUploadError:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise InvalidUploadError("Uploaded file is empty.")
    return tmp_path

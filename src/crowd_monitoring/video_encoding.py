"""Browser-compatible video encoding utilities.

Converts generated OpenCV videos (which often default to mp4v) to standard
HTML5/browser-compatible H.264 (libx264, yuv420p, +faststart) using ffmpeg when
available. Falls back gracefully without blocking or crashing job execution if
ffmpeg is missing or fails.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import tempfile
import cv2

logger = logging.getLogger(__name__)

_CACHED_FOURCC: Optional[int] = None


def get_preferred_video_fourcc() -> int:
    """Return the best available fourcc for browser playback (avc1/H264 preferred, mp4v fallback)."""
    global _CACHED_FOURCC
    if _CACHED_FOURCC is not None:
        return _CACHED_FOURCC

    for code in ("avc1", "H264", "X264", "mp4v"):
        try:
            fourcc = cv2.VideoWriter.fourcc(*code)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            writer = cv2.VideoWriter(tmp_path, fourcc, 10.0, (64, 64))
            opened = writer.isOpened()
            writer.release()
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
            if opened:
                logger.info("Selected video fourcc codec: %s", code)
                _CACHED_FOURCC = fourcc
                return fourcc
        except Exception:
            continue

    _CACHED_FOURCC = cv2.VideoWriter.fourcc(*"mp4v")
    return _CACHED_FOURCC


def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ensure_browser_compatible_video(
    input_path: Path,
    output_path: Optional[Path] = None,
    overwrite: bool = True,
) -> Path:
    """Ensure an MP4 video file uses browser-compatible H.264 / YUV420p codec.

    If `ffmpeg` is present on the system, runs:
        ffmpeg -y -i input_path -c:v libx264 -pix_fmt yuv420p -movflags +faststart output_path

    If `output_path` is not specified, transcodes in-place via a temporary file.
    If `ffmpeg` is not installed or transcoding fails, returns the original `input_path`
    so job execution is never blocked.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        logger.warning("Input video for browser conversion does not exist: %s", input_path)
        return input_path

    if not is_ffmpeg_available():
        logger.info("ffmpeg CLI not available; returning raw output video: %s", input_path)
        return input_path

    target_path = Path(output_path) if output_path is not None else input_path.with_name(f"{input_path.stem}_h264.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target_path),
    ]

    try:
        logger.info("Transcoding video to H.264/yuv420p via ffmpeg: %s -> %s", input_path.name, target_path.name)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        if res.returncode == 0 and target_path.exists() and target_path.stat().st_size > 0:
            if output_path is None:
                # Replace original file with the transcoded file
                target_path.replace(input_path)
                return input_path
            return target_path

        logger.warning("ffmpeg transcoding returned exit code %d: %s", res.returncode, res.stderr[-200:])
    except Exception as exc:
        logger.warning("ffmpeg transcoding failed: %s", exc)

    return input_path

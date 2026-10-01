"""Local-then-upload recording of a running live feed session.

One ``LiveFeedRecorder`` is created per feed session when S3 storage is
enabled. Detection-view frames are written to a local MP4, while heatmap and
combined views are saved as timestamped JPEG snapshots. Finalization uploads
completed files to S3 and removes local copies only after successful uploads.

Recording is best-effort and must never take down a live feed session: every
public method swallows and logs its own exceptions.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2

from crowd_monitoring.video_encoding import ensure_browser_compatible_video

from . import storage
from .settings import get_settings

logger = logging.getLogger(__name__)

_RECORDING_DIR = Path("data/live_recordings")


class LiveFeedRecorder:
    def __init__(self, feed_id: str):
        self.feed_id = feed_id
        self.session_uid = f"{feed_id}_{uuid.uuid4().hex[:8]}"
        self._writer: Optional[cv2.VideoWriter] = None
        self._local_path: Optional[Path] = None
        self._frame_count = 0
        self._start_wall = time.time()
        self._finalized = False
        self._lock = threading.Lock()
        self.s3_key: Optional[str] = None
        self.download_url: Optional[str] = None
        _RECORDING_DIR.mkdir(parents=True, exist_ok=True)

    def write_detection_frame(self, frame_bgr) -> None:
        """Append one detection-view frame, opening the writer lazily."""
        with self._lock:
            if self._finalized:
                return
            try:
                if self._writer is None:
                    h, w = frame_bgr.shape[:2]
                    self._local_path = _RECORDING_DIR / f"{self.session_uid}_detection.mp4"
                    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                    fps = get_settings().live_feed_recording_fps
                    self._writer = cv2.VideoWriter(str(self._local_path), fourcc, fps, (w, h))
                    logger.info("Started live recording for feed %s at %s", self.feed_id, self._local_path)
                self._writer.write(frame_bgr)
                self._frame_count += 1
            except Exception:
                logger.exception("Failed writing detection frame for feed %s recording", self.feed_id)

    def save_snapshot(self, frame_bgr, view: str) -> None:
        """Save one heatmap/combined frame as a timestamped local JPEG."""
        if self._finalized:
            return
        try:
            snap_dir = _RECORDING_DIR / self.session_uid / view
            snap_dir.mkdir(parents=True, exist_ok=True)
            ts_ms = int((time.time() - self._start_wall) * 1000)
            path = snap_dir / f"{ts_ms:010d}.jpg"
            cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception:
            logger.exception("Failed saving %s snapshot for feed %s recording", view, self.feed_id)

    def finalize(self) -> dict[str, Any]:
        """Close the writer, upload finished files, and return summary."""
        with self._lock:
            if self._finalized:
                return self.get_summary()
            self._finalized = True
            try:
                if self._writer is not None:
                    self._writer.release()
                    self._writer = None
            except Exception:
                logger.exception("Error releasing VideoWriter for feed %s", self.feed_id)

        # Transcode local detection video to browser-compatible H.264/yuv420p
        if self._local_path is not None and self._local_path.exists() and self._frame_count > 0:
            try:
                ensure_browser_compatible_video(self._local_path)
            except Exception:
                logger.exception("Failed to transcode live recording %s to H.264", self._local_path)

        if not storage.is_s3_enabled():
            if self._local_path is not None and self._local_path.exists():
                self.download_url = f"/api/feeds/recordings/{self.session_uid}/video"
            return self.get_summary()

        if self._local_path is not None and self._local_path.exists() and self._frame_count > 0:
            key = f"live_feeds/{self.session_uid}/detection.mp4"
            try:
                storage.upload_file(self._local_path, key)
                self._local_path.unlink(missing_ok=True)
                self.s3_key = key
                self.download_url = storage.presigned_url(key)
                logger.info(
                    "Uploaded live recording for feed %s -> %s (%d frames)",
                    self.feed_id, key, self._frame_count,
                )
            except Exception:
                logger.exception(
                    "Failed to upload live recording for feed %s -- local file kept at %s",
                    self.feed_id, self._local_path,
                )

        snap_root = _RECORDING_DIR / self.session_uid
        if snap_root.exists():
            for view_dir in snap_root.iterdir():
                if not view_dir.is_dir():
                    continue
                for jpg_path in sorted(view_dir.glob("*.jpg")):
                    key = f"live_feeds/{self.session_uid}/{view_dir.name}/{jpg_path.name}"
                    try:
                        storage.upload_file(jpg_path, key)
                        jpg_path.unlink(missing_ok=True)
                    except Exception:
                        logger.exception(
                            "Failed to upload snapshot %s for feed %s -- kept locally",
                            jpg_path, self.feed_id,
                        )
            _cleanup_empty_dirs(snap_root)

        return self.get_summary()

    def get_summary(self) -> dict[str, Any]:
        duration_s = round(time.time() - self._start_wall, 2)
        url = self.download_url
        if not url and self.s3_key and storage.is_s3_enabled():
            try:
                url = storage.presigned_url(self.s3_key)
            except Exception:
                pass
        if not url and self._local_path and self._local_path.exists():
            url = f"/api/feeds/recordings/{self.session_uid}/video"

        return {
            "feed_id": self.feed_id,
            "session_uid": self.session_uid,
            "frame_count": self._frame_count,
            "duration_s": duration_s,
            "s3_key": self.s3_key,
            "download_url": url,
            "is_finalized": self._finalized,
        }


def _cleanup_empty_dirs(root: Path) -> None:
    try:
        for view_dir in root.iterdir():
            if view_dir.is_dir() and not any(view_dir.iterdir()):
                view_dir.rmdir()
        if root.exists() and not any(root.iterdir()):
            root.rmdir()
    except Exception:
        pass


def recover_orphaned_recordings() -> None:
    """Upload local live-feed files left by a previous crashed process."""
    if not storage.is_s3_enabled() or not _RECORDING_DIR.exists():
        return

    for mp4_path in _RECORDING_DIR.glob("*_detection.mp4"):
        session_uid = mp4_path.stem.removesuffix("_detection")
        key = f"live_feeds/{session_uid}/detection.mp4"
        try:
            ensure_browser_compatible_video(mp4_path)
            storage.upload_file(mp4_path, key)
            mp4_path.unlink(missing_ok=True)
            logger.info("Recovered orphaned live recording %s -> %s", mp4_path, key)
        except Exception:
            logger.exception("Failed to recover orphaned recording %s -- will retry next startup", mp4_path)

    for session_dir in _RECORDING_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        for view_dir in session_dir.iterdir():
            if not view_dir.is_dir():
                continue
            for jpg_path in view_dir.glob("*.jpg"):
                key = f"live_feeds/{session_dir.name}/{view_dir.name}/{jpg_path.name}"
                try:
                    storage.upload_file(jpg_path, key)
                    jpg_path.unlink(missing_ok=True)
                except Exception:
                    logger.exception("Failed to recover orphaned snapshot %s -- will retry next startup", jpg_path)
        _cleanup_empty_dirs(session_dir)

"""Automatic primary -> backup failover for live (drone) feeds.

`FailoverFrameSource` wraps a live primary source (normally a reconnecting
`RTSPSource`) together with a factory for a backup source (an archived
video, typically the S3 standby clip). It presents one uninterrupted
`frames()` stream to `RealtimeFeedSession`'s producer thread:

    PRIMARY READER THREAD                     frames() (producer thread)
      primary.frames() --set()--> LatestSlot --get()--> fresh primary frame?  -> yield (LIVE)
      (own reconnect loop,                               stalled > failover_after_s? -> yield backup frame
       retries forever)                                                               (BACKUP_ACTIVE)

The primary is read on its own thread so a blocking/stalled `read()` never
freezes delivery: while the drone is down the reader keeps retrying in the
background, and once it has delivered frames steadily for `recover_after_s`
the stream switches back to LIVE on its own. `set_mode("backup")` lets an
operator force the backup regardless of primary health.

Every yielded frame is re-stamped (monotonic frame numbers, wall-clock
timestamps) and letterboxed to one fixed output size, so downstream
consumers -- the tracker, the recorder's fixed-size VideoWriter, the WS
frame stream -- never see a discontinuity when the source switches.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .frame_source import Frame, FrameSource, SourceMetadata
from .latest_slot import LatestSlot

logger = logging.getLogger(__name__)

SOURCE_STATUS_LIVE = "LIVE"
SOURCE_STATUS_BACKUP = "BACKUP_ACTIVE"
SOURCE_STATUS_SIGNAL_LOST = "SIGNAL_LOST"  # primary down and backup unavailable

MODE_AUTO = "auto"
MODE_BACKUP = "backup"

StatusSink = Callable[[str, str], None]  # (new_status, reason)


def _letterbox(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w == width and h == height:
        return frame_bgr
    scale = min(width / float(w), height / float(h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((height, width, 3), dtype=frame_bgr.dtype)
    x0, y0 = (width - new_w) // 2, (height - new_h) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


class FailoverFrameSource(FrameSource):
    def __init__(
        self,
        primary: FrameSource,
        backup_factory: Callable[[], FrameSource],
        *,
        failover_after_s: float = 0.8,
        recover_after_s: float = 1.0,
        backup_fps: float = 25.0,
        primary_retry_max_s: float = 5.0,
        on_status_change: Optional[StatusSink] = None,
    ):
        self._primary = primary
        self._backup_factory = backup_factory
        self._failover_after_s = failover_after_s
        self._recover_after_s = recover_after_s
        self._default_backup_fps = backup_fps
        self._primary_retry_max_s = primary_retry_max_s
        self._on_status_change = on_status_change

        self._stop = threading.Event()
        self._primary_slot: LatestSlot[Frame] = LatestSlot()
        self._primary_thread: Optional[threading.Thread] = None
        self._primary_opened = False
        self._primary_last_frame_at = 0.0

        self._backup: Optional[FrameSource] = None
        self._backup_iter: Optional[Iterator[Frame]] = None
        self._backup_fps = backup_fps
        self._backup_retry_at = 0.0

        self._metadata: Optional[SourceMetadata] = None
        self._out_size: Optional[tuple[int, int]] = None
        self._status_lock = threading.Lock()
        self.source_status = SOURCE_STATUS_LIVE
        self.mode = MODE_AUTO
        self.failover_count = 0
        self.status_changed_at: Optional[float] = None

    @property
    def metadata(self) -> Optional[SourceMetadata]:
        return self._metadata

    # -- lifecycle --------------------------------------------------------

    def open(self) -> SourceMetadata:
        if self._metadata is not None:
            return self._metadata
        self._stop.clear()

        primary_meta: Optional[SourceMetadata] = None
        try:
            primary_meta = self._primary.open()
            self._primary_opened = True
            self._primary_last_frame_at = time.monotonic()
        except Exception as exc:
            # Don't fail the feed: start on the backup straight away and let
            # the reader thread keep retrying the drone in the background.
            logger.warning("Primary source unavailable at start (%s); starting on backup", exc)
            self._primary_last_frame_at = time.monotonic() - self._failover_after_s - 1.0

        if primary_meta is not None and primary_meta.width and primary_meta.height:
            self._out_size = (primary_meta.width, primary_meta.height)

        self._metadata = SourceMetadata(
            description=primary_meta.description if primary_meta else "failover:primary-unavailable",
            width=self._out_size[0] if self._out_size else None,
            height=self._out_size[1] if self._out_size else None,
            fps=primary_meta.fps if primary_meta else None,
            total_frames=None,
            is_live=True,
        )

        self._primary_thread = threading.Thread(
            target=self._run_primary_reader, name="crowd-monitoring-failover-primary", daemon=True
        )
        self._primary_thread.start()
        return self._metadata

    def close(self) -> None:
        self._stop.set()
        self._primary_slot.close()
        try:
            self._primary.close()
        except Exception:
            logger.exception("Error closing primary source")
        if self._primary_thread is not None and self._primary_thread.is_alive():
            self._primary_thread.join(timeout=2.0)
        self._close_backup()

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_AUTO, MODE_BACKUP):
            raise ValueError(f"Unsupported failover mode '{mode}'")
        logger.info("Failover mode set to %s", mode)
        self.mode = mode

    # -- primary reader (own thread, retries forever) ----------------------

    def _run_primary_reader(self) -> None:
        try:
            self._read_primary_forever()
        finally:
            # Covers a reopen that raced with close(): never leak the handle.
            try:
                self._primary.close()
            except Exception:
                pass

    def _read_primary_forever(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            if not self._primary_opened:
                try:
                    self._primary.open()
                    self._primary_opened = True
                    backoff = 1.0
                    logger.info("Primary source reachable again")
                except Exception as exc:
                    logger.debug("Primary still unavailable: %s", exc)
                    if self._stop.wait(backoff):
                        return
                    backoff = min(backoff * 2, self._primary_retry_max_s)
                    continue
            try:
                for frame in self._primary.frames():
                    if self._stop.is_set():
                        return
                    self._primary_last_frame_at = time.monotonic()
                    self._primary_slot.set(frame)
            except Exception:
                logger.exception("Primary source read error")
            # frames() only returns on stop, on EOF of finite media, or on an
            # error: drop the handle and go back to retrying the open.
            if self._stop.is_set():
                return
            try:
                self._primary.close()
            except Exception:
                pass
            self._primary_opened = False
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, self._primary_retry_max_s)

    # -- backup -----------------------------------------------------------

    def _close_backup(self) -> None:
        if self._backup is not None:
            try:
                self._backup.close()
            except Exception:
                logger.exception("Error closing backup source")
        self._backup = None
        self._backup_iter = None

    def _next_backup_frame(self) -> Optional[Frame]:
        """Next backup frame, looping the archive at EOF. None if the backup
        can't be opened right now (retried every few seconds)."""
        for _ in range(2):  # at most one EOF -> reopen per call
            if self._backup_iter is None:
                if time.monotonic() < self._backup_retry_at:
                    return None
                try:
                    self._backup = self._backup_factory()
                    meta = self._backup.open()
                    self._backup_fps = meta.fps or self._default_backup_fps
                    self._backup_iter = iter(self._backup.frames())
                except Exception as exc:
                    logger.error("Backup source unavailable: %s", exc)
                    self._close_backup()
                    self._backup_retry_at = time.monotonic() + self._primary_retry_max_s
                    return None
            try:
                return next(self._backup_iter)
            except StopIteration:
                self._close_backup()
            except Exception:
                logger.exception("Backup source read error")
                self._close_backup()
                self._backup_retry_at = time.monotonic() + 1.0
                return None
        return None

    # -- status -----------------------------------------------------------

    def _set_status(self, status: str, reason: str) -> None:
        with self._status_lock:
            if status == self.source_status:
                return
            previous = self.source_status
            self.source_status = status
            self.status_changed_at = time.time()
            if status == SOURCE_STATUS_BACKUP:
                self.failover_count += 1
        log = logger.info if status == SOURCE_STATUS_LIVE else logger.warning
        log("Feed source status %s -> %s (%s)", previous, status, reason)
        if self._on_status_change is not None:
            try:
                self._on_status_change(status, reason)
            except Exception:
                logger.exception("Error in failover on_status_change callback")

    # -- frame stream -----------------------------------------------------

    def _restamp(self, frame: Frame, frame_number: int, wall_start: float, fps: Optional[float]) -> Frame:
        if self._out_size is None:
            self._out_size = (frame.width, frame.height)
        w, h = self._out_size
        bgr = _letterbox(frame.frame_bgr, w, h)
        return Frame(
            frame_bgr=bgr,
            frame_number=frame_number,
            timestamp_s=time.monotonic() - wall_start,
            width=w,
            height=h,
            source_fps=fps,
        )

    def frames(self) -> Iterator[Frame]:
        if self._metadata is None:
            raise RuntimeError("FailoverFrameSource.frames() called before open()")
        wall_start = time.monotonic()
        frame_number = 0
        last_version = 0
        recovering_since: Optional[float] = None
        last_primary_seen: Optional[float] = None
        next_backup_due = 0.0

        while not self._stop.is_set():
            on_backup = self.source_status != SOURCE_STATUS_LIVE
            # While on backup, only peek at the primary so backup pacing isn't
            # throttled by waiting on a dead link.
            frame, last_version = self._primary_slot.get(last_version, timeout=0.0 if on_backup else 0.1)
            if frame is None and self._primary_slot.closed:
                return
            now = time.monotonic()

            if frame is not None and self.mode == MODE_AUTO:
                if not on_backup:
                    yield self._restamp(frame, frame_number, wall_start, frame.source_fps)
                    frame_number += 1
                    continue
                # Primary is back: require a steady run before switching, so a
                # single stray frame from a flapping link doesn't bounce us.
                if recovering_since is None or (last_primary_seen and now - last_primary_seen > 1.0):
                    recovering_since = now
                last_primary_seen = now
                if now - recovering_since >= self._recover_after_s:
                    self._close_backup()
                    recovering_since = None
                    self._set_status(SOURCE_STATUS_LIVE, "primary stream recovered")
                    yield self._restamp(frame, frame_number, wall_start, frame.source_fps)
                    frame_number += 1
                    continue

            stalled = now - self._primary_last_frame_at > self._failover_after_s
            if self.mode == MODE_AUTO and not on_backup and not stalled:
                continue  # brief gap, still within the failover grace period

            if not on_backup:
                reason = "manual override" if self.mode == MODE_BACKUP else (
                    f"no primary frames for {now - self._primary_last_frame_at:.1f}s"
                )
                self._set_status(SOURCE_STATUS_BACKUP, reason)
                next_backup_due = now

            if now < next_backup_due:
                self._stop.wait(min(next_backup_due - now, 0.05))
                continue

            backup_frame = self._next_backup_frame()
            if backup_frame is None:
                self._set_status(SOURCE_STATUS_SIGNAL_LOST, "primary down and backup unavailable")
                self._stop.wait(0.25)
                continue
            if self.source_status == SOURCE_STATUS_SIGNAL_LOST:
                self._set_status(SOURCE_STATUS_BACKUP, "backup source available")
            next_backup_due = max(now, next_backup_due) + 1.0 / max(1.0, self._backup_fps)
            yield self._restamp(backup_frame, frame_number, wall_start, self._backup_fps)
            frame_number += 1

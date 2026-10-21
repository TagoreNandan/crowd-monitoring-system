"""Frame-source abstraction: a decoded-frame stream plus source metadata.

`FrameSource` is the shape shared by every place frames come from --
uploaded/local video today (`FileVideoSource`), and camera/RTSP/RTMP/WebRTC
later -- so the buffer, inference worker, and pipeline never need to know
which kind of source produced a given frame.

Frames are always produced one at a time via a generator: nothing here reads
an entire file/stream into memory. `SourceMetadata` fields that a source
can't determine (e.g. total frame count for a live camera) are `None`
rather than a guessed value.
"""

from __future__ import annotations

import abc
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceMetadata:
    """Static/best-effort properties of a frame source, known at open time.

    `fps` and `total_frames` are `None` when the source can't report them
    (a live stream has no `total_frames`; some backends/codecs don't report
    `fps`) -- callers must treat that as genuinely unknown, never substitute
    a hardcoded fallback here.
    """

    description: str
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    total_frames: Optional[int]
    is_live: bool = False


@dataclass
class Frame:
    """One decoded frame plus the bookkeeping consumers need, independent of
    which `FrameSource` produced it."""

    frame_bgr: np.ndarray
    frame_number: int  # 0-based, monotonically increasing within this source's lifetime
    timestamp_s: float  # elapsed time since the source started, in seconds
    width: int
    height: int
    source_fps: Optional[float] = None


class FrameSource(abc.ABC):
    """Base contract for anything that can be opened, iterated for frames,
    and closed. `FileVideoSource` below is the only concrete implementation
    today; a future `CameraSource`/`RTSPSource` implements the same three
    methods and drops in anywhere a `FrameSource` is expected."""

    @abc.abstractmethod
    def open(self) -> SourceMetadata:
        """Open the underlying source and return what's known about it.
        Must be called before `frames()`."""
        raise NotImplementedError

    @abc.abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Yield frames one at a time. Must not buffer the whole source in
        memory -- decode-and-yield, not decode-all-then-return."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Release any underlying handle. Safe to call more than once."""
        raise NotImplementedError

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class FileVideoSource(FrameSource):
    """Reads a local/uploaded video file frame-by-frame via
    `cv2.VideoCapture`. Frames are decoded one at a time as `frames()` is
    iterated -- the file is never loaded into memory as a whole, so this
    class imposes no resolution/duration/size ceiling of its own.

    With `loop=True` the file never ends: at the last frame it seeks back to
    frame 0 (`CAP_PROP_POS_FRAMES`) and keeps playing, with frame numbers and
    timestamps continuing monotonically so real-time pacing stays correct.
    """

    def __init__(self, path: str | Path, loop: bool = False):
        self._path = Path(path)
        self.loop = loop
        self.loop_count = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[SourceMetadata] = None

    @property
    def metadata(self) -> Optional[SourceMetadata]:
        return self._metadata

    def open(self) -> SourceMetadata:
        if self._cap is not None:
            assert self._metadata is not None
            return self._metadata

        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self._path}")
        self._cap = cap

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if raw_fps and raw_fps > 0 else None

        raw_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(raw_total) if raw_total and raw_total > 0 else None

        raw_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        raw_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = int(raw_width) if raw_width and raw_width > 0 else None
        height = int(raw_height) if raw_height and raw_height > 0 else None

        self._metadata = SourceMetadata(
            description=f"video:{self._path.name}",
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            is_live=False,
        )
        return self._metadata

    def frames(self) -> Iterator[Frame]:
        if self._cap is None:
            raise RuntimeError("FileVideoSource.frames() called before open()")
        cap = self._cap
        fps = self._metadata.fps if self._metadata else None
        wall_start = time.monotonic()
        frame_number = 0
        loop_offset_s = 0.0
        last_timestamp_s = 0.0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                if not self.loop or frame_number == 0:
                    return
                cap = self._rewind(cap)
                if cap is None:
                    return
                loop_offset_s = last_timestamp_s + (1.0 / fps if fps else 0.0)
                ok, frame_bgr = cap.read()
                if not ok:
                    return

            # Prefer the container's own playback position (correct even
            # with variable frame rate or dropped frames); fall back to a
            # frame-count/fps estimate, then wall-clock, when the backend
            # can't report it.
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms and pos_ms > 0:
                timestamp_s = loop_offset_s + pos_ms / 1000.0
            elif fps:
                timestamp_s = frame_number / fps
            else:
                timestamp_s = time.monotonic() - wall_start
            last_timestamp_s = timestamp_s

            h, w = frame_bgr.shape[:2]
            yield Frame(
                frame_bgr=frame_bgr,
                frame_number=frame_number,
                timestamp_s=timestamp_s,
                width=w,
                height=h,
                source_fps=fps,
            )
            frame_number += 1

    def _rewind(self, cap: cv2.VideoCapture) -> Optional[cv2.VideoCapture]:
        """Seek back to frame 0; reopen the file if the backend can't seek."""
        self.loop_count += 1
        if cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            return cap
        cap.release()
        cap = cv2.VideoCapture(str(self._path))
        self._cap = cap if cap.isOpened() else None
        return self._cap

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CameraSource(FrameSource):
    """Reads frames live from a local camera device (e.g. webcam via device index 0)."""

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[SourceMetadata] = None

    @property
    def metadata(self) -> Optional[SourceMetadata]:
        return self._metadata

    def open(self) -> SourceMetadata:
        if self._cap is not None:
            assert self._metadata is not None
            return self._metadata

        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera device index: {self._device_index}")
        self._cap = cap

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if raw_fps and raw_fps > 0 else 30.0

        raw_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        raw_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = int(raw_width) if raw_width and raw_width > 0 else None
        height = int(raw_height) if raw_height and raw_height > 0 else None

        self._metadata = SourceMetadata(
            description=f"camera:{self._device_index}",
            width=width,
            height=height,
            fps=fps,
            total_frames=None,
            is_live=True,
        )
        return self._metadata

    def frames(self) -> Iterator[Frame]:
        if self._cap is None:
            raise RuntimeError("CameraSource.frames() called before open()")
        cap = self._cap
        fps = self._metadata.fps if self._metadata else 30.0
        wall_start = time.monotonic()
        frame_number = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                return

            timestamp_s = time.monotonic() - wall_start
            h, w = frame_bgr.shape[:2]
            yield Frame(
                frame_bgr=frame_bgr,
                frame_number=frame_number,
                timestamp_s=timestamp_s,
                width=w,
                height=h,
                source_fps=fps,
            )
            frame_number += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


_LIVE_STREAM_SCHEMES = ("rtsp", "rtsps", "rtmp", "rtmps", "srt", "udp", "tcp", "rtp")

_FFMPEG_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"


def ensure_rtsp_tcp_transport() -> None:
    """Force RTSP over TCP for every OpenCV/FFmpeg capture in this process.

    OpenCV's FFmpeg backend (notably the Windows wheels) defaults RTSP to
    UDP, which drops packets heavily with several concurrent drone streams.
    Other operator-set options in the env var are kept; only a missing
    `rtsp_transport` is added. Non-RTSP inputs ignore the option, and OpenCV
    re-reads the variable on every capture open.
    """
    current = os.environ.get(_FFMPEG_OPTIONS_ENV, "").strip()
    if "rtsp_transport" in current:
        return
    os.environ[_FFMPEG_OPTIONS_ENV] = f"{current}|rtsp_transport;tcp" if current else "rtsp_transport;tcp"


ensure_rtsp_tcp_transport()


def is_live_stream_url(url: str) -> bool:
    """True for transport schemes that are inherently live (RTSP/RTMP/SRT/...).
    HTTP(S) URLs are ambiguous (an mp4 on S3 vs. an HLS playlist), so they are
    classified after opening, from whether the container reports a length."""
    scheme = url.split("://", 1)[0].lower() if "://" in url else ""
    return scheme in _LIVE_STREAM_SCHEMES


class RTSPSource(FrameSource):
    """Reads frames from an RTSP/RTMP/HTTP(S) stream URL via OpenCV.

    Live streams never end on a dropped or stalled read. `frames()` logs a
    disconnect warning, reconnects with exponential backoff (1s, 2s, 4s, ...
    capped at `max_backoff_s`) and resumes yielding with `frame_number`
    continuing where it left off, so the consuming producer thread never
    exits just because a drone link hiccupped. Finite media (e.g. a presigned
    S3 mp4) still ends at EOF so the caller's own loop/complete logic applies.

    With `loop=True`, finite media (the S3 backup clips) instead seek back
    to frame 0 at EOF and keep playing; if the HTTP backend can't seek, the
    URL is reopened. Either way `frames()` never ends until `close()`.

    FFmpeg open/read timeouts bound how long a single `read()` can block on a
    dead socket, and `last_frame_at`/`seconds_since_last_frame` expose link
    health to callers such as `FailoverFrameSource`.
    """

    def __init__(
        self,
        url: str,
        display_name: str | None = None,
        *,
        reconnect: Optional[bool] = None,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 5.0,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
        loop: bool = False,
    ):
        self._url = url
        self.loop = loop
        self.loop_count = 0
        self._display_name = display_name
        # None = decide after open(): reconnect for live streams, not for finite media.
        self._reconnect = reconnect
        self._initial_backoff_s = max(0.05, initial_backoff_s)
        self._max_backoff_s = max(self._initial_backoff_s, max_backoff_s)
        self._open_timeout_ms = open_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()
        self._stop = threading.Event()
        self._metadata: Optional[SourceMetadata] = None
        self._last_frame_at: Optional[float] = None
        self.connected = False
        self.reconnect_count = 0

    @property
    def metadata(self) -> Optional[SourceMetadata]:
        return self._metadata

    @property
    def url(self) -> str:
        return self._url

    @property
    def last_frame_at(self) -> Optional[float]:
        """`time.monotonic()` of the last successfully decoded frame (or of
        the last successful (re)connect, before the first frame arrives)."""
        return self._last_frame_at

    @property
    def seconds_since_last_frame(self) -> Optional[float]:
        return None if self._last_frame_at is None else time.monotonic() - self._last_frame_at

    @property
    def reconnect_enabled(self) -> bool:
        if self._reconnect is not None:
            return self._reconnect
        return bool(self._metadata and self._metadata.is_live)

    def _clean_url(self) -> str:
        return self._url.split("?")[0] if "?" in self._url else self._url

    def _open_capture(self) -> cv2.VideoCapture:
        if self._url.lower().startswith(("rtsp://", "rtsps://")):
            # Re-assert in case something cleared the env var since import.
            ensure_rtsp_tcp_transport()

        params: list[int] = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self._open_timeout_ms)]
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self._read_timeout_ms)]
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG, params) if params else cv2.VideoCapture(self._url)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open network stream: {self._clean_url()}")
        try:
            # Keep at most one decoded frame queued so we always read the newest.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def open(self) -> SourceMetadata:
        self._stop.clear()
        with self._cap_lock:
            if self._cap is not None:
                assert self._metadata is not None
                return self._metadata

            cap = self._open_capture()
            self._cap = cap
            self.connected = True
            self._last_frame_at = time.monotonic()

            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = float(raw_fps) if raw_fps and raw_fps > 0 else None

            raw_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            raw_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = int(raw_width) if raw_width and raw_width > 0 else None
            height = int(raw_height) if raw_height and raw_height > 0 else None

            raw_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_frames = int(raw_total) if raw_total and raw_total > 0 else None
            is_live = is_live_stream_url(self._url) or total_frames is None

            desc = self._display_name or f"rtsp:{self._clean_url()}"

            self._metadata = SourceMetadata(
                description=desc,
                width=width,
                height=height,
                fps=fps,
                total_frames=None if is_live else total_frames,
                is_live=is_live,
            )
            return self._metadata

    def _reopen(self) -> None:
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            if self._stop.is_set():
                return
            self._cap = self._open_capture()
            self.connected = True
            self._last_frame_at = time.monotonic()

    def frames(self) -> Iterator[Frame]:
        if self._cap is None:
            raise RuntimeError("RTSPSource.frames() called before open()")
        fps = self._metadata.fps if self._metadata else None
        desc = self._metadata.description if self._metadata else self._clean_url()
        wall_start = time.monotonic()
        frame_number = 0
        frames_since_rewind = 0

        while not self._stop.is_set():
            with self._cap_lock:
                cap = self._cap
                ok, frame_bgr = cap.read() if cap is not None else (False, None)

            if ok and frame_bgr is not None:
                self._last_frame_at = time.monotonic()
                timestamp_s = self._last_frame_at - wall_start
                h, w = frame_bgr.shape[:2]
                yield Frame(
                    frame_bgr=frame_bgr,
                    frame_number=frame_number,
                    timestamp_s=timestamp_s,
                    width=w,
                    height=h,
                    source_fps=fps,
                )
                frame_number += 1
                frames_since_rewind += 1
                continue

            if self._stop.is_set():
                return
            if not self.reconnect_enabled:
                if not self.loop or frame_number == 0:
                    return
                # Finite media at EOF: rewind and keep playing. A rewind that
                # yields no frames before the next EOF means seeking is
                # broken for this stream -- reopen instead of spinning.
                self.loop_count += 1
                rewound = False
                if frames_since_rewind > 0:
                    with self._cap_lock:
                        rewound = self._cap is not None and bool(self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0))
                frames_since_rewind = 0
                if not rewound:
                    try:
                        self._reopen()
                    except Exception as exc:
                        logger.warning("Could not restart looping media %s: %s", desc, exc)
                        if self._stop.wait(self._initial_backoff_s):
                            return
                continue

            self.connected = False
            since = self.seconds_since_last_frame or 0.0
            logger.warning(
                "Stream %s disconnected or stalled (%.1fs since last frame); reconnecting", desc, since
            )
            backoff = self._initial_backoff_s
            while not self._stop.is_set():
                if self._stop.wait(backoff):
                    return
                try:
                    self._reopen()
                except Exception as exc:
                    logger.warning(
                        "Reconnect to %s failed (%s); next attempt in %.1fs",
                        desc, exc, min(backoff * 2, self._max_backoff_s),
                    )
                    backoff = min(backoff * 2, self._max_backoff_s)
                    continue
                self.reconnect_count += 1
                logger.info("Stream %s reconnected (attempt #%d)", desc, self.reconnect_count)
                break

    def close(self) -> None:
        """Stop any in-progress reconnect loop and release the capture. If a
        `read()` is in flight on another thread, the release is left to
        `frames()`, which exits as soon as that read returns (bounded by the
        FFmpeg read timeout) -- releasing a capture mid-read can crash."""
        self._stop.set()
        acquired = self._cap_lock.acquire(timeout=(self._read_timeout_ms / 1000.0) + 1.0)
        if not acquired:
            return
        try:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self.connected = False
        finally:
            self._cap_lock.release()

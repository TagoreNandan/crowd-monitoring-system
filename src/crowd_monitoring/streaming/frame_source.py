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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


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
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
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

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                return

            # Prefer the container's own playback position (correct even
            # with variable frame rate or dropped frames); fall back to a
            # frame-count/fps estimate, then wall-clock, when the backend
            # can't report it.
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms and pos_ms > 0:
                timestamp_s = pos_ms / 1000.0
            elif fps:
                timestamp_s = frame_number / fps
            else:
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


class RTSPSource(FrameSource):
    """Reads frames live from an RTSP/RTMP stream URL via OpenCV."""

    def __init__(self, url: str):
        self._url = url
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[SourceMetadata] = None

    @property
    def metadata(self) -> Optional[SourceMetadata]:
        return self._metadata

    def open(self) -> SourceMetadata:
        if self._cap is not None:
            assert self._metadata is not None
            return self._metadata

        cap = cv2.VideoCapture(self._url)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open network stream: {self._url}")
        self._cap = cap

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if raw_fps and raw_fps > 0 else None

        raw_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        raw_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = int(raw_width) if raw_width and raw_width > 0 else None
        height = int(raw_height) if raw_height and raw_height > 0 else None

        self._metadata = SourceMetadata(
            description=f"rtsp:{self._url}",
            width=width,
            height=height,
            fps=fps,
            total_frames=None,
            is_live=True,
        )
        return self._metadata

    def frames(self) -> Iterator[Frame]:
        if self._cap is None:
            raise RuntimeError("RTSPSource.frames() called before open()")
        cap = self._cap
        fps = self._metadata.fps if self._metadata else None
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


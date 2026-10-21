"""RTSPSource auto-reconnect and FailoverFrameSource primary/backup switching.

These drive the real threading/timing code against fake sources (and a fake
cv2.VideoCapture) so they run offline in well under a few seconds."""

from __future__ import annotations

import os
import threading
import time
from typing import Iterator, List, Optional

import numpy as np
import pytest

from crowd_monitoring.streaming import frame_source as fs
from crowd_monitoring.streaming import (
    SOURCE_STATUS_BACKUP,
    SOURCE_STATUS_LIVE,
    SOURCE_STATUS_SIGNAL_LOST,
    FailoverFrameSource,
    Frame,
    FrameSource,
    RTSPSource,
    SourceMetadata,
)


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class _FakeCapture:
    def __init__(self, n_frames: Optional[int], opened: bool = True, frame_count: float = 0.0):
        self._remaining = n_frames
        self._opened = opened
        self._frame_count = frame_count
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self._remaining is not None:
            if self._remaining <= 0:
                return False, None
            self._remaining -= 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def get(self, prop):
        if prop == fs.cv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        if prop == fs.cv2.CAP_PROP_FPS:
            return 25.0
        return 8.0

    def set(self, *args):
        return True

    def release(self):
        self.released = True


class _ControllableSource(FrameSource):
    """Yields solid-colour frames while `alive` is set; stalls otherwise."""

    def __init__(self, value: int, size=(64, 48), fail_open: bool = False, fps: float = 50.0):
        self.value = value
        self.size = size
        self.alive = threading.Event()
        self.alive.set()
        self.fail_open = fail_open
        self.fps = fps
        self._closed = threading.Event()

    def open(self) -> SourceMetadata:
        if self.fail_open:
            raise RuntimeError("unreachable")
        self._closed.clear()
        return SourceMetadata("fake", self.size[0], self.size[1], self.fps, None, True)

    def frames(self) -> Iterator[Frame]:
        n = 0
        w, h = self.size
        while not self._closed.is_set():
            if self.alive.is_set():
                yield Frame(np.full((h, w, 3), self.value, np.uint8), n, n / self.fps, w, h, self.fps)
                n += 1
            time.sleep(1.0 / self.fps)

    def close(self) -> None:
        self._closed.set()


class _Collector:
    def __init__(self, source: FailoverFrameSource):
        self.source = source
        self.frames: List[Frame] = []
        self.statuses: List[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        for frame in self.source.frames():
            self.frames.append(frame)
            self.statuses.append(self.source.source_status)

    def start(self):
        self._thread.start()
        return self

    def wait_for(self, predicate, timeout=5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def stop(self):
        self.source.close()
        self._thread.join(timeout=3.0)


def _failover(primary, backup_factory, events=None, **kw):
    kw.setdefault("failover_after_s", 0.3)
    kw.setdefault("recover_after_s", 0.2)
    return FailoverFrameSource(
        primary,
        backup_factory,
        on_status_change=(lambda s, r: events.append(s)) if events is not None else None,
        **kw,
    )


# --------------------------------------------------------------------------
# RTSPSource
# --------------------------------------------------------------------------


def test_rtsp_source_reconnects_and_continues_frame_numbers(monkeypatch):
    caps = [_FakeCapture(3), _FakeCapture(0, opened=False), _FakeCapture(2)]
    opened: list = []

    def fake_capture(*args, **kwargs):
        cap = caps.pop(0)
        opened.append(cap)
        return cap

    monkeypatch.setattr(fs.cv2, "VideoCapture", fake_capture)
    source = RTSPSource("rtsp://drone.local/live", initial_backoff_s=0.05, max_backoff_s=0.1)
    meta = source.open()
    assert meta.is_live is True

    numbers = []
    for frame in source.frames():
        numbers.append(frame.frame_number)
        if len(numbers) == 5:
            break
    source.close()

    assert numbers == [0, 1, 2, 3, 4]
    assert source.reconnect_count == 1
    assert opened[0].released  # stale handle released before reconnecting
    assert source.last_frame_at is not None


def test_rtsp_source_sets_tcp_transport_for_rtsp(monkeypatch):
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    monkeypatch.setattr(fs.cv2, "VideoCapture", lambda *a, **k: _FakeCapture(1))
    source = RTSPSource("rtsp://drone.local/live")
    source.open()
    source.close()
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"


def test_rtsp_source_finite_http_media_ends_at_eof_without_reconnect(monkeypatch):
    monkeypatch.setattr(fs.cv2, "VideoCapture", lambda *a, **k: _FakeCapture(4, frame_count=4))
    source = RTSPSource("https://bucket.s3.amazonaws.com/standby.mp4?sig=x", initial_backoff_s=0.05)
    meta = source.open()
    assert meta.is_live is False
    assert len(list(source.frames())) == 4
    assert source.reconnect_count == 0
    source.close()


def test_rtsp_source_close_interrupts_reconnect_backoff(monkeypatch):
    caps = [_FakeCapture(1)] + [_FakeCapture(0, opened=False) for _ in range(50)]
    monkeypatch.setattr(fs.cv2, "VideoCapture", lambda *a, **k: caps.pop(0))
    source = RTSPSource("rtsp://drone.local/live", initial_backoff_s=0.5, max_backoff_s=5.0)
    source.open()
    done = threading.Event()

    def consume():
        for _ in source.frames():
            pass
        done.set()

    threading.Thread(target=consume, daemon=True).start()
    time.sleep(0.2)
    source.close()
    assert done.wait(2.0), "frames() kept retrying after close()"


# --------------------------------------------------------------------------
# FailoverFrameSource
# --------------------------------------------------------------------------


def test_failover_switches_to_backup_and_recovers_to_live():
    primary = _ControllableSource(value=200)
    backup = _ControllableSource(value=50)
    events: list = []
    src = _failover(primary, lambda: backup, events)
    src.open()
    c = _Collector(src).start()
    try:
        assert c.wait_for(lambda: len(c.frames) >= 5)
        assert src.source_status == SOURCE_STATUS_LIVE
        assert int(c.frames[-1].frame_bgr[0, 0, 0]) == 200

        primary.alive.clear()  # drone link drops
        assert c.wait_for(lambda: src.source_status == SOURCE_STATUS_BACKUP, timeout=3.0)
        n = len(c.frames)
        assert c.wait_for(lambda: len(c.frames) >= n + 5), "backup frames not delivered"
        assert int(c.frames[-1].frame_bgr[0, 0, 0]) == 50

        primary.alive.set()  # drone comes back
        assert c.wait_for(lambda: src.source_status == SOURCE_STATUS_LIVE, timeout=3.0)
        n = len(c.frames)
        assert c.wait_for(lambda: len(c.frames) >= n + 3)
        assert int(c.frames[-1].frame_bgr[0, 0, 0]) == 200
    finally:
        c.stop()

    assert events == [SOURCE_STATUS_BACKUP, SOURCE_STATUS_LIVE]
    numbers = [f.frame_number for f in c.frames]
    assert numbers == list(range(len(numbers))), "frame numbers must stay monotonic across switches"
    assert src.failover_count == 1


def test_failover_waits_for_grace_period_before_switching():
    primary = _ControllableSource(value=200)
    src = _failover(primary, lambda: _ControllableSource(value=50), failover_after_s=1.0)
    src.open()
    c = _Collector(src).start()
    try:
        assert c.wait_for(lambda: len(c.frames) >= 3)
        primary.alive.clear()
        time.sleep(0.5)
        assert src.source_status == SOURCE_STATUS_LIVE  # short stutter: no failover yet
        assert c.wait_for(lambda: src.source_status == SOURCE_STATUS_BACKUP, timeout=3.0)
    finally:
        c.stop()


def test_failover_starts_on_backup_when_primary_unreachable_at_open():
    primary = _ControllableSource(value=200, fail_open=True)
    src = _failover(primary, lambda: _ControllableSource(value=50))
    meta = src.open()
    assert meta.is_live is True
    c = _Collector(src).start()
    try:
        assert c.wait_for(lambda: len(c.frames) >= 3)
        assert src.source_status == SOURCE_STATUS_BACKUP
        assert int(c.frames[-1].frame_bgr[0, 0, 0]) == 50

        primary.fail_open = False  # background retry finds the drone
        assert c.wait_for(lambda: src.source_status == SOURCE_STATUS_LIVE, timeout=5.0)
    finally:
        c.stop()


def test_manual_backup_override_and_return_to_auto():
    primary = _ControllableSource(value=200)
    src = _failover(primary, lambda: _ControllableSource(value=50))
    src.open()
    c = _Collector(src).start()
    try:
        assert c.wait_for(lambda: len(c.frames) >= 3)
        src.set_mode("backup")
        assert c.wait_for(lambda: src.source_status == SOURCE_STATUS_BACKUP, timeout=2.0)
        time.sleep(0.5)
        assert src.source_status == SOURCE_STATUS_BACKUP  # healthy primary ignored while forced
        src.set_mode("auto")
        assert c.wait_for(lambda: src.source_status == SOURCE_STATUS_LIVE, timeout=2.0)
    finally:
        c.stop()
    with pytest.raises(ValueError):
        src.set_mode("bogus")


def test_backup_frames_are_letterboxed_to_primary_size():
    primary = _ControllableSource(value=200, size=(64, 48))
    backup = _ControllableSource(value=50, size=(100, 30))
    src = _failover(primary, lambda: backup)
    src.open()
    c = _Collector(src).start()
    try:
        primary.alive.clear()
        assert c.wait_for(lambda: src.source_status == SOURCE_STATUS_BACKUP, timeout=3.0)
        n = len(c.frames)
        assert c.wait_for(lambda: len(c.frames) >= n + 2)
    finally:
        c.stop()
    assert {(f.width, f.height) for f in c.frames} == {(64, 48)}
    assert {f.frame_bgr.shape for f in c.frames} == {(48, 64, 3)}


def test_signal_lost_when_primary_down_and_backup_unavailable():
    primary = _ControllableSource(value=200)

    def broken_backup():
        raise RuntimeError("S3 unavailable")

    events: list = []
    src = _failover(primary, broken_backup, events)
    src.open()
    c = _Collector(src).start()
    try:
        assert c.wait_for(lambda: len(c.frames) >= 2)
        primary.alive.clear()
        assert c.wait_for(lambda: src.source_status == SOURCE_STATUS_SIGNAL_LOST, timeout=3.0)
        primary.alive.set()
        assert c.wait_for(lambda: src.source_status == SOURCE_STATUS_LIVE, timeout=3.0)
    finally:
        c.stop()
    assert events == [SOURCE_STATUS_BACKUP, SOURCE_STATUS_SIGNAL_LOST, SOURCE_STATUS_LIVE]


# --------------------------------------------------------------------------
# looping backups + sub-second switch
# --------------------------------------------------------------------------


def test_file_video_source_loops_with_monotonic_numbers_and_timestamps(tmp_path):
    import cv2

    path = tmp_path / "backup.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"png "), 10.0, (32, 24))
    for i in range(5):
        writer.write(np.full((24, 32, 3), i * 40, np.uint8))
    writer.release()

    src = fs.FileVideoSource(path, loop=True)
    src.open()
    frames = []
    for frame in src.frames():
        frames.append(frame)
        if len(frames) == 13:
            break
    src.close()

    assert [f.frame_number for f in frames] == list(range(13))
    assert all(b.timestamp_s > a.timestamp_s for a, b in zip(frames, frames[1:]))
    assert src.loop_count == 2
    # content really restarted from the first frame
    assert int(frames[5].frame_bgr[0, 0, 0]) == int(frames[0].frame_bgr[0, 0, 0])


class _SeekableCapture(_FakeCapture):
    def __init__(self, n):
        super().__init__(n, frame_count=n)
        self._n = n
        self.seeks = 0

    def set(self, prop, value):
        if prop == fs.cv2.CAP_PROP_POS_FRAMES and value == 0:
            self._remaining = self._n
            self.seeks += 1
            return True
        return True


def test_rtsp_source_loops_finite_media_by_seeking_to_zero(monkeypatch):
    cap = _SeekableCapture(3)
    monkeypatch.setattr(fs.cv2, "VideoCapture", lambda *a, **k: cap)
    src = RTSPSource("https://bucket.s3.amazonaws.com/backup_1.mp4?sig=x", loop=True)
    assert src.open().is_live is False
    numbers = []
    for frame in src.frames():
        numbers.append(frame.frame_number)
        if len(numbers) == 10:
            break
    src.close()
    assert numbers == list(range(10))
    assert cap.seeks == 3


def test_failover_switches_within_one_second_at_default_threshold():
    primary = _ControllableSource(value=200)
    src = FailoverFrameSource(primary, lambda: _ControllableSource(value=50))  # library default 0.8s
    src.open()
    c = _Collector(src).start()
    try:
        assert c.wait_for(lambda: len(c.frames) >= 3)
        primary.alive.clear()
        t0 = time.monotonic()
        assert c.wait_for(lambda: src.source_status == SOURCE_STATUS_BACKUP, timeout=3.0)
        elapsed = time.monotonic() - t0
    finally:
        c.stop()
    assert 0.7 <= elapsed < 1.2, elapsed


def test_tcp_transport_is_added_without_dropping_operator_options(monkeypatch):
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "stimeout;5000000")
    fs.ensure_rtsp_tcp_transport()
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "stimeout;5000000|rtsp_transport;tcp"
    fs.ensure_rtsp_tcp_transport()  # idempotent
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"].count("rtsp_transport") == 1


def test_operator_chosen_rtsp_transport_is_respected(monkeypatch):
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp")
    fs.ensure_rtsp_tcp_transport()
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;udp"


@pytest.mark.parametrize(
    "path",
    [r"C:\Users\Founder\Downloads\video.mp4", "C:/Users/Founder/video.mp4", r"D:\drone.mp4", r"\\nas\share\v.mp4"],
)
def test_windows_paths_are_not_stream_urls(path):
    # A drive letter must never be read as a URL scheme ("c:").
    assert fs.is_live_stream_url(path) is False

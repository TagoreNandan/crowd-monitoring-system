"""Bounded, latest-frame-wins buffer between a frame producer (a
`FrameSource` being read) and a slower consumer (an inference worker).

A dense-crowd model runs far slower than most video frame rates, so an
unbounded producer/consumer queue would grow without limit and eventually
exhaust memory on a long or high-resolution video -- long before decoding
itself would. This buffer instead holds at most `maxsize` pending frames:
once full, `put()` drops the oldest pending frame to make room for the
newest one, so a slow consumer always works on the most recent frame
available rather than catching up through a backlog of stale ones.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class BufferStats:
    """Running counters for observability only -- never fed into any
    count/risk metric."""

    frames_put: int = 0
    frames_dropped: int = 0
    frames_consumed: int = 0


class LatestFrameBuffer(Generic[T]):
    """Thread-safe bounded buffer holding at most `maxsize` pending items.

    `put()` never blocks and never raises: once full, the oldest pending
    item is dropped to make room. `get()` blocks (optionally with a
    timeout) until an item is available or the buffer is closed.
    """

    def __init__(self, maxsize: int = 1):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._maxsize = maxsize
        self._queue: "queue.Queue[T]" = queue.Queue(maxsize=maxsize)
        self._closed_lock = threading.Lock()
        self._closed = False
        self.stats = BufferStats()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def closed(self) -> bool:
        with self._closed_lock:
            return self._closed

    def put(self, item: T) -> None:
        """Enqueue `item`; if the buffer is already full, drop the oldest
        pending item first ("latest frame wins"). No-op once closed."""
        if self.closed:
            return
        while True:
            try:
                self._queue.put_nowait(item)
                self.stats.frames_put += 1
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.stats.frames_dropped += 1
                except queue.Empty:
                    continue  # a concurrent get() won the race; retry the put

    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Return the next item, blocking up to `timeout` seconds (or
        indefinitely if `timeout` is None). Returns `None` if `timeout`
        elapses, or once the buffer is closed and drained -- callers should
        treat `None` as "stop" only after also checking `closed`."""
        poll_interval = 0.1
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            if self.closed and self._queue.empty():
                return None
            if deadline is None:
                wait = poll_interval
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                wait = min(poll_interval, remaining)
            try:
                item = self._queue.get(timeout=wait)
                self.stats.frames_consumed += 1
                return item
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                continue

    def close(self) -> None:
        """Signal shutdown: `put()` becomes a no-op and a blocked/future
        `get()` returns `None` once the buffer drains, instead of blocking
        forever. Safe to call more than once and from either producer or
        consumer side."""
        with self._closed_lock:
            self._closed = True

    def qsize(self) -> int:
        return self._queue.qsize()

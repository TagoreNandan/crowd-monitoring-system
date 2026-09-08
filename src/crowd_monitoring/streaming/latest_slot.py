"""Single-slot, multi-reader, thread-safe "latest value" primitive.

This is NOT a queue. `LatestFrameBuffer` (see `buffer.py`) already gives one
consumer bounded, drop-oldest semantics; `RealtimeFeedSession` (see
`realtime.py`) needs something stronger: **two independent consumers** (a
fast person-detection loop and a slow DM-Count loop) that must each see
"whatever frame is current right now" without either one starving, blocking,
or seeing the other's backlog.

A queue can't do this cleanly -- if two consumers pull from the same queue
they split the frames between them; if each gets its own queue, a slow
consumer still lets its own queue grow into a stale backlog. `LatestSlot`
solves both: `set()` always just overwrites the one slot (never blocks,
never grows), and `get()` blocks a caller only until a frame *newer than the
last one it already saw* appears, so a slow reader simply skips every frame
produced while it was busy -- exactly "drop stale frames, never build a
backlog," per reader, for free.
"""

from __future__ import annotations

import threading
import time
from typing import Generic, Optional, Tuple, TypeVar

T = TypeVar("T")


class LatestSlot(Generic[T]):
    """Holds at most one value. `set()` overwrites it and wakes any blocked
    readers. `get(last_version)` blocks until a value newer than
    `last_version` is set (or `timeout`/`close()`), then returns it together
    with its version number so the caller can pass that back in on the next
    call to only ever be woken by genuinely new values."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._value: Optional[T] = None
        self._version: int = 0
        self._closed = False

    def set(self, value: T) -> None:
        with self._cond:
            if self._closed:
                return
            self._value = value
            self._version += 1
            self._cond.notify_all()

    def get(self, last_version: int = 0, timeout: Optional[float] = None) -> Tuple[Optional[T], int]:
        """Returns `(value, version)`. `value` is `None` only if `timeout`
        elapsed or the slot closed with nothing newer available -- callers
        should check `closed` to distinguish "keep waiting" from "stop"."""
        with self._cond:
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._version <= last_version and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None, last_version
                self._cond.wait(timeout=remaining)
            if self._version <= last_version:
                return None, last_version
            return self._value, self._version

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

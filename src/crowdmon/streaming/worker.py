"""Inference-worker foundation: pulls frames from a `LatestFrameBuffer` and
runs each one through the existing, unmodified `CrowdAnalysisPipeline`,
reporting results through a plain callback ("event sink") rather than being
coupled to any transport.

This module deliberately knows nothing about FastAPI or WebSockets: a
caller (e.g. a future live-camera job, or the existing video job runner)
supplies `on_result`/`on_error` callbacks and decides what to do with each
event -- forward it over the existing WebSocket `ConnectionManager`, store
it, log it, etc. No inference logic, model, or analytics is duplicated
here; `_run` does nothing but `buffer.get()` -> `pipeline.analyze()` ->
callback.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from ..pipeline import CrowdAnalysisPipeline, FrameResult
from .buffer import LatestFrameBuffer
from .frame_source import Frame

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class WorkerResult:
    """One inference result plus the source frame it came from, and enough
    bookkeeping for a callback to build a metrics/event payload without
    re-deriving anything the pipeline or buffer already tracked."""

    frame: Frame
    result: FrameResult
    frames_processed: int
    frames_dropped: int


@dataclass
class WorkerError:
    frame: Optional[Frame]
    error: Exception


EventSink = Callable[[WorkerResult], None]
ErrorSink = Callable[[WorkerError], None]


class InferenceWorker(Generic[T]):
    """Consumes frames from a `LatestFrameBuffer` and analyzes each one with
    an existing `CrowdAnalysisPipeline` instance.

    Two ways to run it:
      - `start()`/`stop()` run the consume loop on a dedicated background
        thread, for a producer that feeds the buffer concurrently (e.g. a
        future live camera/RTSP source).
      - `run_until_empty()` runs the same loop synchronously on the calling
        thread until the buffer reports closed-and-drained -- useful for a
        producer that has already finished (e.g. a fully-read file) and for
        deterministic tests.

    Because the worker only ever holds the single frame it's currently
    analyzing (everything else lives in the bounded buffer, which drops
    stale frames itself), this never accumulates unbounded memory no matter
    how much faster the producer is than inference.
    """

    def __init__(
        self,
        pipeline: CrowdAnalysisPipeline,
        buffer: LatestFrameBuffer[Frame],
        on_result: EventSink,
        on_error: Optional[ErrorSink] = None,
        growth_ratio_fn: Optional[Callable[[], Optional[float]]] = None,
        get_timeout_s: float = 0.5,
    ):
        self._pipeline = pipeline
        self._buffer = buffer
        self._on_result = on_result
        self._on_error = on_error
        self._growth_ratio_fn = growth_ratio_fn
        self._get_timeout_s = get_timeout_s
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.frames_processed = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Begin consuming on a background thread. No-op if already running."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="crowdmon-inference-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Ask the background thread to stop and wait for it to exit. Does
        NOT close the buffer itself -- whichever side owns the buffer's
        lifecycle (typically the producer) decides when that happens."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if not self._thread.is_alive():
                self._thread = None

    def run_until_empty(self) -> None:
        """Synchronously drain the buffer on the calling thread until it is
        closed and empty. Intended for a producer that has already finished
        feeding the buffer (e.g. an already-fully-read file)."""
        self._run()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            frame = self._buffer.get(timeout=self._get_timeout_s)
            if frame is None:
                if self._buffer.closed:
                    return
                continue
            self._process(frame)

    def _process(self, frame: Frame) -> None:
        try:
            growth_ratio = self._growth_ratio_fn() if self._growth_ratio_fn is not None else None
            result = self._pipeline.analyze(frame.frame_bgr, growth_ratio=growth_ratio)
        except Exception as exc:
            logger.exception("Inference worker failed on frame %s", frame.frame_number)
            if self._on_error is not None:
                self._on_error(WorkerError(frame=frame, error=exc))
            return

        self.frames_processed += 1
        self._on_result(
            WorkerResult(
                frame=frame,
                result=result,
                frames_processed=self.frames_processed,
                frames_dropped=self._buffer.stats.frames_dropped,
            )
        )

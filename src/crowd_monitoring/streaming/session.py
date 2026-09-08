"""Streaming session orchestrator.

Connects:
  FrameSource -> LatestFrameBuffer -> InferenceWorker -> CrowdAnalysisPipeline -> Callbacks

Manages the background producer reading from `FrameSource`, pushing frames into
a bounded `LatestFrameBuffer`, and running `InferenceWorker` on a background thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from ..pipeline import CrowdAnalysisPipeline
from .buffer import LatestFrameBuffer
from .frame_source import FrameSource, SourceMetadata
from .worker import InferenceWorker, WorkerError, WorkerResult

logger = logging.getLogger(__name__)


class StreamSession:
    """Manages the full lifecycle of a streaming feed session.

    Handles concurrent producer loop, bounded frame queueing, inference worker,
    event callbacks, error handling, and clean shutdown.
    """

    def __init__(
        self,
        session_id: str,
        source: FrameSource,
        pipeline: CrowdAnalysisPipeline,
        on_result: Optional[Callable[[WorkerResult], None]] = None,
        on_error: Optional[Callable[[WorkerError], None]] = None,
        buffer_maxsize: int = 1,
        loop_source: bool = False,
    ):
        self.session_id = session_id
        self._source = source
        self._pipeline = pipeline
        self._on_result_cb = on_result
        self._on_error_cb = on_error
        self._buffer_maxsize = buffer_maxsize
        self._loop_source = loop_source

        self._buffer: Optional[LatestFrameBuffer] = None
        self._worker: Optional[InferenceWorker] = None
        self._producer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.status: str = "created"
        self.metadata: Optional[SourceMetadata] = None
        self.latest_result: Optional[WorkerResult] = None
        self.last_error: Optional[WorkerError] = None
        self.frames_produced: int = 0
        self.error_count: int = 0
        self.started_at: Optional[float] = None
        self.stopped_at: Optional[float] = None

    @property
    def is_running(self) -> bool:
        return self.status == "running" and self._worker is not None and self._worker.is_running

    def start(self) -> SourceMetadata:
        """Open source, initialize buffer and worker, start producer/consumer threads."""
        if self.status == "running":
            assert self.metadata is not None
            return self.metadata

        self._stop_event.clear()
        self.metadata = self._source.open()

        self._buffer = LatestFrameBuffer(maxsize=self._buffer_maxsize)
        self._worker = InferenceWorker(
            pipeline=self._pipeline,
            buffer=self._buffer,
            on_result=self._handle_worker_result,
            on_error=self._handle_worker_error,
        )

        self._worker.start()
        self._producer_thread = threading.Thread(
            target=self._run_producer,
            name=f"crowd-monitoring-producer-{self.session_id}",
            daemon=True,
        )
        self.started_at = time.time()
        self.status = "running"
        self._producer_thread.start()

        logger.info("StreamSession %s started (source=%s)", self.session_id, self.metadata.description)
        return self.metadata

    def stop(self, timeout: float = 3.0) -> None:
        """Signal shutdown, wait for producer and worker to terminate, close resources."""
        if self.status in ("stopped", "completed", "failed"):
            return

        logger.info("Stopping StreamSession %s...", self.session_id)
        self._stop_event.set()

        if self._buffer is not None:
            self._buffer.close()

        if self._worker is not None:
            self._worker.stop(timeout=timeout)

        if self._producer_thread is not None and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=timeout)

        try:
            self._source.close()
        except Exception:
            logger.exception("Error closing source for session %s", self.session_id)

        self.stopped_at = time.time()
        if self.status == "running":
            self.status = "stopped"

        logger.info("StreamSession %s stopped cleanly.", self.session_id)

    def _handle_worker_result(self, result: WorkerResult) -> None:
        self.latest_result = result
        if self._on_result_cb is not None:
            try:
                self._on_result_cb(result)
            except Exception:
                logger.exception("Error in StreamSession on_result callback for session %s", self.session_id)

    def _handle_worker_error(self, err: WorkerError) -> None:
        self.error_count += 1
        self.last_error = err
        if self._on_error_cb is not None:
            try:
                self._on_error_cb(err)
            except Exception:
                logger.exception("Error in StreamSession on_error callback for session %s", self.session_id)

    def _run_producer(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    for frame in self._source.frames():
                        if self._stop_event.is_set():
                            break
                        if self._buffer is not None:
                            self._buffer.put(frame)
                        self.frames_produced += 1

                    if not self._loop_source or self._stop_event.is_set():
                        break

                    # Re-open if looping
                    self._source.close()
                    self._source.open()
                except Exception:
                    logger.exception("Producer loop error in StreamSession %s", self.session_id)
                    self.error_count += 1
                    break

            if not self._stop_event.is_set() and self.status == "running":
                self.status = "completed"
        finally:
            if self._buffer is not None:
                self._buffer.close()

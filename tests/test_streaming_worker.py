"""Inference-worker tests.

Most tests here use a lightweight fake pipeline (same style as
`backend/tests/test_jobs_video.py`'s `_fake_frame_result`) so they run fast
and don't require downloading model weights. One test
(`test_worker_invokes_real_crowd_analysis_pipeline`) uses the real,
session-scoped `pipeline` fixture from `tests/conftest.py` to prove the
worker genuinely drives `CrowdAnalysisPipeline.analyze()` and not a
duplicated stand-in -- it's skipped if the pretrained weights can't be
obtained (offline runner), same as the other real-pipeline tests in this
suite.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from crowd_monitoring.streaming import Frame, InferenceWorker, LatestFrameBuffer, WorkerResult


class _FakePipeline:
    """Stands in for `CrowdAnalysisPipeline`: records every frame it was
    asked to analyze and returns a minimal `FrameResult`-shaped object."""

    def __init__(self, delay_s: float = 0.0, fail_on: set[int] | None = None):
        self.delay_s = delay_s
        self.fail_on = fail_on or set()
        self.calls: list[np.ndarray] = []

    def analyze(self, frame_bgr, resize=True, growth_ratio=None):
        if self.delay_s:
            time.sleep(self.delay_s)
        call_index = len(self.calls)
        self.calls.append(frame_bgr)
        if call_index in self.fail_on:
            raise RuntimeError(f"synthetic failure on call {call_index}")
        count = float(frame_bgr.sum())
        return SimpleNamespace(inference=SimpleNamespace(count=count, latency_ms=1.0))


def _make_frame(n: int) -> Frame:
    return Frame(
        frame_bgr=np.full((2, 2, 3), n, dtype=np.uint8),
        frame_number=n,
        timestamp_s=float(n),
        width=2,
        height=2,
        source_fps=30.0,
    )


def test_run_until_empty_processes_all_buffered_frames_then_returns():
    buf = LatestFrameBuffer(maxsize=4)
    for i in range(3):
        buf.put(_make_frame(i))
    buf.close()

    pipeline = _FakePipeline()
    results: list[WorkerResult] = []
    worker = InferenceWorker(pipeline=pipeline, buffer=buf, on_result=results.append)

    worker.run_until_empty()

    assert len(results) == 3
    assert [r.frame.frame_number for r in results] == [0, 1, 2]
    assert [r.frames_processed for r in results] == [1, 2, 3]
    assert len(pipeline.calls) == 3


def test_worker_does_not_buffer_more_than_the_bounded_buffer_allows():
    """Feed frames much faster than a deliberately slow pipeline can consume
    them; the buffer must stay bounded (stale frames get dropped) rather
    than the worker/queue accumulating an ever-growing backlog."""
    buf = LatestFrameBuffer(maxsize=1)
    pipeline = _FakePipeline(delay_s=0.02)
    results: list[WorkerResult] = []
    worker = InferenceWorker(pipeline=pipeline, buffer=buf, on_result=results.append)

    worker.start()
    try:
        for i in range(50):
            buf.put(_make_frame(i))
            assert buf.qsize() <= 1
    finally:
        buf.close()
        worker.stop(timeout=5.0)

    assert not worker.is_running
    assert buf.stats.frames_dropped > 0
    assert len(results) < 50  # far fewer results than frames produced
    assert len(results) == len(pipeline.calls)  # every processed frame reached the pipeline


def test_worker_reports_errors_via_error_sink_and_keeps_running():
    buf = LatestFrameBuffer(maxsize=4)
    for i in range(3):
        buf.put(_make_frame(i))
    buf.close()

    pipeline = _FakePipeline(fail_on={1})  # second frame fails
    results: list[WorkerResult] = []
    errors = []
    worker = InferenceWorker(pipeline=pipeline, buffer=buf, on_result=results.append, on_error=errors.append)

    worker.run_until_empty()

    assert len(errors) == 1
    assert errors[0].frame.frame_number == 1
    assert isinstance(errors[0].error, RuntimeError)
    # the worker kept going after the failure and still processed frame 2
    assert [r.frame.frame_number for r in results] == [0, 2]


def test_worker_clean_shutdown_via_stop_without_closing_buffer():
    buf = LatestFrameBuffer(maxsize=1)
    pipeline = _FakePipeline()
    results: list[WorkerResult] = []
    worker = InferenceWorker(pipeline=pipeline, buffer=buf, on_result=results.append, get_timeout_s=0.1)

    worker.start()
    buf.put(_make_frame(0))
    time.sleep(0.2)
    worker.stop(timeout=2.0)

    assert not worker.is_running
    assert not buf.closed  # stop() must not touch the buffer's lifecycle
    assert len(results) == 1


def test_worker_passes_growth_ratio_from_supplied_callable():
    buf = LatestFrameBuffer(maxsize=1)
    buf.put(_make_frame(0))
    buf.close()

    captured = {}

    class _RecordingPipeline(_FakePipeline):
        def analyze(self, frame_bgr, resize=True, growth_ratio=None):
            captured["growth_ratio"] = growth_ratio
            return super().analyze(frame_bgr, resize=resize, growth_ratio=growth_ratio)

    worker = InferenceWorker(
        pipeline=_RecordingPipeline(),
        buffer=buf,
        on_result=lambda r: None,
        growth_ratio_fn=lambda: 0.42,
    )
    worker.run_until_empty()

    assert captured["growth_ratio"] == 0.42


def test_worker_invokes_real_crowd_analysis_pipeline(pipeline, sample_image):
    """`pipeline`/`sample_image` come from `tests/conftest.py`: a real
    `CrowdAnalysisPipeline` (CPU, DM-Count/SHA) and a synthetic RGB-ish
    frame. Confirms the worker drives the actual, unmodified pipeline
    (returns a real `FrameResult`), not a stand-in."""
    from crowd_monitoring.pipeline import FrameResult

    buf = LatestFrameBuffer(maxsize=1)
    buf.put(
        Frame(frame_bgr=sample_image, frame_number=0, timestamp_s=0.0, width=sample_image.shape[1], height=sample_image.shape[0])
    )
    buf.close()

    results: list[WorkerResult] = []
    worker = InferenceWorker(pipeline=pipeline, buffer=buf, on_result=results.append)
    worker.run_until_empty()

    assert len(results) == 1
    assert isinstance(results[0].result, FrameResult)
    assert results[0].result.inference.count >= 0

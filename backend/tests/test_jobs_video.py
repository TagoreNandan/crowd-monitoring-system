"""Tests for the async video job pipeline: POST /api/jobs/video schedules
background processing (jobs/video_runner.py) instead of running inference
inline. Both cv2.VideoCapture and PipelineRuntime.analyze are faked here --
no real model weights are loaded and no real video is decoded, per the
Phase 2A test requirements.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


def _fake_frame_result(count: float) -> SimpleNamespace:
    """A minimal stand-in for `crowdmon.pipeline.FrameResult` carrying only
    the attributes `jobs/video_runner.py` actually reads."""
    return SimpleNamespace(
        inference=SimpleNamespace(count=count, latency_ms=2.0),
        clusters=SimpleNamespace(n_clusters=1, largest_cluster_size=int(count)),
        risk=SimpleNamespace(status="SAFE", score=5.0, components={"relative_density_per_100px2": 0.05}),
    )


class _FakeCapture:
    """Stands in for `cv2.VideoCapture`: yields `n_frames` tiny in-memory
    frames then reports end-of-stream. Touches no filesystem/codec."""

    def __init__(self, n_frames: int):
        self._remaining = n_frames
        self.total_frames = n_frames
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total_frames)
        return 0.0

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _patch_video_pipeline(
    monkeypatch,
    client,
    n_frames: int = 3,
    fail_after: Optional[int] = None,
    analyze_delay_s: float = 0.0,
):
    """Fakes both the OpenCV capture and `PipelineRuntime.analyze` used by
    the app's single shared runtime, so a job runs fast and deterministically.

    `fail_after`: if set, the (fail_after + 1)-th call to `analyze` raises,
    simulating an inference failure mid-video.
    """
    app_state = client.app.state.app_state
    runtime = app_state.pipeline_runtime

    capture_holder: dict[str, _FakeCapture] = {}

    def fake_video_capture(path):
        cap = _FakeCapture(n_frames)
        capture_holder["cap"] = cap
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)

    call_count = {"n": 0}

    async def fake_analyze(frame, growth_ratio=None):
        if analyze_delay_s:
            import asyncio

            await asyncio.sleep(analyze_delay_s)
        call_count["n"] += 1
        if fail_after is not None and call_count["n"] > fail_after:
            raise RuntimeError("synthetic inference failure")
        return _fake_frame_result(count=float(call_count["n"]))

    monkeypatch.setattr(runtime, "analyze", fake_analyze)
    return capture_holder


def _patch_person_detector(monkeypatch, client, boxes_per_frame: int):
    """Fakes the app's shared `PersonDetector` so `visible_count` is a fixed,
    known number distinct from whatever `_fake_frame_result`'s DM-Count
    `estimated_count` reports for that frame."""
    from crowdmon.detection import PersonDetection

    detector = client.app.state.app_state.person_detector
    # `is_available` is a read-only property on the class -- patch it there
    # (monkeypatch reverts automatically), not as an instance attribute.
    monkeypatch.setattr(type(detector), "is_available", True)

    def fake_detect(frame):
        return [PersonDetection(x1=0, y1=0, x2=5, y2=5, confidence=0.99) for _ in range(boxes_per_frame)]

    monkeypatch.setattr(detector, "detect", fake_detect)


def _wait_for_status(client, job_id: str, terminal_statuses, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/api/jobs/{job_id}").json()
        if last["status"] in terminal_statuses:
            return last
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {terminal_statuses} in time (last={last})")


def test_create_video_job_returns_202_with_pending_or_running_job(monkeypatch, client):
    _patch_video_pipeline(monkeypatch, client, n_frames=3)
    files = {"video": ("clip.mp4", b"fake video bytes for interface testing", "video/mp4")}
    resp = client.post("/api/jobs/video", files=files)

    assert resp.status_code == 202
    data = resp.json()
    assert data["job_type"] == "video"
    assert data["status"] in ("pending", "running")
    assert data["video_result"] is None

    lookup = client.get(f"/api/jobs/{data['job_id']}")
    assert lookup.status_code == 200
    assert lookup.json()["job_id"] == data["job_id"]

    _wait_for_status(client, data["job_id"], {"completed", "failed"})


def test_video_job_runner_completes_with_progress_and_summary(monkeypatch, client):
    capture_holder = _patch_video_pipeline(monkeypatch, client, n_frames=4)
    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]

    final = _wait_for_status(client, job_id, {"completed", "failed"})

    assert final["status"] == "completed"
    assert final["progress"] == 1.0
    assert final["current_frame"] == 4
    assert final["total_frames"] == 4
    assert final["peak_count"] == 4.0
    assert final["started_at"] is not None
    assert final["completed_at"] is not None

    summary = final["video_result"]
    assert summary["frames_processed"] == 4
    assert summary["total_frames"] == 4
    assert summary["final_metrics"]["frame_number"] == 4
    assert summary["final_metrics"]["risk_status"] == "SAFE"

    # progress was stored incrementally, not just at completion
    assert final["latest_metrics"]["frame_number"] == 4

    # VideoCapture is always released once processing finishes
    assert capture_holder["cap"].released is True


def test_video_job_visible_count_stays_distinct_from_estimated_count(monkeypatch, client):
    """visible_count (person detector boxes) and estimated_count (DM-Count)
    must never be conflated -- here they're deliberately different numbers
    end to end, through JobStore and the job.completed WebSocket event."""
    _patch_video_pipeline(monkeypatch, client, n_frames=3)
    _patch_person_detector(monkeypatch, client, boxes_per_frame=7)
    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]

    final = _wait_for_status(client, job_id, {"completed", "failed"})

    assert final["status"] == "completed"
    final_metrics = final["video_result"]["final_metrics"]
    assert final_metrics["visible_count"] == 7
    assert final_metrics["estimated_count"] == 3.0  # frame 3 of `_fake_frame_result`
    assert final_metrics["visible_count"] != final_metrics["estimated_count"]

    # Same distinction must be visible in incrementally-stored progress too.
    assert final["latest_metrics"]["visible_count"] == 7
    assert final["latest_metrics"]["estimated_count"] == 3.0


def test_video_job_visible_count_is_null_when_detector_unavailable(monkeypatch, client):
    """Simulates an unavailable PersonDetector -> visible_count must be
    reported as `null` (unavailable), never a fabricated 0, while
    estimated_count (DM-Count) is unaffected."""
    detector = client.app.state.app_state.person_detector
    monkeypatch.setattr(type(detector), "is_available", False)

    _patch_video_pipeline(monkeypatch, client, n_frames=2)
    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]

    final = _wait_for_status(client, job_id, {"completed", "failed"})

    assert final["status"] == "completed"
    assert final["video_result"]["final_metrics"]["visible_count"] is None
    assert final["video_result"]["final_metrics"]["estimated_count"] == 2.0


def test_video_job_temp_file_removed_after_completion(monkeypatch, client):
    _patch_video_pipeline(monkeypatch, client, n_frames=2)
    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]

    final = _wait_for_status(client, job_id, {"completed", "failed"})
    assert final["status"] == "completed"

    record = client.app.state.app_state.job_store.get(job_id)
    assert record.temp_path is not None
    assert not record.temp_path.exists()


def test_video_job_failure_marks_failed_and_cleans_up(monkeypatch, client):
    capture_holder = _patch_video_pipeline(monkeypatch, client, n_frames=5, fail_after=1)
    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]

    final = _wait_for_status(client, job_id, {"completed", "failed"})

    assert final["status"] == "failed"
    assert final["error"] is not None
    assert "synthetic inference failure" in final["error"]
    assert final["video_result"] is None

    assert capture_holder["cap"].released is True
    record = client.app.state.app_state.job_store.get(job_id)
    assert not record.temp_path.exists()


def test_video_job_unopenable_video_marks_failed(monkeypatch, client):
    app_state = client.app.state.app_state

    class _UnopenableCapture:
        def isOpened(self):
            return False

        def get(self, prop_id):
            return 0.0

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: _UnopenableCapture())

    async def fake_analyze(frame, growth_ratio=None):
        raise AssertionError("analyze should never be called for an unopenable video")

    monkeypatch.setattr(app_state.pipeline_runtime, "analyze", fake_analyze)

    files = {"video": ("bad.mp4", b"not really a video", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]

    final = _wait_for_status(client, job_id, {"completed", "failed"})
    assert final["status"] == "failed"
    assert "Could not open" in final["error"]


def test_video_job_websocket_receives_progress_and_completed(monkeypatch, client):
    # `job.started` fires the instant the background task gets scheduled,
    # which can race ahead of this test's own `websocket_connect()` call
    # (pub/sub has no backlog for late joiners, same as `/ws/system`'s
    # heartbeat). The 50ms per-frame delay guarantees the *later* events --
    # job.progress and job.completed -- are still pending when we connect,
    # which is what this test asserts. `job.started`'s envelope/order is
    # covered deterministically by
    # `test_run_video_job_emits_started_progress_completed_in_order` below.
    _patch_video_pipeline(monkeypatch, client, n_frames=3, analyze_delay_s=0.05)
    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]

    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        ack = ws.receive_json()
        assert ack["type"] == "connection.ack"
        assert ack["job_id"] == job_id

        initial_status = ws.receive_json()
        assert initial_status["type"] == "job.status"

        seen_types = []
        completed_envelope = None
        for _ in range(12):
            msg = ws.receive_json()
            seen_types.append(msg["type"])
            assert msg["job_id"] == job_id
            if msg["type"] == "job.completed":
                completed_envelope = msg
                break

        assert "job.progress" in seen_types
        assert "job.completed" in seen_types
        assert completed_envelope is not None
        assert completed_envelope["data"]["frames_processed"] == 3
        assert completed_envelope["data"]["final_metrics"]["frame_number"] == 3


class _RecordingConnectionManager:
    """Fake `ConnectionManager` that just records every broadcast envelope,
    used to assert the exact job.* event order/content deterministically --
    no real socket, no scheduling race."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    @staticmethod
    def job_channel(job_id: str) -> str:
        return f"job:{job_id}"

    async def broadcast(self, channel: str, envelope) -> None:
        self.events.append((envelope.type.value, envelope.data))


def test_run_video_job_emits_started_progress_completed_in_order(monkeypatch, client):
    import asyncio

    from backend.app.jobs.enums import JobType
    from backend.app.jobs.store import JobStore
    from backend.app.jobs.video_runner import run_video_job

    app_state = client.app.state.app_state
    runtime = app_state.pipeline_runtime
    _patch_video_pipeline(monkeypatch, client, n_frames=3)

    job_store = JobStore()
    job = job_store.create(job_type=JobType.VIDEO, source_description="clip.mp4", temp_path=Path("/tmp/unused.mp4"))
    conn_manager = _RecordingConnectionManager()

    asyncio.run(
        run_video_job(job.job_id, job.temp_path, runtime, job_store, conn_manager, app_state.person_detector)
    )

    record = job_store.get(job.job_id)
    assert record.status.value == "completed"

    event_types = [event_type for event_type, _ in conn_manager.events]
    assert event_types[0] == "job.started"
    assert event_types[-1] == "job.completed"
    assert "job.progress" in event_types

    started_data = conn_manager.events[0][1]
    assert started_data["job_type"] == "video"
    assert started_data["source_description"] == "clip.mp4"

    completed_data = conn_manager.events[-1][1]
    assert completed_data["frames_processed"] == 3


def test_video_job_websocket_receives_job_error_on_failure(monkeypatch, client):
    _patch_video_pipeline(monkeypatch, client, n_frames=5, fail_after=0, analyze_delay_s=0.05)
    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]

    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        ws.receive_json()  # ack
        ws.receive_json()  # initial job.status

        seen_types = []
        error_envelope = None
        for _ in range(12):
            msg = ws.receive_json()
            seen_types.append(msg["type"])
            if msg["type"] == "job.error":
                error_envelope = msg
                break

        assert "job.error" in seen_types
        assert error_envelope["job_id"] == job_id
        assert error_envelope["data"]["code"] == "video_processing_failed"
        assert "synthetic inference failure" in error_envelope["data"]["message"]


def test_create_video_job_empty_upload_returns_400(client):
    files = {"video": ("empty.mp4", b"", "video/mp4")}
    resp = client.post("/api/jobs/video", files=files)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_upload"

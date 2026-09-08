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
    """A minimal stand-in for `crowd_monitoring.pipeline.FrameResult` carrying only
    the attributes `jobs/video_runner.py` actually reads -- including a
    tiny real `density_map` array, since the heatmap/combined video writers
    draw directly from it (see `draw_density_heatmap_overlay`)."""
    return SimpleNamespace(
        inference=SimpleNamespace(count=count, latency_ms=2.0, density_map=np.zeros((4, 4), dtype=np.float32)),
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

    Also forces the DM-Count density-pass cadence to "every frame": the fake
    capture below reports no FPS (matching a real backend that can't read
    one), which would otherwise make `run_video_job` fall back to its
    every-15-frames cadence and only call `analyze` once for these
    short (2-5 frame) fake videos. Cadence throttling itself is covered by
    `test_density_cadence_throttles_heavy_analyze_calls` below; the rest of
    this module is testing per-frame data flow (visible vs. estimated counts,
    peak tracking, failure/progress semantics), which is clearest when every
    frame really does call `analyze` once, like the pre-cadence behavior did.
    """
    import backend.app.jobs.video_runner as video_runner_module

    monkeypatch.setattr(video_runner_module, "_density_cadence_frames", lambda *a, **kw: 1)
    # Same reasoning for the detector cadence (added later): the fake capture
    # reports no FPS, so it would otherwise fall back to every-2-frames and
    # only detect on half these short fake videos.
    monkeypatch.setattr(video_runner_module, "_detector_cadence_frames", lambda *a, **kw: 1)

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
    from crowd_monitoring.detection import PersonDetection

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


def test_video_job_annotated_video_retained_and_downloadable(monkeypatch, client):
    """Unlike the original upload (always deleted once processing finishes),
    the annotated (boxes + stats panel) output video must survive job
    completion and be servable -- otherwise the browser has nothing real to
    play back."""
    _patch_video_pipeline(monkeypatch, client, n_frames=2)
    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]

    final = _wait_for_status(client, job_id, {"completed", "failed"})
    assert final["status"] == "completed"

    record = client.app.state.app_state.job_store.get(job_id)
    assert record.annotated_video_path is not None
    assert record.annotated_video_path.exists()

    video_resp = client.get(f"/api/jobs/{job_id}/video")
    assert video_resp.status_code == 200
    assert video_resp.headers["content-type"] == "video/mp4"
    assert len(video_resp.content) > 0


def test_video_job_heatmap_and_combined_videos_retained_and_downloadable(monkeypatch, client):
    """The Heatmap and Combined video views (added alongside the plain
    detection view) must be real, separately-servable output videos built
    in the same processing pass -- not a placeholder, not a duplicate of
    the detection video."""
    _patch_video_pipeline(monkeypatch, client, n_frames=2)
    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]

    final = _wait_for_status(client, job_id, {"completed", "failed"})
    assert final["status"] == "completed"

    record = client.app.state.app_state.job_store.get(job_id)
    assert record.heatmap_video_path is not None and record.heatmap_video_path.exists()
    assert record.combined_video_path is not None and record.combined_video_path.exists()

    for view in ("detection", "heatmap", "combined"):
        resp = client.get(f"/api/jobs/{job_id}/video/{view}")
        assert resp.status_code == 200, view
        assert resp.headers["content-type"] == "video/mp4"
        assert len(resp.content) > 0

    resp = client.get(f"/api/jobs/{job_id}/video/not-a-real-view")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "video_not_available"


def test_video_job_annotated_video_cleaned_up_on_failure(monkeypatch, client):
    _patch_video_pipeline(monkeypatch, client, n_frames=5, fail_after=1)
    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]

    final = _wait_for_status(client, job_id, {"completed", "failed"})
    assert final["status"] == "failed"

    record = client.app.state.app_state.job_store.get(job_id)
    assert record.annotated_video_path is None

    video_resp = client.get(f"/api/jobs/{job_id}/video")
    assert video_resp.status_code == 404
    assert video_resp.json()["error"]["code"] == "video_not_available"


def test_get_job_video_404_before_completion(client):
    resp = client.get("/api/jobs/does-not-exist/video")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "job_not_found"


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


def test_density_cadence_frames_is_time_based_not_frame_count_based():
    """The whole point of the cadence fix: a 60fps source and a 30fps source
    both get the expensive DM-Count pass refreshed at roughly the same
    real-world rate, not the same frame count."""
    from backend.app.jobs.video_runner import _density_cadence_frames

    assert _density_cadence_frames(source_fps=60.0, density_analysis_hz=2.0) == 30
    assert _density_cadence_frames(source_fps=30.0, density_analysis_hz=2.0) == 15
    assert _density_cadence_frames(source_fps=None, density_analysis_hz=2.0) > 0  # falls back, never crashes/zero
    assert _density_cadence_frames(source_fps=60.0, density_analysis_hz=0) > 0  # never zero/negative


def test_detector_cadence_frames_is_time_based_not_frame_count_based():
    """Same idea as density cadence, for the person detector: a 60fps and a
    120fps source both get the detector refreshed at roughly the same
    real-world rate, not the same frame count."""
    from backend.app.jobs.video_runner import _detector_cadence_frames

    assert _detector_cadence_frames(source_fps=60.0, video_detection_hz=20.0) == 3
    assert _detector_cadence_frames(source_fps=120.0, video_detection_hz=20.0) == 6
    assert _detector_cadence_frames(source_fps=None, video_detection_hz=20.0) > 0  # falls back, never crashes/zero
    assert _detector_cadence_frames(source_fps=60.0, video_detection_hz=0) > 0  # never zero/negative
    # A source slower than the target detection rate just gets detected
    # every frame (cadence 1), never "skipped into oblivion".
    assert _detector_cadence_frames(source_fps=10.0, video_detection_hz=20.0) == 1


def test_density_and_detector_cadences_both_throttle_heavy_calls(monkeypatch, client):
    """This is the actual performance fix under test: for a source that
    reports a real FPS, both `PipelineRuntime.analyze` (DM-Count, ~1/sec)
    AND the person detector (~20/sec) must be called far less often than
    once per frame -- neither pretends every one of a 30fps source's frames
    was freshly analyzed."""
    app_state = client.app.state.app_state
    runtime = app_state.pipeline_runtime

    n_frames = 30

    class _FakeCaptureWithFps(_FakeCapture):
        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FPS:
                return 30.0
            return super().get(prop_id)

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: _FakeCaptureWithFps(n_frames))

    analyze_calls = {"n": 0}

    async def fake_analyze(frame, growth_ratio=None):
        analyze_calls["n"] += 1
        return _fake_frame_result(count=float(analyze_calls["n"]))

    monkeypatch.setattr(runtime, "analyze", fake_analyze)

    detect_calls = {"n": 0}
    _patch_person_detector(monkeypatch, client, boxes_per_frame=1)
    detector = app_state.person_detector
    original_detect = detector.detect

    def counting_detect(frame):
        detect_calls["n"] += 1
        return original_detect(frame)

    monkeypatch.setattr(detector, "detect", counting_detect)

    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]
    final = _wait_for_status(client, job_id, {"completed", "failed"})

    assert final["status"] == "completed"
    assert final["current_frame"] == n_frames
    # Detector: ~20/sec of a 30fps, 1s source -> cadence 2 frames -> ~15
    # real detector calls, not 30. Still far more often than density (below),
    # and every processed frame still gets a rendered/written overlay frame
    # (see the video_overlay sync test) -- it just doesn't always come from
    # a *fresh* detection.
    assert 10 <= detect_calls["n"] <= 20
    assert detect_calls["n"] < n_frames
    summary = final["video_result"]
    assert summary["frames_detected"] == detect_calls["n"]
    # DM-Count: ~1/sec of a 30fps, 1s source -> cadence 30 frames -> ~1 real
    # analyze call, dramatically less than once per frame.
    assert 1 <= analyze_calls["n"] <= 2
    assert analyze_calls["n"] < n_frames
    assert summary["density_updates"] == analyze_calls["n"]


def test_video_overlay_uses_same_video_time_and_metrics_as_stored_result(monkeypatch, client):
    """Regression test for a real desync bug: the burned-in overlay's "Time"
    line and every other overlay value must come from the exact same
    `VideoFrameMetrics` object stored/broadcast for that frame -- never
    independently computed. Spies on `render_video_overlay` to prove every
    kwarg it receives for the last processed frame equals the corresponding
    field on the job's final `latest_metrics`, and that `video_time_s` is
    real frame_number/fps math, not wall-clock elapsed time (which has no
    fixed relationship to the video's own duration)."""
    import backend.app.jobs.video_runner as video_runner_module

    monkeypatch.setattr(video_runner_module, "_density_cadence_frames", lambda *a, **kw: 1)

    class _FakeCaptureWithFps(_FakeCapture):
        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FPS:
                return 30.0
            return super().get(prop_id)

    n_frames = 5
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: _FakeCaptureWithFps(n_frames))

    app_state = client.app.state.app_state
    runtime = app_state.pipeline_runtime

    async def fake_analyze(frame, growth_ratio=None):
        return _fake_frame_result(count=42.0)

    monkeypatch.setattr(runtime, "analyze", fake_analyze)

    captured_calls: list[dict] = []
    original_render = video_runner_module.render_video_overlay

    def spy_render(frame_bgr, detections, **kwargs):
        captured_calls.append(kwargs)
        return original_render(frame_bgr, detections, **kwargs)

    monkeypatch.setattr(video_runner_module, "render_video_overlay", spy_render)

    files = {"video": ("clip.mp4", b"fake video bytes", "video/mp4")}
    job_id = client.post("/api/jobs/video", files=files).json()["job_id"]
    final = _wait_for_status(client, job_id, {"completed", "failed"})

    assert final["status"] == "completed"
    # `render_video_overlay` is called twice per frame -- once for the
    # detection-only output, once for the heatmap+boxes "combined" output
    # (see jobs/video_runner.py) -- both built from the exact same
    # `latest_metrics` for that frame, so every captured call (not just one
    # per frame) must satisfy the same invariants.
    assert len(captured_calls) == n_frames * 2

    for call_kwargs in captured_calls:
        i = call_kwargs["frame_number"]
        assert 1 <= i <= n_frames
        # Real frame_number/fps math -- not wall-clock processing time.
        # `FileVideoSource` numbers decoded frames 0-based internally (the
        # first frame's own video position is 0.0s), while `frame_number`
        # here (== frames_processed) is the 1-based count of frames handled
        # so far -- hence `(i - 1)`, not `i`.
        # `_build_metrics` rounds to 3 decimal places for the stored/broadcast
        # value, so compare with a tolerance matching that rounding.
        assert call_kwargs["video_time_s"] == pytest.approx((i - 1) / 30.0, abs=1e-3)

    # The two calls for the SAME frame must be identical (both sourced from
    # the one `latest_metrics` object, never independently computed).
    last_frame_calls = [c for c in captured_calls if c["frame_number"] == n_frames]
    assert len(last_frame_calls) == 2
    assert last_frame_calls[0] == last_frame_calls[1]

    last_call = last_frame_calls[-1]
    stored_metrics = final["latest_metrics"]
    shared_fields = [
        "frame_number", "video_time_s", "visible_count", "estimated_count",
        "peak_count", "fps", "latency_ms", "density", "cluster_count",
        "largest_cluster", "risk_status", "risk_score",
    ]
    for key in shared_fields:
        assert last_call[key] == stored_metrics[key], f"overlay/{key} desynced from stored metrics"

from crowdmon.system_status import StatusTracker


def test_offline_before_model_loaded():
    tracker = StatusTracker()
    assert tracker.snapshot().state() == "OFFLINE"


def test_online_after_model_and_inference():
    tracker = StatusTracker()
    tracker.model_ready("DM-Count", "SHA", "cpu")
    tracker.record_inference(latency_ms=50.0)
    snap = tracker.snapshot()
    assert snap.model_loaded is True
    assert snap.state() == "ONLINE"
    assert snap.frames_processed == 1


def test_degraded_after_error():
    tracker = StatusTracker()
    tracker.model_ready("DM-Count", "SHA", "cpu")
    tracker.record_inference(latency_ms=50.0)
    tracker.record_error("simulated failure")
    snap = tracker.snapshot()
    assert snap.error_count == 1
    assert snap.state() == "DEGRADED"


def test_fps_computed_over_multiple_inferences():
    tracker = StatusTracker()
    tracker.model_ready("DM-Count", "SHA", "cpu")
    for _ in range(5):
        tracker.record_inference(latency_ms=10.0)
    snap = tracker.snapshot()
    assert snap.fps >= 0

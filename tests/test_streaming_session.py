import time
from types import SimpleNamespace
import numpy as np
import pytest

from crowd_monitoring.streaming import FileVideoSource, StreamSession


class _FakePipeline:
    def __init__(self):
        self.calls = []

    def analyze(self, frame_bgr, resize=True, growth_ratio=None):
        self.calls.append(frame_bgr)
        count = float(frame_bgr.sum())
        return SimpleNamespace(
            inference=SimpleNamespace(count=count, latency_ms=1.0),
            localization=SimpleNamespace(points_xy=[]),
            clusters=SimpleNamespace(n_clusters=0, largest_cluster_size=0, average_cluster_size=0.0),
            risk=SimpleNamespace(status="SAFE", score=0.0, components={}),
        )


@pytest.fixture
def dummy_video(tmp_path):
    import cv2
    path = tmp_path / "stream_clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    rng = np.random.default_rng(42)
    for _ in range(5):
        frame = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_stream_session_runs_and_completes(dummy_video):
    source = FileVideoSource(dummy_video)
    pipeline = _FakePipeline()
    results = []

    session = StreamSession(
        session_id="test_sess_1",
        source=source,
        pipeline=pipeline,
        on_result=results.append,
    )

    metadata = session.start()
    assert metadata.total_frames == 5
    assert session.is_running or session.status in ("running", "completed")

    # Give producer and worker time to process frames
    time.sleep(0.5)
    session.stop(timeout=2.0)

    assert session.status in ("completed", "stopped")
    assert len(results) > 0
    assert session.latest_result is not None

"""RealtimeFeedSession frame skipping: the detector runs on every Nth source
frame, and the frames in between are still delivered with held boxes."""

import threading
import time

import numpy as np

from crowd_monitoring.detection import PersonDetection
from crowd_monitoring.streaming.frame_source import Frame, FrameSource, SourceMetadata
from crowd_monitoring.streaming.realtime import RealtimeFeedSession


class _ListSource(FrameSource):
    def __init__(self, count: int):
        self._count = count

    def open(self) -> SourceMetadata:
        return SourceMetadata(description="list", width=64, height=48, fps=None, total_frames=self._count)

    def frames(self):
        for i in range(self._count):
            # Slow enough that the fast loop sees every frame.
            time.sleep(0.01)
            yield Frame(frame_bgr=np.zeros((48, 64, 3), np.uint8), frame_number=i, timestamp_s=i / 30, width=64, height=48)

    def close(self) -> None:
        pass


class _CountingDetector:
    is_available = True

    def __init__(self):
        self.calls = 0

    def detect(self, frame_bgr):
        self.calls += 1
        return [PersonDetection(x1=5, y1=5, x2=20, y2=40, confidence=0.9)]


class _NoopPipeline:
    def analyze(self, frame_bgr):
        raise RuntimeError("density not under test")


def _run(stride: int, count: int = 30):
    detector = _CountingDetector()
    results = []
    done = threading.Event()

    def on_fast(result):
        results.append(result)
        if result.frame.frame_number == count - 1:
            done.set()

    session = RealtimeFeedSession(
        "stride", _ListSource(count), _NoopPipeline(), detector,
        on_fast_result=on_fast, loop_source=False, ego_motion_compensation=False,
        detection_stride=stride, density_hz=0.1,
    )
    session.start()
    assert done.wait(5.0)
    session.stop()
    return detector, results


def test_stride_runs_detector_on_every_third_frame_but_delivers_all():
    detector, results = _run(stride=3)
    assert len(results) == 30
    detected = [r.frame.frame_number for r in results if r.detected]
    assert detected == list(range(0, 30, 3))
    assert detector.calls == 10
    held = [r for r in results if not r.detected]
    assert held and all(r.visible_count == 1 and len(r.detections) == 1 for r in held)


def test_stride_one_detects_every_frame():
    detector, results = _run(stride=1, count=12)
    assert detector.calls == 12
    assert all(r.detected for r in results)


def test_stride_detects_immediately_after_source_rewind():
    session = RealtimeFeedSession("s", _ListSource(1), _NoopPipeline(), _CountingDetector(), detection_stride=4)
    assert session._should_detect(0, None)
    assert not session._should_detect(102, 100)
    assert session._should_detect(104, 100)
    assert session._should_detect(2, 100)  # looped back to the start

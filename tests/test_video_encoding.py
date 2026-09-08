import cv2
import numpy as np
import pytest

from crowd_monitoring.video_encoding import ensure_browser_compatible_video, is_ffmpeg_available


@pytest.fixture
def raw_mp4(tmp_path):
    path = tmp_path / "raw.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    rng = np.random.default_rng(7)
    for _ in range(3):
        frame = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_ensure_browser_compatible_video(raw_mp4, tmp_path):
    out_path = tmp_path / "out_browser.mp4"
    result = ensure_browser_compatible_video(raw_mp4, out_path)
    assert result.exists()
    assert result.stat().st_size > 0

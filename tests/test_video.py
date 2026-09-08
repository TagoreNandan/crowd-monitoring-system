import cv2
import numpy as np
import pytest

from crowd_monitoring.video import process_source


@pytest.fixture
def image_dir(tmp_path):
    rng = np.random.default_rng(4)
    for i in range(3):
        img = (rng.random((64, 96, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{i}.jpg"), img)
    return tmp_path


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (96, 64))
    rng = np.random.default_rng(5)
    for _ in range(5):
        frame = (rng.random((64, 96, 3)) * 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_process_image_directory(pipeline, image_dir):
    frames = list(process_source(pipeline, str(image_dir)))
    assert len(frames) == 3
    for pf in frames:
        assert pf.result.inference.count >= 0
    # peak count should track the max seen so far
    assert frames[-1].state.peak_count == max(f.result.inference.count for f in frames)


def test_process_video_file(pipeline, video_file):
    frames = list(process_source(pipeline, str(video_file)))
    assert len(frames) == 5
    assert all(pf.result.inference.count >= 0 for pf in frames)


def test_frame_skip_reduces_analyzed_frames(pipeline, video_file):
    all_frames = list(process_source(pipeline, str(video_file), frame_skip=0))
    skipped = list(process_source(pipeline, str(video_file), frame_skip=1))
    assert len(skipped) < len(all_frames)


def test_max_frames_limit(pipeline, image_dir):
    frames = list(process_source(pipeline, str(image_dir), max_frames=2))
    assert len(frames) == 2


def test_missing_source_raises(pipeline):
    with pytest.raises(FileNotFoundError):
        list(process_source(pipeline, "/no/such/path/at/all"))

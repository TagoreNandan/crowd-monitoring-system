import inspect

import cv2
import numpy as np
import pytest

from crowdmon.streaming import FileVideoSource


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (96, 64))
    rng = np.random.default_rng(7)
    for _ in range(6):
        frame = (rng.random((64, 96, 3)) * 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_open_reports_metadata(video_file):
    source = FileVideoSource(video_file)
    try:
        metadata = source.open()
        assert metadata.total_frames == 6
        assert metadata.width == 96
        assert metadata.height == 64
        assert metadata.fps is not None and metadata.fps > 0
        assert metadata.is_live is False
        assert "clip.mp4" in metadata.description
    finally:
        source.close()


def test_frames_yields_incrementally_without_buffering_whole_file(video_file):
    """frames() must be a generator, not something that eagerly decodes and
    returns a list -- that's what keeps memory bounded for large videos."""
    source = FileVideoSource(video_file)
    source.open()
    try:
        gen = source.frames()
        assert inspect.isgenerator(gen)
        first = next(gen)
        assert first.frame_number == 0
        assert first.width == 96
        assert first.height == 64
        assert first.frame_bgr.shape == (64, 96, 3)
    finally:
        source.close()


def test_frame_iteration_covers_all_frames_with_sequential_numbers(video_file):
    source = FileVideoSource(video_file)
    source.open()
    try:
        frames = list(source.frames())
    finally:
        source.close()

    assert len(frames) == 6
    assert [f.frame_number for f in frames] == list(range(6))
    assert all(f.source_fps == frames[0].source_fps for f in frames)
    # timestamps are non-decreasing across the stream
    assert all(b.timestamp_s >= a.timestamp_s for a, b in zip(frames, frames[1:]))


def test_frames_before_open_raises(video_file):
    source = FileVideoSource(video_file)
    with pytest.raises(RuntimeError):
        list(source.frames())


def test_open_missing_file_raises(tmp_path):
    source = FileVideoSource(tmp_path / "does_not_exist.mp4")
    with pytest.raises(RuntimeError, match="Could not open"):
        source.open()


def test_close_is_idempotent_and_releases(video_file):
    source = FileVideoSource(video_file)
    source.open()
    source.close()
    source.close()  # must not raise


def test_context_manager_opens_and_closes(video_file):
    with FileVideoSource(video_file) as source:
        frames = list(source.frames())
    assert len(frames) == 6

"""Video / webcam / image-directory processing loop.

Wraps CrowdAnalysisPipeline and adds the state that only makes sense across
multiple frames: count history, peak count, and a rolling growth-rate used
by the risk engine's "trend" signal. Everything updates automatically as
frames arrive -- callers don't need to manage counters by hand.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .pipeline import CrowdAnalysisPipeline, FrameResult
from .system_status import GLOBAL_STATUS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class StreamState:
    frame_index: int = 0
    peak_count: float = 0.0
    peak_frame_index: int = 0
    count_history: deque = field(default_factory=lambda: deque(maxlen=600))  # (timestamp, count)
    fps_estimate: float = 0.0

    def update(self, count: float):
        now = time.time()
        self.count_history.append((now, count))
        if count > self.peak_count:
            self.peak_count = count
            self.peak_frame_index = self.frame_index

    def growth_ratio(self, window_s: float = 5.0) -> Optional[float]:
        if len(self.count_history) < 2:
            return None
        now = self.count_history[-1][0]
        window = [c for t, c in self.count_history if now - t <= window_s]
        if len(window) < 2 or window[0] <= 1e-6:
            return None
        return (window[-1] - window[0]) / window[0]


def _open_source(source):
    """source: video file path, webcam index (int or "0"), or a directory of images."""
    if isinstance(source, (int,)) or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {source}")
        return ("stream", cap, f"webcam:{source}")

    path = Path(source)
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise RuntimeError(f"No images found in directory {path}")
        return ("images", images, f"directory:{path}")

    if path.is_file():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file {path}")
        return ("stream", cap, f"video:{path.name}")

    raise FileNotFoundError(f"Source not found: {source}")


@dataclass
class ProcessedFrame:
    frame_bgr: np.ndarray
    result: FrameResult
    state: StreamState


def process_source(
    pipeline: CrowdAnalysisPipeline,
    source,
    frame_skip: int = 0,
    max_frames: Optional[int] = None,
    resize: bool = True,
) -> Iterator[ProcessedFrame]:
    """Yield a ProcessedFrame for each frame that gets analyzed.

    frame_skip: analyze every (frame_skip + 1)-th frame; skipped frames are
        not run through the model (keeps live video responsive on CPU).
    """
    kind, handle, description = _open_source(source)
    GLOBAL_STATUS.source_ready(description)
    state = StreamState()

    try:
        if kind == "images":
            for img_path in handle:
                if max_frames is not None and state.frame_index >= max_frames:
                    break
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                result = pipeline.analyze(frame, resize=resize, growth_ratio=state.growth_ratio())
                state.update(result.inference.count)
                yield ProcessedFrame(frame_bgr=frame, result=result, state=state)
                state.frame_index += 1
        else:
            cap = handle
            skip_counter = 0
            while True:
                if max_frames is not None and state.frame_index >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                if skip_counter > 0:
                    skip_counter -= 1
                    state.frame_index += 1
                    continue
                skip_counter = frame_skip

                result = pipeline.analyze(frame, resize=resize, growth_ratio=state.growth_ratio())
                state.update(result.inference.count)
                yield ProcessedFrame(frame_bgr=frame, result=result, state=state)
                state.frame_index += 1
            cap.release()
    finally:
        GLOBAL_STATUS.source_closed()

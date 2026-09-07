"""Streaming/source layer: frame sources, bounded latest-frame-wins
buffer, inference worker, and stream session orchestrator.
"""

from __future__ import annotations

from .buffer import BufferStats, LatestFrameBuffer
from .frame_source import CameraSource, FileVideoSource, Frame, FrameSource, RTSPSource, SourceMetadata
from .session import StreamSession
from .worker import ErrorSink, EventSink, InferenceWorker, WorkerError, WorkerResult

__all__ = [
    "BufferStats",
    "LatestFrameBuffer",
    "Frame",
    "FileVideoSource",
    "CameraSource",
    "RTSPSource",
    "FrameSource",
    "SourceMetadata",
    "StreamSession",
    "ErrorSink",
    "EventSink",
    "InferenceWorker",
    "WorkerError",
    "WorkerResult",
]

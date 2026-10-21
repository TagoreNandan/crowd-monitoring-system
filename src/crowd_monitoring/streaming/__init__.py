"""Streaming/source layer: frame sources, bounded latest-frame-wins
buffer, inference worker, and stream session orchestrator.
"""

from __future__ import annotations

from .buffer import BufferStats, LatestFrameBuffer
from .failover import (
    SOURCE_STATUS_BACKUP,
    SOURCE_STATUS_LIVE,
    SOURCE_STATUS_SIGNAL_LOST,
    FailoverFrameSource,
)
from .frame_source import (
    CameraSource,
    FileVideoSource,
    Frame,
    FrameSource,
    RTSPSource,
    SourceMetadata,
    is_live_stream_url,
)
from .latest_slot import LatestSlot
from .realtime import DensityResult, FastResult, RealtimeError, RealtimeFeedSession
from .session import StreamSession
from .worker import ErrorSink, EventSink, InferenceWorker, WorkerError, WorkerResult

__all__ = [
    "BufferStats",
    "LatestFrameBuffer",
    "LatestSlot",
    "Frame",
    "FileVideoSource",
    "CameraSource",
    "RTSPSource",
    "FailoverFrameSource",
    "SOURCE_STATUS_LIVE",
    "SOURCE_STATUS_BACKUP",
    "SOURCE_STATUS_SIGNAL_LOST",
    "is_live_stream_url",
    "FrameSource",
    "SourceMetadata",
    "StreamSession",
    "ErrorSink",
    "EventSink",
    "InferenceWorker",
    "WorkerError",
    "WorkerResult",
    "RealtimeFeedSession",
    "FastResult",
    "DensityResult",
    "RealtimeError",
]

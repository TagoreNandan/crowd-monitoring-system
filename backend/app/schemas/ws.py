"""Typed WebSocket message contract.

Every message on every channel uses the same envelope shape:

    {"type": "...", "job_id": "...", "ts": 0, "data": {...}}

`type` selects which `data` schema applies: connection lifecycle events
(ack / status / heartbeat / error), plus the video job lifecycle events
(`job.started` / `job.progress` / `job.completed` / `job.error`) published
by `jobs/video_runner.py`. `job.progress` carries only compact per-frame
metrics (see `VideoFrameMetrics`) -- never a detection array, density map,
or any binary payload; video/image bytes never go over these sockets.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# `job.progress`/`job.completed` reuse the exact same typed metrics/summary
# the REST job-lookup response uses (see `schemas/jobs.py`) -- one shape,
# not a WS-only duplicate of it.
from .jobs import VideoFrameMetrics, VideoJobSummary  # noqa: F401


class WSEventType(str, Enum):
    CONNECTION_ACK = "connection.ack"
    SYSTEM_STATUS = "system.status"
    JOB_STATUS = "job.status"
    JOB_STARTED = "job.started"
    JOB_PROGRESS = "job.progress"
    JOB_COMPLETED = "job.completed"
    JOB_ERROR = "job.error"
    FEED_STATUS = "feed.status"
    FEED_PROGRESS = "feed.progress"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"


class WSEnvelope(BaseModel):
    type: WSEventType
    job_id: Optional[str] = None
    ts: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)


class ConnectionAckData(BaseModel):
    channel: str
    message: str = "connected"


class SystemStatusData(BaseModel):
    """Mirrors `crowdmon.system_status.SystemStatus` -- the same
    real, locally-observed state exposed by GET /api/status."""

    state: str
    model_loaded: bool
    model_name: str
    model_weights: str
    device: str
    fps: float
    frames_processed: int
    error_count: int


class JobStatusData(BaseModel):
    status: str
    progress: float
    message: Optional[str] = None


class JobStartedData(BaseModel):
    """`job.started` payload, sent once when the video job runner begins
    processing (i.e. when the job leaves `pending` for `running`)."""

    job_type: str
    source_description: str = ""


class ErrorData(BaseModel):
    code: str
    message: str

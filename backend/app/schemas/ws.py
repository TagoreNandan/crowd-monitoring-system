"""Typed WebSocket message contract.

Every message on every channel uses the same envelope shape:

    {"type": "...", "job_id": "...", "ts": 0, "data": {...}}

`type` selects which `data` schema applies. This phase defines the
connection lifecycle events (ack / status / heartbeat / error) that later
phases build on -- it does not yet stream per-frame metrics, detection
arrays, or any binary payload.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class WSEventType(str, Enum):
    CONNECTION_ACK = "connection.ack"
    SYSTEM_STATUS = "system.status"
    JOB_STATUS = "job.status"
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


class ErrorData(BaseModel):
    code: str
    message: str

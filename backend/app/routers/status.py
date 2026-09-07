"""GET /api/status -- a typed passthrough of the existing
`crowdmon.system_status.GLOBAL_STATUS` singleton, unchanged. This is the
same real, locally-observed state (model loaded, fps, latency, error count,
ONLINE/DEGRADED/OFFLINE) the Flask dashboard's `/status` route already
exposes -- reused here, not reimplemented.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from crowdmon.system_status import GLOBAL_STATUS

router = APIRouter(prefix="/api", tags=["status"])


class SystemStatusResponse(BaseModel):
    state: str
    model_loaded: bool
    model_name: str
    model_weights: str
    device: str
    source_connected: bool
    source_description: str
    inference_running: bool
    last_inference_ts: Optional[float]
    last_latency_ms: Optional[float]
    fps: float
    frames_processed: int
    error_count: int
    last_error: str


@router.get("/status", response_model=SystemStatusResponse)
async def get_status() -> SystemStatusResponse:
    snapshot = GLOBAL_STATUS.snapshot().to_dict()
    return SystemStatusResponse(**snapshot)

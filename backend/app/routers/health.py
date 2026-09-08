"""GET /api/health -- cheap liveness/readiness check.

Deliberately minimal: "is the process up, and is the model loaded." For the
richer, real-time-observed state (fps, latency, error count, ...) see
GET /api/status, which is a direct passthrough of
`crowd_monitoring.system_status.GLOBAL_STATUS`.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..dependencies import get_pipeline_runtime
from ..pipeline_runtime import PipelineRuntime

router = APIRouter(prefix="/api", tags=["health"])

_PROCESS_STARTED_AT = time.time()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


@router.get("/health", response_model=HealthResponse)
async def get_health(runtime: PipelineRuntime = Depends(get_pipeline_runtime)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=runtime.is_ready,
        uptime_seconds=round(time.time() - _PROCESS_STARTED_AT, 3),
    )

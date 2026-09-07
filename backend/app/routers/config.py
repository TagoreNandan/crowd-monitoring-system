"""GET /api/config -- current pipeline configuration.

Read-only in this phase: there is no PATCH endpoint and no runtime model
hot-swap. Values reflect the pipeline's actual resolved state where
possible (see `PipelineRuntime.describe_config`).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_pipeline_runtime
from ..pipeline_runtime import PipelineRuntime
from ..schemas.config import AppConfigResponse

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config", response_model=AppConfigResponse)
async def get_config(runtime: PipelineRuntime = Depends(get_pipeline_runtime)) -> AppConfigResponse:
    return AppConfigResponse(**runtime.describe_config())

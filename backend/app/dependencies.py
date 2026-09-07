from __future__ import annotations

from fastapi import Request

from .jobs.store import JobStore
from .pipeline_runtime import PipelineRuntime
from .state import AppState
from .ws.manager import ConnectionManager


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state


def get_pipeline_runtime(request: Request) -> PipelineRuntime:
    return get_app_state(request).pipeline_runtime


def get_job_store(request: Request) -> JobStore:
    return get_app_state(request).job_store


def get_connection_manager(request: Request) -> ConnectionManager:
    return get_app_state(request).connection_manager

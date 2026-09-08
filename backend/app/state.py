"""Everything the backend needs at runtime, held off the FastAPI `app`
object as one explicit object (`app.state.app_state`) rather than scattered
module-level globals -- routers and tests reach it through a single
dependency (see `dependencies.py`).
"""

from __future__ import annotations

from typing import Any

from crowd_monitoring.detection import PersonDetector

from .jobs.store import JobStore
from .pipeline_runtime import PipelineRuntime
from .settings import Settings
from .ws.manager import ConnectionManager


class AppState:
    def __init__(self, settings: Settings, pipeline_kwargs: dict[str, Any]):
        self.settings = settings
        self.pipeline_kwargs = pipeline_kwargs
        self.pipeline_runtime = PipelineRuntime(pipeline_kwargs)
        self.job_store = JobStore(ttl_seconds=settings.job_ttl_seconds)
        self.connection_manager = ConnectionManager()
        # Separate, optional model from the DM-Count pipeline above -- see
        # crowd_monitoring.detection. Lazily loads its own YOLOv8n model on first
        # use; never constructed if nothing calls it. Shared across image
        # jobs, video jobs, and real-time feeds so the model is only loaded
        # once per process.
        self.person_detector = PersonDetector(
            confidence_threshold=settings.person_detection_confidence,
            imgsz=settings.person_detection_imgsz,
        )

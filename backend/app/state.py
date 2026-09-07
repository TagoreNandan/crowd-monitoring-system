"""Everything the backend needs at runtime, held off the FastAPI `app`
object as one explicit object (`app.state.app_state`) rather than scattered
module-level globals -- routers and tests reach it through a single
dependency (see `dependencies.py`).
"""

from __future__ import annotations

from typing import Any

from crowdmon.detection import PersonDetector

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
        # crowdmon.detection. Lazily loads its own Caffe model on first use;
        # never constructed if the video job runner never calls it.
        self.person_detector = PersonDetector(confidence_threshold=settings.person_detection_confidence)

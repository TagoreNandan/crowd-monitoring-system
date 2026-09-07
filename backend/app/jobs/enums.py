from __future__ import annotations

from enum import Enum


class JobType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
